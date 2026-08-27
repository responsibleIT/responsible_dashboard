"""
Generative model perplexity prediction using the exported GP surrogate.

Uses 2-anchor local interpolation: evaluates the real model at thresholds
0.0 and 10.0, then predicts perplexity for all intermediate thresholds
via a trained Gaussian Process.  FLOPs are computed directly by pruning.
"""

import json
import math
from pathlib import Path

import numpy as np
import joblib

_DIR = Path(__file__).resolve().parent

# HuggingFace repo mapping for preset models
PRESET_MODEL_REPOS = {
    "gpt2-xl": "openai-community/gpt2-xl",
    "bloom-1.7b": "bigscience/bloom-1b7",
    "opt-1.3b": "facebook/opt-1.3b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
}

# Lazy-loaded globals
_gp_ppl = None
_scaler_ppl = None
_metadata = None


def _load_artefacts():
    global _gp_ppl, _scaler_ppl, _metadata
    if _gp_ppl is not None:
        return
    _gp_ppl = joblib.load(_DIR / "gp_perplexity.joblib")
    _scaler_ppl = joblib.load(_DIR / "scaler_perplexity.joblib")
    with open(_DIR / "metadata.json", "r", encoding="utf-8") as f:
        _metadata = json.load(f)


def get_anchor_thresholds() -> list[float]:
    """Return the thresholds at which the real model must be evaluated."""
    _load_artefacts()
    return _metadata["anchor_thresholds"]


def get_validation_models() -> list[str]:
    _load_artefacts()
    return _metadata.get("validation_models", [])


def get_test_models() -> list[str]:
    _load_artefacts()
    return _metadata.get("test_models", [])


def get_preset_models() -> list[str]:
    """Return all models that can be selected from the dropdown (val + test)."""
    return get_validation_models() + get_test_models()


def get_preset_repo(name: str) -> str:
    """Map a short preset model name to its HuggingFace repo."""
    return PRESET_MODEL_REPOS.get(name, name)


# ── Pareto / knee helpers (from V3 notebook) ───────────────────────────

def _find_pareto_front(ppl_values, flops_values):
    """Find Pareto-optimal points: minimize perplexity, maximize FLOPs reduction."""
    n = len(ppl_values)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if ppl_values[j] <= ppl_values[i] and flops_values[j] >= flops_values[i]:
                if ppl_values[j] < ppl_values[i] or flops_values[j] > flops_values[i]:
                    is_pareto[i] = False
                    break
    return np.where(is_pareto)[0]


def _find_knee_point(ppl_pareto, flops_pareto):
    """Find knee point via max distance from line connecting endpoints."""
    if len(ppl_pareto) < 3:
        return 0
    ppl_n = (ppl_pareto - ppl_pareto.min()) / (ppl_pareto.max() - ppl_pareto.min() + 1e-10)
    flops_n = (flops_pareto - flops_pareto.min()) / (flops_pareto.max() - flops_pareto.min() + 1e-10)
    p1 = np.array([ppl_n[0], flops_n[0]])
    p2 = np.array([ppl_n[-1], flops_n[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return 0
    line_unit = line_vec / line_len
    dists = [
        np.linalg.norm(
            (np.array([ppl_n[i], flops_n[i]]) - p1)
            - np.dot(np.array([ppl_n[i], flops_n[i]]) - p1, line_unit) * line_unit
        )
        for i in range(len(ppl_n))
    ]
    return int(np.argmax(dists))


# ── Main prediction function ──────────────────────────────────────────

def predict_perplexity_curve(
    anchor_perplexities: dict[float, float],
    flops_at_thresholds: dict[float, float],
    total_params: int,
    total_layers: int,
    original_flops: float,
    thresholds: np.ndarray | None = None,
) -> dict:
    """
    Predict perplexity curve for a new model using the GP surrogate.

    Parameters
    ----------
    anchor_perplexities : dict
        Mapping threshold -> measured perplexity (must include all anchor
        thresholds, i.e. {0.0: ppl_base, 10.0: ppl_at_10}).
    flops_at_thresholds : dict
        Mapping threshold -> flops_reduction_pct (from actual pruning).
    total_params, total_layers, original_flops :
        Architecture metadata of the model.
    thresholds : array, optional
        Thresholds to predict at (default: 0.0 to 10.0, step 0.1).

    Returns
    -------
    dict with keys: thresholds, perplexity, perplexity_std,
    flops_reduction_pct, pareto_indices, knee_threshold,
    knee_perplexity, knee_flops_reduction.
    """
    _load_artefacts()

    if thresholds is None:
        thresholds = np.arange(0.0, 10.05, 0.1)

    sorted_anchors = sorted(_metadata["anchor_thresholds"])

    # Pre-compute anchor log-perplexities
    anchor_log_ppls = {}
    for at in sorted_anchors:
        at_r = round(at, 1)
        anchor_log_ppls[at_r] = math.log(anchor_perplexities[at_r])

    # Build local interpolation features (same as V3 notebook)
    rows = []
    for t in thresholds:
        left_t = round(max(a for a in sorted_anchors if a <= t + 1e-9), 1)
        right_t = round(min(a for a in sorted_anchors if a >= t - 1e-9), 1)
        span = right_t - left_t
        rel_pos = (t - left_t) / span if span > 1e-9 else 0.0
        rows.append([
            t,
            anchor_log_ppls[left_t],
            anchor_log_ppls[right_t],
            span,
            rel_pos,
            float(total_params),
            float(total_layers),
            float(original_flops),
        ])

    X_ppl = np.array(rows)
    X_scaled = _scaler_ppl.transform(X_ppl)
    y_ppl_log, y_ppl_std = _gp_ppl.predict(X_scaled, return_std=True)
    ppl_pred = np.exp(y_ppl_log)

    # Actual FLOPs reduction from pruning
    flops_actual = np.array([
        flops_at_thresholds.get(round(t, 1), 0.0) for t in thresholds
    ])

    # Pareto front
    pareto_idx = _find_pareto_front(ppl_pred, flops_actual)
    pareto_idx = pareto_idx[np.argsort(ppl_pred[pareto_idx])]
    knee_in_pareto = _find_knee_point(ppl_pred[pareto_idx], flops_actual[pareto_idx])
    knee_idx = pareto_idx[knee_in_pareto]

    return {
        "thresholds": thresholds.tolist(),
        "perplexity": ppl_pred.tolist(),
        "perplexity_std": (np.exp(y_ppl_log + y_ppl_std) - ppl_pred).tolist(),
        "flops_reduction_pct": flops_actual.tolist(),
        "pareto_indices": pareto_idx.tolist(),
        "knee_threshold": float(thresholds[knee_idx]),
        "knee_perplexity": float(ppl_pred[knee_idx]),
        "knee_flops_reduction": float(flops_actual[knee_idx]),
    }
