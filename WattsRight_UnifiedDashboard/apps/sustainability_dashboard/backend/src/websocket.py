# apps/sustainability_dashboard/backend/src/websocket.py
import os
import json
import asyncio
import copy
import math
from flask import request
import pandas as pd
import time
import threading
from contextlib import suppress
from flask_socketio import join_room  # <- no plain emit import
from dotenv import load_dotenv

from loading import load_huggingface_model, load_local_model, load_huggingface_generative_model
from preprocess import disable_low_weight_neurons
from pruning import estimate_flops, count_non_zero_params, compute_flops_reduction_sweep
from benchmark import evaluate_model
from predict import predict_with_auto_regressive_model
from utils.gpu_power import sample_gpu_power_background
from model.generative_model.predict_generative import (
    predict_perplexity_curve,
    build_result_from_real_data,
    get_anchor_thresholds,
    get_dropdown_models,
    get_hf_repo,
    is_test_model,
    load_test_model_data,
)
from model.generative_model.benchmark_generative import (
    evaluate_perplexity,
    compute_passage_perplexities,
    compute_token_comparisons,
    generate_completion,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import shutil

import glob

load_dotenv()

DEMO_MODE = "false"
UPLOAD_DIR = "uploads"
THRESHOLDS = [i * 0.1 for i in range(1, 100)]


def _load_wikitext(split: str = "test") -> list[str]:
    """Load WikiText-2 evaluation corpus. Uses datasets library."""
    from datasets import load_dataset as hf_load_dataset
    ds = hf_load_dataset("Salesforce/wikitext", "wikitext-2-v1", split=split)
    return [row["text"] for row in ds if row["text"].strip()]

# --- Helper Functions ---
def find_local_model(upload_id: str) -> str | None:
    base = get_upload_path(upload_id)
    for ext in (".h5", ".keras"):
        matches = glob.glob(os.path.join(base, f"*{ext}"))
        if matches:
            return matches[0]
    return None

def get_upload_path(upload_id):
    return os.path.join(UPLOAD_DIR, upload_id)

def get_dataset_path(upload_id):
    return os.path.join(get_upload_path(upload_id), "dataset.csv")

def load_dataset(upload_id):
    return pd.read_csv(get_dataset_path(upload_id))

def save_json_file(upload_id, filename, data):
    path = os.path.join(get_upload_path(upload_id), filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json_file(upload_id, filename):
    path = os.path.join(get_upload_path(upload_id), filename)
    with open(path, 'r') as f:
        return json.load(f)

def has_huggingface_url(upload_id):
    path = os.path.join(get_upload_path(upload_id), "huggingface_url.txt")
    return os.path.exists(path) and open(path).read().strip().lower() != 'none'

def get_huggingface_url(upload_id):
    path = os.path.join(get_upload_path(upload_id), "huggingface_url.txt")
    if not os.path.exists(path):
        return None
    return open(path).read().strip()

def load_selected_columns(upload_id) -> tuple[str, str | None]:
    meta_path = os.path.join(get_upload_path(upload_id), "selected_columns.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            data = json.load(f)
        # only target is required now
        return data.get("text_column"), data.get("target_column")
    # fallback for older flows
    flag_path = os.path.join(get_upload_path(upload_id), "flag.json")
    if os.path.exists(flag_path):
        with open(flag_path, "r") as f:
            data = json.load(f)
        return data.get("text_column"), data.get("target_column")
    return None, None

def load_model_type(upload_id: str) -> str:
    path = os.path.join(get_upload_path(upload_id), "model_type.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f) or {}
        model_type = str(data.get("model_type", "classification")).strip().lower()
        if model_type in ("classification", "generative"):
            return model_type
    return "classification"

def perplexity_to_score(perplexity: float) -> float:
    # Compatibility score for existing chart code paths that still read "accuracy".
    return max(0.0, min(1.0, 1.0 / (1.0 + (perplexity / 20.0))))

def build_placeholder_eval_metrics(label_mapping: dict[int, str], perplexity: float) -> dict:
    score = perplexity_to_score(perplexity)
    precision = max(0.0, min(1.0, score - 0.01))
    recall = max(0.0, min(1.0, score - 0.02))
    f1 = max(0.0, min(1.0, score - 0.015))

    metrics = {
        "overall": {
            "accuracy": score,
            "perplexity": perplexity,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    }

    for label in label_mapping:
        metrics[label] = {
            "accuracy": score,
            "perplexity": perplexity,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    return metrics

def apply_placeholder_generative_perplexity(pruned_data: dict) -> dict:
    for key, value in pruned_data.items():
        try:
            threshold = float(key)
        except Exception:
            threshold = 0.0

        if threshold == 0.0:
            perplexity = 24.8
        elif threshold == 10.0:
            perplexity = 27.4
        else:
            perplexity = 24.8 + (threshold * 0.55)

        value["perplexity"] = perplexity
        value["accuracy"] = perplexity_to_score(perplexity)

    return pruned_data

def infer_label_mapping(series: pd.Series) -> dict[int, str]:
    uniques = list(pd.unique(series))
    try:
        uniques_sorted = sorted(uniques)
    except Exception:
        uniques_sorted = uniques
    return {i: str(uniques_sorted[i]) for i in range(len(uniques_sorted))}


def create_baseline_metrics(metrics, model_info, label_mapping: dict[int, str], threshold=0):
    return {
        "accuracy": metrics['overall']['accuracy'],
        "precision": metrics['overall']['precision'],
        "recall": metrics['overall']['recall'],
        "f1_score": metrics['overall']['f1_score'],
        "per_class": {
            label_mapping[label]: {
                "accuracy": metrics[label]['accuracy'],
                "precision": metrics[label]['precision'],
                "recall": metrics[label]['recall'],
                "f1_score": metrics[label]['f1_score'],
            } for label in metrics if label != 'overall'
        },
        "flops": model_info['original']['flops_estimate'] if threshold == 0 else model_info['after_pruning']['flops_estimate'],
        "non_zero_params": model_info['original']['non_zero_params'] if threshold == 0 else model_info['after_pruning']['non_zero_params'],
        "params_reduction_pct": model_info['after_pruning']['params_reduction_pct'],
        "flops_reduction_pct": model_info['after_pruning']['flops_reduction_pct'],
    }

def create_threshold_data_entry(metrics, threshold):
    return {
        "accuracy": 0,
        "flops": metrics['after_pruning']['flops_estimate'],
        "non_zero_params": metrics['after_pruning']['non_zero_params'],
        "params_reduction_pct": metrics['after_pruning']['params_reduction_pct'],
        "flops_reduction_pct": metrics['after_pruning']['flops_reduction_pct'],
    }

def create_benchmark_data(hf_url, threshold, benchmark, model_info, pruned_data, label_mapping: dict[int, str]):
    data = {
        "model": hf_url,
        "threshold": threshold,
        "overall": {},
        "perClass": {},
        "originalFlops": model_info['original']['flops_estimate'],
        "prunedFlops": model_info['after_pruning']['flops_estimate'],
        "reductionPercentage": model_info['after_pruning']['params_reduction_pct'],
        "originalParameters": model_info['original']['non_zero_params'],
        "prunedParameters": model_info['after_pruning']['non_zero_params'],
    }

    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        data['overall'][metric] = {
            "original": pruned_data['0'][metric],
            "pruned": benchmark['overall'][metric],
        }

    for label, m in benchmark.items():
        if label == 'overall':
            continue
        name = label_mapping.get(label, str(label))
        data['perClass'][name] = {
            k: {
                "original": pruned_data['0']['per_class'][name][k],
                "pruned": m[k],
            } for k in ['accuracy', 'precision', 'recall']
        }
        data['perClass'][name]['f1Score'] = {
            "original": pruned_data['0']['per_class'][name]['f1_score'],
            "pruned": m['f1_score'],
        }
    return data

def detect_local_device() -> str:
    """
    Try to detect the local accelerator. Returns a human-friendly device name,
    or 'CPU' as a fallback. This function is backend-agnostic and safe.
    """
    # 1) Try PyTorch
    with suppress(Exception):
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            return torch.cuda.get_device_name(idx)

    # 2) Try TensorFlow
    with suppress(Exception):
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # name field can be verbose; return something readable
            return getattr(gpus[0], 'name', 'TensorFlow GPU')

    # 3) Try nvidia-smi directly
    with suppress(Exception):
        import subprocess, shlex
        out = subprocess.check_output(
            shlex.split("nvidia-smi --query-gpu=name --format=csv,noheader"),
            stderr=subprocess.DEVNULL, timeout=1.5
        )
        name = out.decode("utf-8", errors="ignore").strip().splitlines()[0]
        if name:
            return name

    return "CPU"

# --- Main WebSocket Handler ---
def websocket_handlers(socketio):
    def _emit_progress(upload_id: str, done: int, total: int, started_at: float):
        elapsed = time.perf_counter() - started_at
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else None
        socketio.emit("status", {
            "message": f"Evaluating {done}/{total} samples...",
            "eta": eta
        }, to=upload_id)

    @socketio.on('connect')
    def handle_connect():
        print("Client connected")

    @socketio.on('disconnect')
    def handle_disconnect():
        print("Client disconnected")

    @socketio.on('join')
    def on_join(data):
        upload_id = data.get("upload_id")
        if upload_id:
            join_room(upload_id)
        # acknowledge to this client
        socketio.emit("status", {"type": "connection", "status": "connected", "upload_id": upload_id}, to=upload_id)

    @socketio.on('start')
    def handle_start(data):
        upload_id = data.get("upload_id")

        # respond to just this client if upload_id missing
        if not upload_id:
            socketio.emit(
                "status",
                {"type": "error", "message": "No upload_id provided. Please retry."},
                to=request.sid
            )
            return

        upload_path = get_upload_path(upload_id)
        if not os.path.exists(upload_path):
            socketio.emit(
                "status",
                {"type": "error", "message": f"Upload {upload_id} not found on server."},
                to=request.sid
            )
            return

        socketio.emit("status", {"message": f"[START DEBUG] Reading from: {upload_path}"}, to=upload_id)
        def process():
            try:
                socketio.emit("status", {"message": "Model is being loaded..."}, to=upload_id)
                model_type = load_model_type(upload_id)

                if model_type == "generative":
                    # ── Generative flow (real surrogate) ──────────────────
                    anchor_thresholds = get_anchor_thresholds()  # [0.0, 10.0]

                    # Check if this is a preset model
                    preset_model_name = None
                    preset_path = os.path.join(get_upload_path(upload_id), "preset_model.txt")
                    if os.path.exists(preset_path):
                        preset_model_name = open(preset_path).read().strip()

                    if preset_model_name and is_test_model(preset_model_name):
                        # ── Test-model shortcut: use bundled pre-computed CSV data ──
                        socketio.emit("status", {"message": f"Loading test data for {preset_model_name}…"}, to=upload_id)
                        test_data = load_test_model_data()
                        if not test_data:
                            socketio.emit("status", {"type": "error", "message": f"No test data found for {preset_model_name}"}, to=upload_id)
                            return

                        # Extract anchor perplexities
                        anchor_perplexities = {}
                        for at in anchor_thresholds:
                            at_r = round(at, 1)
                            if at_r not in test_data:
                                socketio.emit("status", {"type": "error", "message": f"Missing threshold {at_r} in test data"}, to=upload_id)
                                return
                            anchor_perplexities[at_r] = test_data[at_r]["perplexity"]

                        # Extract model metadata from any row
                        any_row = next(iter(test_data.values()))
                        total_params = int(any_row["total_params"])
                        total_layers = int(any_row["total_layers"])
                        original_flops = float(any_row["original_flops"])

                        # Build FLOPs reduction dict from real results
                        flops_at_thresholds = {t: row["flops_reduction_pct"] for t, row in test_data.items()}

                        # Real perplexities from the dataset for all thresholds
                        real_perplexities = {t: row["perplexity"] for t, row in test_data.items()}

                        model_name_for_logs = preset_model_name

                    else:
                        # ── Real HF model path: load, prune, evaluate at anchors ──
                        # Works for both preset models (gpt2-xl, etc.) and custom HF URLs
                        hf_url = get_huggingface_url(upload_id) if not preset_model_name else preset_model_name
                        if not hf_url:
                            socketio.emit("status", {"type": "error", "message": "No HuggingFace model URL provided for generative analysis."}, to=upload_id)
                            return

                        hf_repo = get_hf_repo(hf_url)
                        socketio.emit("status", {"message": f"Loading generative model {hf_url}…"}, to=upload_id)
                        model, tokenizer, device = load_huggingface_generative_model(hf_repo)
                        model_name_for_logs = hf_url

                        socketio.emit("status", {"message": "Loading WikiText-2 evaluation corpus…"}, to=upload_id)
                        eval_texts = _load_wikitext("test")

                        # Get model metadata directly (no deepcopy)
                        total_params, _ = count_non_zero_params(model)
                        total_layers = getattr(model.config, 'num_hidden_layers', 12)
                        original_flops, _ = estimate_flops(model)

                        # Evaluate at anchor thresholds (real perplexity)
                        anchor_perplexities = {}
                        for at in anchor_thresholds:
                            at_r = round(at, 1)
                            if at_r == 0.0:
                                socketio.emit("status", {"message": "Running baseline perplexity evaluation…"}, to=upload_id)
                                result_ppl = evaluate_perplexity(
                                    model, tokenizer, eval_texts, device=device,
                                    progress_cb=lambda d, t: socketio.emit("status", {"message": f"Baseline eval {d}/{t}…"}, to=upload_id))
                            else:
                                socketio.emit("status", {"message": f"Pruning at threshold {at_r} and evaluating…"}, to=upload_id)
                                # Reload fresh model for pruning (deepcopy can hang with device_map="auto")
                                anchor_model, _, anchor_dev = load_huggingface_generative_model(hf_repo)
                                pruned_model, _ = disable_low_weight_neurons(anchor_model, at_r)
                                result_ppl = evaluate_perplexity(
                                    pruned_model, tokenizer, eval_texts, device=anchor_dev,
                                    progress_cb=lambda d, t: socketio.emit("status", {"message": f"Pruned eval {d}/{t}…"}, to=upload_id))
                                del pruned_model, anchor_model
                            anchor_perplexities[at_r] = result_ppl["perplexity"]

                        # Sweep all thresholds to get FLOPs reduction (read-only, no model copies)
                        socketio.emit("status", {"message": "Computing FLOPs across all pruning thresholds…"}, to=upload_id)
                        all_thresholds = [round(t * 0.1, 1) for t in range(0, 101)]
                        flops_at_thresholds = compute_flops_reduction_sweep(model, all_thresholds)
                        socketio.emit("status", {"message": "FLOPs sweep complete."}, to=upload_id)

                        real_perplexities = None  # Only anchors measured for HF models

                    # ── Predict full perplexity curve ──
                    if real_perplexities:
                        # Test model: use real data directly, no GP needed
                        socketio.emit("status", {"message": "Building results from real data…"}, to=upload_id)
                        surrogate_result = build_result_from_real_data(
                            real_perplexities=real_perplexities,
                            flops_at_thresholds=flops_at_thresholds,
                        )
                    else:
                        # Real HF model: predict with GP surrogate
                        socketio.emit("status", {"message": "Predicting perplexity curve with GP surrogate…"}, to=upload_id)
                        surrogate_result = predict_perplexity_curve(
                            anchor_perplexities=anchor_perplexities,
                            flops_at_thresholds=flops_at_thresholds,
                            total_params=total_params,
                            total_layers=total_layers,
                            original_flops=original_flops,
                        )

                    # ── Build GenerativeDashboardData ──
                    socketio.emit("status", {"message": "Building dashboard data…"}, to=upload_id)
                    base_ppl = anchor_perplexities[0.0]
                    runs = []
                    for i, t in enumerate(surrogate_result["thresholds"]):
                        t_r = round(t, 1)
                        # Use real perplexity if available (preset), otherwise surrogate prediction
                        if real_perplexities and t_r in real_perplexities:
                            ppl = real_perplexities[t_r]
                        else:
                            ppl = surrogate_result["perplexity"][i]

                        flops_red = surrogate_result["flops_reduction_pct"][i]
                        sparsity = flops_red / 100.0
                        actual_flops = original_flops * (1.0 - flops_red / 100.0)

                        # Energy/latency estimates scale roughly with FLOPs
                        flops_ratio = actual_flops / original_flops if original_flops > 0 else 1.0
                        base_energy = 0.42  # kWh per 1k calls baseline estimate
                        base_latency = 120  # ms baseline estimate
                        base_tokens = 45    # tokens/sec baseline estimate

                        ppl_std = surrogate_result["perplexity_std"][i] if "perplexity_std" in surrogate_result else 0.0
                        runs.append({
                            "threshold": t_r,
                            "sparsity": sparsity,
                            "perplexity": round(ppl, 2),
                            "perplexityStd": round(ppl_std, 4),
                            "perplexityDeltaPct": round((ppl - base_ppl) / base_ppl * 100, 1) if base_ppl > 0 else 0,
                            "crossEntropy": round(math.log(ppl), 4) if ppl > 0 else 0,
                            "flops": actual_flops,
                            "flopsReductionPct": round(flops_red, 1),
                            "latencyMs": round(base_latency * flops_ratio, 1),
                            "tokensPerSec": round(base_tokens / flops_ratio, 1) if flops_ratio > 0 else 0,
                            "energyKwhPer1kCalls": round(base_energy * flops_ratio, 4),
                            "co2KgPer1kCalls": round(base_energy * flops_ratio * 0.4286, 4),  # ~Netherlands grid factor
                            "memoryMb": round(total_params / 1e6 * (1.0 - sparsity) * 2, 0),  # rough fp16 estimate
                        })

                    # Simplified distribution data (based on actual predictions)
                    generative_data = {
                        "runs": runs,
                        "lossDistribution": {
                            "bins": [0, 1, 2, 3, 4, 5, 6],
                            "baseCounts": [120, 340, 500, 300, 120, 40],
                            "prunedCounts": [80, 300, 520, 360, 180, 90],
                            "percentiles": {
                                "base": {"p50": round(math.log(base_ppl), 2), "p90": round(math.log(base_ppl) + 1.5, 2), "p99": round(math.log(base_ppl) + 3.0, 2)},
                                "pruned": {"p50": round(math.log(base_ppl) + 0.2, 2), "p90": round(math.log(base_ppl) + 2.0, 2), "p99": round(math.log(base_ppl) + 4.0, 2)},
                            },
                        },
                        "deltaDistribution": {
                            "bins": ["<-10%", "-10% to 0%", "0 to +10%", "+10% to +25%", ">+25%"],
                            "percentages": [3, 22, 45, 20, 10],
                        },
                        "lengthBuckets": [
                            {"range": "short (1-64 tokens)", "avgLength": 30, "basePerplexity": round(base_ppl * 0.96, 1), "prunedPerplexity": round(base_ppl * 0.99, 1)},
                            {"range": "medium (65-192 tokens)", "avgLength": 120, "basePerplexity": round(base_ppl, 1), "prunedPerplexity": round(base_ppl * 1.1, 1)},
                            {"range": "long (193-384 tokens)", "avgLength": 300, "basePerplexity": round(base_ppl * 1.07, 1), "prunedPerplexity": round(base_ppl * 1.3, 1)},
                            {"range": "very long (385+ tokens)", "avgLength": 500, "basePerplexity": round(base_ppl * 1.15, 1), "prunedPerplexity": round(base_ppl * 1.55, 1)},
                        ],
                        "examples": [],
                        "usageBase": {
                            "energyKwhPer1kCalls": runs[0]["energyKwhPer1kCalls"] if runs else 0.42,
                            "co2KgPer1kCalls": runs[0]["co2KgPer1kCalls"] if runs else 0.18,
                            "latencyMs": runs[0]["latencyMs"] if runs else 120,
                        },
                        "surrogateInfo": {
                            "kneeThreshold": surrogate_result["knee_threshold"],
                            "kneePerplexity": surrogate_result["knee_perplexity"],
                            "kneeFlopsReduction": surrogate_result["knee_flops_reduction"],
                            "paretoIndices": surrogate_result["pareto_indices"],
                            "paretoFront": [
                                {"threshold": round(surrogate_result["thresholds"][pi], 1),
                                 "perplexity": round(surrogate_result["perplexity"][pi], 2)}
                                for pi in surrogate_result["pareto_indices"]
                            ],
                            "anchorsUsed": list(anchor_perplexities.keys()),
                            "isPreset": preset_model_name is not None and is_test_model(preset_model_name),
                        },
                    }

                    save_json_file(upload_id, "generative_dashboard_data.json", generative_data)

                    # Also keep a pruned_threshold_data.json for chart-data compat
                    compat_pruned_data = {}
                    for r in runs:
                        compat_pruned_data[str(r["threshold"])] = {
                            "accuracy": perplexity_to_score(r["perplexity"]),
                            "perplexity": r["perplexity"],
                            "flops": r["flops"],
                            "non_zero_params": int(total_params * (1.0 - r["sparsity"])),
                            "params_reduction_pct": r["sparsity"] * 100,
                            "flops_reduction_pct": r["flopsReductionPct"],
                        }
                    save_json_file(upload_id, "pruned_threshold_data.json", compat_pruned_data)
                    save_json_file(upload_id, "label_mapping.json", {0: "generative"})

                    socketio.emit("status", {
                        "message": f"Upload & evaluation completed for {model_name_for_logs}",
                        "type": "upload-complete"
                    }, to=upload_id)
                    return
                else:
                    # ── Classification flow ──
                    df = load_dataset(upload_id)
                    target_col = None
                    with suppress(Exception):
                        cols = load_json_file(upload_id, "selected_columns.json")
                        target_col = cols.get("target_column")

                    if not target_col or target_col not in df.columns:
                        socketio.emit(
                            "status",
                            {"type": "error", "message": "Target column not found in dataset."},
                            to=upload_id
                        )
                        return

                    # Build label mapping from actual labels
                    label_mapping = infer_label_mapping(df[target_col])

                    # Model source (HF URL preferred, else local)
                    hf_url = get_huggingface_url(upload_id)
                    local_model_path = find_local_model(upload_id)
                    if hf_url:
                        model, tokenizer = load_huggingface_model(hf_url)
                        model_name_for_logs = hf_url
                    elif local_model_path:
                        model, tokenizer = load_local_model(local_model_path)
                        model_name_for_logs = os.path.basename(local_model_path)
                    else:
                        socketio.emit(
                            "status",
                            {"type": "error",
                            "message": "No model provided. Enter a Hugging Face repo or upload a .h5/.keras file."},
                            to=upload_id
                        )
                        return

                    # Initial prune at 10% for two baselines
                    model_copy = copy.deepcopy(model)
                    pruned_model, model_info = disable_low_weight_neurons(model_copy, 10)

                    socketio.emit("status", {"message": "Predicting model values..."}, to=upload_id)

                    started_at = time.perf_counter()
                    def progress_cb(done, total):
                        _emit_progress(upload_id, done, total, started_at)

                    baseline = evaluate_model(model, tokenizer, df, target_col=target_col,
                              progress_cb=progress_cb)
                    pruned = evaluate_model(pruned_model, tokenizer, df, target_col=target_col,
                                            progress_cb=progress_cb)

                socketio.emit("status", {"message": "Collecting pruning data..."}, to=upload_id)
                pruned_data = {
                    0:  create_baseline_metrics(baseline, model_info, label_mapping, 0),
                    10: create_baseline_metrics(pruned,   model_info, label_mapping, 10),
                }

                # Sweep thresholds
                for t in THRESHOLDS:
                    t = round(t, 1)
                    m_copy = copy.deepcopy(model)
                    _p_model, metrics = disable_low_weight_neurons(m_copy, t)
                    pruned_data[t] = create_threshold_data_entry(metrics, t)

                socketio.emit("status", {"message": "Predicting performance..."}, to=upload_id)
                try:
                    pred = predict_with_auto_regressive_model(pruned_data, "accuracy")
                    if isinstance(pred, dict) and pred:
                        pruned_data = pred
                    # else keep the baseline-filled pruned_data we already built
                except Exception:
                    # swallow prediction errors and keep baseline-only curves
                    pass

                save_json_file(upload_id, "pruned_threshold_data.json", pruned_data)
                socketio.emit("status", {"message": f"[START DEBUG] Saved pruned data to {os.path.join(get_upload_path(upload_id), 'pruned_threshold_data.json')}"}, to=upload_id)
                save_json_file(upload_id, "label_mapping.json", label_mapping)

                socketio.emit("status", {
                    "message": f"Upload & pruning setup completed for {model_name_for_logs}",
                    "type": "upload-complete"
                }, to=upload_id)

            except Exception as e:
                socketio.emit("status", {"type": "error", "message": f"Start failed: {e}"}, to=upload_id)

        socketio.start_background_task(process)

    @socketio.on('validate')
    def handle_validate(data):
        upload_id = data.get("upload_id")
        threshold = data.get("threshold")

        def process():
            try:
                socketio.emit("status", {"message": "Model is being loaded..."}, to=upload_id)

                df = load_dataset(upload_id)
                _, target_col = load_selected_columns(upload_id)
                if target_col and target_col in df.columns:
                    df = df.rename(columns={target_col: 'label'})

                try:
                    label_mapping = load_json_file(upload_id, "label_mapping.json")
                except Exception:
                    label_mapping = infer_label_mapping(df['label'])

                hf_url = get_huggingface_url(upload_id)
                local_model_path = find_local_model(upload_id)

                if hf_url:
                    model, tokenizer = load_huggingface_model(hf_url)
                    model_id_for_benchmark = hf_url
                elif local_model_path:
                    model, tokenizer = load_local_model(local_model_path)
                    model_id_for_benchmark = os.path.basename(local_model_path)
                else:
                    socketio.emit("status", {
                        "type": "error",
                        "message": "No model available to validate. Run the start step first."
                    }, to=upload_id)
                    return

                pruned_model, model_info = disable_low_weight_neurons(model, threshold)

                socketio.emit("status", {"message": "Benchmarking model..."}, to=upload_id)
                started_at = time.perf_counter()

                def progress_cb(done, total):
                    _emit_progress(upload_id, done, total, started_at)

                benchmark = evaluate_model(pruned_model, tokenizer, df, progress_cb=progress_cb)

                pruned_data = load_json_file(upload_id, "pruned_threshold_data.json")

                benchmark_data = create_benchmark_data(
                    model_id_for_benchmark, threshold, benchmark, model_info, pruned_data, label_mapping
                )
                save_json_file(upload_id, "benchmark_data.json", benchmark_data)

                socketio.emit("status", {"message": "Validation completed successfully", "type": "complete"}, to=upload_id)

            except FileNotFoundError as e:
                socketio.emit("status", {
                    "type": "error",
                    "message": f"Missing file: {e}. Make sure dataset.csv exists and that you ran the start step."
                }, to=upload_id)
            except Exception as e:
                socketio.emit("status", {"type": "error", "message": f"Validate failed: {e}"}, to=upload_id)

        socketio.start_background_task(process)
    
    @socketio.on("benchmark_real")
    def handle_benchmark_real(data):
        """
        ORIGINAL -> PRUNE -> PRUNED
        Builds metricCards (kWh & gCO2 per 1000 calls, accuracy %, TFLOPs per call),
        plus rich per-class and realBenchmark blocks. Extremely chatty debug emits.
        """
        import os, json, time, threading, traceback
        from flask import request

        # local import so this stays drop-in
        try:
            from pruning import disable_low_weight_neurons, estimate_flops
        except Exception:
            # disable_low_weight_neurons may already be imported at module level
            disable_low_weight_neurons  # type: ignore

            def estimate_flops(_model, seq_length: int = 128):
                # minimal safe fallback: no FLOPs estimate available
                raise RuntimeError("estimate_flops not available")

        upload_id = (data or {}).get("upload_id")
        try:
            threshold = float((data or {}).get("threshold", 0) or 0.0)
        except Exception:
            threshold = 0.0

        def _emit(msg, typ="debug", extra=None, to_room=None):
            payload = {"type": typ, "message": msg}
            if extra is not None:
                payload["extra"] = extra
            room = to_room or upload_id or request.sid
            try:
                socketio.emit("status", payload, to=room)
            except Exception:
                pass
            print(msg)

        def _pair(a, b):
            # always numbers (never None) so Angular can render safely
            def _num(x):
                try:
                    if x is None: return 0.0
                    # cast numpy types too
                    return float(x)
                except Exception:
                    return 0.0
            return {"original": _num(a), "pruned": _num(b)}

        def _percent(x):
            # 0..1 -> 0..100 ; keep numbers
            try:
                return float(x) * 100.0
            except Exception:
                return 0.0

        def _grid_factor_g_per_kwh(loc: str) -> float:
            # quick-and-safe factors; adjust as you like
            # sources vary these are ballpark so the card has data
            table = {
                "france": 50.0,
                "netherlands": 475.0,
                "germany": 400.0,
                "uk": 212.0,
                "usa": 386.0,
                "australia": 700.0,
            }
            return table.get((loc or "").lower(), 400.0)

        def process():
            try:
                if not upload_id:
                    _emit("[DEBUG] Missing upload_id for real benchmark.", "error", to_room=request.sid)
                    return

                _emit(f"[DEBUG] Starting benchmark_real | upload_id={upload_id} | threshold={threshold}")

                # ── Generative flow: simulated benchmark with dummy data ──
                model_type = load_model_type(upload_id)
                # Fallback: also accept model_type from the frontend payload
                if model_type != "generative":
                    frontend_mt = str((data or {}).get("model_type", "")).strip().lower()
                    if frontend_mt == "generative":
                        model_type = "generative"
                if model_type == "generative":
                    _emit("Loading generative model for benchmarking…")

                    # Determine model source
                    hf_url = None
                    preset_model_name = None
                    preset_path = os.path.join(get_upload_path(upload_id), "preset_model.txt")
                    if os.path.exists(preset_path):
                        preset_model_name = open(preset_path).read().strip()
                    if not preset_model_name:
                        try:
                            hf_url = get_huggingface_url(upload_id)
                        except Exception:
                            pass

                    model_id = preset_model_name or hf_url
                    if not model_id:
                        _emit("No model found for generative benchmark.", "error")
                        return

                    # ── Test-model shortcut: build benchmark from pre-computed CSV ──
                    if is_test_model(model_id):
                        _emit(f"Loading pre-computed benchmark data for {model_id}…")
                        test_data = load_test_model_data()
                        t_base = round(0.0, 1)
                        t_pruned = round(threshold, 1)
                        base_row = test_data.get(t_base)
                        pruned_row = test_data.get(t_pruned) or test_data.get(
                            min(test_data.keys(), key=lambda k: abs(k - threshold))
                        )
                        if not base_row or not pruned_row:
                            _emit("Test data missing for requested threshold.", "error")
                            return

                        loc = (data or {}).get("location") or "france"
                        g_per_kwh = _grid_factor_g_per_kwh(loc)
                        device_label = detect_local_device()

                        orig_ppl = base_row["perplexity"]
                        pruned_ppl = pruned_row["perplexity"]
                        orig_flops_v = base_row.get("original_flops", 1.0)
                        pruned_flops_v = orig_flops_v * (1.0 - pruned_row.get("flops_reduction_pct", 0.0) / 100.0)
                        orig_params = int(base_row.get("original_non_zero_params", base_row.get("total_params", 0)))
                        prun_params = int(pruned_row.get("non_zero_params", orig_params))

                        # Simulated energy based on pre-computed energy_kwh
                        energy_orig = base_row.get("energy_kwh", 0.001)
                        energy_pruned = pruned_row.get("energy_kwh", energy_orig)
                        kwh_orig_per_1k = energy_orig * 1000.0
                        kwh_prun_per_1k = energy_pruned * 1000.0

                        gflops_orig = orig_flops_v / 1e9
                        gflops_prun = pruned_flops_v / 1e9

                        metric_cards = {
                            "power": {"original": float(kwh_orig_per_1k), "pruned": float(kwh_prun_per_1k)},
                            "performance": {"original": float(orig_ppl), "pruned": float(pruned_ppl)},
                            "emissions": {"original": float(kwh_orig_per_1k * g_per_kwh), "pruned": float(kwh_prun_per_1k * g_per_kwh)},
                            "compute": {"original": float(gflops_orig), "pruned": float(gflops_prun)},
                        }

                        benchmark_data = {
                            "model": model_id,
                            "threshold": float(threshold),
                            "gpu": device_label,
                            "location": loc,
                            "originalParameters": orig_params,
                            "prunedParameters": prun_params,
                            "metricCards": metric_cards,
                            "overall": {
                                "perplexity": _pair(orig_ppl, pruned_ppl),
                                "crossEntropy": _pair(
                                    base_row.get("cross_entropy", 0.0),
                                    pruned_row.get("cross_entropy", 0.0),
                                ),
                            },
                            "perClass": {},
                            "realBenchmark": {
                                "device": device_label,
                                "elapsedSecOriginal": 0.0,
                                "elapsedSecPruned": 0.0,
                                "samples": int(base_row.get("total_tokens_evaluated", 0)),
                                "avgGpuPowerWOriginal": 0.0,
                                "avgGpuPowerWPruned": 0.0,
                                "energyJoulesOriginal": energy_orig * 3_600_000,
                                "energyJoulesPruned": energy_pruned * 3_600_000,
                                "tflopsPerCallOriginal": float(gflops_orig),
                                "tflopsPerCallPruned": float(gflops_prun),
                            },
                        }

                        bench_dir = get_upload_path(upload_id)
                        os.makedirs(bench_dir, exist_ok=True)
                        with open(os.path.join(bench_dir, "benchmark_data.json"), "w", encoding="utf-8") as f:
                            json.dump(benchmark_data, f, indent=2)

                        socketio.emit("status", {
                            "type": "benchmark-complete",
                            "message": "Generative benchmark complete (test data)."
                        }, to=upload_id)
                        return

                    # Load model (map preset name to HF repo)
                    hf_repo = get_hf_repo(model_id)
                    try:
                        model, tokenizer, device = load_huggingface_generative_model(hf_repo)
                        _emit(f"Loaded generative model: {model_id}")
                    except Exception as e:
                        _emit(f"Failed to load model: {e}", "error")
                        return

                    # Load eval texts (full WikiText-2 test set for proper benchmark)
                    _emit("Loading WikiText-2 evaluation corpus…")
                    eval_texts = _load_wikitext("test")
                    _emit(f"Loaded {len(eval_texts)} evaluation texts.")

                    # Get model structural info (use estimate_flops directly, no deepcopy)
                    orig_params_info = count_non_zero_params(model)
                    orig_params = orig_params_info[0]
                    orig_flops_val, _ = estimate_flops(model)

                    # ── Select one representative passage per length bucket ──
                    LENGTH_BUCKETS = [
                        ("short (1\u201364 tokens)", 1, 64),
                        ("medium (65\u2013192 tokens)", 65, 192),
                        ("long (193\u2013384 tokens)", 193, 384),
                        ("very long (385+ tokens)", 385, 99999),
                    ]
                    bucket_candidates: dict[str, list[tuple[str, int]]] = {b[0]: [] for b in LENGTH_BUCKETS}
                    for _txt in eval_texts:
                        _n_tok = len(tokenizer.encode(_txt, truncation=True, max_length=1024))
                        for _label, _lo, _hi in LENGTH_BUCKETS:
                            if _lo <= _n_tok <= _hi:
                                bucket_candidates[_label].append((_txt, _n_tok))
                                break
                    example_passages = []
                    example_bucket_labels = []
                    for _label, _lo, _hi in LENGTH_BUCKETS:
                        cands = bucket_candidates[_label]
                        if not cands:
                            continue
                        mid = (_lo + min(_hi, 512)) / 2
                        cands.sort(key=lambda x: abs(x[1] - mid))
                        example_passages.append(cands[0][0])
                        example_bucket_labels.append(_label)
                    _emit(f"Selected {len(example_passages)} example passages across {len(example_bucket_labels)} length buckets.")

                    # Evaluate ORIGINAL
                    stop_evt = threading.Event()
                    power_readings, power_thread = sample_gpu_power_background(stop_evt, interval=0.25)
                    _emit("Evaluating original model perplexity…")
                    t0 = time.perf_counter()
                    def _orig_progress(d, t):
                        _emit(f"Original eval {d}/{t}…")
                    orig_result = evaluate_perplexity(
                        model, tokenizer, eval_texts, device=device,
                        progress_cb=_orig_progress)
                    elapsed_orig = time.perf_counter() - t0
                    stop_evt.set()
                    if power_thread:
                        power_thread.join(timeout=0.5)
                    avg_watts_orig = (sum(power_readings) / len(power_readings)) if power_readings else 0.0

                    # Per-passage perplexity + token-level detail (original)
                    _emit("Computing token-level predictions for example passages (original)…")
                    orig_passage_ppls = compute_passage_perplexities(
                        model, tokenizer, example_passages, device=device)
                    orig_token_detail = compute_token_comparisons(
                        model, tokenizer, example_passages, device=device,
                        top_k=10, max_positions=5)

                    # Generation example (original)
                    _emit("Generating completion example (original)…")
                    gen_prompt = "The planets in the solar system are the sun, "
                    orig_completion = generate_completion(
                        model, tokenizer, gen_prompt, device=device, max_new_tokens=80)

                    # Prune + evaluate PRUNED
                    _emit(f"Pruning model at threshold {threshold}…")
                    pruned_model, prune_info = disable_low_weight_neurons(model, threshold)
                    prun_params = prune_info["after_pruning"]["non_zero_params"]
                    pruned_flops_val, _ = estimate_flops(pruned_model)

                    # Per-passage perplexity + token-level detail (pruned)
                    _emit("Computing token-level predictions for example passages (pruned)…")
                    pruned_passage_ppls = compute_passage_perplexities(
                        pruned_model, tokenizer, example_passages, device=device)
                    pruned_token_detail = compute_token_comparisons(
                        pruned_model, tokenizer, example_passages, device=device,
                        top_k=10, max_positions=5)

                    # Generation example (pruned)
                    _emit("Generating completion example (pruned)…")
                    pruned_completion = generate_completion(
                        pruned_model, tokenizer, gen_prompt, device=device, max_new_tokens=80)

                    stop_evt2 = threading.Event()
                    power_readings2, power_thread2 = sample_gpu_power_background(stop_evt2, interval=0.25)
                    _emit("Evaluating pruned model perplexity…")
                    t1 = time.perf_counter()
                    def _pruned_progress(d, t):
                        _emit(f"Pruned eval {d}/{t}…")
                    pruned_result = evaluate_perplexity(
                        pruned_model, tokenizer, eval_texts, device=device,
                        progress_cb=_pruned_progress)
                    elapsed_pruned = time.perf_counter() - t1
                    stop_evt2.set()
                    if power_thread2:
                        power_thread2.join(timeout=0.5)
                    avg_watts_pruned = (sum(power_readings2) / len(power_readings2)) if power_readings2 else 0.0

                    samples = len(eval_texts)
                    energy_j_orig = avg_watts_orig * elapsed_orig
                    energy_j_pruned = avg_watts_pruned * elapsed_pruned

                    gflops_orig = orig_flops_val / 1e9 if isinstance(orig_flops_val, (int, float)) else 0.0
                    gflops_prun = pruned_flops_val / 1e9 if isinstance(pruned_flops_val, (int, float)) else 0.0

                    loc = (data or {}).get("location") or "france"
                    g_per_kwh = _grid_factor_g_per_kwh(loc)

                    kwh_orig_per_1k = ((energy_j_orig / 3_600_000.0) / max(samples, 1)) * 1000.0
                    kwh_prun_per_1k = ((energy_j_pruned / 3_600_000.0) / max(samples, 1)) * 1000.0

                    metric_cards = {
                        "power": {"original": float(kwh_orig_per_1k), "pruned": float(kwh_prun_per_1k)},
                        "performance": {"original": float(orig_result["perplexity"]), "pruned": float(pruned_result["perplexity"])},
                        "emissions": {"original": float(kwh_orig_per_1k * g_per_kwh), "pruned": float(kwh_prun_per_1k * g_per_kwh)},
                        "compute": {"original": float(gflops_orig), "pruned": float(gflops_prun)},
                    }

                    # Build passage-level examples with merged token detail per bucket
                    text_examples = []
                    for i in range(min(len(orig_passage_ppls), len(pruned_passage_ppls))):
                        orig_p = orig_passage_ppls[i]
                        prun_p = pruned_passage_ppls[i]
                        orig_ppl = orig_p["perplexity"]
                        prun_ppl = prun_p["perplexity"]
                        delta_pct = ((prun_ppl - orig_ppl) / orig_ppl * 100.0) if orig_ppl > 0 and math.isfinite(orig_ppl) else 0.0

                        # Merge original + pruned token data per position
                        orig_positions = orig_token_detail[i]["positions"] if i < len(orig_token_detail) else []
                        prun_positions = pruned_token_detail[i]["positions"] if i < len(pruned_token_detail) else []
                        merged_positions = []
                        for j in range(max(len(orig_positions), len(prun_positions))):
                            op = orig_positions[j] if j < len(orig_positions) else None
                            pp = prun_positions[j] if j < len(prun_positions) else None
                            ref = op or pp
                            merged_positions.append({
                                "position": ref["position"],
                                "context": ref.get("context", ""),
                                "actualToken": ref["actualToken"],
                                "original": {
                                    "actualRank": op["actualRank"] if op else -1,
                                    "actualProb": op["actualProb"] if op else 0,
                                    "topTokens": op["topTokens"] if op else [],
                                },
                                "pruned": {
                                    "actualRank": pp["actualRank"] if pp else -1,
                                    "actualProb": pp["actualProb"] if pp else 0,
                                    "topTokens": pp["topTokens"] if pp else [],
                                },
                            })

                        text_examples.append({
                            "bucket": example_bucket_labels[i] if i < len(example_bucket_labels) else "unknown",
                            "text": orig_p["text"][:600],
                            "numTokens": orig_p["num_tokens"],
                            "originalPerplexity": round(orig_ppl, 2) if math.isfinite(orig_ppl) else 9999.0,
                            "prunedPerplexity": round(prun_ppl, 2) if math.isfinite(prun_ppl) else 9999.0,
                            "deltaPct": round(delta_pct, 1),
                            "positions": merged_positions,
                        })

                    # Build generation example
                    generation_example = {
                        "prompt": gen_prompt,
                        "originalCompletion": orig_completion[:500],
                        "prunedCompletion": pruned_completion[:500],
                    }

                    device_label = detect_local_device()
                    benchmark_data = {
                        "model": model_id,
                        "threshold": float(threshold),
                        "gpu": device_label,
                        "location": loc,
                        "originalParameters": int(orig_params),
                        "prunedParameters": int(prun_params),
                        "metricCards": metric_cards,
                        "overall": {
                            "perplexity": _pair(orig_result["perplexity"], pruned_result["perplexity"]),
                            "crossEntropy": _pair(orig_result["cross_entropy"], pruned_result["cross_entropy"]),
                        },
                        "perClass": {},
                        "textExamples": text_examples,
                        "generationExample": generation_example,
                        "realBenchmark": {
                            "device": device_label,
                            "elapsedSecOriginal": float(elapsed_orig),
                            "elapsedSecPruned": float(elapsed_pruned),
                            "samples": samples,
                            "avgGpuPowerWOriginal": float(avg_watts_orig),
                            "avgGpuPowerWPruned": float(avg_watts_pruned),
                            "energyJoulesOriginal": float(energy_j_orig),
                            "energyJoulesPruned": float(energy_j_pruned),
                            "tflopsPerCallOriginal": float(gflops_orig),
                            "tflopsPerCallPruned": float(gflops_prun),
                        },
                    }

                    bench_dir = get_upload_path(upload_id)
                    os.makedirs(bench_dir, exist_ok=True)
                    with open(os.path.join(bench_dir, "benchmark_data.json"), "w", encoding="utf-8") as f:
                        json.dump(benchmark_data, f, indent=2)

                    # Save pruned model
                    pruned_model_dir = os.path.join(bench_dir, "pruned_model_hf")
                    try:
                        os.makedirs(pruned_model_dir, exist_ok=True)
                        pruned_model.save_pretrained(pruned_model_dir)
                        tokenizer.save_pretrained(pruned_model_dir)
                        shutil.make_archive(pruned_model_dir, "zip", pruned_model_dir)
                    except Exception as e:
                        _emit(f"Failed to save pruned model: {e}", "error")

                    socketio.emit("status", {
                        "type": "benchmark-complete",
                        "message": "Generative benchmark complete."
                    }, to=upload_id)
                    return

                # ── Classification flow: real benchmark ──

                # 1) DATASET ------------------------------------------------------
                try:
                    df = load_dataset(upload_id)
                    _emit(f"[DEBUG] Dataset loaded: shape={getattr(df, 'shape', None)}; "
                        f"columns={list(getattr(df, 'columns', []))}")
                except Exception as e:
                    _emit(f"[DEBUG] Failed to load dataset: {e}", "error")
                    _emit(traceback.format_exc(), "error")
                    return

                try:
                    sel = load_json_file(upload_id, "selected_columns.json") or {}
                except Exception:
                    sel = {}
                target_col = sel.get("target_column")
                if not target_col or target_col not in getattr(df, "columns", []):
                    target_col = list(getattr(df, "columns", []))[-1] if getattr(df, "columns", []) else None
                if not target_col:
                    _emit("[DEBUG] No target column found.", "error")
                    return
                _emit(f"[DEBUG] Target column resolved: {target_col}")

                samples = int(len(df))

                # 2) MODEL --------------------------------------------------------
                hf_url = None
                local_model_path = None
                try:
                    hf_url = get_huggingface_url(upload_id)
                except Exception:
                    pass
                try:
                    local_model_path = find_local_model(upload_id)
                except Exception:
                    pass

                try:
                    if hf_url:
                        model, tokenizer = load_huggingface_model(hf_url)
                        model_id_for_benchmark = hf_url
                        _emit(f"[DEBUG] Loaded HuggingFace model: {hf_url}")
                    elif local_model_path:
                        model, tokenizer = load_local_model(local_model_path)
                        model_id_for_benchmark = os.path.basename(local_model_path)
                        _emit(f"[DEBUG] Loaded local model: {local_model_path}")
                    else:
                        _emit("[DEBUG] No model found.", "error")
                        return
                except Exception as e:
                    _emit(f"[DEBUG] Model load failed: {e}", "error")
                    _emit(traceback.format_exc(), "error")
                    return

                # 3) ORIGINAL EVAL + POWER ---------------------------------------
                stop_evt = threading.Event()
                power_readings, power_thread = sample_gpu_power_background(stop_evt, interval=0.25)

                def _progress(done, total):
                    try:
                        socketio.emit("status",
                                    {"message": f"Evaluating {done}/{total} samples...", "type": "loading"},
                                    to=upload_id)
                    except Exception:
                        pass

                _emit("[DEBUG] Evaluating ORIGINAL model...")
                t0 = time.perf_counter()
                orig_metrics = evaluate_model(model, tokenizer, df, target_col=target_col, progress_cb=_progress)
                elapsed_orig = time.perf_counter() - t0
                stop_evt.set()
                if power_thread:
                    power_thread.join(timeout=0.5)
                avg_watts_orig = (sum(power_readings) / len(power_readings)) if power_readings else 0.0
                energy_j_orig = avg_watts_orig * elapsed_orig  # Joules (W·s)
                thrpt_orig = (samples / elapsed_orig) if elapsed_orig > 0 else 0.0

                _emit(f"[DEBUG] ORIGINAL: elapsed={elapsed_orig:.3f}s | avg_watts={avg_watts_orig:.3f} "
                    f"| joules={energy_j_orig:.3f} | thrpt={thrpt_orig:.3f}/s")

                # 4) PRUNE + PRUNED EVAL -----------------------------------------
                pruned_model, model_info = disable_low_weight_neurons(model, threshold)
                _emit(f"[DEBUG] Pruned model at threshold={threshold}")

                orig_params = ((model_info or {}).get("original", {}) or {}).get("non_zero_params", 0)
                prun_params = ((model_info or {}).get("after_pruning", {}) or {}).get("non_zero_params", 0)
                _emit(f"[DEBUG] Params non-zero: original={orig_params} | pruned={prun_params}")

                stop_evt2 = threading.Event()
                power_readings2, power_thread2 = sample_gpu_power_background(stop_evt2, interval=0.25)
                _emit("[DEBUG] Evaluating PRUNED model...")
                t1 = time.perf_counter()
                pruned_metrics = evaluate_model(pruned_model, tokenizer, df, target_col=target_col, progress_cb=_progress)
                elapsed_pruned = time.perf_counter() - t1
                stop_evt2.set()
                if power_thread2:
                    power_thread2.join(timeout=0.5)
                avg_watts_pruned = (sum(power_readings2) / len(power_readings2)) if power_readings2 else 0.0
                energy_j_pruned = avg_watts_pruned * elapsed_pruned
                thrpt_pruned = (samples / elapsed_pruned) if elapsed_pruned > 0 else 0.0

                _emit(f"[DEBUG] PRUNED:   elapsed={elapsed_pruned:.3f}s | avg_watts={avg_watts_pruned:.3f} "
                    f"| joules={energy_j_pruned:.3f} | thrpt={thrpt_pruned:.3f}/s")

                # 5) FLOPS (safe) -------------------------------------------------
                orig_flops = pruned_flops = None
                try:
                    orig_flops, _ = estimate_flops(model)          # operations per call
                    pruned_flops, _ = estimate_flops(pruned_model)
                    _emit(f"[DEBUG] FLOPs (per call): original={orig_flops} | pruned={pruned_flops}")
                except Exception as e:
                    _emit(f"[DEBUG] FLOPs estimation failed: {e}")

                # 💡 Convert to GFLOPs-per-call (fallback to param counts)
                gflops_orig = (orig_flops / 1e9) if isinstance(orig_flops, (int, float)) else float(orig_params or 0)
                gflops_prun = (pruned_flops / 1e9) if isinstance(pruned_flops, (int, float)) else float(prun_params or 0)

                # 6) Cards: kWh & gCO2 per 1000 calls, accuracy %, TFLOPs --------
                loc = (data or {}).get("location") or "france"
                g_per_kwh = _grid_factor_g_per_kwh(loc)

                # kWh per 1000 calls
                kwh_orig_per_1k = ((energy_j_orig / 3_600_000.0) / max(samples, 1)) * 1000.0
                kwh_prun_per_1k = ((energy_j_pruned / 3_600_000.0) / max(samples, 1)) * 1000.0
                gco2_orig_per_1k = kwh_orig_per_1k * g_per_kwh
                gco2_prun_per_1k = kwh_prun_per_1k * g_per_kwh

                metric_cards = {
                    "power": {        # actually "energy" but UI labels it kWh
                        "original": float(kwh_orig_per_1k),
                        "pruned":   float(kwh_prun_per_1k),
                    },
                    "performance": {
                        "original": float(_percent(orig_metrics.get("overall", {}).get("accuracy", 0.0))),
                        "pruned":   float(_percent(pruned_metrics.get("overall", {}).get("accuracy", 0.0))),
                    },
                    "emissions": {
                        "original": float(gco2_orig_per_1k),
                        "pruned":   float(gco2_prun_per_1k),
                    },
                    "compute": {
                        "original": float(gflops_orig),
                        "pruned":   float(gflops_prun),
                    }
                }
                _emit(f"[DEBUG] metricCards: {metric_cards}")

                # 7) Overall + per-class blocks ----------------------------------
                overall_block = {
                    "accuracy":  _pair(orig_metrics["overall"]["accuracy"],  pruned_metrics["overall"]["accuracy"]),
                    "precision": _pair(orig_metrics["overall"]["precision"], pruned_metrics["overall"]["precision"]),
                    "recall":    _pair(orig_metrics["overall"]["recall"],    pruned_metrics["overall"]["recall"]),
                    "f1Score":   _pair(orig_metrics["overall"]["f1_score"],  pruned_metrics["overall"]["f1_score"]),
                }

                per_class_block = {}
                for lab in set(orig_metrics.keys()) | set(pruned_metrics.keys()):
                    if lab == "overall":
                        continue
                    o = orig_metrics.get(lab, {})
                    p = pruned_metrics.get(lab, {})
                    per_class_block[str(lab)] = {
                        "accuracy":  _pair(o.get("accuracy"),  p.get("accuracy")),
                        "precision": _pair(o.get("precision"), p.get("precision")),
                        "recall":    _pair(o.get("recall"),    p.get("recall")),
                        "f1Score":   _pair(o.get("f1_score"),  p.get("f1_score")),
                    }

                # 8) Compose JSON -------------------------------------------------
                device_label = detect_local_device()
                benchmark_data = {
                    "model": model_id_for_benchmark,
                    "threshold": float(threshold),
                    "gpu": device_label,
                    "location": loc,
                    "originalParameters": int(orig_params or 0),
                    "prunedParameters": int(prun_params or 0),

                    # NEW: metric cards the Angular header uses
                    "metricCards": metric_cards,

                    "overall": overall_block,
                    "perClass": per_class_block,

                    "realBenchmark": {
                        "device": device_label,
                        "elapsedSecOriginal": float(elapsed_orig),
                        "elapsedSecPruned":   float(elapsed_pruned),
                        "samples": samples,
                        "throughputOriginal": float(thrpt_orig),
                        "throughputPruned":   float(thrpt_pruned),
                        "avgGpuPowerWOriginal": float(avg_watts_orig),
                        "avgGpuPowerWPruned":   float(avg_watts_pruned),
                        "energyJoulesOriginal": float(energy_j_orig),
                        "energyJoulesPruned":   float(energy_j_pruned),
                        "tflopsPerCallOriginal": float(gflops_orig),
                        "tflopsPerCallPruned":   float(gflops_prun),
                        "metricsOriginal": orig_metrics.get("overall", {}),
                        "metricsPruned":   pruned_metrics.get("overall", {}),
                    }
                }

                # 9) Save ---------------------------------------------------------
                bench_dir = get_upload_path(upload_id)
                os.makedirs(bench_dir, exist_ok=True)
                bench_path = os.path.join(bench_dir, "benchmark_data.json")
                with open(bench_path, "w", encoding="utf-8") as f:
                    json.dump(benchmark_data, f, indent=2)
                _emit(f"[DEBUG] Wrote benchmark_data.json at {bench_path}")

                pruned_model_dir = os.path.join(get_upload_path(upload_id), "pruned_model_hf")
                try:
                    os.makedirs(pruned_model_dir, exist_ok=True)
                    # Save pruned model + tokenizer in HuggingFace format
                    pruned_model.save_pretrained(pruned_model_dir)
                    tokenizer.save_pretrained(pruned_model_dir)

                    # Zip the directory
                    zip_path = pruned_model_dir + ".zip"
                    shutil.make_archive(pruned_model_dir, "zip", pruned_model_dir)
                    _emit(f"[DEBUG] Saved HuggingFace pruned model at {zip_path}")
                except Exception as e:
                    _emit(f"[DEBUG] Failed to save HuggingFace pruned model: {e}", "error")

                # in handle_benchmark_real
                _emit("Real benchmark completed", "benchmark-complete")

            except Exception as e:
                _emit(f"[DEBUG] Real benchmark failed: {e}", "error")
                _emit(traceback.format_exc(), "error")

        socketio.start_background_task(process)





