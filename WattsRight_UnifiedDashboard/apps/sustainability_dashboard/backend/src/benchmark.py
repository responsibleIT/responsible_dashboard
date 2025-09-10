# benchmark.py (drop-in)

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from preprocess import preprocess

try:
    import torch
except Exception:  # torch is optional at import time
    torch = None


def _infer_label_to_index(y):
    """
    Create a deterministic mapping from raw labels -> integer indices.
    Tries to preserve natural ordering when possible.
    """
    # Convert to strings for stable, comparable values
    uniq = list({str(v) for v in y})
    try:
        uniq_sorted = sorted(uniq)
    except Exception:
        uniq_sorted = uniq
    return {lab: i for i, lab in enumerate(uniq_sorted)}


def _ensure_device_tensor(x):
    if torch is None:
        return x
    # No-op: tokenizer already returns tensors on CPU by default.
    # You can add model.device handling here if needed.
    return x


def evaluate_model(
    model,
    tokenizer,
    df,
    text_col: str = "text",
    target_col: str = "label",
    label_to_index: dict | None = None,
):
    """
    Evaluate a text classification model on a dataframe.

    Parameters
    ----------
    model : transformers.PreTrainedModel (or similar)
    tokenizer : transformers.PreTrainedTokenizer (or similar)
    df : pd.DataFrame
        Must contain at least `text_col` and `target_col`.
    text_col : str
        Name of the column containing the input text.
    target_col : str
        Name of the column containing the ground-truth label.
    label_to_index : dict[str, int] | None
        Optional mapping from raw labels (stringified) to integer indices.
        If None, a deterministic mapping will be inferred from the data.

    Returns
    -------
    dict
        {
          0: {accuracy, f1_score, precision, recall},
          1: {...},
          ...
          "overall": {...}
        }
        Per-class keys are integer indices (0..K-1) so the caller can
        map them to display names separately.
    """
    predictions: list[int] = []
    true_labels: list[int] = []

    # Build mapping if not provided
    if label_to_index is None:
        label_to_index = _infer_label_to_index(df[target_col].tolist())

    # Inverse mapping is used only by the caller (websocket), so we keep
    # keys here as integer indices.
    # Note: callers that already renamed columns to 'text'/'label' can
    # still call this with defaults.

    # Use no-grad for speed/memory
    use_torch_no_grad = torch is not None and hasattr(torch, "no_grad")

    iterator = df.itertuples(index=False)
    if use_torch_no_grad:
        no_grad_ctx = torch.no_grad()
    else:
        # Dummy context manager
        class _NullCtx:
            def __enter__(self): return None
            def __exit__(self, *args): return False
        no_grad_ctx = _NullCtx()

    with no_grad_ctx:
        for row in iterator:
            # Access fields safely by name
            row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(df.columns, row))
            raw_text = row_dict[text_col]
            raw_label = row_dict[target_col]

            txt = preprocess(str(raw_text))
            encoded = tokenizer(txt, return_tensors="pt")
            encoded = _ensure_device_tensor(encoded)

            output = model(**encoded)
            # a) transformers logits -> output.logits
            # b) generic tuple -> output[0]
            logits = getattr(output, "logits", None)
            if logits is None:
                logits = output[0]
            # batch size is 1; take [0]
            scores = logits[0].detach().cpu().numpy()
            probs = softmax(scores)

            pred_idx = int(np.argmax(probs))
            predictions.append(pred_idx)

            # map true label to index (stringify for mapping key)
            tl = label_to_index.get(str(raw_label))
            if tl is None:
                # unseen label at eval time — extend mapping consistently
                tl = len(label_to_index)
                label_to_index[str(raw_label)] = tl
            true_labels.append(int(tl))

    # Overall metrics
    overall_accuracy = accuracy_score(true_labels, predictions)
    overall_f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)
    overall_precision = precision_score(true_labels, predictions, average="weighted", zero_division=0)
    overall_recall = recall_score(true_labels, predictions, average="weighted", zero_division=0)

    # Per-class metrics keyed by integer index
    unique_indices = sorted(set(true_labels))
    class_metrics: dict[int, dict[str, float]] = {}
    for idx in unique_indices:
        cls_true = [1 if y == idx else 0 for y in true_labels]
        cls_pred = [1 if p == idx else 0 for p in predictions]

        # For single-class binarized vectors:
        cls_acc = accuracy_score(cls_true, cls_pred)
        # Use binary f1/precision/recall per class to avoid weighted-by-support
        cls_f1 = f1_score(cls_true, cls_pred, zero_division=0)
        cls_prec = precision_score(cls_true, cls_pred, zero_division=0)
        cls_rec = recall_score(cls_true, cls_pred, zero_division=0)

        class_metrics[idx] = {
            "accuracy": cls_acc,
            "f1_score": cls_f1,
            "precision": cls_prec,
            "recall": cls_rec,
        }

    class_metrics["overall"] = {
        "accuracy": overall_accuracy,
        "f1_score": overall_f1,
        "precision": overall_precision,
        "recall": overall_recall,
    }

    return class_metrics
