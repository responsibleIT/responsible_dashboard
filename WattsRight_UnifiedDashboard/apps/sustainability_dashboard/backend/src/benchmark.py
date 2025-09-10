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
except Exception:
    torch = None


def _infer_label_to_index(labels):
    """Deterministic mapping raw_label(str) -> integer index."""
    uniq = list({str(v) for v in labels})
    try:
        uniq_sorted = sorted(uniq)
    except Exception:
        uniq_sorted = uniq
    return {lab: i for i, lab in enumerate(uniq_sorted)}


def evaluate_model(
    model,
    tokenizer,
    df,
    *,
    target_col: str,
    feature_cols: list[str] | None = None,
    label_to_index: dict[str, int] | None = None,
):
    """
    Evaluate a text classifier using ALL non-target columns as input features.

    - target_col: name of the ground-truth label column.
    - feature_cols: optional explicit list; if None -> all columns except target_col.
    - label_to_index: optional mapping raw label (as string) -> integer class index.
    """
    # Which columns feed the text model?
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    # Stable mapping for metrics keyed by integers (0..K-1)
    if label_to_index is None:
        label_to_index = _infer_label_to_index(df[target_col].tolist())

    predictions: list[int] = []
    true_labels: list[int] = []

    use_no_grad = torch is not None and hasattr(torch, "no_grad")
    null_ctx = (
        torch.no_grad() if use_no_grad
        else type("Null", (), {"__enter__": lambda *_: None, "__exit__": lambda *_: False})()
    )

    with null_ctx:
        for _, row in df.iterrows():
            # Concatenate all non-target features into one string
            pieces = []
            for c in feature_cols:
                val = row[c]
                if val is None:
                    continue
                s = str(val).strip()
                if s:
                    pieces.append(s)
            joined_text = " ".join(pieces)

            # Preprocess + tokenize
            txt = preprocess(joined_text)
            encoded = tokenizer(txt, return_tensors="pt", truncation=True)
            output = model(**encoded)

            logits = getattr(output, "logits", None)
            if logits is None:
                logits = output[0]
            scores = logits[0].detach().cpu().numpy()
            probs = softmax(scores)
            pred_idx = int(np.argmax(probs))
            predictions.append(pred_idx)

            # map true label to index
            raw_label = str(row[target_col])
            tl = label_to_index.get(raw_label)
            if tl is None:
                tl = len(label_to_index)
                label_to_index[raw_label] = tl
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

        cls_acc = accuracy_score(cls_true, cls_pred)
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
