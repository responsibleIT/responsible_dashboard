def evaluate_model(
    model,
    tokenizer,
    df,
    *,
    target_col: str = "label",
    progress_cb=None,
    batch_size: int = 32,
    max_rows: int | None = None,
    max_length: int = 256,
):
    """
    Evaluate a text classifier by concatenating ALL columns except `target_col`
    into a single input string per row.

    Args:
        model, tokenizer: Hugging Face-style model & tokenizer.
        df (pd.DataFrame): dataset containing the target col + feature cols.
        target_col (str): name of the target/label column.
        progress_cb (callable): optional function(done:int, total:int).
        batch_size (int): batch size for tokenization/inference.
        max_rows (int|None): evaluate at most this many rows (for speed).
        max_length (int): tokenizer truncation length.

    Returns:
        dict: { 'overall': {...}, <class_label>: {...}, ... }
    """
    # Local imports so this stays drop-in regardless of file-level imports
    import numpy as np
    import pandas as pd
    from scipy.special import softmax
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    try:
        import torch
        has_torch = True
    except Exception:
        torch = None
        has_torch = False

    if df is None or len(df) == 0:
        # Safe empty result
        return {
            'overall': {'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}
        }

    # Determine device if torch is present
    device = "cpu"
    if has_torch and hasattr(model, "parameters"):
        try:
            device = next(model.parameters()).device  # type: ignore[attr-defined]
        except Exception:
            device = "cpu"

    total = len(df) if max_rows is None else min(max_rows, len(df))

    # Build texts (concat all non-target columns) and collect labels
    feature_cols = [c for c in df.columns if c != target_col]
    texts: list[str] = []
    labels: list = []
    for _, row in df.iloc[:total].iterrows():
        parts = [str(row[c]) for c in feature_cols if pd.notna(row[c])]
        texts.append(" ".join(parts))
        labels.append(row[target_col])

    # Preprocess texts (you have a helper already)
    from preprocess import preprocess
    texts = [preprocess(t) for t in texts]

    # Inference
    predictions: list[int] = []
    if has_torch:
        model.eval()  # no-op for non-torch
    done = 0

    # no_grad context only if torch is available
    no_grad_ctx = torch.no_grad() if has_torch else contextlib.nullcontext()  # type: ignore
    with no_grad_ctx:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]

            # Tokenize (batched)
            enc = tokenizer(
                batch_texts,
                return_tensors="pt" if has_torch else None,
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            # Move tensors to device if torch available & device != cpu
            if has_torch and device != "cpu":
                for k in list(enc.keys()):
                    try:
                        enc[k] = enc[k].to(device)  # type: ignore
                    except Exception:
                        pass

            # Forward pass
            outputs = model(**enc)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                # older models return tuple
                logits = outputs[0]

            # Get numpy logits (CPU)
            if has_torch:
                scores = logits.detach().cpu().numpy()
            else:
                # If a non-torch model is used, assume numpy already
                scores = np.array(logits)

            # Softmax & argmax
            probs = softmax(scores, axis=1)
            preds = np.argmax(probs, axis=1).tolist()
            predictions.extend(preds)

            done = end
            if callable(progress_cb):
                try:
                    progress_cb(done, total)
                except Exception:
                    pass

    # --- Metrics (overall) ---
    overall_accuracy = accuracy_score(labels, predictions)
    overall_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    overall_precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    overall_recall = recall_score(labels, predictions, average='weighted', zero_division=0)

    metrics: dict = {
        'overall': {
            'accuracy': overall_accuracy,
            'f1_score': overall_f1,
            'precision': overall_precision,
            'recall': overall_recall
        }
    }

    # --- Per-class ---
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == lab]
        lab_true = [labels[i] for i in idxs]
        lab_pred = [predictions[i] for i in idxs]

        m_acc = accuracy_score(lab_true, lab_pred)
        m_f1 = f1_score(lab_true, lab_pred, average='weighted', zero_division=0)
        m_prec = precision_score(lab_true, lab_pred, average='weighted', zero_division=0)
        m_rec = recall_score(lab_true, lab_pred, average='weighted', zero_division=0)

        metrics[lab] = {
            'accuracy': m_acc,
            'f1_score': m_f1,
            'precision': m_prec,
            'recall': m_rec
        }

    return metrics
