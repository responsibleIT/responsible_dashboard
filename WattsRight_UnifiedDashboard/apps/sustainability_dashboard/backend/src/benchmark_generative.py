"""
Perplexity benchmark for causal language models (generative).

Adapted from experiment_setup/benchmark.py for dashboard use.
Adds a progress callback and returns detailed metrics.
"""

import math
import time

import torch


def evaluate_perplexity(model, tokenizer, texts, max_length=1024,
                        progress_cb=None, device=None):
    """
    Compute perplexity of a causal LM on a list of texts.

    Returns dict with: perplexity, cross_entropy, bits_per_token, total_tokens.
    """
    if device is None:
        device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0
    n_texts = len(texts)

    with torch.no_grad():
        for i, text in enumerate(texts):
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids, labels=input_ids)
            n_tokens = input_ids.shape[1] - 1
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

            if progress_cb and ((i + 1) % 50 == 0 or (i + 1) == n_texts):
                progress_cb(i + 1, n_texts)

    if total_tokens == 0:
        return {
            "perplexity": float("inf"),
            "cross_entropy": float("inf"),
            "bits_per_token": float("inf"),
            "total_tokens": 0,
        }

    avg_loss = total_loss / total_tokens
    return {
        "perplexity": math.exp(avg_loss),
        "cross_entropy": avg_loss,
        "bits_per_token": avg_loss / math.log(2),
        "total_tokens": total_tokens,
    }
