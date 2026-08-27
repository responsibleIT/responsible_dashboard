"""
Perplexity benchmark for causal language models (generative).

Adapted from experiment_setup/benchmark.py for dashboard use.
Adds a progress callback and returns detailed metrics.
"""

import math

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
            loss_val = outputs.loss.item()
            # Skip NaN / inf losses (can happen with heavily-pruned models)
            if not math.isfinite(loss_val):
                continue
            n_tokens = input_ids.shape[1] - 1
            total_loss += loss_val * n_tokens
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
    # Clamp to avoid OverflowError in math.exp for heavily-pruned models
    clamped_loss = min(avg_loss, 700.0)
    return {
        "perplexity": math.exp(clamped_loss),
        "cross_entropy": avg_loss,
        "bits_per_token": avg_loss / math.log(2),
        "total_tokens": total_tokens,
    }


def compute_passage_perplexities(model, tokenizer, texts, device=None,
                                  max_length=1024):
    """
    Compute per-passage perplexity for a list of texts.

    Returns list of dicts: {text, perplexity, cross_entropy, num_tokens}.
    """
    if device is None:
        device = next(model.parameters()).device

    results = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids, labels=input_ids)
            loss_val = outputs.loss.item()
            n_tokens = input_ids.shape[1] - 1

            if not math.isfinite(loss_val):
                ppl = float("inf")
            else:
                ppl = math.exp(min(loss_val, 700.0))

            results.append({
                "text": text,
                "perplexity": ppl,
                "cross_entropy": loss_val,
                "num_tokens": n_tokens,
            })
    return results


def compute_token_comparisons(model, tokenizer, texts, device=None,
                              max_length=512, top_k=10, max_positions=20):
    """
    For each text, compute the top-k predicted tokens at each position
    for both the ground-truth next token. Returns a list (one per text)
    of dicts with token-level detail.

    Each result dict:
      text: str
      positions: list of {
        position: int,
        actual_token: str,
        actual_rank: int,           # rank of the correct token (1-based)
        actual_prob: float,
        top_tokens: [{token, prob}, ...]  # top-k predictions
      }
    """
    if device is None:
        device = next(model.parameters()).device

    results = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            seq_len = input_ids.shape[1]
            if seq_len < 2:
                continue

            outputs = model(input_ids)
            logits = outputs.logits[0]  # (seq_len, vocab_size)

            # Sample positions spread across the passage
            n_positions = min(max_positions, seq_len - 1)
            if seq_len - 1 <= max_positions:
                pos_indices = list(range(seq_len - 1))
            else:
                step = (seq_len - 1) / n_positions
                pos_indices = [int(round(i * step)) for i in range(n_positions)]

            positions = []
            for pos in pos_indices:
                # logits[pos] predicts the token at pos+1
                probs = torch.softmax(logits[pos], dim=-1)
                top_probs, top_ids = torch.topk(probs, top_k)

                actual_id = input_ids[0, pos + 1].item()
                actual_prob = probs[actual_id].item()

                # Find rank of actual token
                sorted_probs, sorted_ids = torch.sort(probs, descending=True)
                rank_mask = (sorted_ids == actual_id).nonzero(as_tuple=True)[0]
                actual_rank = (rank_mask[0].item() + 1) if len(rank_mask) > 0 else -1

                top_tokens = []
                for j in range(top_k):
                    tok_id = top_ids[j].item()
                    tok_str = tokenizer.decode([tok_id])
                    top_tokens.append({
                        "token": tok_str,
                        "prob": round(top_probs[j].item(), 4),
                        "isActual": tok_id == actual_id,
                    })

                # Context: preceding tokens + target
                ctx_start = max(0, pos - 3)
                ctx_ids = input_ids[0, ctx_start:pos + 1].tolist()
                context_str = tokenizer.decode(ctx_ids).strip()
                if ctx_start > 0:
                    context_str = "\u2026" + context_str

                positions.append({
                    "position": pos,
                    "context": context_str,
                    "actualToken": tokenizer.decode([actual_id]),
                    "actualRank": actual_rank,
                    "actualProb": round(actual_prob, 4),
                    "topTokens": top_tokens,
                })

            results.append({
                "text": text,
                "positions": positions,
            })
    return results


def generate_completion(model, tokenizer, prompt, device=None,
                        max_new_tokens=60):
    """
    Generate a text completion for a single prompt.
    Uses nucleus sampling with a fixed seed so both models get the same
    randomness, plus aggressive repetition controls.
    Returns the generated continuation string.
    """
    if device is None:
        device = next(model.parameters()).device

    with torch.no_grad():
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=256)
        input_ids = enc["input_ids"].to(device)

        torch.manual_seed(42)
        if input_ids.is_cuda:
            torch.cuda.manual_seed(42)

        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
