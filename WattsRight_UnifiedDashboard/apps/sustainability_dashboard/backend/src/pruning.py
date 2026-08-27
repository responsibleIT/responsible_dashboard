import torch

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def count_layers(model):
    return len([name for name, param in model.named_parameters() if 'layer' in name])

def count_non_zero_params(model):
    total_params = 0
    non_zero_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            non_zero_params += (param.data != 0).sum().item()
    
    return non_zero_params, total_params

def estimate_flops(model, seq_length=128):
    config = model.config
    hidden_size = getattr(config, 'hidden_size', 768)
    num_layers = getattr(config, 'num_hidden_layers', 12)
    
    flops_per_token = 0
    
    flops_per_token += 4 * num_layers * hidden_size * hidden_size
    flops_per_token += 2 * num_layers * seq_length * hidden_size
    
    ffn_dim = getattr(config, 'intermediate_size', 4 * hidden_size)
    flops_per_token += 2 * num_layers * hidden_size * ffn_dim
    
    total_flops = flops_per_token * seq_length
    
    non_zero_params, total_params = count_non_zero_params(model)
    sparsity_ratio = non_zero_params / total_params if total_params > 0 else 0
    
    adjusted_flops = total_flops * sparsity_ratio
    
    return adjusted_flops, total_flops


def compute_flops_reduction_sweep(model, thresholds, seq_length=128):
    """
    Compute FLOPs reduction % at many thresholds WITHOUT modifying the model.
    Read-only: counts how many params would survive each threshold.
    Returns dict {threshold: flops_reduction_pct}.
    """
    config = model.config
    hidden_size = getattr(config, 'hidden_size', 768)
    num_layers = getattr(config, 'num_hidden_layers', 12)

    flops_per_token = 0
    flops_per_token += 4 * num_layers * hidden_size * hidden_size
    flops_per_token += 2 * num_layers * seq_length * hidden_size
    ffn_dim = getattr(config, 'intermediate_size', 4 * hidden_size)
    flops_per_token += 2 * num_layers * hidden_size * ffn_dim
    total_flops = flops_per_token * seq_length

    # Pre-compute per-layer data for fast threshold sweep
    layer_data = []  # (max_abs, abs_flat_tensor, numel, is_prunable)
    total_params = 0
    orig_non_zero = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        numel = param.numel()
        total_params += numel
        nz = (param.data != 0).sum().item()
        orig_non_zero += nz
        if 'weight' in name and len(param.shape) > 1:
            abs_data = torch.abs(param.data).flatten()
            max_w = abs_data.max().item()
            layer_data.append((max_w, abs_data, numel, True))
        else:
            layer_data.append((0.0, None, numel, False))

    if total_params == 0 or orig_non_zero == 0:
        return {t: 0.0 for t in thresholds}

    orig_sparsity = orig_non_zero / total_params
    orig_flops = total_flops * orig_sparsity

    results = {}
    for t_pct in thresholds:
        if t_pct == 0.0:
            results[t_pct] = 0.0
            continue
        surviving = 0
        for max_w, abs_data, numel, is_prunable in layer_data:
            if is_prunable:
                thresh_val = max_w * (t_pct / 100.0)
                surviving += (abs_data >= thresh_val).sum().item()
            else:
                surviving += numel  # bias/non-prunable params stay
        sparsity = surviving / total_params
        pruned_flops = total_flops * sparsity
        reduction = (1.0 - pruned_flops / orig_flops) * 100.0 if orig_flops > 0 else 0.0
        results[t_pct] = reduction
    return results


def disable_low_weight_neurons(model, threshold_percentage=0.5):
    original_number_of_layers = count_layers(model)
    original_non_zero_params, original_total_params = count_non_zero_params(model)
    original_flops_estimate, original_total_flops = estimate_flops(model)
    
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:
            max_weight = torch.abs(param.data).max().item()
            threshold = max_weight * (threshold_percentage / 100)
            
            mask = torch.abs(param.data) >= threshold
            param.data[~mask] = 0.0
    
    pruned_non_zero_params, pruned_total_params = count_non_zero_params(model)
    pruned_flops_estimate, pruned_total_flops = estimate_flops(model)
    
    return model, {
        "original": {
            "total_layers": original_number_of_layers,
            "non_zero_params": original_non_zero_params,
            "total_params": original_total_params,
            "flops_estimate": original_flops_estimate
        },
        "after_pruning": {
            "non_zero_params": pruned_non_zero_params,
            "total_params": pruned_total_params,
            "flops_estimate": pruned_flops_estimate,
            "params_reduction_pct": (1 - pruned_non_zero_params/original_non_zero_params)*100,
            "flops_reduction_pct": (1 - pruned_flops_estimate/original_flops_estimate)*100
        }
    }