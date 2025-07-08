import numpy as np
import tensorflow as tf

def count_non_zero_params_keras(model):
    """Counts non-zero parameters and total parameters in a Keras model."""
    total_params = model.count_params() # Total trainable and non-trainable
    non_zero_params = 0
    trainable_params_total = 0

    for layer in model.layers:
        layer_weights = layer.get_weights() # List of numpy arrays (weights, biases)
        for weights_array in layer_weights:
            non_zero_params += np.count_nonzero(weights_array)
            # Only count trainable params for a more focused 'total' if needed, 
            # but model.count_params() is standard.
            # if layer.trainable: # Keras model.count_params() includes non-trainable. Let's stick to that for total.
            #     trainable_params_total += weights_array.size
                
    return non_zero_params, total_params


def prune_keras_model_weights(model, threshold_percentage=0.5):
    """
    Prunes a Keras model by zeroing out weights below a certain percentage of max weight per layer.
    This is a simple magnitude pruning.
    Args:
        model: The Keras model to prune.
        threshold_percentage: Percentage of the maximum absolute weight in a layer.
                              Weights below this threshold (relative to max) will be zeroed.
    Returns:
        The pruned Keras model (modified in-place).
    """
    for layer in model.layers:
        if layer.get_weights(): # Check if the layer has weights (e.g., Dense, Conv2D)
            original_weights = layer.get_weights()
            pruned_weights = []
            for weights_array in original_weights:
                if weights_array.ndim > 1: # Typically kernels (weights), not biases
                    abs_weights = np.abs(weights_array)
                    max_weight_in_array = np.max(abs_weights)
                    if max_weight_in_array == 0: # Avoid division by zero if all weights are already zero
                        threshold_value = 0
                    else:
                        threshold_value = max_weight_in_array * (threshold_percentage / 100.0)
                    
                    # Create a mask for weights to keep
                    mask = abs_weights >= threshold_value
                    # Zero out weights below the threshold
                    pruned_array = weights_array * mask 
                    pruned_weights.append(pruned_array.astype(weights_array.dtype))
                else: # Keep biases or 1D weights as they are (or implement specific pruning for them)
                    pruned_weights.append(weights_array)
            layer.set_weights(pruned_weights)
    return model

def get_keras_model_metrics(model):
    """
    Calculates metrics for a Keras model (non-zero params, total params).
    A simplified FLOPs estimation can be added here if a good method is chosen.
    For now, focuses on parameters.
    """
    non_zero, total = count_non_zero_params_keras(model)
    
    # Simplified FLOPs proxy: assume FLOPs are proportional to non-zero params
    # This is a very rough estimate.
    # A more accurate Keras FLOPs counter would require iterating through layer types and their ops.
    # For example, tf.profiler.profile can give FLOPs for a concrete function execution.
    # Here, we'll use non_zero parameters as a proxy for computational load after pruning.
    
    return {
        "non_zero_params": non_zero,
        "total_params": total,
        "flops_estimate_proxy": non_zero # Using non_zero_params as a FLOPs proxy
    }

def calculate_reduction_metrics(original_metrics, pruned_metrics):
    """Calculates reduction percentages."""
    original_nz_params = original_metrics["non_zero_params"]
    pruned_nz_params = pruned_metrics["non_zero_params"]
    
    original_flops_proxy = original_metrics["flops_estimate_proxy"]
    pruned_flops_proxy = pruned_metrics["flops_estimate_proxy"]

    params_reduction_pct = 0
    if original_nz_params > 0:
        params_reduction_pct = (1 - pruned_nz_params / original_nz_params) * 100
    
    flops_reduction_pct = 0
    if original_flops_proxy > 0:
        flops_reduction_pct = (1 - pruned_flops_proxy / original_flops_proxy) * 100
        
    return {
        "params_reduction_percentage": params_reduction_pct,
        "flops_reduction_percentage": flops_reduction_pct # Based on proxy
    }