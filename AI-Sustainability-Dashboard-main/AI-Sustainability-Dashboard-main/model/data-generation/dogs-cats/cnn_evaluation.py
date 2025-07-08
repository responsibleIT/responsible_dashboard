import numpy as np
import pandas as pd
import tensorflow as tf
# import copy # No longer explicitly needed with clone_model as primary
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from cnn_pruning_utils import prune_keras_model_weights, get_keras_model_metrics, calculate_reduction_metrics

def evaluate_cnn_model(model, image_dataset_loader_func, dataset_df):
    # ... (This function remains the same as the previous version) ...
    predictions_classes = []
    
    images_for_eval, labels_for_eval = image_dataset_loader_func(dataset_df)

    if not images_for_eval:
        print("No images provided for evaluation by the loader function.")
        metrics_keys = ['overall_accuracy', 'overall_f1', 'overall_precision', 'overall_recall',
                        'cat_accuracy', 'cat_f1', 'cat_precision', 'cat_recall',
                        'dog_accuracy', 'dog_f1', 'dog_precision', 'dog_recall']
        return {key: 0 for key in metrics_keys}

    if isinstance(images_for_eval, list) and len(images_for_eval) > 0:
        try:
            batch_of_images = np.vstack(images_for_eval)
            raw_predictions = model.predict(batch_of_images, verbose=0)
        except ValueError as ve:
            print(f"ValueError during np.vstack or model.predict: {ve}")
            raw_predictions = np.array([])
    elif isinstance(images_for_eval, np.ndarray) and images_for_eval.ndim == 4 :
        raw_predictions = model.predict(images_for_eval, verbose=0)
    else:
        print(f"Warning: images_for_eval has unexpected type/shape: {type(images_for_eval)}. No predictions made.")
        raw_predictions = np.array([])

    for raw_pred in raw_predictions:
        predicted_class = 0 if raw_pred[0] >= 0.5 else 1
        predictions_classes.append(predicted_class)
    
    true_labels = labels_for_eval

    unique_classes = sorted(np.unique(true_labels)) if len(true_labels) > 0 else [0, 1]
    class_names = { 0: "cat", 1: "dog" }
    
    metrics = {}
    if len(true_labels) > 0 and len(predictions_classes) == len(true_labels):
        metrics['overall_accuracy'] = accuracy_score(true_labels, predictions_classes)
        metrics['overall_f1'] = f1_score(true_labels, predictions_classes, average='weighted', zero_division=0)
        metrics['overall_precision'] = precision_score(true_labels, predictions_classes, average='weighted', zero_division=0)
        metrics['overall_recall'] = recall_score(true_labels, predictions_classes, average='weighted', zero_division=0)
        
        cm_labels = [0, 1] 
        cm = confusion_matrix(true_labels, predictions_classes, labels=cm_labels)
        
        for class_idx in cm_labels:
            class_name = class_names.get(class_idx)
            class_tp = cm[class_idx, class_idx]
            class_actual_total = np.sum(cm[class_idx, :])
            metrics[f'{class_name}_accuracy'] = class_tp / class_actual_total if class_actual_total > 0 else 0
            metrics[f'{class_name}_f1'] = f1_score(true_labels, predictions_classes, labels=[class_idx], average='binary', zero_division=0)
            metrics[f'{class_name}_precision'] = precision_score(true_labels, predictions_classes, labels=[class_idx], average='binary', zero_division=0)
            metrics[f'{class_name}_recall'] = recall_score(true_labels, predictions_classes, labels=[class_idx], average='binary', zero_division=0)
    else: 
        print("Warning: No valid predictions or label mismatch, returning zeroed metrics for this evaluation.")
        for key_base in ['overall', 'cat', 'dog']:
            for metric_suffix in ['accuracy', 'f1', 'precision', 'recall']:
                metrics[f'{key_base}_{metric_suffix}'] = 0
    return metrics

def evaluate_thresholds_cnn(image_dataset_loader_func, dataset_df, 
                            threshold_values_0_to_10, # Renamed for clarity
                            original_model):
    """
    Evaluates a Keras CNN model over a range of pruning thresholds.
    Args:
        image_dataset_loader_func: Function to load and preprocess images from dataset_df.
        dataset_df: DataFrame with image info for evaluation.
        threshold_values_0_to_10: List of pruning values (0.0 to 10.0). This value is directly used as the percentage for pruning.
        original_model: The original, unpruned Keras CNN model.
    """
    print("Evaluating baseline (original) model...")
    baseline_images, baseline_labels = image_dataset_loader_func(dataset_df)
    
    if not baseline_images:
        print("Failed to load baseline images for evaluation. Aborting threshold evaluation.")
        return pd.DataFrame()

    baseline_eval_metrics = evaluate_cnn_model(original_model, lambda df: (baseline_images, baseline_labels), dataset_df)
    original_model_stats = get_keras_model_metrics(original_model)
    
    results = {
        'threshold': [], 'flops_proxy': [], 'non_zero_params': [],
        'params_reduction_pct': [], 'flops_reduction_pct_proxy': []
    }
    for key in baseline_eval_metrics.keys(): results[key] = []
    
    # Store baseline results (threshold 0)
    results['threshold'].append(0.0) # Baseline is threshold 0.0
    results['flops_proxy'].append(original_model_stats['flops_estimate_proxy'])
    results['non_zero_params'].append(original_model_stats['non_zero_params'])
    results['params_reduction_pct'].append(0)
    results['flops_reduction_pct_proxy'].append(0)
    for key, value in baseline_eval_metrics.items(): results[key].append(value)
    
    print(f"Baseline metrics:")
    print(f"  Overall accuracy: {baseline_eval_metrics.get('overall_accuracy', 0):.4f}")
    print(f"  Params: {original_model_stats['non_zero_params']:,}/{original_model_stats['total_params']:,}")
    print(f"  FLOPs Proxy (non-zero params): {original_model_stats['flops_estimate_proxy']:,.0f}")
    
    # threshold_values_0_to_10 is e.g. [0.0, 0.1, 0.2, ..., 10.0]
    for current_threshold_as_percentage in threshold_values_0_to_10:
        if current_threshold_as_percentage == 0.0: # Baseline already processed
            continue

        print(f"\nEvaluating with pruning threshold: {current_threshold_as_percentage:.1f}% (of max weight in layer)")
        
        pruned_model = None
        try:
            pruned_model = tf.keras.models.clone_model(original_model)
            pruned_model.set_weights(original_model.get_weights())
        except Exception as clone_exc:
            print(f"  Error during tf.keras.models.clone_model: {clone_exc}. Trying from_config as fallback.")
            try:
                model_config = original_model.get_config()
                if 'layers' in model_config: 
                    for layer_cfg in model_config['layers']:
                        if 'config' in layer_cfg: layer_cfg['config'].pop('batch_input_shape', None)
                pruned_model = tf.keras.Model.from_config(model_config)
                pruned_model.set_weights(original_model.get_weights())
            except Exception as from_config_exc:
                print(f"  FATAL: Error with both clone_model and from_config: {from_config_exc}")
                print(f"  Skipping threshold {current_threshold_as_percentage}%.")
                continue 
        if pruned_model is None:
            print(f"  FATAL: Pruned model could not be created for threshold {current_threshold_as_percentage}%. Skipping.")
            continue

        # `current_threshold_as_percentage` is directly used by prune_keras_model_weights
        pruned_model = prune_keras_model_weights(pruned_model, threshold_percentage=current_threshold_as_percentage)
        
        evaluation_metrics = evaluate_cnn_model(pruned_model, lambda df: (baseline_images, baseline_labels), dataset_df)
        pruned_model_stats = get_keras_model_metrics(pruned_model)
        reduction_stats = calculate_reduction_metrics(original_model_stats, pruned_model_stats)
        
        results['threshold'].append(current_threshold_as_percentage) # This is already the 0-10 value
        results['flops_proxy'].append(pruned_model_stats['flops_estimate_proxy'])
        results['non_zero_params'].append(pruned_model_stats['non_zero_params'])
        results['params_reduction_pct'].append(reduction_stats['params_reduction_percentage'])
        results['flops_reduction_pct_proxy'].append(reduction_stats['flops_reduction_percentage'])
        
        for key, value in evaluation_metrics.items(): results[key].append(value)
        
        print(f"  Overall accuracy: {evaluation_metrics.get('overall_accuracy',0):.4f}")
        print(f"  Params: {pruned_model_stats['non_zero_params']:,}/{pruned_model_stats['total_params']:,} " +
              f"(Reduction: {reduction_stats['params_reduction_percentage']:.2f}%)")
        print(f"  FLOPs Proxy: {pruned_model_stats['flops_estimate_proxy']:,.0f} " +
              f"(Reduction: {reduction_stats['flops_reduction_percentage']:.2f}%)")
        
        del pruned_model
        tf.keras.backend.clear_session()
    
    results_df = pd.DataFrame(results)
    return results_df