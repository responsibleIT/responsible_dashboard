import numpy as np
import pandas as pd
import torch
import copy
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from preprocess import preprocess, count_non_zero_params, estimate_flops, disable_low_weight_neurons

def evaluate_model(model, tokenizer, validation_df):
    predictions = []
    true_labels = []
    
    for index, row in validation_df.iterrows():
        text = row['text']
        true_label = row['label']
        
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        
        predicted_class = np.argmax(scores)
        predictions.append(predicted_class)
        true_labels.append(true_label)

    unique_classes = sorted(validation_df['label'].unique())
    class_names = {
        0: "ham",
        1: "spam", 
    }
    
    overall_accuracy = accuracy_score(true_labels, predictions)
    overall_f1 = f1_score(true_labels, predictions, average='weighted')
    overall_precision = precision_score(true_labels, predictions, average='weighted')
    overall_recall = recall_score(true_labels, predictions, average='weighted')
    metrics = {
        'overall_accuracy': overall_accuracy,
        'overall_f1': overall_f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall
    }
    
    cm = confusion_matrix(true_labels, predictions)
    
    for class_idx in unique_classes:
        class_name = class_names.get(class_idx, f"class_{class_idx}")

        class_accuracy = cm[class_idx, class_idx] / np.sum(cm[class_idx, :]) if np.sum(cm[class_idx, :]) > 0 else 0
        class_f1 = f1_score(true_labels, predictions, labels=[class_idx], average='micro')
        class_precision = precision_score(true_labels, predictions, labels=[class_idx], average='micro')
        class_recall = recall_score(true_labels, predictions, labels=[class_idx], average='micro')
        
        metrics[f'{class_name}_accuracy'] = class_accuracy
        metrics[f'{class_name}_f1'] = class_f1
        metrics[f'{class_name}_precision'] = class_precision
        metrics[f'{class_name}_recall'] = class_recall
    
    return metrics

def evaluate_thresholds(validation_path, thresholds, tokenizer, original_model):
    validation_df = pd.read_csv(validation_path)
    
    baseline_metrics = evaluate_model(original_model, tokenizer, validation_df)
    baseline_params, total_params = count_non_zero_params(original_model)
    baseline_flops, total_flops = estimate_flops(original_model)
    
    results = {
        'threshold': [],
        'flops': [],
        'non_zero_params': [],
        'params_reduction_pct': [],
        'flops_reduction_pct': []
    }
    
    for key in baseline_metrics.keys():
        results[key] = []
    
    results['threshold'].append(0)
    results['flops'].append(baseline_flops)
    results['non_zero_params'].append(baseline_params)
    results['params_reduction_pct'].append(0)
    results['flops_reduction_pct'].append(0)
    
    for key, value in baseline_metrics.items():
        results[key].append(value)
    
    print(f"Baseline metrics:")
    print(f"Overall accuracy: {baseline_metrics['overall_accuracy']:.4f}")
    print(f"Baseline params: {baseline_params:,}/{total_params:,}")
    print(f"Baseline FLOPs: {baseline_flops:,.0f}")
    
    for threshold in thresholds[1:]:
        print(f"\nEvaluating threshold: {threshold}%")
        
        model_copy = copy.deepcopy(original_model)
        
        pruned_model, metrics = disable_low_weight_neurons(model_copy, threshold_percentage=threshold)
        
        evaluation_metrics = evaluate_model(pruned_model, tokenizer, validation_df)
        
        results['threshold'].append(threshold)
        results['flops'].append(metrics['after_pruning']['flops_estimate'])
        results['non_zero_params'].append(metrics['after_pruning']['non_zero_params'])
        results['params_reduction_pct'].append(metrics['after_pruning']['params_reduction_percentage'])
        results['flops_reduction_pct'].append(metrics['after_pruning']['flops_reduction_percentage'])
        
        for key, value in evaluation_metrics.items():
            results[key].append(value)
        
        print(f"Overall accuracy: {evaluation_metrics['overall_accuracy']:.4f}")
        print(f"Params: {metrics['after_pruning']['non_zero_params']:,}/{metrics['original']['total_params']:,} " +
              f"(Reduction: {metrics['after_pruning']['params_reduction_percentage']:.2f}%)")
        print(f"FLOPs: {metrics['after_pruning']['flops_estimate']:,.0f} " +
              f"(Reduction: {metrics['after_pruning']['flops_reduction_percentage']:.2f}%)")
        
        del model_copy
        del pruned_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    results_df = pd.DataFrame(results)
    
    return results_df