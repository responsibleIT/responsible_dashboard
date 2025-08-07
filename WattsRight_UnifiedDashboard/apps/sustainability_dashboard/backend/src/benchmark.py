import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from preprocess import preprocess

def evaluate_model(model, tokenizer, df):
    predictions = []
    true_labels = []
    
    for index, row in df.iterrows():
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
    
    overall_accuracy = accuracy_score(true_labels, predictions)
    overall_f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    overall_precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    overall_recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)

    # Calculate the same metrics for each class
    unique_labels = np.unique(true_labels)
    class_metrics = {}
    for label in unique_labels:
        class_indices = [i for i, l in enumerate(true_labels) if l == label]
        class_predictions = [predictions[i] for i in class_indices]
        class_true_labels = [true_labels[i] for i in class_indices]

        class_accuracy = accuracy_score(class_true_labels, class_predictions)
        class_f1 = f1_score(class_true_labels, class_predictions, average='weighted', zero_division=0)
        class_precision = precision_score(class_true_labels, class_predictions, average='weighted', zero_division=0)
        class_recall = recall_score(class_true_labels, class_predictions, average='weighted', zero_division=0)

        class_metrics[label] = {
            'accuracy': class_accuracy,
            'f1_score': class_f1,
            'precision': class_precision,
            'recall': class_recall
        }

    class_metrics['overall'] = {
        'accuracy': overall_accuracy,
        'f1_score': overall_f1,
        'precision': overall_precision,
        'recall': overall_recall
    }
    
    return class_metrics