from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate key binary classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary containing accuracy, precision, recall, and F1 score
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def confusion_matrix_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    Calculate and return confusion matrix statistics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary with TP, FP, TN, FN counts
    """
    cm = confusion_matrix(y_true, y_pred)

    # Extract values from confusion matrix
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle case where certain classes might be missing in the predictions
        tn = fp = fn = tp = 0

    return {
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }


def calculate_metrics_from_confusion(confusion_stats: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate metrics directly from confusion matrix statistics.

    Args:
        confusion_stats: Dictionary with TP, FP, TN, FN counts

    Returns:
        Dictionary containing accuracy, precision, recall, and F1 score
    """
    tp = confusion_stats['true_positives']
    fp = confusion_stats['false_positives']
    tn = confusion_stats['true_negatives']
    fn = confusion_stats['false_negatives']

    # Avoid division by zero
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def print_classification_report(metrics: Dict[str, float], confusion: Dict[str, int]) -> None:
    """
    Print a formatted classification report with metrics and confusion matrix stats.

    Args:
        metrics: Dictionary containing accuracy, precision, recall, and F1 score
        confusion: Dictionary with TP, FP, TN, FN counts
    """
    print("\n=== Classification Report ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")

    print("\n=== Confusion Matrix ===")
    print(f"True Positives:  {confusion['true_positives']}")
    print(f"False Positives: {confusion['false_positives']}")
    print(f"True Negatives:  {confusion['true_negatives']}")
    print(f"False Negatives: {confusion['false_negatives']}")