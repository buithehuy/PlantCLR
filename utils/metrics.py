import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate supervised evaluation metrics.
    Args:
        y_true: 1D array of true labels
        y_pred: 1D array of predicted labels
        y_prob: 2D array of predicted probabilities per class (optional for AUC)
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    if y_prob is not None:
        try:
            # Multi-class macro-averaged ROC-AUC
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
        except ValueError:
            # Thrown if validation set is too small and not missing a class
            metrics['auc_roc'] = 0.0
            
    return metrics
