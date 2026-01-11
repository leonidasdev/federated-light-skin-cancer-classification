"""
Metrics Utilities
=================

Metrics calculation for classification tasks.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    balanced_accuracy_score,
)


class MetricsCalculator:
    """
    Calculator for classification metrics.
    
    Computes various metrics relevant for skin cancer classification,
    including handling class imbalance.
    
    Args:
        num_classes: Number of classes
        class_names: Optional list of class names
        average: Averaging strategy for multi-class metrics
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        class_names: Optional[List[str]] = None,
        average: str = "macro",
    ):
        self.num_classes = num_classes
        self.class_names = class_names
        self.average = average
    
    def calculate(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate all metrics.
        
        Args:
            y_pred: Predicted labels
            y_true: Ground truth labels
            y_prob: Optional predicted probabilities for AUC
            
        Returns:
            Dictionary of metrics
        """
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(
                y_true, y_pred, average=self.average, zero_division=0
            ),
            'recall': recall_score(
                y_true, y_pred, average=self.average, zero_division=0
            ),
            'f1': f1_score(
                y_true, y_pred, average=self.average, zero_division=0
            ),
        }
        
        # Calculate per-class metrics
        metrics['precision_per_class'] = precision_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        metrics['recall_per_class'] = recall_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        metrics['f1_per_class'] = f1_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        
        # AUC if probabilities provided
        if y_prob is not None:
            try:
                metrics['auc'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average=self.average
                )
            except ValueError:
                metrics['auc'] = 0.0
        
        return metrics
    
    def get_confusion_matrix(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> np.ndarray:
        """
        Get confusion matrix.
        
        Args:
            y_pred: Predicted labels
            y_true: Ground truth labels
            
        Returns:
            Confusion matrix as numpy array
        """
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> str:
        """
        Get detailed classification report.
        
        Args:
            y_pred: Predicted labels
            y_true: Ground truth labels
            
        Returns:
            Classification report string
        """
        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            zero_division=0,
        )


def calculate_class_weights(labels: np.ndarray) -> np.ndarray:
    """
    Calculate class weights inversely proportional to frequency.
    
    Args:
        labels: Array of labels
        
    Returns:
        Array of class weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    return weights
