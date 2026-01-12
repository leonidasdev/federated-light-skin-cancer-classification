"""
Evaluation Utilities.

Comprehensive evaluation metrics for skin cancer classification including
accuracy, F1-score, AUC-ROC, confusion matrix, and per-class metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)

from ..data.datasets import UNIFIED_CLASSES

logger = logging.getLogger(__name__)


# Class names for reporting
CLASS_NAMES = list(UNIFIED_CLASSES.values())


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    auc_macro: Optional[float]
    confusion_matrix: np.ndarray
    per_class_metrics: Dict[str, Dict[str, float]]
    predictions: np.ndarray
    labels: np.ndarray
    probabilities: Optional[np.ndarray]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "auc_macro": self.auc_macro,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "per_class_metrics": self.per_class_metrics,
        }


class ModelEvaluator:
    """
    Comprehensive model evaluator for skin cancer classification.
    
    Computes multiple metrics and provides detailed analysis.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int = 7,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate.
            device: Device to run evaluation on.
            num_classes: Number of classes.
            class_names: Names of classes for reporting.
        """
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or CLASS_NAMES[:num_classes]
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        compute_auc: bool = True,
    ) -> EvaluationResults:
        """
        Evaluate model on given dataloader.
        
        Args:
            dataloader: DataLoader with test/validation data.
            compute_auc: Whether to compute AUC-ROC.
            
        Returns:
            EvaluationResults with all metrics.
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        for images, labels in dataloader:
            images = images.to(self.device)
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
        
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        probabilities = np.array(all_probabilities)
        
        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        balanced_acc = balanced_accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average="macro", zero_division=0)
        recall = recall_score(labels, predictions, average="macro", zero_division=0)
        f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
        f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
        
        # AUC-ROC (one-vs-rest)
        auc = None
        if compute_auc:
            try:
                # Check if all classes are present
                unique_labels = np.unique(labels)
                if len(unique_labels) >= 2:
                    auc = roc_auc_score(
                        labels,
                        probabilities,
                        multi_class="ovr",
                        average="macro",
                    )
            except ValueError as e:
                logger.warning(f"Could not compute AUC: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions, labels=range(self.num_classes))
        
        # Per-class metrics
        per_class = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = labels == i
            if class_mask.sum() > 0:
                class_pred = predictions[class_mask]
                class_true = labels[class_mask]
                per_class[class_name] = {
                    "accuracy": (class_pred == class_true).mean(),
                    "precision": precision_score([i] * len(class_pred), class_pred, labels=[i], zero_division=0),
                    "recall": recall_score(class_true, class_pred, labels=[i], average="micro", zero_division=0),
                    "support": int(class_mask.sum()),
                }
            else:
                per_class[class_name] = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "support": 0,
                }
        
        return EvaluationResults(
            accuracy=accuracy,
            balanced_accuracy=balanced_acc,
            precision_macro=precision,
            recall_macro=recall,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            auc_macro=auc,
            confusion_matrix=cm,
            per_class_metrics=per_class,
            predictions=predictions,
            labels=labels,
            probabilities=probabilities,
        )
    
    def print_report(self, results: EvaluationResults) -> None:
        """Print formatted evaluation report."""
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:          {results.accuracy:.4f}")
        print(f"  Balanced Accuracy: {results.balanced_accuracy:.4f}")
        print(f"  Precision (macro): {results.precision_macro:.4f}")
        print(f"  Recall (macro):    {results.recall_macro:.4f}")
        print(f"  F1 (macro):        {results.f1_macro:.4f}")
        print(f"  F1 (weighted):     {results.f1_weighted:.4f}")
        if results.auc_macro is not None:
            print(f"  AUC-ROC (macro):   {results.auc_macro:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print("-" * 60)
        print(f"{'Class':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'Support':>10}")
        print("-" * 60)
        for class_name, metrics in results.per_class_metrics.items():
            print(
                f"{class_name:<15} {metrics['accuracy']:>10.4f} "
                f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                f"{metrics['support']:>10}"
            )
        
        print("\nConfusion Matrix:")
        print(results.confusion_matrix)
        print("=" * 60)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    num_classes: int = 7,
    class_names: Optional[List[str]] = None,
    print_report: bool = True,
) -> EvaluationResults:
    """
    Convenience function to evaluate a model.
    
    Args:
        model: Model to evaluate.
        dataloader: DataLoader with test data.
        device: Device to use.
        num_classes: Number of classes.
        class_names: Class names for reporting.
        print_report: Whether to print results.
        
    Returns:
        EvaluationResults.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
    )
    
    results = evaluator.evaluate(dataloader)
    
    if print_report:
        evaluator.print_report(results)
    
    return results


def compute_federated_metrics(
    client_results: List[EvaluationResults],
    client_weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute aggregated metrics from multiple clients.
    
    Args:
        client_results: List of evaluation results from each client.
        client_weights: Optional weights for each client (default: equal).
        
    Returns:
        Dictionary of aggregated metrics.
    """
    if client_weights is None:
        client_weights = [1.0 / len(client_results)] * len(client_results)
    
    # Normalize weights
    total = sum(client_weights)
    weights = [w / total for w in client_weights]
    
    metrics = {
        "accuracy": sum(r.accuracy * w for r, w in zip(client_results, weights)),
        "balanced_accuracy": sum(r.balanced_accuracy * w for r, w in zip(client_results, weights)),
        "f1_macro": sum(r.f1_macro * w for r, w in zip(client_results, weights)),
        "f1_weighted": sum(r.f1_weighted * w for r, w in zip(client_results, weights)),
    }
    
    # AUC if available
    auc_values = [r.auc_macro for r in client_results if r.auc_macro is not None]
    if auc_values:
        metrics["auc_macro"] = np.mean(auc_values)
    
    return metrics


def compare_results(
    centralized: EvaluationResults,
    federated: EvaluationResults,
) -> Dict[str, Dict[str, float]]:
    """
    Compare centralized and federated results.
    
    Args:
        centralized: Results from centralized training.
        federated: Results from federated training.
        
    Returns:
        Comparison dictionary.
    """
    comparison = {}
    
    metrics = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]
    
    for metric in metrics:
        cent_val = getattr(centralized, metric)
        fed_val = getattr(federated, metric)
        diff = fed_val - cent_val
        rel_diff = diff / cent_val if cent_val != 0 else 0
        
        comparison[metric] = {
            "centralized": cent_val,
            "federated": fed_val,
            "absolute_diff": diff,
            "relative_diff_pct": rel_diff * 100,
        }
    
    return comparison


def print_comparison(comparison: Dict[str, Dict[str, float]]) -> None:
    """Print formatted comparison between centralized and federated."""
    print("\n" + "=" * 80)
    print("CENTRALIZED vs FEDERATED COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<20} {'Centralized':>12} {'Federated':>12} {'Diff':>10} {'Rel %':>10}")
    print("-" * 80)
    
    for metric, values in comparison.items():
        print(
            f"{metric:<20} {values['centralized']:>12.4f} "
            f"{values['federated']:>12.4f} {values['absolute_diff']:>+10.4f} "
            f"{values['relative_diff_pct']:>+10.2f}%"
        )
    print("=" * 80)
