# =============================================================================
# Evaluation Utilities
# =============================================================================
"""
Evaluation Utilities.

Comprehensive evaluation metrics for skin cancer classification including
accuracy, F1-score, AUC-ROC, confusion matrix, and per-class metrics.
"""

# =============================================================================
# Imports
# =============================================================================

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader

from ..data.datasets import CLASS_NAMES

logger = logging.getLogger(__name__)

__all__ = [
    "EvaluationResults",
    "ModelEvaluator",
    "compare_results",
    "compute_federated_metrics",
    "evaluate_model",
    "print_comparison",
]

# =============================================================================
# Constants
# =============================================================================

# Default class names for reporting (strings)
DEFAULT_CLASS_NAMES = list(CLASS_NAMES)

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EvaluationResults:
    """Container for evaluation results.

    Attributes:
        accuracy: Overall classification accuracy.
        balanced_accuracy: Class-balanced accuracy (macro-averaged recall).
        precision_macro: Macro-averaged precision.
        recall_macro: Macro-averaged recall (sensitivity).
        f1_macro: Macro-averaged F1 score.
        f1_weighted: Weighted F1 score (by class support).
        auc_macro: Macro-averaged AUC-ROC, or None if unavailable.
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes).
        per_class_metrics: Dict mapping class name to per-class metric dict.
        predictions: Array of predicted class indices.
        labels: Array of ground-truth class indices.
        probabilities: Predicted class probabilities, or None.
    """

    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    auc_macro: float | None
    confusion_matrix: np.ndarray
    per_class_metrics: dict[str, dict[str, float]]
    predictions: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray | None

    def to_dict(self) -> dict[str, Any]:
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


# =============================================================================
# Model Evaluator
# =============================================================================


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
        class_names: list[str] | None = None,
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
        self.class_names = class_names or DEFAULT_CLASS_NAMES[:num_classes]

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        compute_auc: bool = True,
    ) -> EvaluationResults:
        """
        Evaluate model on given dataloader.

        Computes comprehensive metrics for skin cancer classification including
        accuracy, balanced accuracy, F1 scores, and optional AUC-ROC.

        Args:
            dataloader: DataLoader with test/validation data. Should yield
                (images, labels) tuples where images are normalized tensors.
            compute_auc: Whether to compute AUC-ROC. Set to False for faster
                evaluation or when only hard predictions are needed.

        Returns:
            EvaluationResults dataclass containing:
                - accuracy: Overall classification accuracy
                - balanced_accuracy: Mean per-class accuracy (handles imbalance)
                - precision_macro: Macro-averaged precision
                - recall_macro: Macro-averaged recall (sensitivity)
                - f1_macro: Macro-averaged F1 score
                - f1_weighted: Class-weighted F1 score
                - auc_macro: Macro AUC-ROC (None if compute_auc=False or error)
                - confusion_matrix: NxN numpy array of predictions vs labels
                - per_class_metrics: Dict mapping class names to their metrics
                - predictions: Array of predicted class indices
                - labels: Array of true class indices
                - probabilities: Array of predicted class probabilities

        Example:
            >>> evaluator = ModelEvaluator(model, device, num_classes=7)
            >>> results = evaluator.evaluate(test_loader)
            >>> print(f"Balanced Accuracy: {results.balanced_accuracy:.2%}")
            Balanced Accuracy: 84.52%
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

        # Compute metrics (cast to native Python types to satisfy type checkers)
        accuracy = float(accuracy_score(labels, predictions))
        balanced_acc = float(balanced_accuracy_score(labels, predictions))
        precision = float(precision_score(labels, predictions, average="macro", zero_division=0))
        recall = float(recall_score(labels, predictions, average="macro", zero_division=0))
        f1_macro = float(f1_score(labels, predictions, average="macro", zero_division=0))
        f1_weighted = float(f1_score(labels, predictions, average="weighted", zero_division=0))

        # AUC-ROC (one-vs-rest) - compute per-class AUCs and average only
        # over classes that have valid support in the test set. This makes the
        # macro AUC robust to missing classes (common in non-IID splits).
        auc = None
        if compute_auc:
            try:
                # Binarize labels per class. `label_binarize` may return a
                # scipy sparse matrix for certain inputs; convert to a
                # NumPy array to allow safe indexing (`y_true_bin[:, i]`).
                y_true_bin = label_binarize(labels, classes=range(self.num_classes))
                # `label_binarize` may return a sparse matrix; use an `Any`
                # typed temporary to avoid static type complaints and then
                # attempt `.toarray()` with fallback to `np.asarray`.
                y_true_bin_any: Any = y_true_bin
                try:
                    y_true_bin = y_true_bin_any.toarray()
                except Exception:
                    y_true_bin = np.asarray(y_true_bin_any)
                per_class_aucs = []
                for i in range(self.num_classes):
                    # class has at least one positive and at least one negative sample
                    pos_count = int(y_true_bin[:, i].sum())
                    if 0 < pos_count < len(labels):
                        try:
                            auc_i = float(roc_auc_score(y_true_bin[:, i], probabilities[:, i]))
                            # sanity: only accept finite auc values within [0,1]
                            if np.isfinite(auc_i) and 0.0 <= auc_i <= 1.0:
                                per_class_aucs.append(auc_i)
                        except Exception:
                            # skip class if roc_auc_score fails for this class
                            logger.debug(f"Skipping ROC AUC for class {i} (insufficient variation)")

                auc = float(np.mean(per_class_aucs)) if per_class_aucs else None
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")

        # Confusion matrix
        cm = confusion_matrix(labels, predictions, labels=range(self.num_classes))

        # Per-class metrics (one-vs-rest approach)
        per_class = {}
        for i, class_name in enumerate(self.class_names):
            # Create binary labels: 1 if class i, 0 otherwise
            binary_labels = (labels == i).astype(int)
            binary_preds = (predictions == i).astype(int)

            class_support = binary_labels.sum()
            if class_support > 0:
                # Compute metrics for this class vs all others
                precision_val = precision_score(binary_labels, binary_preds, zero_division=0)
                recall_val = recall_score(binary_labels, binary_preds, zero_division=0)
                f1_val = f1_score(binary_labels, binary_preds, zero_division=0)
                # Accuracy for samples that are actually this class
                class_mask = labels == i
                class_accuracy = float((predictions[class_mask] == i).mean())

                per_class[class_name] = {
                    "accuracy": class_accuracy,
                    "precision": float(precision_val),
                    "recall": float(recall_val),
                    "f1": float(f1_val),
                    "support": int(class_support),
                }
            else:
                per_class[class_name] = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
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
        """Print formatted evaluation report.

        Args:
            results: Evaluation results to display.
        """
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)

        print("\nOverall Metrics:")
        print(f"  Accuracy:          {results.accuracy:.4f}")
        print(f"  Balanced Accuracy: {results.balanced_accuracy:.4f}")
        print(f"  Precision (macro): {results.precision_macro:.4f}")
        print(f"  Recall (macro):    {results.recall_macro:.4f}")
        print(f"  F1 (macro):        {results.f1_macro:.4f}")
        print(f"  F1 (weighted):     {results.f1_weighted:.4f}")
        if results.auc_macro is not None:
            print(f"  AUC-ROC (macro):   {results.auc_macro:.4f}")

        print("\nPer-Class Metrics:")
        print("-" * 70)
        print(f"{'Class':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 70)
        for class_name, metrics in results.per_class_metrics.items():
            print(
                f"{class_name:<15} {metrics['accuracy']:>10.4f} "
                f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                f"{metrics.get('f1', 0.0):>10.4f} {metrics['support']:>10}"
            )

        print("\nConfusion Matrix:")
        print(results.confusion_matrix)
        print("=" * 60)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | None = None,
    num_classes: int = 7,
    class_names: list[str] | None = None,
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
    client_results: list[EvaluationResults],
    client_weights: list[float] | None = None,
) -> dict[str, float]:
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
        metrics["auc_macro"] = float(np.mean(auc_values))

    return metrics


def compare_results(
    centralized: EvaluationResults,
    federated: EvaluationResults,
) -> dict[str, dict[str, float]]:
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


def print_comparison(comparison: dict[str, dict[str, float]]) -> None:
    """Print formatted comparison between centralized and federated.

    Args:
        comparison: Dict mapping metric names to dicts with 'centralized',
            'federated', 'absolute_diff', and 'relative_diff_pct' keys.
    """
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
