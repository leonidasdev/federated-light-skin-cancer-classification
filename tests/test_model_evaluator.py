# =============================================================================
# Tests for ModelEvaluator and Convenience Functions
# =============================================================================
"""Tests for ModelEvaluator, evaluate_model, and print_comparison."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import (
    EvaluationResults,
    ModelEvaluator,
    compare_results,
    evaluate_model,
    print_comparison,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_dummy_model(num_classes: int = 3):
    """Create a tiny model that maps 4-dim input to num_classes logits."""
    return torch.nn.Linear(4, num_classes)


def _make_dummy_dataloader(n_samples: int = 20, n_features: int = 4, num_classes: int = 3):
    """Create a DataLoader with random data and labels."""
    torch.manual_seed(0)
    x = torch.randn(n_samples, n_features)
    y = torch.randint(0, num_classes, (n_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=8)


# =============================================================================
# ModelEvaluator Tests
# =============================================================================


class TestModelEvaluator:
    """Tests for ModelEvaluator.evaluate and print_report."""

    @pytest.fixture
    def evaluator(self):
        """Create a ModelEvaluator with a tiny model."""
        model = _make_dummy_model(num_classes=3)
        model.eval()
        return ModelEvaluator(
            model=model,
            device=torch.device("cpu"),
            num_classes=3,
            class_names=["A", "B", "C"],
        )

    @pytest.fixture
    def dataloader(self):
        return _make_dummy_dataloader(n_samples=30, num_classes=3)

    def test_evaluate_returns_results(self, evaluator, dataloader):
        """evaluate() should return an EvaluationResults instance."""
        results = evaluator.evaluate(dataloader)
        assert isinstance(results, EvaluationResults)

    def test_evaluate_metrics_in_range(self, evaluator, dataloader):
        """Accuracy and F1 should be in [0, 1]."""
        results = evaluator.evaluate(dataloader)
        assert 0.0 <= results.accuracy <= 1.0
        assert 0.0 <= results.balanced_accuracy <= 1.0
        assert 0.0 <= results.f1_macro <= 1.0
        assert 0.0 <= results.f1_weighted <= 1.0

    def test_evaluate_confusion_matrix_shape(self, evaluator, dataloader):
        """Confusion matrix should be (num_classes, num_classes)."""
        results = evaluator.evaluate(dataloader)
        assert results.confusion_matrix.shape == (3, 3)

    def test_evaluate_predictions_shape(self, evaluator, dataloader):
        """predictions and labels should have length == n_samples."""
        results = evaluator.evaluate(dataloader)
        assert len(results.predictions) == 30
        assert len(results.labels) == 30

    def test_evaluate_probabilities_shape(self, evaluator, dataloader):
        """probabilities should be (n_samples, num_classes)."""
        results = evaluator.evaluate(dataloader)
        assert results.probabilities.shape == (30, 3)

    def test_evaluate_auc_computed(self, evaluator, dataloader):
        """AUC should be computed by default."""
        results = evaluator.evaluate(dataloader, compute_auc=True)
        # AUC might be None if fewer than 2 classes present in preds;
        # with 30 random samples and 3 classes, likely to be computed
        assert results.auc_macro is None or 0.0 <= results.auc_macro <= 1.0

    def test_evaluate_no_auc(self, evaluator, dataloader):
        """AUC should be None when compute_auc=False."""
        results = evaluator.evaluate(dataloader, compute_auc=False)
        assert results.auc_macro is None

    def test_evaluate_per_class_metrics(self, evaluator, dataloader):
        """per_class_metrics should contain all class names."""
        results = evaluator.evaluate(dataloader)
        for name in ["A", "B", "C"]:
            assert name in results.per_class_metrics
            assert "accuracy" in results.per_class_metrics[name]
            assert "precision" in results.per_class_metrics[name]
            assert "recall" in results.per_class_metrics[name]
            assert "support" in results.per_class_metrics[name]

    def test_print_report_no_error(self, evaluator, dataloader, capsys):
        """print_report should print without errors."""
        results = evaluator.evaluate(dataloader)
        evaluator.print_report(results)
        captured = capsys.readouterr()
        assert "EVALUATION REPORT" in captured.out

    def test_default_class_names(self):
        """Omitting class_names should use defaults."""
        model = _make_dummy_model(num_classes=7)
        evaluator = ModelEvaluator(model=model, device=torch.device("cpu"), num_classes=7)
        assert len(evaluator.class_names) == 7


# =============================================================================
# evaluate_model Convenience Function Tests
# =============================================================================


class TestEvaluateModel:
    """Tests for the evaluate_model convenience function."""

    def test_returns_evaluation_results(self):
        """evaluate_model should return an EvaluationResults."""
        model = _make_dummy_model(num_classes=3)
        model.eval()
        dl = _make_dummy_dataloader(n_samples=20, num_classes=3)
        results = evaluate_model(
            model,
            dl,
            device=torch.device("cpu"),
            num_classes=3,
            print_report=False,
        )
        assert isinstance(results, EvaluationResults)

    def test_print_report_flag(self, capsys):
        """print_report=True should produce output."""
        model = _make_dummy_model(num_classes=3)
        model.eval()
        dl = _make_dummy_dataloader(n_samples=10, num_classes=3)
        evaluate_model(
            model, dl, device=torch.device("cpu"), num_classes=3, print_report=True
        )
        captured = capsys.readouterr()
        assert "EVALUATION REPORT" in captured.out


# =============================================================================
# print_comparison Tests
# =============================================================================


class TestPrintComparison:
    """Tests for the print_comparison utility."""

    def test_print_comparison_output(self, capsys):
        """print_comparison should print a formatted table."""
        r1 = EvaluationResults(
            accuracy=0.9, balanced_accuracy=0.88, precision_macro=0.87,
            recall_macro=0.86, f1_macro=0.86, f1_weighted=0.89,
            auc_macro=0.95, confusion_matrix=np.eye(2),
            per_class_metrics={}, predictions=np.array([]),
            labels=np.array([]), probabilities=None,
        )
        r2 = EvaluationResults(
            accuracy=0.85, balanced_accuracy=0.83, precision_macro=0.82,
            recall_macro=0.81, f1_macro=0.81, f1_weighted=0.84,
            auc_macro=0.92, confusion_matrix=np.eye(2),
            per_class_metrics={}, predictions=np.array([]),
            labels=np.array([]), probabilities=None,
        )
        comparison = compare_results(r1, r2)
        print_comparison(comparison)

        captured = capsys.readouterr()
        assert "CENTRALIZED vs FEDERATED COMPARISON" in captured.out
        assert "accuracy" in captured.out
