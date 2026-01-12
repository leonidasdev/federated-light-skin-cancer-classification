"""
Tests for Evaluation Metrics and Visualization.
"""

import pytest
import numpy as np


class TestEvaluationResults:
    """Tests for EvaluationResults dataclass."""
    
    def test_results_to_dict(self):
        """Test results serialization."""
        from src.evaluation.metrics import EvaluationResults
        
        results = EvaluationResults(
            accuracy=0.85,
            balanced_accuracy=0.80,
            precision_macro=0.82,
            recall_macro=0.78,
            f1_macro=0.80,
            f1_weighted=0.84,
            auc_macro=0.90,
            confusion_matrix=np.array([[10, 2], [3, 15]]),
            per_class_metrics={"class_0": {"accuracy": 0.83}},
            predictions=np.array([0, 1, 1, 0]),
            labels=np.array([0, 1, 0, 0]),
            probabilities=None,
        )
        
        result_dict = results.to_dict()
        
        assert result_dict["accuracy"] == 0.85
        assert result_dict["f1_macro"] == 0.80
        assert isinstance(result_dict["confusion_matrix"], list)


class TestMetricsCalculation:
    """Tests for metrics calculation functions."""
    
    def test_compare_results(self):
        """Test comparison between centralized and federated results."""
        from src.evaluation.metrics import EvaluationResults, compare_results
        
        cent_results = EvaluationResults(
            accuracy=0.90,
            balanced_accuracy=0.88,
            precision_macro=0.87,
            recall_macro=0.86,
            f1_macro=0.86,
            f1_weighted=0.89,
            auc_macro=0.95,
            confusion_matrix=np.eye(2),
            per_class_metrics={},
            predictions=np.array([]),
            labels=np.array([]),
            probabilities=None,
        )
        
        fed_results = EvaluationResults(
            accuracy=0.85,
            balanced_accuracy=0.83,
            precision_macro=0.82,
            recall_macro=0.81,
            f1_macro=0.81,
            f1_weighted=0.84,
            auc_macro=0.92,
            confusion_matrix=np.eye(2),
            per_class_metrics={},
            predictions=np.array([]),
            labels=np.array([]),
            probabilities=None,
        )
        
        comparison = compare_results(cent_results, fed_results)
        
        assert "accuracy" in comparison
        assert comparison["accuracy"]["centralized"] == 0.90
        assert comparison["accuracy"]["federated"] == 0.85
        assert comparison["accuracy"]["absolute_diff"] == pytest.approx(-0.05)
    
    def test_compute_federated_metrics(self):
        """Test aggregation of metrics from multiple clients."""
        from src.evaluation.metrics import EvaluationResults, compute_federated_metrics
        
        # Create mock results for 2 clients
        client1 = EvaluationResults(
            accuracy=0.80,
            balanced_accuracy=0.78,
            precision_macro=0.75,
            recall_macro=0.73,
            f1_macro=0.74,
            f1_weighted=0.79,
            auc_macro=0.85,
            confusion_matrix=np.eye(2),
            per_class_metrics={},
            predictions=np.array([]),
            labels=np.array([]),
            probabilities=None,
        )
        
        client2 = EvaluationResults(
            accuracy=0.90,
            balanced_accuracy=0.88,
            precision_macro=0.85,
            recall_macro=0.83,
            f1_macro=0.84,
            f1_weighted=0.89,
            auc_macro=0.92,
            confusion_matrix=np.eye(2),
            per_class_metrics={},
            predictions=np.array([]),
            labels=np.array([]),
            probabilities=None,
        )
        
        # Equal weighting
        aggregated = compute_federated_metrics([client1, client2])
        
        assert aggregated["accuracy"] == pytest.approx(0.85, rel=0.01)
        assert aggregated["f1_macro"] == pytest.approx(0.79, rel=0.01)
    
    def test_weighted_federated_metrics(self):
        """Test weighted aggregation."""
        from src.evaluation.metrics import EvaluationResults, compute_federated_metrics
        
        client1 = EvaluationResults(
            accuracy=0.80, balanced_accuracy=0.0, precision_macro=0.0,
            recall_macro=0.0, f1_macro=0.80, f1_weighted=0.0,
            auc_macro=None, confusion_matrix=np.eye(2),
            per_class_metrics={}, predictions=np.array([]),
            labels=np.array([]), probabilities=None,
        )
        
        client2 = EvaluationResults(
            accuracy=0.90, balanced_accuracy=0.0, precision_macro=0.0,
            recall_macro=0.0, f1_macro=0.90, f1_weighted=0.0,
            auc_macro=None, confusion_matrix=np.eye(2),
            per_class_metrics={}, predictions=np.array([]),
            labels=np.array([]), probabilities=None,
        )
        
        # Client 2 has more weight (3:1)
        aggregated = compute_federated_metrics(
            [client1, client2],
            client_weights=[1.0, 3.0]
        )
        
        # Weighted: (0.8 * 0.25) + (0.9 * 0.75) = 0.875
        assert aggregated["accuracy"] == pytest.approx(0.875, rel=0.01)


class TestVisualization:
    """Tests for visualization functions (without actual plotting)."""
    
    def test_plotting_available_check(self):
        """Test that plotting availability is checked."""
        from src.evaluation.visualization import check_plotting_available
        
        # Should return True if matplotlib is installed
        result = check_plotting_available()
        assert isinstance(result, bool)
