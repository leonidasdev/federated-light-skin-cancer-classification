"""
Utilities Tests
===============

Unit tests for utility functions.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import MetricsCalculator, calculate_class_weights
from src.utils.seed import set_seed


class TestMetricsCalculator:
    """Tests for metrics calculator."""
    
    @pytest.fixture
    def calculator(self):
        return MetricsCalculator(num_classes=3)
    
    def test_perfect_predictions(self, calculator):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        
        metrics = calculator.calculate(y_pred, y_true)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_random_predictions(self, calculator):
        """Test metrics with random predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        
        metrics = calculator.calculate(y_pred, y_true)
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_confusion_matrix_shape(self, calculator):
        """Test confusion matrix has correct shape."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        
        cm = calculator.get_confusion_matrix(y_pred, y_true)
        
        assert cm.shape == (3, 3)


class TestClassWeights:
    """Tests for class weight calculation."""
    
    def test_balanced_weights(self):
        """Test weights with balanced classes."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        weights = calculate_class_weights(labels)
        
        # All weights should be equal for balanced classes
        assert np.allclose(weights, weights[0])
    
    def test_imbalanced_weights(self):
        """Test weights with imbalanced classes."""
        labels = np.array([0, 0, 0, 0, 1, 2])
        weights = calculate_class_weights(labels)
        
        # Minority classes should have higher weights
        assert weights[1] > weights[0]
        assert weights[2] > weights[0]


class TestSeed:
    """Tests for seed utilities."""
    
    def test_reproducibility(self):
        """Test that setting seed produces reproducible results."""
        import torch
        
        set_seed(42)
        tensor1 = torch.randn(10)
        
        set_seed(42)
        tensor2 = torch.randn(10)
        
        assert torch.allclose(tensor1, tensor2)
