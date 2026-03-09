# =============================================================================
# Tests for Visualization Module
# =============================================================================
"""Tests for src.evaluation.visualization."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.evaluation.visualization import (
    check_plotting_available,
    plot_client_comparison,
    plot_communication_cost,
    plot_confusion_matrix,
    plot_fl_vs_centralized,
    plot_noniid_distribution,
    plot_training_curves,
)

# Use non-interactive backend for all tests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _close_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def training_history():
    return {
        "train_loss": [1.0, 0.8, 0.6, 0.4],
        "val_loss": [1.1, 0.9, 0.7, 0.5],
        "train_accuracy": [0.3, 0.5, 0.6, 0.7],
        "val_accuracy": [0.25, 0.45, 0.55, 0.65],
    }


# =============================================================================
# Tests
# =============================================================================


class TestCheckPlottingAvailable:
    def test_returns_true(self):
        assert check_plotting_available() is True


class TestPlotTrainingCurves:
    def test_runs_without_error(self, training_history):
        with patch.object(plt, "show"):
            plot_training_curves(training_history, title="Test Curves")

    def test_saves_to_file(self, training_history, tmp_path):
        save_path = tmp_path / "curves.png"
        with patch.object(plt, "show"):
            plot_training_curves(training_history, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_partial_history(self):
        with patch.object(plt, "show"):
            plot_training_curves({"train_loss": [1.0, 0.5]})

    def test_empty_history(self):
        with patch.object(plt, "show"):
            plot_training_curves({})


class TestPlotConfusionMatrix:
    def test_runs_without_error(self):
        cm = np.array([[10, 2], [3, 15]])
        with patch.object(plt, "show"):
            plot_confusion_matrix(cm, class_names=["A", "B"])

    def test_normalized(self):
        cm = np.array([[10, 0], [0, 10]])
        with patch.object(plt, "show"):
            plot_confusion_matrix(cm, class_names=["A", "B"], normalize=True)

    def test_unnormalized(self):
        cm = np.array([[10, 2], [3, 15]])
        with patch.object(plt, "show"):
            plot_confusion_matrix(cm, class_names=["A", "B"], normalize=False)

    def test_saves_to_file(self, tmp_path):
        cm = np.array([[5, 1], [2, 8]])
        save_path = tmp_path / "cm.png"
        with patch.object(plt, "show"):
            plot_confusion_matrix(cm, class_names=["X", "Y"], save_path=save_path)
        assert save_path.exists()

    def test_zero_row(self):
        cm = np.array([[0, 0], [3, 15]])
        with patch.object(plt, "show"):
            plot_confusion_matrix(cm, class_names=["A", "B"], normalize=True)


class TestPlotClientComparison:
    def test_runs_without_error(self):
        metrics = {0: {"accuracy": 0.8}, 1: {"accuracy": 0.9}}
        with patch.object(plt, "show"):
            plot_client_comparison(metrics)

    def test_custom_metric(self):
        metrics = {0: {"f1": 0.7}, 1: {"f1": 0.85}}
        with patch.object(plt, "show"):
            plot_client_comparison(metrics, metric_name="f1")

    def test_saves_to_file(self, tmp_path):
        metrics = {0: {"accuracy": 0.8}}
        save_path = tmp_path / "clients.png"
        with patch.object(plt, "show"):
            plot_client_comparison(metrics, save_path=save_path)
        assert save_path.exists()

    def test_missing_metric_defaults_zero(self):
        metrics = {0: {"accuracy": 0.8}, 1: {}}
        with patch.object(plt, "show"):
            plot_client_comparison(metrics, metric_name="accuracy")


class TestPlotNoniidDistribution:
    def test_runs_without_error(self):
        dists = {0: {0: 50, 1: 30}, 1: {0: 20, 1: 60}}
        with patch.object(plt, "show"):
            plot_noniid_distribution(dists, class_names=["A", "B"])

    def test_saves_to_file(self, tmp_path):
        dists = {0: {0: 100, 1: 50}}
        save_path = tmp_path / "dist.png"
        with patch.object(plt, "show"):
            plot_noniid_distribution(dists, class_names=["A", "B"], save_path=save_path)
        assert save_path.exists()

    def test_empty_client(self):
        dists = {0: {0: 0, 1: 0}, 1: {0: 50, 1: 50}}
        with patch.object(plt, "show"):
            plot_noniid_distribution(dists, class_names=["A", "B"])


class TestPlotFlVsCentralized:
    def test_runs_without_error(self):
        fl = {"val_accuracy": [0.5, 0.6, 0.7]}
        cent = {"val_accuracy": [0.55, 0.65, 0.75]}
        with patch.object(plt, "show"):
            plot_fl_vs_centralized(fl, cent)

    def test_saves_to_file(self, tmp_path):
        fl = {"val_accuracy": [0.5, 0.6]}
        cent = {"val_accuracy": [0.55, 0.65]}
        save_path = tmp_path / "comparison.png"
        with patch.object(plt, "show"):
            plot_fl_vs_centralized(fl, cent, save_path=save_path)
        assert save_path.exists()

    def test_loss_metric(self):
        fl = {"val_loss": [1.0, 0.8]}
        cent = {"val_loss": [0.9, 0.7]}
        with patch.object(plt, "show"):
            plot_fl_vs_centralized(fl, cent, metric="val_loss")

    def test_empty_histories(self):
        with patch.object(plt, "show"):
            plot_fl_vs_centralized({}, {})


class TestPlotCommunicationCost:
    def test_runs_without_error(self):
        with patch.object(plt, "show"):
            plot_communication_cost([1, 2, 3], [10.0, 10.0, 10.0])

    def test_saves_to_file(self, tmp_path):
        save_path = tmp_path / "comm.png"
        with patch.object(plt, "show"):
            plot_communication_cost([1, 2], [5.0, 5.0], save_path=save_path)
        assert save_path.exists()
