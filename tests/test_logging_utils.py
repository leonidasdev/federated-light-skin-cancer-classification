# =============================================================================
# Tests for Logging and Metrics Utilities
# =============================================================================
"""Tests for MetricsTracker, ExperimentLogger, TensorBoardLogger, and setup_logging."""

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from src.utils.logging_utils import (
    ExperimentLogger,
    MetricsTracker,
    TensorBoardLogger,
    setup_logging,
)


# =============================================================================
# setup_logging
# =============================================================================


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging(name="test_setup")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_setup"

    def test_console_handler_added(self):
        logger = setup_logging(name="test_console")
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_file_handler_added(self, tmp_path):
        log_file = tmp_path / "test.log"
        logger = setup_logging(name="test_file", log_file=log_file)
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        logger.info("hello")
        # Flush handlers
        for h in logger.handlers:
            h.flush()
        assert log_file.exists()
        assert "hello" in log_file.read_text()

    def test_clears_existing_handlers(self):
        logger = setup_logging(name="test_clear")
        n1 = len(logger.handlers)
        # Call again — should clear and re-add
        setup_logging(name="test_clear")
        assert len(logger.handlers) == n1

    def test_log_level(self):
        logger = setup_logging(level=logging.DEBUG, name="test_level")
        assert logger.level == logging.DEBUG


# =============================================================================
# MetricsTracker
# =============================================================================


class TestMetricsTracker:
    def test_init_creates_dirs(self, tmp_path):
        tracker = MetricsTracker(tmp_path, "exp1")
        assert tracker.metrics_dir.exists()

    def test_log_records_metrics(self, tmp_path):
        tracker = MetricsTracker(tmp_path, "exp1")
        tracker.log(1, loss=0.5, accuracy=0.8)
        tracker.log(2, loss=0.3, accuracy=0.9)
        assert tracker.metrics["loss"] == [0.5, 0.3]
        assert tracker.metrics["accuracy"] == [0.8, 0.9]

    def test_log_writes_csv(self, tmp_path):
        tracker = MetricsTracker(tmp_path, "exp1")
        tracker.log(1, loss=0.5, accuracy=0.8)
        tracker.log(2, loss=0.3, accuracy=0.9)
        # CSV should be written
        assert tracker.csv_path.exists()
        content = tracker.csv_path.read_text()
        assert "step" in content
        assert "loss" in content

    def test_csv_header_change(self, tmp_path):
        """When new metrics are added, CSV should be re-written with new headers."""
        tracker = MetricsTracker(tmp_path, "exp1")
        tracker.log(1, loss=0.5)
        tracker.log(2, loss=0.3, accuracy=0.9)
        content = tracker.csv_path.read_text()
        assert "accuracy" in content

    def test_get_best_max(self, tmp_path):
        tracker = MetricsTracker(tmp_path, "exp1")
        tracker.log(1, accuracy=0.7)
        tracker.log(2, accuracy=0.9)
        tracker.log(3, accuracy=0.8)
        best_val, best_step = tracker.get_best("accuracy", mode="max")
        assert best_val == 0.9
        assert best_step == 2

    def test_get_best_min(self, tmp_path):
        tracker = MetricsTracker(tmp_path, "exp1")
        tracker.log(1, loss=0.5)
        tracker.log(2, loss=0.2)
        tracker.log(3, loss=0.3)
        best_val, best_step = tracker.get_best("loss", mode="min")
        assert best_val == 0.2
        assert best_step == 2

    def test_get_best_empty(self, tmp_path):
        tracker = MetricsTracker(tmp_path, "exp1")
        val, step = tracker.get_best("nonexistent")
        assert val is None
        assert step is None

    def test_get_summary(self, tmp_path):
        tracker = MetricsTracker(tmp_path, "exp1")
        tracker.log(1, loss=0.5, accuracy=0.7)
        tracker.log(2, loss=0.3, accuracy=0.9)
        summary = tracker.get_summary()
        assert summary["experiment_name"] == "exp1"
        assert summary["num_steps"] == 2
        assert summary["loss_final"] == 0.3
        assert summary["loss_best"] == 0.3  # min for loss
        assert summary["accuracy_best"] == 0.9
        assert "accuracy_mean" in summary
        assert "accuracy_std" in summary

    def test_save(self, tmp_path):
        tracker = MetricsTracker(tmp_path, "exp1")
        tracker.log(1, loss=0.5)
        tracker.save()
        json_path = tracker.metrics_dir / "exp1_metrics.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "metadata" in data
        assert "metrics" in data
        assert "summary" in data
        assert "end_time" in data["metadata"]

    def test_context_manager(self, tmp_path):
        with MetricsTracker(tmp_path, "ctx_exp") as tracker:
            tracker.log(1, loss=0.5)
        json_path = tracker.metrics_dir / "ctx_exp_metrics.json"
        assert json_path.exists()

    def test_read_existing_csv_no_file(self, tmp_path):
        tracker = MetricsTracker(tmp_path, "exp_nocsv")
        data = tracker._read_existing_csv()
        assert data == []


# =============================================================================
# ExperimentLogger
# =============================================================================


class TestExperimentLogger:
    def test_init_creates_dirs(self, tmp_path):
        el = ExperimentLogger("test_exp", output_dir=str(tmp_path))
        assert el.output_dir.exists()
        assert (el.output_dir / "experiment.log").exists()

    def test_log_methods(self, tmp_path):
        el = ExperimentLogger("test_exp", output_dir=str(tmp_path))
        el.info("info msg")
        el.debug("debug msg")
        el.warning("warn msg")
        el.error("err msg")
        # No errors raised

    def test_log_metrics(self, tmp_path):
        el = ExperimentLogger("test_exp", output_dir=str(tmp_path))
        el.log_metrics(1, loss=0.5, accuracy=0.8)
        el.log_metrics(2, loss=0.3, accuracy=0.9)
        assert el.metrics.metrics["loss"] == [0.5, 0.3]

    def test_log_config(self, tmp_path):
        el = ExperimentLogger("test_exp", output_dir=str(tmp_path))
        el.log_config({"lr": 0.001, "epochs": 10})
        config_path = el.output_dir / "config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data["lr"] == 0.001

    def test_finish(self, tmp_path):
        el = ExperimentLogger("test_exp", output_dir=str(tmp_path))
        el.log_metrics(1, loss=0.5)
        summary = el.finish()
        assert "experiment_name" in summary
        # JSON file should be saved
        json_path = el.metrics.metrics_dir / "test_exp_metrics.json"
        assert json_path.exists()


# =============================================================================
# TensorBoardLogger
# =============================================================================


class TestTensorBoardLogger:
    def test_disabled(self):
        tb = TensorBoardLogger(log_dir=Path("/tmp/tb"), enabled=False)
        assert not tb.enabled
        # These should not raise
        tb.log_scalar("tag", 1.0, 1)
        tb.log_scalars("main", {"a": 1.0}, 1)
        tb.log_histogram("tag", [1, 2, 3], 1)
        tb.close()

    def test_enabled_without_tensorboard(self, tmp_path, monkeypatch):
        """If tensorboard is not importable, should gracefully disable."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "tensorboard" in name:
                raise ImportError("mock")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        tb = TensorBoardLogger(log_dir=tmp_path, enabled=True)
        # Should degrade gracefully
        assert not tb.enabled or tb.writer is None
        tb.close()
