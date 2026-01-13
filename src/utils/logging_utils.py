"""
Logging and Metrics Utilities.

Centralized logging configuration and metrics tracking for experiments.
"""

import logging
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    name: str = "dscatnet_fl",
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level.
        log_file: Optional file path for logging.
        name: Logger name.
        
    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class MetricsTracker:
    """
    Track and persist training metrics.
    
    Provides functionality to log, save, and analyze training metrics
    across multiple experiments.
    """
    
    def __init__(self, output_dir: Path, experiment_name: str):
        """
        Initialize metrics tracker.
        
        Args:
            output_dir: Directory to save metrics.
            experiment_name: Name of the experiment.
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.metadata: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
        }
        
        # CSV writer for real-time logging
        self.csv_path = self.metrics_dir / f"{experiment_name}_metrics.csv"
        self.csv_file = None
        self.csv_writer = None
    
    def log(self, step: int, **kwargs) -> None:
        """
        Log metrics for a step.
        
        Args:
            step: Training step (epoch/round).
            **kwargs: Metric name-value pairs.
        """
        for name, value in kwargs.items():
            self.metrics[name].append(value)
        
        # Write to CSV
        self._write_csv_row(step, kwargs)
    
    def _write_csv_row(self, step: int, metrics: Dict[str, float]) -> None:
        """Write a row to the CSV file."""
        if self.csv_file is None:
            self.csv_file = open(self.csv_path, "w", newline="")
            headers = ["step"] + list(metrics.keys())
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=headers)
            self.csv_writer.writeheader()
        
        row = {"step": step, **metrics}

        # Check if headers match (ensure writer exists before accessing attributes)
        if self.csv_writer is None or set(row.keys()) != set(self.csv_writer.fieldnames):
            # Reopen with new headers
            try:
                if self.csv_file:
                    self.csv_file.close()
            except Exception:
                pass

            existing_data = self._read_existing_csv()

            all_headers = set(["step"])
            for data in existing_data:
                all_headers.update(data.keys())
            all_headers.update(metrics.keys())

            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=sorted(all_headers))
            self.csv_writer.writeheader()
            for data in existing_data:
                self.csv_writer.writerow(data)

        # At this point csv_writer is guaranteed to be not None
        assert self.csv_writer is not None
        self.csv_writer.writerow(row)
        if self.csv_file:
            self.csv_file.flush()
    
    def _read_existing_csv(self) -> List[Dict]:
        """Read existing CSV data."""
        data = []
        try:
            with open(self.csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
        except FileNotFoundError:
            pass
        return data
    
    def get_best(self, metric_name: str, mode: str = "max") -> tuple:
        """
        Get best value and step for a metric.
        
        Args:
            metric_name: Name of the metric.
            mode: 'max' or 'min'.
            
        Returns:
            Tuple of (best_value, best_step).
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None, None
        
        values = self.metrics[metric_name]
        if mode == "max":
            best_idx = max(range(len(values)), key=lambda i: values[i])
        else:
            best_idx = min(range(len(values)), key=lambda i: values[i])
        
        return values[best_idx], best_idx + 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all metrics."""
        summary = {
            "experiment_name": self.experiment_name,
            "num_steps": max(len(v) for v in self.metrics.values()) if self.metrics else 0,
        }
        
        import numpy as np
        
        for name, values in self.metrics.items():
            if values:
                summary[f"{name}_final"] = values[-1]
                summary[f"{name}_best"] = max(values) if "loss" not in name.lower() else min(values)
                summary[f"{name}_mean"] = np.mean(values)
                summary[f"{name}_std"] = np.std(values)
        
        return summary
    
    def save(self) -> None:
        """Save metrics to JSON file."""
        # Update metadata
        self.metadata["end_time"] = datetime.now().isoformat()
        
        # Prepare data
        data = {
            "metadata": self.metadata,
            "metrics": dict(self.metrics),
            "summary": self.get_summary(),
        }
        
        # Save JSON
        json_path = self.metrics_dir / f"{self.experiment_name}_metrics.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        
        # Close CSV
        if self.csv_file:
            self.csv_file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()


class ExperimentLogger:
    """
    High-level experiment logging combining metrics and text logs.
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "./outputs",
        log_level: int = logging.INFO,
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment.
            output_dir: Output directory.
            log_level: Logging level.
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup text logger
        log_file = self.output_dir / "experiment.log"
        self.logger = setup_logging(
            level=log_level,
            log_file=log_file,
            name=experiment_name,
        )
        
        # Setup metrics tracker
        self.metrics = MetricsTracker(self.output_dir, experiment_name)
        
        # Log start
        self.logger.info("=" * 60)
        self.logger.info(f"Starting experiment: {experiment_name}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 60)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def log_metrics(self, step: int, **kwargs) -> None:
        """
        Log metrics for a step.
        
        Args:
            step: Training step.
            **kwargs: Metric name-value pairs.
        """
        self.metrics.log(step, **kwargs)
        
        # Also log as text
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in kwargs.items())
        self.logger.info(f"Step {step}: {metrics_str}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Configuration saved to {config_path}")
    
    def finish(self) -> Dict[str, Any]:
        """Finish experiment and return summary."""
        summary = self.metrics.get_summary()
        self.metrics.save()
        
        self.logger.info("=" * 60)
        self.logger.info("Experiment completed")
        self.logger.info(f"Summary: {json.dumps(summary, indent=2)}")
        self.logger.info("=" * 60)
        
        return summary


class TensorBoardLogger:
    """
    TensorBoard logging wrapper (optional).
    
    Only imports tensorboard if available.
    """
    
    def __init__(self, log_dir: Path, enabled: bool = True):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs.
            enabled: Whether logging is enabled.
        """
        self.enabled = enabled
        self.writer = None
        
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(log_dir))
            except ImportError:
                logging.warning("TensorBoard not available. Install with: pip install tensorboard")
                self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalars."""
        if self.enabled and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log histogram."""
        if self.enabled and self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def close(self) -> None:
        """Close the writer."""
        if self.writer:
            self.writer.close()
