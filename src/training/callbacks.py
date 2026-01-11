"""
Training Callbacks
==================

Callback functions for training monitoring and control.
"""

from typing import Dict, Optional
import numpy as np


class EarlyStopping:
    """
    Early stopping to terminate training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max' depending on the metric
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False
    
    def __call__(
        self,
        trainer,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
    ) -> None:
        if val_metrics is None:
            return
        
        score = val_metrics.get('accuracy', val_metrics.get('loss'))
        
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                trainer.logger.info(
                    f"Early stopping triggered after {trainer.current_epoch + 1} epochs"
                )
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


class ModelCheckpoint:
    """
    Save model checkpoint when validation metric improves.
    
    Args:
        filepath: Path to save checkpoint
        monitor: Metric to monitor
        mode: 'min' or 'max'
        save_best_only: Only save when metric improves
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = "accuracy",
        mode: str = "max",
        save_best_only: bool = True,
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score: Optional[float] = None
    
    def __call__(
        self,
        trainer,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
    ) -> None:
        if val_metrics is None:
            return
        
        score = val_metrics.get(self.monitor, 0)
        
        should_save = False
        if not self.save_best_only:
            should_save = True
        elif self.best_score is None:
            should_save = True
            self.best_score = score
        elif self._is_improvement(score):
            should_save = True
            self.best_score = score
        
        if should_save:
            trainer.save_checkpoint(self.filepath)
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score
        else:
            return score < self.best_score


class LearningRateScheduler:
    """
    Wrapper callback for learning rate schedulers.
    
    Args:
        scheduler: PyTorch learning rate scheduler
        step_on: When to step ('epoch' or 'batch')
    """
    
    def __init__(
        self,
        scheduler,
        step_on: str = "epoch",
    ):
        self.scheduler = scheduler
        self.step_on = step_on
    
    def __call__(
        self,
        trainer,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
    ) -> None:
        if self.step_on == "epoch":
            self.scheduler.step()
        
        current_lr = self.scheduler.get_last_lr()[0]
        trainer.logger.info(f"Learning rate: {current_lr:.2e}")
