"""
Checkpoint Utilities.

Helpers for saving, loading, and managing model checkpoints.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manage model checkpoints with automatic cleanup.
    
    Keeps track of checkpoints and optionally removes old ones
    to save disk space.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 5,
        keep_best: bool = True,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints.
            max_checkpoints: Maximum number of checkpoints to keep.
            keep_best: Whether to always keep the best checkpoint.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        
        self.checkpoints: List[Path] = []
        self.best_checkpoint: Optional[Path] = None
        self.best_metric: Optional[float] = None
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save.
            optimizer: Optional optimizer state.
            scheduler: Optional scheduler state.
            epoch: Current epoch/round.
            metrics: Optional metrics dict.
            is_best: Whether this is the best model so far.
            filename: Custom filename.
            
        Returns:
            Path to saved checkpoint.
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        
        logger.debug(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_checkpoint = best_path
            logger.info(f"Saved best model (epoch {epoch})")
        
        # Cleanup old checkpoints
        self._cleanup()
        
        return checkpoint_path
    
    def _cleanup(self) -> None:
        """Remove old checkpoints to stay under max limit."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Keep the most recent checkpoints
        to_remove = self.checkpoints[:-self.max_checkpoints]
        self.checkpoints = self.checkpoints[-self.max_checkpoints:]
        
        for path in to_remove:
            # Don't remove best checkpoint
            if self.keep_best and path == self.best_checkpoint:
                continue
            if path.exists():
                path.unlink()
                logger.debug(f"Removed old checkpoint: {path}")
    
    def load(
        self,
        model: nn.Module,
        checkpoint_path: Optional[Path] = None,
        load_best: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            model: Model to load weights into.
            checkpoint_path: Path to checkpoint (or None for best/latest).
            load_best: Whether to load best model.
            optimizer: Optional optimizer to restore.
            scheduler: Optional scheduler to restore.
            
        Returns:
            Checkpoint dict with metadata.
        """
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.checkpoint_dir / "best_model.pt"
            else:
                # Find latest
                checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
                if not checkpoints:
                    raise FileNotFoundError("No checkpoints found")
                checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model from {checkpoint_path}")
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.debug("Restored optimizer state")
        
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.debug("Restored scheduler state")
        
        return checkpoint
    
    def get_checkpoints(self) -> List[Path]:
        """Get list of all checkpoint paths."""
        return list(self.checkpoint_dir.glob("checkpoint_*.pt"))


def save_model_for_inference(
    model: nn.Module,
    save_path: Path,
    config: Optional[Dict[str, Any]] = None,
    include_optimizer: bool = False,
) -> None:
    """
    Save model in a format suitable for inference.
    
    Args:
        model: Trained model.
        save_path: Path to save.
        config: Optional model configuration.
        include_optimizer: Whether to include optimizer (usually False for inference).
    """
    save_dict = {
        "model_state_dict": model.state_dict(),
    }
    
    if config:
        save_dict["config"] = config
    
    torch.save(save_dict, save_path)
    logger.info(f"Saved model for inference: {save_path}")


def load_model_for_inference(
    model: nn.Module,
    load_path: Path,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Load model for inference.
    
    Args:
        model: Model architecture (uninitialized weights ok).
        load_path: Path to saved model.
        device: Device to load onto.
        
    Returns:
        Model with loaded weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(load_path, map_location=device)
    
    # Handle both full checkpoint and inference-only saves
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {load_path}")
    return model
