# =============================================================================
# Utility Functions
# =============================================================================
"""
Utility Functions.

Common helper functions used across the project.
"""

# =============================================================================
# Imports
# =============================================================================

import logging
import platform
import random
import sys
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "autocast",
    "collect_environment_info",
    "compute_class_weights",
    "count_parameters",
    "create_grad_scaler",
    "format_size",
    "format_time",
    "get_device",
    "set_seed",
]

# =============================================================================
# Reproducibility
# =============================================================================


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.debug("Random seed set to %d", seed)


# =============================================================================
# Device Utilities
# =============================================================================


def get_device(device: str | None = None) -> torch.device:
    """
    Get torch device.

    Args:
        device: Device string ('cuda', 'cpu', or None for auto).

    Returns:
        torch.device object.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug("Using device: %s", device)
    return torch.device(device)


# =============================================================================
# AMP Compatibility
# =============================================================================
# Use ``torch.amp.autocast`` if available (PyTorch >= 2.0),
# otherwise fall back to ``torch.cuda.amp.autocast``.

try:
    _HAS_TORCH_AMP_AUTOCAST = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
except Exception:
    _HAS_TORCH_AMP_AUTOCAST = False


def autocast() -> "torch.autocast":
    """Return the appropriate autocast context manager for the current PyTorch version.

    Returns:
        Context manager for automatic mixed precision.
    """
    if _HAS_TORCH_AMP_AUTOCAST:
        return torch.amp.autocast("cuda")  # type: ignore[attr-defined]
    return torch.cuda.amp.autocast()  # type: ignore[attr-defined]


def create_grad_scaler() -> Any:
    """Create a GradScaler instance, handling PyTorch version differences.

    Returns:
        A ``torch.amp.GradScaler`` (or ``torch.cuda.amp.GradScaler`` on older PyTorch versions).
    """
    amp_mod = getattr(torch, "amp", None)
    scaler_cls = getattr(amp_mod, "GradScaler", None) if amp_mod else None

    if scaler_cls is not None:
        try:
            return scaler_cls(device_type="cuda")
        except TypeError:
            return scaler_cls()

    from torch.cuda.amp import GradScaler as _GradScaler  # type: ignore[attr-defined]

    return _GradScaler()


# =============================================================================
# Model Utilities
# =============================================================================


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model.
        trainable_only: Whether to count only trainable parameters.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def compute_class_weights(
    label_counts: dict[int, int],
    num_classes: int,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets.

    Uses the formula: weight_c = N_total / (C * N_c)

    Args:
        label_counts: Mapping of class index to sample count.
        num_classes: Total number of classes.

    Returns:
        Float tensor of shape (num_classes,) with per-class weights.
    """
    total = sum(label_counts.values())
    weights = torch.zeros(num_classes)
    for cls, count in label_counts.items():
        if 0 <= cls < num_classes and count > 0:
            weights[cls] = total / (num_classes * count)
        elif 0 <= cls < num_classes and count == 0:
            logger.warning("Class %d has 0 samples; weight set to 0.0", cls)
    return weights


# =============================================================================
# Formatting Utilities
# =============================================================================


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted string (e.g., "2h 15m 30s").
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


def format_size(size_bytes: int) -> str:
    """
    Format bytes into human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string (e.g., "1.5 GB").
    """
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size) < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


# =============================================================================
# Environment Info
# =============================================================================


def collect_environment_info() -> dict[str, Any]:
    """Collect hardware and software environment information for reproducibility.

    Returns:
        Dictionary with python, pytorch, cuda, and GPU details.
    """
    info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        info["cudnn_version"] = str(torch.backends.cudnn.version())

    try:
        import timm as _timm

        info["timm_version"] = _timm.__version__
    except (ImportError, AttributeError):
        pass

    try:
        import flwr

        info["flower_version"] = flwr.__version__
    except (ImportError, AttributeError):
        pass

    return info
