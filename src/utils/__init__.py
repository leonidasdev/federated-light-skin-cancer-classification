# =============================================================================
# Utils Module
# =============================================================================
"""
Utils Module.

Provides utility functions for logging, checkpoints, and common operations.
"""

# =============================================================================
# Logging Imports
# =============================================================================

# =============================================================================
# Checkpoint Imports
# =============================================================================
from .checkpoints import (
    CheckpointManager,
    load_model_for_inference,
    save_model_for_inference,
)

# =============================================================================
# Config Schema Imports
# =============================================================================
from .config_schema import (
    ConfigType,
    ExperimentConfig,
    FLConfig,
    ModelConfig,
    validate_config,
    validate_config_dict,
)

# =============================================================================
# Helper Imports
# =============================================================================
from .helpers import (
    autocast,
    compute_class_weights,
    count_parameters,
    create_grad_scaler,
    format_size,
    format_time,
    get_device,
    set_seed,
)
from .logging_utils import (
    ExperimentLogger,
    MetricsTracker,
    TensorBoardLogger,
    setup_logging,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Logging
    "setup_logging",
    "MetricsTracker",
    "ExperimentLogger",
    "TensorBoardLogger",
    # Checkpoints
    "CheckpointManager",
    "save_model_for_inference",
    "load_model_for_inference",
    # Helpers
    "set_seed",
    "get_device",
    "autocast",
    "count_parameters",
    "format_time",
    "format_size",
    "compute_class_weights",
    "create_grad_scaler",
    # Config Schema
    "ConfigType",
    "validate_config",
    "validate_config_dict",
    "ExperimentConfig",
    "FLConfig",
    "ModelConfig",
]
