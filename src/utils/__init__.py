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

from .logging_utils import (
    setup_logging,
    MetricsTracker,
    ExperimentLogger,
    TensorBoardLogger,
)

# =============================================================================
# Checkpoint Imports
# =============================================================================

from .checkpoints import (
    CheckpointManager,
    save_model_for_inference,
    load_model_for_inference,
)

# =============================================================================
# Helper Imports
# =============================================================================

from .helpers import (
    set_seed,
    get_device,
    autocast,
    count_parameters,
    format_time,
    format_size,
)

# =============================================================================
# Config Schema Imports
# =============================================================================

from .config_schema import (
    ConfigType,
    validate_config,
    validate_config_dict,
    get_default_config,
    ExperimentConfig,
    FLConfig,
    ModelConfig,
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
    # Config Schema
    "ConfigType",
    "validate_config",
    "validate_config_dict",
    "get_default_config",
    "ExperimentConfig",
    "FLConfig",
    "ModelConfig",
]
