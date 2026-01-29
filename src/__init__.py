# =============================================================================
# Federated Learning for Skin Cancer Classification with DSCATNet
# =============================================================================
"""
Federated Learning for Skin Cancer Classification with DSCATNet.

This package provides the complete implementation of DSCATNet (Dual-Scale
Cross-Attention Vision Transformer) adapted for federated learning using
the Flower framework.

Modules:
    models: DSCATNet architecture and components
    federated: FL client, server, and simulation infrastructure
    centralized: Baseline centralized training
    data: Dataset loaders and preprocessing pipelines
    evaluation: Metrics and visualization utilities
    utils: Helper functions and checkpoint management

Supported Datasets:
    - HAM10000: 10,015 dermoscopy images, 7 classes
    - ISIC 2018: ~10k images, 7 classes
    - ISIC 2019: ~25k images, 8 classes + UNK
    - ISIC 2020: ~33k images, binary classification
    - PAD-UFES-20: 2,298 clinical images, 6 classes

Author: Leonardo Chen
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Leonardo Chen"

# =============================================================================
# Package Imports
# =============================================================================

# Make key modules accessible at package level (explicit re-exports)
from . import models as models
from . import federated as federated
from . import data as data
from . import centralized as centralized
from . import evaluation as evaluation
from . import utils as utils
