# =============================================================================
# Model Architectures for Skin Cancer Classification
# =============================================================================
"""
Model architectures for skin cancer classification.

This module provides:
    - DSCATNet: Dual-Scale Cross-Attention Vision Transformer
    - DualScalePatchEmbedding: Multi-resolution patch embedding
    - CrossScaleAttention: Cross-attention between feature scales
"""

# =============================================================================
# Model Imports
# =============================================================================

from .dscatnet import DSCATNet, create_dscatnet, get_model_parameters, set_model_parameters, load_pretrained_vit_weights
from .patch_embedding import DualScalePatchEmbedding
from .cross_attention import CrossScaleAttention, CrossScaleAttentionBlock

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "DSCATNet",
    "create_dscatnet",
    "load_pretrained_vit_weights",
    "get_model_parameters",
    "set_model_parameters",
    "DualScalePatchEmbedding",
    "CrossScaleAttention",
    "CrossScaleAttentionBlock",
]
