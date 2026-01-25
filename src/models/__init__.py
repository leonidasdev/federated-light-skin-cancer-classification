"""
Model architectures for skin cancer classification.

This module provides:
    - DSCATNet: Dual-Scale Cross-Attention Vision Transformer
    - DualScalePatchEmbedding: Multi-resolution patch embedding
    - CrossScaleAttention: Cross-attention between feature scales
"""

from .dscatnet import DSCATNet, create_dscatnet, get_model_parameters, set_model_parameters
from .patch_embedding import DualScalePatchEmbedding
from .cross_attention import CrossScaleAttention, CrossScaleAttentionBlock

__all__ = [
    "DSCATNet",
    "create_dscatnet",
    "get_model_parameters",
    "set_model_parameters",
    "DualScalePatchEmbedding",
    "CrossScaleAttention",
    "CrossScaleAttentionBlock",
]
