"""
Model architectures for skin cancer classification.
"""

from .dscatnet import DSCATNet
from .patch_embedding import DualScalePatchEmbedding
from .cross_attention import CrossScaleAttention

__all__ = ["DSCATNet", "DualScalePatchEmbedding", "CrossScaleAttention"]
