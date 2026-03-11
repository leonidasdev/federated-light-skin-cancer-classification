# =============================================================================
# DSCATNet: Dual-Scale Cross-Attention Vision Transformer
# =============================================================================
"""
DSCATNet: Dual-Scale Cross-Attention Vision Transformer for Skin Cancer Classification.

This is the main model architecture that combines:
1. Dual-scale patch embeddings (8x8 and 16x16)
2. Cross-attention between scales
3. Lightweight transformer encoder
4. CLS token extraction and fusion
5. Softmax classifier

Reference: Adapted for Federated Learning based on the original DSCATNet paper (PLOS ONE 2024)
"""

# =============================================================================
# Imports
# =============================================================================

import torch
from torch import nn
import torch.nn.functional as F
from typing import Any
import logging
import numpy as np

try:
    import timm
except ImportError:
    timm = None  # type: ignore[assignment]

from .patch_embedding import DualScalePatchEmbedding
from .cross_attention import CrossScaleAttentionBlock

# =============================================================================
# Main Model
# =============================================================================


class DSCATNet(nn.Module):
    """
    Dual-Scale Cross-Attention Vision Transformer (DSCATNet).

    A lightweight vision transformer designed for dermoscopic image classification
    that captures both fine-grained local features and global contextual information
    through dual-scale patch embeddings and cross-attention mechanisms.

    Note:
        The original PONE paper (Yadav et al., 2024) reports ~22M parameters for the
        small variant. This implementation yields ~29.4M parameters due to separate
        Q/K/V projections in the bidirectional cross-attention (12 linear projections
        per block) and independent self-attention modules per scale. The architectural
        behavior follows the paper; the difference is in parameter counting.

    Args:
        img_size: Input image size (default: 224)
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 7 for HAM10000)
        embed_dim: Embedding dimension (default: 384)
        depth: Number of transformer blocks (default: 6)
        num_heads: Number of attention heads (default: 6)
        mlp_ratio: MLP hidden dim ratio (default: 4.0)
        fine_patch_size: Fine-scale patch size (default: 8)
        coarse_patch_size: Coarse-scale patch size (default: 16)
        drop_rate: Dropout rate (default: 0.1)
        attn_drop_rate: Attention dropout rate (default: 0.0)
        fusion_method: How to fuse dual-scale features ('concat', 'add', 'attention')
    """

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 7,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        fine_patch_size: int = 8,
        coarse_patch_size: int = 16,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
        fusion_method: str = "concat",
    ):
        super().__init__()

        self.img_size = img_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.fusion_method = fusion_method

        # Dual-scale patch embedding
        self.patch_embed = DualScalePatchEmbedding(
            img_size=img_size,
            fine_patch_size=fine_patch_size,
            coarse_patch_size=coarse_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Dropout after embedding
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks with cross-scale attention
        self.blocks = nn.ModuleList(
            [
                CrossScaleAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(depth)
            ]
        )

        # Final layer normalization
        self.norm_fine = nn.LayerNorm(embed_dim)
        self.norm_coarse = nn.LayerNorm(embed_dim)

        # Fusion and classification head
        if fusion_method == "concat":
            # Concatenate CLS tokens from both scales
            self.fusion = nn.Linear(embed_dim * 2, embed_dim)
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim), nn.Dropout(drop_rate), nn.Linear(embed_dim, num_classes)
            )
        elif fusion_method == "add":
            # Add CLS tokens from both scales
            self.fusion = nn.Identity()
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim), nn.Dropout(drop_rate), nn.Linear(embed_dim, num_classes)
            )
        elif fusion_method == "attention":
            # Learnable attention-based fusion
            self.fusion_attention = nn.Linear(embed_dim, 1)
            self.fusion = nn.Identity()
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim), nn.Dropout(drop_rate), nn.Linear(embed_dim, num_classes)
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize linear and normalization layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input image.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Fused feature representation of shape (B, embed_dim)
        """
        # Create dual-scale patch embeddings
        fine_tokens, coarse_tokens = self.patch_embed(x)

        # Apply dropout
        fine_tokens = self.pos_drop(fine_tokens)
        coarse_tokens = self.pos_drop(coarse_tokens)

        # Process through transformer blocks
        for block in self.blocks:
            fine_tokens, coarse_tokens = block(fine_tokens, coarse_tokens)

        # Apply final layer norm
        fine_tokens = self.norm_fine(fine_tokens)
        coarse_tokens = self.norm_coarse(coarse_tokens)

        # Extract CLS tokens
        fine_cls = fine_tokens[:, 0]  # (B, embed_dim)
        coarse_cls = coarse_tokens[:, 0]  # (B, embed_dim)

        # Fuse dual-scale representations
        if self.fusion_method == "concat":
            fused = torch.cat([fine_cls, coarse_cls], dim=-1)  # (B, embed_dim * 2)
            fused = self.fusion(fused)  # (B, embed_dim)
        elif self.fusion_method == "add":
            fused = fine_cls + coarse_cls  # (B, embed_dim)
        elif self.fusion_method == "attention":
            # Stack CLS tokens
            cls_stack = torch.stack([fine_cls, coarse_cls], dim=1)  # (B, 2, embed_dim)
            # Compute attention weights
            attn_weights = F.softmax(self.fusion_attention(cls_stack), dim=1)  # (B, 2, 1)
            # Weighted sum
            fused = (attn_weights * cls_stack).sum(dim=1)  # (B, embed_dim)

        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Class logits of shape (B, num_classes)
        """
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_config(self) -> dict[str, Any]:
        """Return model configuration dictionary."""
        return {
            "img_size": self.img_size,
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "fusion_method": self.fusion_method,
            "num_parameters": self.get_num_parameters(),
        }


def load_pretrained_vit_weights(model: DSCATNet, variant: str = "small") -> None:
    """
    Load pretrained ViT-Small (ImageNet-21k) weights into DSCATNet.

    Maps compatible layers from timm's vit_small_patch16_224 to DSCATNet's
    dual-stream architecture:
    - ViT blocks 0..depth-1   → fine-scale self-attention + FFN
    - ViT blocks depth..2*depth-1 → coarse-scale self-attention + FFN
    - ViT patch_embed (16×16) → coarse-scale patch embedding
    - ViT pos_embed / cls_token → coarse-scale positional embedding / CLS token

    Cross-attention layers, fine-scale patch embedding, fusion, and classifier
    remain randomly initialized.

    Args:
        model: DSCATNet model instance to load weights into.
        variant: Model variant ('small' or 'paper'). Other variants skip loading.
    """
    logger = logging.getLogger(__name__)

    if variant not in ("small", "paper"):
        logger.warning(
            f"Pretrained ViT weight loading only supported for 'small' and 'paper' variants, got '{variant}'. Skipping."
        )
        return

    if timm is None:
        logger.warning("timm is not installed. Skipping pretrained weight loading.")
        return

    logger.info("Loading pretrained ViT-Small (patch16_224) weights from timm...")
    vit = timm.create_model("vit_small_patch16_224", pretrained=True)
    vit_sd = vit.state_dict()
    del vit  # free memory

    depth = model.depth  # 6 for small
    vit_depth = 12  # ViT-Small has 12 blocks
    mapped_sd: dict[str, torch.Tensor] = {}

    for i in range(depth):
        fine_vit_idx = i
        coarse_vit_idx = i + depth

        # Fine-scale self-attention ← ViT block fine_vit_idx
        mapping = {
            f"blocks.{i}.fine_self_attn.in_proj_weight": f"blocks.{fine_vit_idx}.attn.qkv.weight",
            f"blocks.{i}.fine_self_attn.in_proj_bias": f"blocks.{fine_vit_idx}.attn.qkv.bias",
            f"blocks.{i}.fine_self_attn.out_proj.weight": f"blocks.{fine_vit_idx}.attn.proj.weight",
            f"blocks.{i}.fine_self_attn.out_proj.bias": f"blocks.{fine_vit_idx}.attn.proj.bias",
            f"blocks.{i}.norm_fine_self.weight": f"blocks.{fine_vit_idx}.norm1.weight",
            f"blocks.{i}.norm_fine_self.bias": f"blocks.{fine_vit_idx}.norm1.bias",
            f"blocks.{i}.norm_fine_ffn.weight": f"blocks.{fine_vit_idx}.norm2.weight",
            f"blocks.{i}.norm_fine_ffn.bias": f"blocks.{fine_vit_idx}.norm2.bias",
            f"blocks.{i}.fine_ffn.0.weight": f"blocks.{fine_vit_idx}.mlp.fc1.weight",
            f"blocks.{i}.fine_ffn.0.bias": f"blocks.{fine_vit_idx}.mlp.fc1.bias",
            f"blocks.{i}.fine_ffn.3.weight": f"blocks.{fine_vit_idx}.mlp.fc2.weight",
            f"blocks.{i}.fine_ffn.3.bias": f"blocks.{fine_vit_idx}.mlp.fc2.bias",
        }

        # Coarse-scale self-attention ← ViT block coarse_vit_idx
        if coarse_vit_idx < vit_depth:
            mapping.update(
                {
                    f"blocks.{i}.coarse_self_attn.in_proj_weight": f"blocks.{coarse_vit_idx}.attn.qkv.weight",
                    f"blocks.{i}.coarse_self_attn.in_proj_bias": f"blocks.{coarse_vit_idx}.attn.qkv.bias",
                    f"blocks.{i}.coarse_self_attn.out_proj.weight": f"blocks.{coarse_vit_idx}.attn.proj.weight",
                    f"blocks.{i}.coarse_self_attn.out_proj.bias": f"blocks.{coarse_vit_idx}.attn.proj.bias",
                    f"blocks.{i}.norm_coarse_self.weight": f"blocks.{coarse_vit_idx}.norm1.weight",
                    f"blocks.{i}.norm_coarse_self.bias": f"blocks.{coarse_vit_idx}.norm1.bias",
                    f"blocks.{i}.norm_coarse_ffn.weight": f"blocks.{coarse_vit_idx}.norm2.weight",
                    f"blocks.{i}.norm_coarse_ffn.bias": f"blocks.{coarse_vit_idx}.norm2.bias",
                    f"blocks.{i}.coarse_ffn.0.weight": f"blocks.{coarse_vit_idx}.mlp.fc1.weight",
                    f"blocks.{i}.coarse_ffn.0.bias": f"blocks.{coarse_vit_idx}.mlp.fc1.bias",
                    f"blocks.{i}.coarse_ffn.3.weight": f"blocks.{coarse_vit_idx}.mlp.fc2.weight",
                    f"blocks.{i}.coarse_ffn.3.bias": f"blocks.{coarse_vit_idx}.mlp.fc2.bias",
                }
            )

        for dscatnet_key, vit_key in mapping.items():
            if vit_key in vit_sd:
                mapped_sd[dscatnet_key] = vit_sd[vit_key]

    # Transfer coarse-scale patch embedding (16×16 kernel matches ViT)
    mapped_sd["patch_embed.coarse_embedding.projection.weight"] = vit_sd["patch_embed.proj.weight"]
    mapped_sd["patch_embed.coarse_embedding.projection.bias"] = vit_sd["patch_embed.proj.bias"]

    # Transfer coarse positional embedding and CLS token (same sequence length)
    mapped_sd["patch_embed.coarse_pos_embed"] = vit_sd["pos_embed"]
    mapped_sd["patch_embed.coarse_cls_token"] = vit_sd["cls_token"]

    # Transfer ViT final norm → DSCATNet norm_coarse
    mapped_sd["norm_coarse.weight"] = vit_sd["norm.weight"]
    mapped_sd["norm_coarse.bias"] = vit_sd["norm.bias"]

    # Load mapped weights (strict=False leaves unmapped params as-is)
    missing, _ = model.load_state_dict(mapped_sd, strict=False)

    loaded_count = len(mapped_sd)
    total_count = len(model.state_dict())
    logger.info(
        f"Loaded {loaded_count}/{total_count} parameter tensors from ViT-Small. "
        f"{len(missing)} remaining with random init "
        f"(cross-attention, fine-scale embeddings, fusion, classifier)."
    )


def create_dscatnet(num_classes: int = 7, img_size: int = 224, variant: str = "base", **kwargs) -> DSCATNet:
    """
    Factory function to create DSCATNet variants.

    Variants:
        - tiny: embed_dim=192, depth=4, heads=3 (~5M params) — fast prototyping
        - small: embed_dim=384, depth=6, heads=6 (~29.4M params) — balanced
        - paper: embed_dim=384, depth=6, heads=12 (~29.4M params) — matches paper H=12
        - base: embed_dim=384, depth=8, heads=6 (~39M params) — larger capacity

    Args:
        num_classes: Number of output classes
        img_size: Input image size
        variant: Model variant ('tiny', 'small', 'paper', 'base')
        **kwargs: Additional arguments passed to DSCATNet

    Returns:
        Configured DSCATNet model

    Raises:
        ValueError: If variant is not recognized.
    """
    variants = {
        "tiny": {"embed_dim": 192, "depth": 4, "num_heads": 3, "mlp_ratio": 3.0},
        "small": {"embed_dim": 384, "depth": 6, "num_heads": 6, "mlp_ratio": 4.0},
        "paper": {
            # Matches paper Section 5.8: H=12 heads, D=384 (unified adaptation
            # of paper's asymmetric D=192/768), depth=6, MLP ratio=4.0.
            # Compatible with ViT-Small (patch16_224) pretrained weights.
            "embed_dim": 384,
            "depth": 6,
            "num_heads": 12,
            "mlp_ratio": 4.0,
        },
        "base": {"embed_dim": 384, "depth": 8, "num_heads": 6, "mlp_ratio": 4.0},
    }

    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")

    config = variants[variant]
    config.update(kwargs)

    # Extract pretrained flag before filtering config keys
    pretrained = bool(config.pop("pretrained", False))

    # Filter config to only keys accepted by DSCATNet.__init__
    accepted_keys = {
        "in_channels",
        "embed_dim",
        "depth",
        "num_heads",
        "mlp_ratio",
        "fine_patch_size",
        "coarse_patch_size",
        "drop_rate",
        "attn_drop_rate",
        "fusion_method",
    }

    extra_keys = set(config.keys()) - accepted_keys
    if extra_keys:
        logger = logging.getLogger(__name__)
        logger.debug(f"create_dscatnet: ignoring unknown keys: {sorted(extra_keys)}")

    filtered_config = {k: v for k, v in config.items() if k in accepted_keys}

    model = DSCATNet(img_size=img_size, num_classes=num_classes, **filtered_config)

    if pretrained:
        load_pretrained_vit_weights(model, variant=variant)

    return model


# =============================================================================
# FL Utility Functions
# =============================================================================


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    """Get model parameters as a list of numpy arrays.

    Used by Flower FL framework for parameter serialization.

    Args:
        model: PyTorch model instance.

    Returns:
        List of numpy arrays, one per state dict entry.
    """
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Set model parameters from a list of numpy arrays.

    Used by Flower FL framework for parameter deserialization.
    Ensures tensors are placed on the same device as existing model parameters.

    Args:
        model: PyTorch model instance.
        parameters: List of numpy arrays matching the model's state dict.
    """
    device = next(model.parameters()).device
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v, device=device) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
