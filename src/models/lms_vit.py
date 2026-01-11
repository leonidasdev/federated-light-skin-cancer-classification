"""
LMS-ViT: Lightweight Multi-Scale Vision Transformer
====================================================

Implementation of the LMS-ViT architecture (2025) for skin lesion classification.

Key Features:
- Multi-scale patch embedding for capturing features at different resolutions
- Multi-scale attention mechanism for efficient processing
- Lightweight transformer blocks with depthwise separable operations
- Designed for dermoscopic image classification

Architecture Overview:
    Input Image (224x224x3)
         ↓
    Multi-Scale Patch Embedding (patch sizes: 4, 8, 16)
         ↓
    + CLS Token + Positional Encoding
         ↓
    Lightweight Transformer Blocks (x depth)
         ↓
    CLS Token → Classification Head
         ↓
    Output Logits (num_classes)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .components import (
    MultiScalePatchEmbedding,
    LightweightBlock,
    PositionalEncoding,
    DropPath,
)


class LMSViT(nn.Module):
    """
    Lightweight Multi-Scale Vision Transformer for skin cancer classification.
    
    This architecture is designed specifically for dermoscopic images, using
    multi-scale processing to capture both fine-grained texture details and
    broader structural patterns in skin lesions.
    
    Args:
        img_size: Input image size (default: 224)
        patch_sizes: Tuple of patch sizes for multi-scale embedding (default: (4, 8, 16))
        in_channels: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 7 for HAM10000)
        embed_dim: Embedding dimension (default: 384)
        depth: Number of transformer blocks (default: 12)
        num_heads: Number of attention heads (default: 6)
        mlp_ratio: MLP hidden dim ratio (default: 4.0)
        dropout: Dropout rate (default: 0.1)
        attn_drop: Attention dropout rate (default: 0.0)
        drop_path: Stochastic depth rate (default: 0.1)
        scales: Multi-scale attention scales (default: (1, 2, 4))
    
    Example:
        >>> model = LMSViT(num_classes=7)
        >>> x = torch.randn(2, 3, 224, 224)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([2, 7])
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_sizes: Tuple[int, ...] = (4, 8, 16),
        in_channels: int = 3,
        num_classes: int = 7,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
        scales: Tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        
        # Store configuration
        self.img_size = img_size
        self.patch_sizes = patch_sizes
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        
        # =================================================================
        # 1. Multi-Scale Patch Embedding
        # =================================================================
        # Converts input image to sequence of patch embeddings at multiple scales
        # Different patch sizes capture different levels of detail:
        # - Small patches (4x4): Fine texture, hair, small features
        # - Medium patches (8x8): Local patterns, color variations
        # - Large patches (16x16): Global structure, lesion shape
        self.patch_embed = MultiScalePatchEmbedding(
            img_size=img_size,
            patch_sizes=patch_sizes,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        
        # Calculate number of patches (using largest patch size for token count)
        self.num_patches = (img_size // max(patch_sizes)) ** 2
        
        # =================================================================
        # 2. CLS Token
        # =================================================================
        # Learnable classification token prepended to patch sequence
        # Aggregates global information through attention
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # =================================================================
        # 3. Positional Encoding
        # =================================================================
        # Adds learnable position information to tokens
        # num_patches + 1 for CLS token
        self.pos_encoding = PositionalEncoding(
            num_patches=self.num_patches,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        
        # =================================================================
        # 4. Transformer Encoder
        # =================================================================
        # Stack of Lightweight Transformer Blocks
        # Uses stochastic depth with linearly increasing drop probability
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        self.blocks = nn.ModuleList([
            LightweightBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                scales=scales,
                dropout=dropout,
                attn_drop=attn_drop,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # =================================================================
        # 5. Classification Head
        # =================================================================
        # Projects CLS token to class logits
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        
        # =================================================================
        # Weight Initialization
        # =================================================================
        self._init_weights()
    
    def _init_weights(self) -> None:
        """
        Initialize model weights using truncated normal distribution.
        
        This initialization strategy helps with training stability:
        - Linear layers: truncated normal with std=0.02
        - LayerNorm: bias=0, weight=1
        - CLS token: truncated normal with std=0.02
        """
        # Initialize CLS token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize all other weights
        self.apply(self._init_weights_module)
    
    def _init_weights_module(self, m: nn.Module) -> None:
        """Initialize weights for a single module."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, embed_dim)
        """
        B = x.shape[0]
        
        # Step 1: Multi-scale patch embedding
        # (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        
        # Step 2: Prepend CLS token
        # (1, 1, embed_dim) -> (B, 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # (B, num_patches, embed_dim) -> (B, num_patches + 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Step 3: Add positional encoding
        x = self.pos_encoding(x)
        
        # Step 4: Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Step 5: Layer normalization
        x = self.norm(x)
        
        # Step 6: Extract CLS token for classification
        # (B, num_patches + 1, embed_dim) -> (B, embed_dim)
        cls_output = x[:, 0]
        
        return cls_output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Classification logits of shape (B, num_classes)
        """
        # Extract features
        features = self.forward_features(x)
        
        # Classification head
        logits = self.head(features)
        
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention maps for visualization.
        
        Useful for interpretability and understanding what the model
        focuses on in dermoscopic images.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of attention maps from each transformer block
            
        TODO: Implement attention map extraction
        - Modify forward pass to store attention weights
        - Return attention maps for each layer
        """
        # TODO: Implement attention map extraction for visualization
        # This requires modifying the attention module to return attention weights
        raise NotImplementedError(
            "Attention map extraction not yet implemented. "
            "Requires modifying MultiScaleAttention to return attention weights."
        )
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """
        Get the number of model parameters.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Model Variants
# =============================================================================

def lmsvit_tiny(num_classes: int = 7, **kwargs) -> LMSViT:
    """
    LMS-ViT Tiny variant for resource-constrained environments.
    
    ~5M parameters
    """
    return LMSViT(
        embed_dim=192,
        depth=6,
        num_heads=3,
        num_classes=num_classes,
        **kwargs
    )


def lmsvit_small(num_classes: int = 7, **kwargs) -> LMSViT:
    """
    LMS-ViT Small variant - balanced performance/efficiency.
    
    ~22M parameters
    """
    return LMSViT(
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
        **kwargs
    )


def lmsvit_base(num_classes: int = 7, **kwargs) -> LMSViT:
    """
    LMS-ViT Base variant for maximum performance.
    
    ~86M parameters
    """
    return LMSViT(
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes,
        **kwargs
    )
