"""
DSCATNet: Dual-Scale Cross-Attention Vision Transformer for Skin Cancer Classification.

This is the main model architecture that combines:
1. Dual-scale patch embeddings (8×8 and 16×16)
2. Cross-attention between scales
3. Lightweight transformer encoder
4. Global average pooling
5. Softmax classifier

Reference: Adapted for Federated Learning based on the original DSCATNet paper (PLOS ONE 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .patch_embedding import DualScalePatchEmbedding
from .cross_attention import CrossScaleAttentionBlock


class DSCATNet(nn.Module):
    """
    Dual-Scale Cross-Attention Vision Transformer (DSCATNet).
    
    A lightweight vision transformer designed for dermoscopic image classification
    that captures both fine-grained local features and global contextual information
    through dual-scale patch embeddings and cross-attention mechanisms.
    
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
        fusion_method: str = 'concat'
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
            embed_dim=embed_dim
        )
        
        # Dropout after embedding
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks with cross-scale attention
        self.blocks = nn.ModuleList([
            CrossScaleAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        # Final layer normalization
        self.norm_fine = nn.LayerNorm(embed_dim)
        self.norm_coarse = nn.LayerNorm(embed_dim)
        
        # Fusion and classification head
        if fusion_method == 'concat':
            # Concatenate CLS tokens from both scales
            self.fusion = nn.Linear(embed_dim * 2, embed_dim)
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(drop_rate),
                nn.Linear(embed_dim, num_classes)
            )
        elif fusion_method == 'add':
            # Add CLS tokens from both scales
            self.fusion = nn.Identity()
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(drop_rate),
                nn.Linear(embed_dim, num_classes)
            )
        elif fusion_method == 'attention':
            # Learnable attention-based fusion
            self.fusion_attention = nn.Linear(embed_dim, 1)
            self.fusion = nn.Identity()
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(drop_rate),
                nn.Linear(embed_dim, num_classes)
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
        fine_cls = fine_tokens[:, 0]    # (B, embed_dim)
        coarse_cls = coarse_tokens[:, 0]  # (B, embed_dim)
        
        # Fuse dual-scale representations
        if self.fusion_method == 'concat':
            fused = torch.cat([fine_cls, coarse_cls], dim=-1)  # (B, embed_dim * 2)
            fused = self.fusion(fused)  # (B, embed_dim)
        elif self.fusion_method == 'add':
            fused = fine_cls + coarse_cls  # (B, embed_dim)
        elif self.fusion_method == 'attention':
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
    
    def get_model_config(self) -> Dict[str, Any]:
        """Return model configuration dictionary."""
        return {
            'img_size': self.img_size,
            'num_classes': self.num_classes,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'fusion_method': self.fusion_method,
            'num_parameters': self.get_num_parameters()
        }


def create_dscatnet(
    num_classes: int = 7,
    img_size: int = 224,
    variant: str = 'base',
    **kwargs
) -> DSCATNet:
    """
    Factory function to create DSCATNet variants.
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size
        variant: Model variant ('tiny', 'small', 'base')
        **kwargs: Additional arguments passed to DSCATNet
        
    Returns:
        Configured DSCATNet model
    """
    variants = {
        'tiny': {
            'embed_dim': 192,
            'depth': 4,
            'num_heads': 3,
            'mlp_ratio': 3.0
        },
        'small': {
            'embed_dim': 384,
            'depth': 6,
            'num_heads': 6,
            'mlp_ratio': 4.0
        },
        'base': {
            'embed_dim': 384,
            'depth': 8,
            'num_heads': 6,
            'mlp_ratio': 4.0
        }
    }
    
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")
    
    config = variants[variant]
    config.update(kwargs)
    
    return DSCATNet(
        img_size=img_size,
        num_classes=num_classes,
        **config
    )


# Convenience functions for FL
def get_model_parameters(model: nn.Module) -> list:
    """Get model parameters as a list of numpy arrays (for Flower)."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_parameters(model: nn.Module, parameters: list) -> None:
    """Set model parameters from a list of numpy arrays (for Flower)."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
