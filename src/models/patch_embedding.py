"""
Dual-Scale Patch Embedding Module for DSCATNet.

This module creates patch embeddings at two different scales (8×8 and 16×16)
to capture both fine-grained and coarse-grained features from dermoscopic images.
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Tuple


class PatchEmbedding(nn.Module):
    """
    Single-scale patch embedding layer.
    
    Converts an image into a sequence of patch embeddings using a convolutional layer.
    
    Args:
        img_size: Input image size (assumes square images)
        patch_size: Size of each patch
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension for each patch
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Convolutional projection to create patch embeddings
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Layer normalization for stable training
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.projection(x)
        
        # (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = rearrange(x, 'b e h w -> b (h w) e')
        
        # Apply layer normalization
        x = self.norm(x)
        
        return x


class DualScalePatchEmbedding(nn.Module):
    """
    Dual-Scale Patch Embedding for DSCATNet.
    
    Creates two parallel streams of patch embeddings:
    - Fine-scale: 8×8 patches (more patches, fine-grained details)
    - Coarse-scale: 16×16 patches (fewer patches, global context)
    
    Args:
        img_size: Input image size
        fine_patch_size: Size of fine-scale patches (default: 8)
        coarse_patch_size: Size of coarse-scale patches (default: 16)
        in_channels: Number of input channels
        embed_dim: Embedding dimension for each patch
    """
    
    def __init__(
        self,
        img_size: int = 224,
        fine_patch_size: int = 8,
        coarse_patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384
    ):
        super().__init__()
        
        self.img_size = img_size
        self.fine_patch_size = fine_patch_size
        self.coarse_patch_size = coarse_patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches for each scale
        self.num_fine_patches = (img_size // fine_patch_size) ** 2    # 784 for 224/8
        self.num_coarse_patches = (img_size // coarse_patch_size) ** 2  # 196 for 224/16
        
        # Fine-scale embedding (8×8 patches)
        self.fine_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=fine_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Coarse-scale embedding (16×16 patches)
        self.coarse_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=coarse_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Learnable class tokens for each scale
        self.fine_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.coarse_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings for each scale (+1 for CLS token)
        self.fine_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_fine_patches + 1, embed_dim)
        )
        self.coarse_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_coarse_patches + 1, embed_dim)
        )
        
        # Initialize parameters
        self._init_weights()
        
    def _init_weights(self):
        """Initialize class tokens and positional embeddings."""
        nn.init.trunc_normal_(self.fine_cls_token, std=0.02)
        nn.init.trunc_normal_(self.coarse_cls_token, std=0.02)
        nn.init.trunc_normal_(self.fine_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.coarse_pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create dual-scale patch embeddings.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of:
                - fine_tokens: Shape (B, num_fine_patches + 1, embed_dim)
                - coarse_tokens: Shape (B, num_coarse_patches + 1, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Create patch embeddings at both scales
        fine_patches = self.fine_embedding(x)    # (B, 784, embed_dim)
        coarse_patches = self.coarse_embedding(x)  # (B, 196, embed_dim)
        
        # Expand class tokens for batch
        fine_cls = self.fine_cls_token.expand(batch_size, -1, -1)
        coarse_cls = self.coarse_cls_token.expand(batch_size, -1, -1)
        
        # Prepend class tokens
        fine_tokens = torch.cat([fine_cls, fine_patches], dim=1)
        coarse_tokens = torch.cat([coarse_cls, coarse_patches], dim=1)
        
        # Add positional embeddings
        fine_tokens = fine_tokens + self.fine_pos_embed
        coarse_tokens = coarse_tokens + self.coarse_pos_embed
        
        return fine_tokens, coarse_tokens
    
    def get_num_patches(self) -> Tuple[int, int]:
        """Return the number of patches for each scale (excluding CLS token)."""
        return self.num_fine_patches, self.num_coarse_patches
