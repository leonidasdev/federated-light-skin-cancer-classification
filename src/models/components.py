"""
LMS-ViT Components
==================

Building blocks for the LMS-ViT architecture.
Implements multi-scale patch embedding, attention mechanisms, and lightweight blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for converting images to token sequences.
    
    Uses a convolutional layer to project image patches into embedding space.
    
    Args:
        img_size: Input image size (assumes square images)
        patch_size: Size of each patch
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutional projection: maps patches to embedding dimension
        # kernel_size = stride = patch_size ensures non-overlapping patches
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Project patches: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        
        # Flatten spatial dimensions: (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose to (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x


class MultiScalePatchEmbedding(nn.Module):
    """
    Multi-scale patch embedding for capturing features at different resolutions.
    
    LMS-ViT uses multiple patch sizes to capture both fine and coarse features.
    
    Args:
        img_size: Input image size
        patch_sizes: Tuple of patch sizes for multi-scale embedding
        in_channels: Number of input channels
        embed_dim: Embedding dimension (will be split across scales)
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_sizes: Tuple[int, ...] = (4, 8, 16),
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_sizes = patch_sizes
        self.num_scales = len(patch_sizes)
        
        # Divide embedding dimension across scales
        self.embed_dims = [embed_dim // self.num_scales] * self.num_scales
        # Handle remainder
        self.embed_dims[-1] += embed_dim - sum(self.embed_dims)
        
        # Create patch embedding for each scale
        self.patch_embeds = nn.ModuleList([
            PatchEmbedding(
                img_size=img_size,
                patch_size=ps,
                in_channels=in_channels,
                embed_dim=ed,
            )
            for ps, ed in zip(patch_sizes, self.embed_dims)
        ])
        
        # Projection layers to unify token counts across scales
        # All scales will be projected to the token count of the largest patch size
        self.target_num_patches = (img_size // max(patch_sizes)) ** 2
        
        self.proj_layers = nn.ModuleList()
        for ps in patch_sizes:
            num_patches = (img_size // ps) ** 2
            if num_patches != self.target_num_patches:
                # Use adaptive pooling to match token counts
                self.proj_layers.append(
                    nn.AdaptiveAvgPool1d(self.target_num_patches)
                )
            else:
                self.proj_layers.append(nn.Identity())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale patch embedding.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Multi-scale embeddings of shape (B, num_patches, embed_dim)
        """
        embeddings = []
        
        for patch_embed, proj in zip(self.patch_embeds, self.proj_layers):
            # Get patch embeddings: (B, num_patches_i, embed_dim_i)
            emb = patch_embed(x)
            
            # Unify token counts: transpose, pool, transpose back
            # (B, num_patches_i, embed_dim_i) -> (B, target_num_patches, embed_dim_i)
            emb = proj(emb.transpose(1, 2)).transpose(1, 2)
            
            embeddings.append(emb)
        
        # Concatenate along embedding dimension
        # (B, target_num_patches, embed_dim)
        x = torch.cat(embeddings, dim=-1)
        
        return x


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism for capturing features at different scales.
    
    Implements attention with multi-scale key/value projections for
    efficient processing of dermoscopic images.
    
    Args:
        dim: Input/output dimension
        num_heads: Number of attention heads
        scales: Tuple of downsampling scales for K/V
        qkv_bias: Whether to include bias in QKV projections
        attn_drop: Attention dropout rate
        proj_drop: Output projection dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        scales: Tuple[int, ...] = (1, 2, 4),
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scales = scales
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query projection (full resolution)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Key and Value projections for each scale
        self.kv_projs = nn.ModuleList([
            nn.Linear(dim, dim * 2, bias=qkv_bias)
            for _ in scales
        ])
        
        # Downsampling layers for multi-scale K/V
        self.downsample = nn.ModuleList()
        for s in scales:
            if s > 1:
                self.downsample.append(
                    nn.AvgPool1d(kernel_size=s, stride=s)
                )
            else:
                self.downsample.append(nn.Identity())
        
        # Scale fusion weights (learnable)
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Multi-scale attention forward pass.
        
        Args:
            x: Input tensor of shape (B, N, C)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        
        # Query: (B, N, C) -> (B, num_heads, N, head_dim)
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Multi-scale attention
        attn_outputs = []
        weights = F.softmax(self.scale_weights, dim=0)
        
        for i, (kv_proj, downsample, w) in enumerate(zip(self.kv_projs, self.downsample, weights)):
            # Downsample tokens for this scale
            # (B, N, C) -> (B, C, N) -> pool -> (B, C, N') -> (B, N', C)
            x_scaled = downsample(x.transpose(1, 2)).transpose(1, 2)
            N_scaled = x_scaled.shape[1]
            
            # K, V projections: (B, N', 2*C) -> (B, N', 2, num_heads, head_dim)
            kv = kv_proj(x_scaled).reshape(B, N_scaled, 2, self.num_heads, self.head_dim)
            k, v = kv.unbind(2)  # Each: (B, N', num_heads, head_dim)
            
            # Transpose for attention: (B, num_heads, N', head_dim)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            
            # Attention: (B, num_heads, N, N')
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            
            # Weighted output: (B, num_heads, N, head_dim)
            out = attn @ v
            attn_outputs.append(out * w)
        
        # Combine multi-scale outputs
        x = sum(attn_outputs)
        
        # Reshape: (B, num_heads, N, head_dim) -> (B, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class LightweightMLP(nn.Module):
    """
    Efficient MLP with depthwise separable convolution-like structure.
    
    Uses a bottleneck design to reduce parameters while maintaining capacity.
    
    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension (default: 4x input)
        out_features: Output dimension (default: same as input)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        # Two-layer MLP with GELU activation
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, N, C)
            
        Returns:
            Output tensor of shape (B, N, C_out)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LightweightBlock(nn.Module):
    """
    Lightweight transformer block with multi-scale attention.
    
    Combines multi-scale attention with efficient MLP in a residual structure.
    
    Args:
        dim: Input/output dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        scales: Multi-scale attention scales
        dropout: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Stochastic depth rate
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        scales: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        
        # Layer normalization before attention (Pre-LN)
        self.norm1 = nn.LayerNorm(dim)
        
        # Multi-scale attention
        self.attn = MultiScaleAttention(
            dim=dim,
            num_heads=num_heads,
            scales=scales,
            attn_drop=attn_drop,
            proj_drop=dropout,
        )
        
        # Layer normalization before MLP
        self.norm2 = nn.LayerNorm(dim)
        
        # Lightweight MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LightweightMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout,
        )
        
        # Stochastic depth (drop path)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor of shape (B, N, C)
            
        Returns:
            Output tensor of shape (B, N, C)
        """
        # Attention block with residual
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # MLP block with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) regularization.
    
    Randomly drops entire residual branches during training.
    
    Args:
        drop_prob: Probability of dropping the path
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        # Work with batches of different sizes
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        
        output = x.div(keep_prob) * random_tensor
        return output


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for transformer.
    
    Args:
        num_patches: Number of patches (tokens)
        embed_dim: Embedding dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Initialize with truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (B, N, C) where N includes CLS token
            
        Returns:
            Position-encoded tensor of shape (B, N, C)
        """
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.pos_drop(x)
        return x
