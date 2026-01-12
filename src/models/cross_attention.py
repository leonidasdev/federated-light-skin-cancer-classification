"""
Cross-Scale Attention Module for DSCATNet.

This module implements the cross-attention mechanism that enables information
exchange between fine-scale and coarse-scale representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CrossScaleAttention(nn.Module):
    """
    Cross-Scale Attention mechanism for DSCATNet.
    
    Enables bidirectional information flow between fine-scale and coarse-scale
    token representations through cross-attention.
    
    The fine-scale tokens query the coarse-scale tokens to get global context,
    while coarse-scale tokens query fine-scale tokens to get local details.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to add bias to QKV projections
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection
    """
    
    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        # Projections for fine-scale tokens
        self.fine_q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.fine_k = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.fine_v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        
        # Projections for coarse-scale tokens
        self.coarse_q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.coarse_k = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.coarse_v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        
        # Output projections
        self.fine_proj = nn.Linear(embed_dim, embed_dim)
        self.coarse_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention."""
        B, N, C = x.shape
        # (B, N, C) -> (B, num_heads, N, head_dim)
        return x.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    
    def _cross_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-attention.
        
        Args:
            query: (B, num_heads, N_q, head_dim)
            key: (B, num_heads, N_kv, head_dim)
            value: (B, num_heads, N_kv, head_dim)
            
        Returns:
            Attention output of shape (B, N_q, embed_dim)
        """
        B = query.shape[0]
        
        # Compute attention scores
        attn = (query @ key.transpose(-2, -1)) * self.scale  # (B, heads, N_q, N_kv)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = attn @ value  # (B, heads, N_q, head_dim)
        
        # Reshape back
        out = out.transpose(1, 2).reshape(B, -1, self.embed_dim)  # (B, N_q, embed_dim)
        
        return out
        
    def forward(
        self,
        fine_tokens: torch.Tensor,
        coarse_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional cross-attention between scales.
        
        Args:
            fine_tokens: Fine-scale tokens (B, N_fine, embed_dim)
            coarse_tokens: Coarse-scale tokens (B, N_coarse, embed_dim)
            
        Returns:
            Tuple of updated (fine_tokens, coarse_tokens)
        """
        # === Fine tokens attend to coarse tokens (get global context) ===
        fine_q = self._reshape_for_attention(self.fine_q(fine_tokens))
        coarse_k = self._reshape_for_attention(self.coarse_k(coarse_tokens))
        coarse_v = self._reshape_for_attention(self.coarse_v(coarse_tokens))
        
        fine_out = self._cross_attention(fine_q, coarse_k, coarse_v)
        fine_out = self.proj_drop(self.fine_proj(fine_out))
        
        # === Coarse tokens attend to fine tokens (get local details) ===
        coarse_q = self._reshape_for_attention(self.coarse_q(coarse_tokens))
        fine_k = self._reshape_for_attention(self.fine_k(fine_tokens))
        fine_v = self._reshape_for_attention(self.fine_v(fine_tokens))
        
        coarse_out = self._cross_attention(coarse_q, fine_k, fine_v)
        coarse_out = self.proj_drop(self.coarse_proj(coarse_out))
        
        return fine_out, coarse_out


class CrossScaleAttentionBlock(nn.Module):
    """
    Complete Cross-Scale Attention Block with residual connections and FFN.
    
    Each block performs:
    1. Cross-attention between scales (with residual)
    2. Self-attention within each scale (with residual)
    3. Feed-forward network (with residual)
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to embed_dim
        qkv_bias: Whether to add bias to QKV projections
        drop: Dropout rate
        attn_drop: Attention dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        
        # Layer norms for cross-attention
        self.norm_fine_cross = nn.LayerNorm(embed_dim)
        self.norm_coarse_cross = nn.LayerNorm(embed_dim)
        
        # Cross-scale attention
        self.cross_attn = CrossScaleAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # Layer norms for self-attention
        self.norm_fine_self = nn.LayerNorm(embed_dim)
        self.norm_coarse_self = nn.LayerNorm(embed_dim)
        
        # Self-attention for each scale
        self.fine_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True
        )
        self.coarse_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True
        )
        
        # Layer norms for FFN
        self.norm_fine_ffn = nn.LayerNorm(embed_dim)
        self.norm_coarse_ffn = nn.LayerNorm(embed_dim)
        
        # Feed-forward networks
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.fine_ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(drop)
        )
        self.coarse_ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(drop)
        )
        
    def forward(
        self,
        fine_tokens: torch.Tensor,
        coarse_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the cross-scale attention block.
        
        Args:
            fine_tokens: (B, N_fine, embed_dim)
            coarse_tokens: (B, N_coarse, embed_dim)
            
        Returns:
            Updated (fine_tokens, coarse_tokens)
        """
        # 1. Cross-attention with residual
        fine_norm = self.norm_fine_cross(fine_tokens)
        coarse_norm = self.norm_coarse_cross(coarse_tokens)
        fine_cross, coarse_cross = self.cross_attn(fine_norm, coarse_norm)
        fine_tokens = fine_tokens + fine_cross
        coarse_tokens = coarse_tokens + coarse_cross
        
        # 2. Self-attention with residual
        fine_norm = self.norm_fine_self(fine_tokens)
        fine_self, _ = self.fine_self_attn(fine_norm, fine_norm, fine_norm)
        fine_tokens = fine_tokens + fine_self
        
        coarse_norm = self.norm_coarse_self(coarse_tokens)
        coarse_self, _ = self.coarse_self_attn(coarse_norm, coarse_norm, coarse_norm)
        coarse_tokens = coarse_tokens + coarse_self
        
        # 3. FFN with residual
        fine_tokens = fine_tokens + self.fine_ffn(self.norm_fine_ffn(fine_tokens))
        coarse_tokens = coarse_tokens + self.coarse_ffn(self.norm_coarse_ffn(coarse_tokens))
        
        return fine_tokens, coarse_tokens
