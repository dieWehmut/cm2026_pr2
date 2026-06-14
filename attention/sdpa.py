"""
SDPA Attention Implementation
使用 PyTorch 原生 scaled_dot_product_attention
"""

import torch
import torch.nn.functional as F


def sdpa_attention(q, k, v, attn_mask=None, dropout_p=0.0, training=False):
    """
    SDPA attention implementation using PyTorch's scaled_dot_product_attention.

    Args:
        q: Query tensor, shape [B, num_heads, N, head_dim]
        k: Key tensor, shape [B, num_heads, N, head_dim]
        v: Value tensor, shape [B, num_heads, N, head_dim]
        attn_mask: Attention mask (optional, shape [B, num_heads, N, N] or broadcastable)
        dropout_p: Dropout probability (default 0.0)
        training: Training mode flag (default False)

    Returns:
        Output tensor, shape [B, num_heads, N, head_dim]
    """
    # Use PyTorch's optimized scaled_dot_product_attention
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if training else 0.0,
        is_causal=False
    )

    return out
