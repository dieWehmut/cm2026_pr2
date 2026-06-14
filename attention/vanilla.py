"""
Vanilla Attention Implementation
"""

import torch
import torch.nn.functional as F


def vanilla_attention(q, k, v, attn_mask=None, dropout_p=0.0, training=False):
    """
    Vanilla attention implementation using pure PyTorch.

    Args:
        q: Query tensor, shape [B, num_heads, N, head_dim]
        k: Key tensor, shape [B, num_heads, N, head_dim]
        v: Value tensor, shape [B, num_heads, N, head_dim]
        attn_mask: Attention mask (optional, ignored for API compatibility)
        dropout_p: Dropout probability (default 0.0, ignored for API compatibility)
        training: Training mode flag (default False, ignored for API compatibility)

    Returns:
        Output tensor, shape [B, num_heads, N, head_dim]
    """
    raise NotImplementedError("Vanilla attention is not implemented yet.")
