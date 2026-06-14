"""
Vanilla Attention Implementation
"""

import torch


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
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    attn = torch.softmax(scores, dim=-1)
    if dropout_p and training:
        attn = torch.dropout(attn, dropout_p, train=True)

    return torch.matmul(attn.to(v.dtype), v)
