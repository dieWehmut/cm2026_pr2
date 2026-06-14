"""
Block-sparse attention with Triton kernel.

Uses block indices instead of boolean mask for better efficiency.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math


def sparse_attention(q, k, v, attn_mask=None, topk_ratio=0.5, block_size=64):
    """
    Block-sparse attention with Triton kernel using block indices.

    Args:
        q: Query tensor, shape [B, num_heads, N, head_dim]
        k: Key tensor, shape [B, num_heads, N, head_dim]
        v: Value tensor, shape [B, num_heads, N, head_dim]
        attn_mask: Attention mask (optional, ignored for API compatibility)
        topk_ratio: Ratio of K blocks to select per Q block (default 0.5)
        block_size: Block size for selection (default 64)

    Returns:
        Output tensor, shape [B, num_heads, N, head_dim]
    """
    raise NotImplementedError("Sparse attention is not implemented yet.")
