"""
Block-sparse attention with int8 Q/K quantization (Triton).

Adapted from thu-ml/SpargeAttn (Apache-2.0):
  - Triton_SpargeAttn/triton_kernel_example.py
  - spas_sage_attn/quant_per_block.py

Compared to attention/sparse.py, this variant:
  - Quantizes Q and K to int8 per-block (V stays fp16).
  - Bakes 1.44269504/sqrt(d) into Q's scale so the inner softmax uses exp2.
  - Uses int8 tensor-core matmul (tl.dot of int8 inputs -> int32, dequantized
    by `* q_scale * k_scale`).
  - Optional smooth_k (subtract per-channel K mean) — mathematically a no-op
    for softmax output but improves K's int8 dynamic range.
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def sparse_int8_attention(q, k, v, attn_mask=None, topk_ratio=0.5,
                          block_size=64, smooth_k=True):
    """Block-sparse attention with int8 Q/K quantization.

    Args:
        q: Query tensor, shape [B, num_heads, N, head_dim]
        k: Key tensor, shape [B, num_heads, N, head_dim]
        v: Value tensor, shape [B, num_heads, N, head_dim]
        attn_mask: Attention mask (optional, ignored for API compatibility)
        topk_ratio: Ratio of K blocks to select per Q block (default 0.5)
        block_size: Block size for selection and kernel tiles (default 64)
        smooth_k: Whether to subtract per-channel K mean before quantization (default True)

    Returns:
        Output tensor, shape [B, num_heads, N, head_dim]
    """
    raise NotImplementedError("Sparse int8 attention is not implemented yet.")

