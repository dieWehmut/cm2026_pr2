"""
Triton-based Flash Attention 2 implementation (forward only).
Optimized for inference without backward pass.
"""

import math

import torch
import triton
import triton.language as tl


FA2_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(configs=FA2_CONFIGS, key=["n_q", "head_dim"])
@triton.jit
def _flash_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    sm_scale,
    n_q: tl.constexpr, n_k: tl.constexpr, batch: tl.constexpr, num_heads: tl.constexpr,
    head_dim: tl.constexpr, block_d: tl.constexpr, out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    y_q_dim = batch * num_heads * n_q
    y_k_dim = batch * num_heads * n_k

    desc_q = _maybe_make_tensor_desc(
        q_ptr,
        shape=[y_q_dim, head_dim],
        strides=[head_dim, 1],
        block_shape=[BLOCK_M, block_d],
    )
    desc_k = _maybe_make_tensor_desc(
        k_ptr,
        shape=[y_k_dim, head_dim],
        strides=[head_dim, 1],
        block_shape=[BLOCK_N, block_d],
    )
    desc_v = _maybe_make_tensor_desc(
        v_ptr,
        shape=[y_k_dim, head_dim],
        strides=[head_dim, 1],
        block_shape=[BLOCK_N, block_d],
    )
    desc_o = _maybe_make_tensor_desc(
        o_ptr,
        shape=[y_q_dim, head_dim],
        strides=[head_dim, 1],
        block_shape=[BLOCK_M, block_d],
    )

    q_offset_y = pid_bh * n_q
    k_offset_y = pid_bh * n_k
    q_start_m = pid_m * BLOCK_M
    q = desc_q.load([q_offset_y + q_start_m, 0])

    m_i = tl.zeros((BLOCK_M,), tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M,), tl.float32) + 1.0
    acc = tl.zeros((BLOCK_M, block_d), tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    for start_n in tl.range(0, n_k, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([k_offset_y + start_n, 0]).T
        qk = tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = desc_v.load([k_offset_y + start_n, 0])
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    desc_o.store([q_offset_y + q_start_m, 0], acc.to(out_dtype))


def _next_power_of_2(x):
    return 1 << (x - 1).bit_length()


def flash_attention_2(q, k, v, attn_mask=None, dropout_p=0.0, training=False):
    """
    Flash Attention 2 forward pass using Triton.

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
    if attn_mask is not None:
        raise NotImplementedError("flash_attention_2 does not support attn_mask in this project.")
    if dropout_p and training:
        raise NotImplementedError("flash_attention_2 is inference-only and does not support dropout.")
    if not q.is_cuda:
        from .vanilla import vanilla_attention
        return vanilla_attention(q, k, v)

    B, H, Nq, D = q.shape
    Nk = k.shape[-2]
    assert k.shape == (B, H, Nk, D)
    assert v.shape == (B, H, Nk, D)

    q = q.contiguous().reshape(B * H * Nq, D)
    k = k.contiguous().reshape(B * H * Nk, D)
    v = v.contiguous().reshape(B * H * Nk, D)
    out = torch.empty((B * H * Nq, D), device=q.device, dtype=v.dtype)

    block_d = _next_power_of_2(D)
    if block_d < 16:
        block_d = 16

    out_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }.get(v.dtype, tl.float16)

    grid = lambda META: (triton.cdiv(Nq, META["BLOCK_M"]), B * H)
    sm_scale = 1.0 / math.sqrt(D)

    _flash_attn_fwd_kernel[grid](
        q, k, v, out,
        sm_scale,
        Nq, Nk, B, H,
        D,
        block_d,
        out_dtype,
    )

    return out.view(B, H, Nq, D)
