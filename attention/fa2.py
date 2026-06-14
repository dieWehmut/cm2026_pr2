"""
Triton-based Flash Attention 2 implementation (forward only).
Optimized for inference without backward pass.
"""

import torch
import triton
import triton.language as tl
import math


FA2_CONFIGS = [
    triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn}, num_stages=stages, num_warps=warps)
    for bm in [64, 128]
    for bn in [32, 64, 128]
    for stages in [2, 3, 4]
    for warps in [4, 8]
]


@triton.autotune(configs=FA2_CONFIGS, key=["n_q", "n_k", "head_dim"])
@triton.jit
def _flash_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qn: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kd: tl.constexpr,
    stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vd: tl.constexpr,
    stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_on: tl.constexpr, stride_od: tl.constexpr,
    n_q: tl.constexpr, n_k: tl.constexpr, head_dim: tl.constexpr,
    num_heads: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh - pid_b * num_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qn
        + offs_d[None, :] * stride_qd,
        mask=(offs_m[:, None] < n_q) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in tl.range(0, n_k, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cols = start_n + offs_n
        k = tl.load(
            k_ptr
            + pid_b * stride_kb
            + pid_h * stride_kh
            + cols[:, None] * stride_kn
            + offs_d[None, :] * stride_kd,
            mask=(cols[:, None] < n_k) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        v = tl.load(
            v_ptr
            + pid_b * stride_vb
            + pid_h * stride_vh
            + cols[:, None] * stride_vn
            + offs_d[None, :] * stride_vd,
            mask=(cols[:, None] < n_k) & (offs_d[None, :] < head_dim),
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where((offs_m[:, None] < n_q) & (cols[None, :] < n_k), qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_new = tl.where(offs_m < n_q, m_new, 0.0)
        p = tl.exp2(qk - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_new
        l_i = l_new

    acc = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
    tl.store(
        o_ptr
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_on
        + offs_d[None, :] * stride_od,
        acc,
        mask=(offs_m[:, None] < n_q) & (offs_d[None, :] < head_dim),
    )


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

    out = torch.empty((B, H, Nq, D), device=q.device, dtype=v.dtype)
    block_d = _next_power_of_2(D)
    if block_d < 16:
        block_d = 16

    grid = lambda META: (triton.cdiv(Nq, META["BLOCK_M"]), B * H)
    sm_scale = (1.0 / math.sqrt(D)) * 1.4426950408889634

    _flash_attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Nq, Nk, D, H, sm_scale,
        BLOCK_D=block_d,
    )

    return out

