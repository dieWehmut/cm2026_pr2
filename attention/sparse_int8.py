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

from .sparse import _make_block_indices, _next_power_of_2, _torch_sparse_attention


@triton.jit
def _quantize_per_block_kernel(
    x_ptr, xq_ptr, scale_ptr,
    stride_xb: tl.constexpr, stride_xh: tl.constexpr, stride_xn: tl.constexpr, stride_xd: tl.constexpr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qn: tl.constexpr, stride_qd: tl.constexpr,
    stride_sb: tl.constexpr, stride_sh: tl.constexpr, stride_sn: tl.constexpr,
    n: tl.constexpr, head_dim: tl.constexpr, num_heads: tl.constexpr,
    SCALE_MULT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_block = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh - pid_b * num_heads

    offs_m = pid_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask = (offs_m[:, None] < n) & (offs_d[None, :] < head_dim)
    x = tl.load(
        x_ptr
        + pid_b * stride_xb
        + pid_h * stride_xh
        + offs_m[:, None] * stride_xn
        + offs_d[None, :] * stride_xd,
        mask=mask,
        other=0.0,
    )

    amax = tl.max(tl.max(tl.abs(x), axis=0), axis=0)
    base_scale = tl.maximum(amax / 127.0, 1.0e-8)
    y = x / base_scale
    y = tl.where(y >= 0.0, tl.floor(y + 0.5), -tl.floor(-y + 0.5))
    y = tl.minimum(tl.maximum(y, -127.0), 127.0)

    tl.store(
        xq_ptr
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qn
        + offs_d[None, :] * stride_qd,
        y.to(tl.int8),
        mask=mask,
    )
    tl.store(
        scale_ptr
        + pid_b * stride_sb
        + pid_h * stride_sh
        + pid_block * stride_sn,
        base_scale * SCALE_MULT,
    )


@triton.jit
def _sparse_int8_attn_fwd_kernel(
    q_ptr, k_ptr, qs_ptr, ks_ptr, v_ptr, idx_ptr, o_ptr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qn: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kd: tl.constexpr,
    stride_qsb: tl.constexpr, stride_qsh: tl.constexpr, stride_qsn: tl.constexpr,
    stride_ksb: tl.constexpr, stride_ksh: tl.constexpr, stride_ksn: tl.constexpr,
    stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vd: tl.constexpr,
    stride_ib: tl.constexpr, stride_ih: tl.constexpr, stride_iq: tl.constexpr, stride_it: tl.constexpr,
    stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_on: tl.constexpr, stride_od: tl.constexpr,
    n_q: tl.constexpr, n_k: tl.constexpr, head_dim: tl.constexpr,
    num_heads: tl.constexpr, topk: tl.constexpr,
    USE_INT8_DOT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh - pid_b * num_heads

    offs_m = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qn
        + offs_d[None, :] * stride_qd,
        mask=(offs_m[:, None] < n_q) & (offs_d[None, :] < head_dim),
        other=0,
    )
    q_scale = tl.load(
        qs_ptr
        + pid_b * stride_qsb
        + pid_h * stride_qsh
        + pid_q * stride_qsn
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for t in tl.range(0, topk):
        block_id = tl.load(
            idx_ptr
            + pid_b * stride_ib
            + pid_h * stride_ih
            + pid_q * stride_iq
            + t * stride_it
        )
        cols = block_id * BLOCK_N + offs_n
        k = tl.load(
            k_ptr
            + pid_b * stride_kb
            + pid_h * stride_kh
            + cols[:, None] * stride_kn
            + offs_d[None, :] * stride_kd,
            mask=(cols[:, None] < n_k) & (offs_d[None, :] < head_dim),
            other=0,
        )
        k_scale = tl.load(
            ks_ptr
            + pid_b * stride_ksb
            + pid_h * stride_ksh
            + block_id * stride_ksn
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

        if USE_INT8_DOT:
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.int32).to(tl.float32)
        else:
            qk = tl.dot(q.to(tl.float16), tl.trans(k).to(tl.float16), out_dtype=tl.float32)
        qk = qk * (q_scale * k_scale)
        qk = tl.where((offs_m[:, None] < n_q) & (cols[None, :] < n_k), qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_new = tl.where(offs_m < n_q, m_new, 0.0)
        p = tl.exp2(qk - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
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
    block_indices = _make_block_indices(q, k, topk_ratio, block_size)
    if attn_mask is not None or not q.is_cuda:
        return _torch_sparse_attention(q, k, v, block_indices, block_size, attn_mask)

    B, H, Nq, D = q.shape
    Nk = k.shape[-2]
    num_q_blocks = triton.cdiv(Nq, block_size)
    num_k_blocks = triton.cdiv(Nk, block_size)
    block_d = max(16, _next_power_of_2(D))
    q_int8 = torch.empty((B, H, Nq, D), device=q.device, dtype=torch.int8)
    k_int8 = torch.empty((B, H, Nk, D), device=k.device, dtype=torch.int8)
    q_scale = torch.empty((B, H, num_q_blocks), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((B, H, num_k_blocks), device=k.device, dtype=torch.float32)

    k_for_quant = k - k.float().mean(dim=-2, keepdim=True).to(k.dtype) if smooth_k else k
    q_mult = (1.0 / math.sqrt(D)) * 1.4426950408889634

    num_warps = 4 if block_d <= 64 else 8
    _quantize_per_block_kernel[(num_q_blocks, B * H)](
        q, q_int8, q_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        q_int8.stride(0), q_int8.stride(1), q_int8.stride(2), q_int8.stride(3),
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
        Nq, D, H, q_mult,
        BLOCK_M=block_size, BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=3,
    )
    _quantize_per_block_kernel[(num_k_blocks, B * H)](
        k_for_quant, k_int8, k_scale,
        k_for_quant.stride(0), k_for_quant.stride(1), k_for_quant.stride(2), k_for_quant.stride(3),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        Nk, D, H, 1.0,
        BLOCK_M=block_size, BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=3,
    )

    out = torch.empty((B, H, Nq, D), device=q.device, dtype=v.dtype)
    use_int8_dot = torch.cuda.get_device_capability(q.device)[0] >= 8
    _sparse_int8_attn_fwd_kernel[(num_q_blocks, B * H)](
        q_int8, k_int8, q_scale, k_scale, v, block_indices, out,
        q_int8.stride(0), q_int8.stride(1), q_int8.stride(2), q_int8.stride(3),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
        q_scale.stride(0), q_scale.stride(1), q_scale.stride(2),
        k_scale.stride(0), k_scale.stride(1), k_scale.stride(2),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        block_indices.stride(0), block_indices.stride(1), block_indices.stride(2), block_indices.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Nq, Nk, D, H, block_indices.shape[-1], use_int8_dot,
        BLOCK_M=block_size, BLOCK_N=block_size, BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
