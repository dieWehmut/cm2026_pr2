"""
Block-sparse attention with Triton kernel.

Uses block indices instead of boolean mask for better efficiency.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math


def _next_power_of_2(x):
    return 1 << (x - 1).bit_length()


def _make_block_indices(q, k, topk_ratio, block_size):
    B, H, Nq, D = q.shape
    Nk = k.shape[-2]
    num_q_blocks = triton.cdiv(Nq, block_size)
    num_k_blocks = triton.cdiv(Nk, block_size)
    topk = max(1, min(num_k_blocks, int(num_k_blocks * float(topk_ratio))))

    if topk >= num_k_blocks:
        idx = torch.arange(num_k_blocks, device=q.device, dtype=torch.int32)
        return idx.view(1, 1, 1, num_k_blocks).expand(B, H, num_q_blocks, num_k_blocks).contiguous()

    qf = q.float().contiguous()
    kf = k.float().contiguous()
    pad_q = num_q_blocks * block_size - Nq
    pad_k = num_k_blocks * block_size - Nk
    if pad_q:
        qf = F.pad(qf, (0, 0, 0, pad_q))
    if pad_k:
        kf = F.pad(kf, (0, 0, 0, pad_k))

    q_blocks = qf.view(B, H, num_q_blocks, block_size, D)
    k_blocks = kf.view(B, H, num_k_blocks, block_size, D)

    q_counts = torch.full((num_q_blocks,), block_size, device=q.device, dtype=torch.float32)
    k_counts = torch.full((num_k_blocks,), block_size, device=q.device, dtype=torch.float32)
    if pad_q:
        q_counts[-1] = block_size - pad_q
    if pad_k:
        k_counts[-1] = block_size - pad_k

    q_pool = q_blocks.sum(dim=3) / q_counts.view(1, 1, -1, 1)
    k_pool = k_blocks.sum(dim=3) / k_counts.view(1, 1, -1, 1)
    scores = torch.matmul(q_pool, k_pool.transpose(-2, -1))
    return torch.topk(scores, k=topk, dim=-1).indices.to(torch.int32).contiguous()


def _torch_sparse_attention(q, k, v, block_indices, block_size, attn_mask=None):
    B, H, Nq, D = q.shape
    Nk = k.shape[-2]
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (D ** -0.5)
    keep = torch.zeros((B, H, Nq, Nk), device=q.device, dtype=torch.bool)
    num_q_blocks = block_indices.shape[2]
    for qb in range(num_q_blocks):
        q0 = qb * block_size
        q1 = min(q0 + block_size, Nq)
        for t in range(block_indices.shape[-1]):
            kb = block_indices[:, :, qb, t]
            for b in range(B):
                for h in range(H):
                    k0 = int(kb[b, h].item()) * block_size
                    k1 = min(k0 + block_size, Nk)
                    keep[b, h, q0:q1, k0:k1] = True

    if attn_mask is not None:
        keep = keep & attn_mask if attn_mask.dtype == torch.bool else keep
        if attn_mask.dtype != torch.bool:
            scores = scores + attn_mask

    scores = scores.masked_fill(~keep, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn.to(v.dtype), v)


@triton.jit
def _sparse_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, idx_ptr, o_ptr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qn: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kd: tl.constexpr,
    stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vd: tl.constexpr,
    stride_ib: tl.constexpr, stride_ih: tl.constexpr, stride_iq: tl.constexpr, stride_it: tl.constexpr,
    stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_on: tl.constexpr, stride_od: tl.constexpr,
    n_q: tl.constexpr, n_k: tl.constexpr, head_dim: tl.constexpr,
    num_heads: tl.constexpr, topk: tl.constexpr,
    sm_scale: tl.constexpr,
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
        other=0.0,
    ).to(tl.float16)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for t in range(0, topk):
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
            other=0.0,
        ).to(tl.float16)
        v = tl.load(
            v_ptr
            + pid_b * stride_vb
            + pid_h * stride_vh
            + cols[:, None] * stride_vn
            + offs_d[None, :] * stride_vd,
            mask=(cols[:, None] < n_k) & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float16)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where((offs_m[:, None] < n_q) & (cols[None, :] < n_k), qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_new = tl.where(offs_m < n_q, m_new, 0.0)
        p = tl.exp2(qk - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
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
    block_indices = _make_block_indices(q, k, topk_ratio, block_size)
    if attn_mask is not None or not q.is_cuda:
        return _torch_sparse_attention(q, k, v, block_indices, block_size, attn_mask)

    B, H, Nq, D = q.shape
    Nk = k.shape[-2]
    out = torch.empty((B, H, Nq, D), device=q.device, dtype=v.dtype)
    block_d = max(16, _next_power_of_2(D))
    grid = (triton.cdiv(Nq, block_size), B * H)
    sm_scale = (1.0 / math.sqrt(D)) * 1.4426950408889634
    num_warps = 4 if block_d <= 64 else 8

    _sparse_attn_fwd_kernel[grid](
        q, k, v, block_indices, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        block_indices.stride(0), block_indices.stride(1), block_indices.stride(2), block_indices.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Nq, Nk, D, H, block_indices.shape[-1], sm_scale,
        BLOCK_M=block_size, BLOCK_N=block_size, BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
