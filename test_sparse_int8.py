#!/usr/bin/env python
"""
Test script for sparse_int8_attention.

Compares sparse (fp16) vs sparse_int8 across different sequence lengths,
measuring speed and accuracy.
"""

import argparse
import os
import sys
import glob

def _ensure_triton_cuda_stub_on_library_path():
    """Make libcuda.so stub discoverable for Triton JIT compilation."""
    for pattern in (
        '/usr/local/cuda*/targets/*/lib/stubs',
        '/usr/local/cuda*/lib64/stubs',
    ):
        for stub_dir in glob.glob(pattern):
            if os.path.exists(os.path.join(stub_dir, 'libcuda.so')):
                existing = os.environ.get('LIBRARY_PATH', '')
                if stub_dir not in existing.split(':'):
                    os.environ['LIBRARY_PATH'] = (
                        f"{stub_dir}:{existing}" if existing else stub_dir
                    )
                return

_ensure_triton_cuda_stub_on_library_path()

import torch
from attention.sparse import sparse_attention
from attention.sparse_int8 import sparse_int8_attention
from attention import sdpa_attention


def benchmark_stable(fn, q, k, v, warmup=10, iters=30):
    """Run stable benchmark with outlier removal."""
    for _ in range(warmup):
        _ = fn(q, k, v)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = fn(q, k, v)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    # Remove top/bottom 10% outliers
    times = sorted(times)[iters//10 : -iters//10]
    return sum(times) / len(times)


def compute_metrics(out, ref):
    """Three accuracy metrics on flattened float32 tensors.

    Cosine similarity: sum(O * O') / (sqrt(sum(O^2)) * sqrt(sum(O'^2)))
    Relative L1:       sum(|O - O'|) / sum(|O|)
    RMSE:              sqrt(mean((O - O')^2))
    """
    out_f = out.float().reshape(-1)
    ref_f = ref.float().reshape(-1)
    diff = out_f - ref_f

    cos_sim = (ref_f * out_f).sum() / (
        ref_f.pow(2).sum().sqrt() * out_f.pow(2).sum().sqrt() + 1e-12
    )
    rel_l1 = diff.abs().sum() / (ref_f.abs().sum() + 1e-12)
    rmse = diff.pow(2).mean().sqrt()
    return cos_sim.item(), rel_l1.item(), rmse.item()


def main():
    parser = argparse.ArgumentParser(description='Test sparse_int8_attention performance')
    parser.add_argument('--seq-lens', type=int, nargs='+', default=[1024, 2048, 4096, 8192, 16384, 32768],
                        help='Sequence lengths to test')
    parser.add_argument('--batch', type=int, default=2, help='Batch size')
    parser.add_argument('--num-heads', type=int, default=16, help='Number of heads')
    parser.add_argument('--head-dim', type=int, default=64, help='Head dimension')
    parser.add_argument('--topk-ratio', type=float, default=0.8, help='Top-k ratio for sparse attention')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16'])
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=30, help='Benchmark iterations')
    parser.add_argument('--txt', type=str, default=None,
                        help='Path to write text results (default: sparse_int8_results.txt)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('CUDA is required.')
        sys.exit(1)

    dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16
    B, H, D = args.batch, args.num_heads, args.head_dim

    header_lines = [
        '=' * 110,
        f'Sparse Int8 Test | GPU: {torch.cuda.get_device_name(0)}',
        f'Config: B={B}, H={H}, D={D}, topk={args.topk_ratio}, dtype={args.dtype}',
        f'Warmup: {args.warmup}, Iterations: {args.iters}',
        f"Metrics vs SDPA: CosSim (higher=better), RelL1 = sum|O-O'|/sum|O|, RMSE = sqrt(mean((O-O')^2))",
        '=' * 110,
        '',
        f'{"N":>6} | {"sparse(fp16)":>14} | {"sparse_int8":>14} | {"speedup":>10} | '
        f'{"CosSim":>10} | {"RelL1":>12} | {"RMSE":>12}',
        '-' * 110,
    ]

    txt_lines = list(header_lines)
    for line in header_lines:
        print(line)

    for N in args.seq_lens:
        torch.manual_seed(0)
        q = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
        k = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
        v = torch.randn(B, H, N, D, device='cuda', dtype=dtype)

        # Reference (SDPA)
        with torch.no_grad():
            ref = sdpa_attention(q, k, v).float()

        # Benchmark sparse (fp16)
        t_fp16 = benchmark_stable(
            lambda q, k, v: sparse_attention(q, k, v, topk_ratio=args.topk_ratio),
            q, k, v, args.warmup, args.iters
        )

        # Benchmark sparse_int8
        t_int8 = benchmark_stable(
            lambda q, k, v: sparse_int8_attention(q, k, v, topk_ratio=args.topk_ratio),
            q, k, v, args.warmup, args.iters
        )

        # Compute metrics vs SDPA reference (sparse_int8 only — that's what we evaluate)
        with torch.no_grad():
            out_int8 = sparse_int8_attention(q, k, v, topk_ratio=args.topk_ratio)
            cos_sim, rel_l1, rmse = compute_metrics(out_int8, ref)

        speedup = t_fp16 / t_int8
        symbol = '✓' if speedup > 1.0 else '✗'

        line = (f'{N:6d} | {t_fp16:10.3f} ms | {t_int8:10.3f} ms | {speedup:7.2f}x {symbol} | '
                f'{cos_sim:10.6f} | {rel_l1:8.2e} | {rmse:8.2e}')
        print(line)
        txt_lines.append(line)

    print('-' * 110)
    txt_lines.append('-' * 110)

    txt_path = args.txt or 'output/sparse_int8_results.txt'
    os.makedirs('output', exist_ok=True)
    with open(txt_path, 'w') as f:
        f.write('\n'.join(txt_lines) + '\n')
    print(f'\nResults written to: {txt_path}')


if __name__ == '__main__':
    main()
