#!/usr/bin/env python
"""
Benchmark attention implementations.

Measures forward-pass latency of each attention backend across a sweep of
sequence lengths, head counts, and head dims, and reports the numerical error
of each backend relative to PyTorch SDPA (treated as the reference).

Backends:
  - sdpa        : torch.nn.functional.scaled_dot_product_attention (reference)
  - vanilla     : pure-PyTorch softmax attention
  - triton_fa2  : Triton FlashAttention-2
  - sparse      : Triton block-sparse attention (top-k block selection)

Usage:
  python benchmark_attention.py
  python benchmark_attention.py --seq-lens 1024 4096 --head-dims 64 72 128
  python benchmark_attention.py --topk-ratios 0.3 0.5 1.0 --csv out.csv
"""

import argparse
import glob
import os
import sys


def _ensure_triton_cuda_stub_on_library_path():
    """Triton JIT links against -lcuda; make sure a 64-bit libcuda stub is on
    LIBRARY_PATH. No-op if one can't be found or is already present."""
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

import torch  # noqa: E402

# Make the local `attention` package importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention import (  # noqa: E402
    sdpa_attention,
    vanilla_attention,
    flash_attention_2,
    sparse_attention,
    sparse_int8_attention,
)


def _time_fn(fn, warmup, iters):
    """Return mean latency in milliseconds over `iters` runs (after `warmup`)."""
    # Warmup (also triggers Triton JIT compilation on first call)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter


def _compute_metrics(out, ref):
    """Three accuracy metrics computed on flattened float32 tensors.

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

    return {
        'cos_sim': cos_sim.item(),
        'rel_l1': rel_l1.item(),
        'rmse': rmse.item(),
    }



def _build_backends(topk_ratios):
    """Return ordered list of (name, callable) attention backends to test."""
    backends = [
        ('sdpa', sdpa_attention),
        # ('vanilla', vanilla_attention),
        ('triton_fa2', flash_attention_2),
    ]
    for r in topk_ratios:
        backends.append((f'sparse(topk={r})',
                         lambda q, k, v, r=r: sparse_attention(q, k, v, topk_ratio=r)))
        backends.append((f'sparse_int8(topk={r})',
                         lambda q, k, v, r=r: sparse_int8_attention(q, k, v, topk_ratio=r)))
    return backends


def run_config(B, H, N, D, dtype, backends, warmup, iters):
    """Benchmark all backends for one (B,H,N,D) config. Returns list of rows."""
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
    k = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
    v = torch.randn(B, H, N, D, device='cuda', dtype=dtype)

    # Reference output (SDPA) for error comparison
    with torch.no_grad():
        ref = sdpa_attention(q, k, v).float()

    rows = []
    for name, fn in backends:
        try:
            with torch.no_grad():
                out = fn(q, k, v)
                metrics = _compute_metrics(out, ref)
                ms = _time_fn(lambda: fn(q, k, v), warmup, iters)
            rows.append((name, ms, metrics, None))
        except Exception as e:
            rows.append((name, None, None, str(e).splitlines()[0][:60]))
        finally:
            torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description='Benchmark attention backends.')
    parser.add_argument('--batch', type=int, default=2, help='Batch size (default 2)')
    parser.add_argument('--seq-lens', type=int, nargs='+', default=[2048, 4096, 8192, 16384],
                        help='Sequence lengths to sweep')
    parser.add_argument('--num-heads', type=int, nargs='+', default=[8, 16],
                        help='Head counts to sweep')
    parser.add_argument('--head-dims', type=int, nargs='+', default=[64, 128],
                        help='Head dims to sweep')
    parser.add_argument('--topk-ratios', type=float, nargs='+', default=[0.3, 0.5, 0.8, 0.9, 1.0],
                        help='Sparse attention top-k ratios to test')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=50, help='Timed iterations')
    parser.add_argument('--txt', type=str, default=None, help='Path to write text results (default: benchmark_results.txt)')
    parser.add_argument('--csv', type=str, default=None, help='Optional path to write CSV results')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('CUDA is required for this benchmark.')
        sys.exit(1)

    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.dtype]
    backends = _build_backends(args.topk_ratios)

    print('=' * 110)
    print(f'Attention Benchmark | GPU: {torch.cuda.get_device_name(0)} | '
          f'dtype: {args.dtype} | batch: {args.batch} | warmup: {args.warmup} | iters: {args.iters}')
    print('Metrics vs SDPA: CosSim (higher=better), RelL1 = sum|O-O\'|/sum|O|, RMSE = sqrt(mean((O-O\')^2)). Time = ms/forward.')
    print('=' * 110)

    csv_rows = [('B', 'H', 'N', 'D', 'backend', 'time_ms', 'cos_sim', 'rel_l1', 'rmse', 'error_msg')]
    txt_lines = [
        '=' * 110,
        f'Attention Benchmark Report',
        f'GPU: {torch.cuda.get_device_name(0)}',
        f'dtype: {args.dtype} | batch: {args.batch} | warmup: {args.warmup} iters | measurement: {args.iters} iters',
        f'Metrics vs SDPA: CosSim (higher=better), RelL1 = sum|O-O\'|/sum|O|, RMSE = sqrt(mean((O-O\')^2))',
        '=' * 110,
        '',
    ]

    for D in args.head_dims:
        for H in args.num_heads:
            for N in args.seq_lens:
                header = f'[B={args.batch}, H={H}, N={N}, D={D}]'
                print(f'\n{header}')
                print(f'  {"backend":<22}{"time (ms)":>12}{"vs SDPA":>10}{"CosSim":>14}{"RelL1":>14}{"RMSE":>14}')
                print('  ' + '-' * 86)

                txt_lines.append(f'\n{header}')
                txt_lines.append(f'  {"backend":<22}{"time (ms)":>12}{"vs SDPA":>10}{"CosSim":>14}{"RelL1":>14}{"RMSE":>14}')
                txt_lines.append('  ' + '-' * 86)

                rows = run_config(args.batch, H, N, D, dtype, backends, args.warmup, args.iters)
                sdpa_ms = next((ms for nm, ms, _, _ in rows if nm == 'sdpa' and ms is not None), None)

                for name, ms, metrics, msg in rows:
                    if msg is not None:
                        line = f'  {name:<22}{"FAILED":>12}{"":>10}{"":>14}{"":>14}{"":>14}  {msg}'
                        print(line)
                        txt_lines.append(line)
                    else:
                        speedup = f'{sdpa_ms / ms:.2f}x' if (sdpa_ms and ms) else '-'
                        if name == 'sdpa':
                            cos_str = rel_str = rmse_str = '(ref)'
                        else:
                            cos_str = f'{metrics["cos_sim"]:.6f}'
                            rel_str = f'{metrics["rel_l1"]:.2e}'
                            rmse_str = f'{metrics["rmse"]:.2e}'
                        line = f'  {name:<22}{ms:>12.4f}{speedup:>10}{cos_str:>14}{rel_str:>14}{rmse_str:>14}'
                        print(line)
                        txt_lines.append(line)

                    csv_rows.append((args.batch, H, N, D, name,
                                     f'{ms:.4f}' if ms is not None else '',
                                     f'{metrics["cos_sim"]:.6f}' if metrics is not None else '',
                                     f'{metrics["rel_l1"]:.6e}' if metrics is not None else '',
                                     f'{metrics["rmse"]:.6e}' if metrics is not None else '',
                                     msg or ''))

    # Write txt report
    txt_path = args.txt or 'output/benchmark_results.txt'
    os.makedirs('output', exist_ok=True)
    with open(txt_path, 'w') as f:
        f.write('\n'.join(txt_lines))
    print(f'\nResults written to: {txt_path}')

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            csv.writer(f).writerows(csv_rows)
        print(f'CSV written to: {args.csv}')

    print('\nDone.')



if __name__ == '__main__':
    main()

