#!/usr/bin/env python
"""
Test T2I Inference with different attention modes
调用 PixArt inference.py 进行图片生成，支持选择 attention 类型
"""

import argparse
import os
import glob
import subprocess
import sys
from pathlib import Path


def _ensure_triton_cuda_stub_on_library_path():
    """
    Triton JIT-compiles a small CUDA utility at runtime and links against
    `-lcuda`. On this box the only 64-bit libcuda lives under the CUDA
    toolkit `stubs` dir, so we prepend it to LIBRARY_PATH if libcuda.so
    is not already discoverable. No-op when a stub can't be found.
    """
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


def get_args():
    parser = argparse.ArgumentParser(description='Test Text-to-Image generation with different attention modes')
    parser.add_argument(
        '--attention_mode',
        type=str,
        default='sdpa',
        choices=['sdpa', 'vanilla', 'triton_fa2', 'sparse', 'sparse_int8'],
        help='Attention implementation: sdpa (PyTorch SDPA, default), vanilla (pure PyTorch), triton_fa2 (Triton Flash Attention 2), sparse (block-sparse attention with top-k selection), or sparse_int8 (block-sparse attention with int8 Q/K quantization)'
    )
    parser.add_argument(
        '--topk_ratio',
        type=float,
        default=0.5,
        help='Top-k ratio for sparse attention block selection (default 0.5, only used when --attention_mode sparse or sparse_int8)'
    )
    parser.add_argument(
        '--online_t5',
        action='store_true',
        help='Force using T5 model to compute embeddings online'
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Get the project root directory
    project_root = Path(__file__).parent
    inference_script = project_root / 'PixArt-alpha' / 'scripts' / 'inference.py'

    if not inference_script.exists():
        print(f"Error: inference.py not found at {inference_script}")
        sys.exit(1)

    # Build the command
    cmd = [
        'python',
        str(inference_script),
        '--attention_mode', args.attention_mode,
    ]

    if args.attention_mode == 'sparse' or args.attention_mode == 'sparse_int8':
        cmd.extend(['--topk_ratio', str(args.topk_ratio)])

    if args.online_t5:
        cmd.append('--online_t5')

    # Triton-based kernels (flash attention 2, sparse, sparse_int8) need libcuda discoverable at compile time
    if args.attention_mode in ('triton_fa2', 'sparse', 'sparse_int8'):
        _ensure_triton_cuda_stub_on_library_path()

    # Print command info
    print("=" * 80)
    print("PixArt Text-to-Image Generation")
    print("=" * 80)
    print(f"Attention mode: {args.attention_mode}")
    if args.attention_mode in ['sparse', 'sparse_int8']:
        print(f"Top-k ratio: {args.topk_ratio}")
    print(f"Online T5: {args.online_t5}")
    print("=" * 80)

    # Run the inference script
    try:
        result = subprocess.run(cmd, cwd=str(project_root))
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running inference: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
