import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import re
import time
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

from diffusion.model.utils import prepare_prompt_ar
from diffusion import IDDPM, DPMS, SASolverSampler
from tools.download import find_model
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from diffusion.model.t5 import T5Embedder
# from diffusion.data.datasets import get_chunks, ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST

ASPECT_RATIO_512_TEST = {
     '0.25': [256.0, 1024.0], '0.28': [256.0, 928.0],
     '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
     '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
     '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
     '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
     '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
     '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
     '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
     '2.5': [800.0, 320.0], '3.0': [864.0, 288.0],
     '4.0': [1024.0, 256.0]
     }

ASPECT_RATIO_1024_TEST = {
    '0.25': [512., 2048.], '0.28': [512., 1856.],
    '0.32': [576., 1792.], '0.33': [576., 1728.], '0.35': [576., 1664.], '0.4':  [640., 1600.],
    '0.42':  [640., 1536.], '0.48': [704., 1472.], '0.5': [704., 1408.], '0.52': [704., 1344.],
    '0.57': [768., 1344.], '0.6': [768., 1280.], '0.68': [832., 1216.], '0.72': [832., 1152.],
    '0.78': [896., 1152.], '0.82': [896., 1088.], '0.88': [960., 1088.], '0.94': [960., 1024.],
    '1.0':  [1024., 1024.], '1.07': [1024.,  960.], '1.13': [1088.,  960.], '1.21': [1088.,  896.],
    '1.29': [1152.,  896.], '1.38': [1152.,  832.], '1.46': [1216.,  832.], '1.67': [1280.,  768.],
    '1.75': [1344.,  768.], '2.0':  [1408.,  704.], '2.09':  [1472.,  704.], '2.4':  [1536.,  640.],
    '2.5':  [1600.,  640.], '3.0':  [1728.,  576.],
    '4.0':  [2048.,  512.],
}

def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Accumulate per-image diffusion sampling times (sampling loop only,
# excludes T5 embedding / VAE decode / image saving).
INFER_TIMES = []


def load_precomputed_embeddings(embedding_dir, device):
    """
    Load all pre-computed T5 embeddings from directory
    Returns: list of (prompt, caption_emb, emb_mask) tuples
    """
    embeddings = []
    embedding_files = sorted([f for f in os.listdir(embedding_dir) if f.startswith('prompt_') and f.endswith('.pt')])

    print(f"Loading {len(embedding_files)} pre-computed embeddings from {embedding_dir}")

    for emb_file in tqdm(embedding_files):
        emb_path = os.path.join(embedding_dir, emb_file)
        data = torch.load(emb_path, map_location='cpu')

        prompt = data['prompt']
        caption_emb = data['caption_emb'].to(device)
        emb_mask = data['emb_mask'].to(device)

        embeddings.append((prompt, caption_emb, emb_mask))

    return embeddings

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--t5_path', default='pretrained_models', type=str)
    parser.add_argument('--tokenizer_path', default='pretrained_models/sd-vae-ft-ema', type=str)
    parser.add_argument('--txt_file', default='data/test.txt', type=str)
    parser.add_argument('--model_path', default='pretrained_models/PixArt-XL-2-1024-MS.pth', type=str)
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--save_name', default='test_sample', type=str)
    parser.add_argument('--embedding_dir', default='data/prompt_embeddings', type=str,
                        help='Directory containing pre-saved T5 embeddings')
    parser.add_argument('--online_t5', action='store_true',
                        help='Force using T5 model to compute embeddings online instead of loading pre-saved embeddings')
    parser.add_argument('--attention_mode', default='sdpa', type=str, choices=['sdpa', 'vanilla', 'triton_fa2', 'sparse', 'sparse_int8'],
                        help='Attention implementation: sdpa (torch SDPA), vanilla (pure torch), triton_fa2 (Triton Flash Attention 2), sparse (block-sparse attention with top-k selection), or sparse_int8 (block-sparse attention with int8 Q/K quantization)')
    parser.add_argument('--topk_ratio', default=0.5, type=float,
                        help='Top-k ratio for sparse attention block selection (default 0.5, only used with --attention_mode sparse or sparse_int8)')

    return parser.parse_args()


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)


@torch.inference_mode()
def visualize(items, bs, sample_steps, cfg_scale):

    for chunk in tqdm(list(get_chunks(items, bs)), unit='batch'):

        prompts = []
        if bs == 1:
            prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(chunk[0], base_ratios, device=device, show=False)  # ar for aspect ratio
            if args.image_size == 1024:
                latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
            else:
                hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
                ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
                latent_size_h, latent_size_w = latent_size, latent_size
            prompts.append(prompt_clean.strip())
        else:
            hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
            ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
            for prompt in chunk:
                prompts.append(prepare_prompt_ar(prompt, base_ratios, device=device, show=False)[0].strip())
            latent_size_h, latent_size_w = latent_size, latent_size

        null_y = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]

        with torch.no_grad():
            caption_embs, emb_masks = t5.get_text_embeddings(prompts)
            caption_embs = caption_embs.float()[:, None]
            print('finish embedding')

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t_start = time.perf_counter()

            if args.sampling_algo == 'iddpm':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
                model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                                    cfg_scale=cfg_scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                diffusion = IDDPM(str(sample_steps))
                # Sample images:
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                    device=device
                )
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            elif args.sampling_algo == 'dpm-solver':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                dpm_solver = DPMS(model.forward_with_dpmsolver,
                                  condition=caption_embs,
                                  uncondition=null_y,
                                  cfg_scale=cfg_scale,
                                  model_kwargs=model_kwargs)
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
            elif args.sampling_algo == 'sa-solver':
                # Create sampling noise:
                n = len(prompts)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
                samples = sa_solver.sample(
                    S=25,
                    batch_size=n,
                    shape=(4, latent_size_h, latent_size_w),
                    eta=1,
                    conditioning=caption_embs,
                    unconditional_conditioning=null_y,
                    unconditional_guidance_scale=cfg_scale,
                    model_kwargs=model_kwargs,
                )[0]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _elapsed = time.perf_counter() - _t_start
            INFER_TIMES.extend([_elapsed / len(prompts)] * len(prompts))
            print(f"[timing] sampling: {_elapsed:.3f}s for {len(prompts)} image(s) "
                  f"({_elapsed / len(prompts):.3f}s/image)")

        # Offload DiT model to CPU, load VAE to GPU for decoding
        model.to('cpu')
        torch.cuda.empty_cache()
        vae.to(device)
        samples = vae.decode(samples / 0.18215).sample
        vae.to('cpu')
        # Reload DiT model back to GPU for next iteration
        model.to(device)
        torch.cuda.empty_cache()

        # Save images:
        for i, sample in enumerate(samples):
            save_path = os.path.join(save_root, f"{prompts[i][:100]}.jpg")
            print("Saving path: ", save_path)
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))


@torch.inference_mode()
def visualize_precomputed(embeddings_data, bs, sample_steps, cfg_scale):
    """
    Generate images using pre-computed embeddings
    """
    for idx, (prompt, caption_emb, emb_mask) in enumerate(tqdm(embeddings_data, desc="Generating images")):

        # Prepare aspect ratio
        if bs == 1:
            prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)
            if args.image_size == 1024:
                latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
            else:
                hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
                ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
                latent_size_h, latent_size_w = latent_size, latent_size
        else:
            hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
            ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
            latent_size_h, latent_size_w = latent_size, latent_size

        null_y = model.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None]

        with torch.no_grad():
            # Use pre-computed embeddings
            caption_embs = caption_emb.float()[:, None]
            emb_masks = emb_mask

            print(f'Using pre-computed embedding for: {prompt[:80]}...')

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t_start = time.perf_counter()

            if args.sampling_algo == 'iddpm':
                n = 1
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
                model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                                    cfg_scale=cfg_scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                diffusion = IDDPM(str(sample_steps))
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                    device=device
                )
                samples, _ = samples.chunk(2, dim=0)
            elif args.sampling_algo == 'dpm-solver':
                n = 1
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                dpm_solver = DPMS(model.forward_with_dpmsolver,
                                  condition=caption_embs,
                                  uncondition=null_y,
                                  cfg_scale=cfg_scale,
                                  model_kwargs=model_kwargs)
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
            elif args.sampling_algo == 'sa-solver':
                n = 1
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
                samples = sa_solver.sample(
                    S=25,
                    batch_size=n,
                    shape=(4, latent_size_h, latent_size_w),
                    eta=1,
                    conditioning=caption_embs,
                    unconditional_conditioning=null_y,
                    unconditional_guidance_scale=cfg_scale,
                    model_kwargs=model_kwargs,
                )[0]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _elapsed = time.perf_counter() - _t_start
            INFER_TIMES.append(_elapsed)
            print(f"[timing] sampling: {_elapsed:.3f}s/image")

        torch.cuda.empty_cache()

        # Offload DiT model to CPU, load VAE to GPU for decoding
        model.to('cpu')
        torch.cuda.empty_cache()
        vae.to(device)
        samples = vae.decode(samples / 0.18215).sample
        vae.to('cpu')
        # Reload DiT model back to GPU for next iteration
        model.to(device)
        torch.cuda.empty_cache()

        # Save images
        save_path = os.path.join(save_root, f"{idx:03d}_{prompt[:80]}.jpg")
        print(f"Saving to: {save_path}")
        save_image(samples[0], save_path, nrow=1, normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    args = get_args()

    # Set attention mode before model initialization
    from diffusion.model.nets.PixArt_blocks import set_attention_mode, set_sparse_topk_ratio
    set_attention_mode(args.attention_mode)
    if args.attention_mode in ['sparse', 'sparse_int8']:
        set_sparse_topk_ratio(args.topk_ratio)

    # Setup PyTorch:
    seed = args.seed
    set_env(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']

    # only support fixed latent size currently
    latent_size = args.image_size // 8
    lewei_scale = {512: 1, 1024: 2}     # trick for positional embedding interpolation
    sample_steps_dict = {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    weight_dtype = torch.float16
    print(f"Inference with {weight_dtype}")

    # model setting
    if args.image_size == 512:
        model = PixArt_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)
    else:
        model = PixArtMS_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)

    print(f"Generating sample from ckpt: {args.model_path}")
    state_dict = find_model(args.model_path)
    del state_dict['state_dict']['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    print('Missing keys: ', missing)
    print('Unexpected keys', unexpected)
    model.eval()
    model.to(weight_dtype)
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    # Load VAE to CPU first to save VRAM (offload to GPU only during decode)
    vae = AutoencoderKL.from_pretrained(args.tokenizer_path, use_safetensors=False,).to('cpu')
    vae.eval()
    print("VAE loaded to CPU (will offload to GPU only during decode)")

    # Check if pre-computed embeddings exist and should be used
    embeddings_exist = os.path.exists(args.embedding_dir) and len([f for f in os.listdir(args.embedding_dir) if f.startswith('prompt_') and f.endswith('.pt')]) > 0
    use_precomputed = embeddings_exist and not args.online_t5

    # Load T5 model or pre-computed embeddings
    if use_precomputed:
        print(f"Found pre-computed embeddings in {args.embedding_dir}")
        t5 = None
        embeddings_data = load_precomputed_embeddings(args.embedding_dir, device)
        items = None
    else:
        t5 = T5Embedder(device="cuda", local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float)
        embeddings_data = None
        # Load prompts from txt file
        with open(args.txt_file, 'r') as f:
            items = [item.strip() for item in f.readlines()]

    work_dir = 'output'

    # img save setting
    os.makedirs(work_dir, exist_ok=True)

    save_root = os.path.join(work_dir, f"{datetime.now().date()}_attention_{args.attention_mode}")
    if args.attention_mode in ['sparse', 'sparse_int8']:
        save_root += f"_topk{args.topk_ratio}"
    os.makedirs(save_root, exist_ok=True)

    # Run inference
    if use_precomputed:
        visualize_precomputed(embeddings_data, args.bs, sample_steps, args.cfg_scale)
    else:
        visualize(items, args.bs, sample_steps, args.cfg_scale)

    # Report average inference time (diffusion sampling only)
    if INFER_TIMES:
        n_imgs = len(INFER_TIMES)
        avg = sum(INFER_TIMES) / n_imgs
        config_line = (f"Inference timing ({args.attention_mode}"
                       + (f", topk={args.topk_ratio}" if args.attention_mode in ['sparse', 'sparse_int8'] else "")
                       + f", {args.sampling_algo}, {sample_steps} steps):")
        summary = "\n".join([
            "=" * 80,
            config_line,
            f"  Images:            {n_imgs}",
            f"  Avg sampling time: {avg:.3f} s/image",
            f"  Total sampling:    {sum(INFER_TIMES):.3f} s",
            "=" * 80,
        ])
        print(summary)

        # Write timing to a .txt file alongside the generated images
        timing_path = os.path.join(save_root, "timing.txt")
        with open(timing_path, "w") as f:
            f.write(summary + "\n\n")
            f.write("Per-image sampling time (s):\n")
            for i, t in enumerate(INFER_TIMES):
                f.write(f"  {i:03d}: {t:.3f}\n")
        print(f"Timing written to: {timing_path}")