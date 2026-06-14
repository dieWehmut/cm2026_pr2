import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")
import argparse
import torch
from tqdm import tqdm

from diffusion.model.t5 import T5Embedder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t5_path', default='pretrained_models', type=str)
    parser.add_argument('--txt_file', default='asset/test.txt', type=str)
    parser.add_argument('--output_dir', default='pretrained_models/prompt_embeddings', type=str)

    return parser.parse_args()


@torch.inference_mode()
def save_embeddings(prompts, t5, output_dir):
    """
    Save T5 embeddings for a list of prompts
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(prompts)} prompts...")

    for idx, prompt in enumerate(tqdm(prompts)):
        # Get embedding
        caption_embs, emb_masks = t5.get_text_embeddings([prompt])

        # Save embedding and mask
        output_file = os.path.join(output_dir, f"prompt_{idx:03d}.pt")
        torch.save({
            'prompt': prompt,
            'caption_emb': caption_embs.cpu(),
            'emb_mask': emb_masks.cpu()
        }, output_file)

        print(f"Saved: {output_file}")
        print(f"Prompt: {prompt[:80]}...")

    # Save a summary file with all prompts
    summary_file = os.path.join(output_dir, "prompts_summary.txt")
    with open(summary_file, 'w') as f:
        for idx, prompt in enumerate(prompts):
            f.write(f"{idx:03d}: {prompt}\n")

    print(f"\nAll embeddings saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    args = get_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load T5 model
    print("Loading T5 model...")
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float)
    print("T5 model loaded successfully")

    # Load prompts
    print(f"Loading prompts from: {args.txt_file}")
    with open(args.txt_file, 'r') as f:
        prompts = [item.strip() for item in f.readlines() if item.strip()]

    print(f"Found {len(prompts)} prompts")

    # Save embeddings
    save_embeddings(prompts, t5, args.output_dir)
