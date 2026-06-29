"""Extract A_global (256x256) attention from ImageNet-64 VQ-f4 baseline checkpoint.

For each image: random perm order → forward with return_attentions → remap to physical.
Output: A_global.npy (256, 256) float32 averaged over N images.

Usage:
    python extract_image_attention.py \
        --ckpt ~/nanogpt-learned-order/out/smoke/.../ckpt_10k.pt \
        --data /tmp/imagenet64_sanity/output/train.bin \
        --n-images 500 --out probe_results_image_large/
"""

from __future__ import annotations

import os, sys, argparse, pickle
from pathlib import Path

import numpy as np
import torch

# Add nanogpt-learned-order to path for AOGPT imports
_NANO_ROOT = Path(__file__).resolve().parent.parent / "nanogpt-learned-order"
sys.path.insert(0, str(_NANO_ROOT))

from AOGPT import AOGPTConfig, AOGPT


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # vocab_size only in model_args, not config
    m_args = ckpt.get("model_args", {})
    cfg = ckpt["config"]
    model_args = {}
    for k in ["block_size", "vocab_size", "n_layer", "n_head", "n_embd",
              "dropout", "bias", "block_order_block_len", "order_impl"]:
        if k in m_args:
            model_args[k] = m_args[k]
        elif k in cfg:
            model_args[k] = cfg[k]
    model = AOGPT(AOGPTConfig(**model_args))
    sd = ckpt["model"]
    if all(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(
            f"Missing keys when loading ckpt (extractor was loading random weights!): {missing[:5]}..."
        )
    if unexpected:
        print(f"  [warn] unexpected keys ignored: {unexpected[:5]}")
    model.crop_block_size(model_args["block_size"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, model_args


def extract_A_global(model, data_tokens, n_images: int, device: str):
    """Average token-level attention over n_images with random orders.

    Returns A (T, T) in physical coordinates (row-major 16x16 grid).
    A[i, j] = mean attention from physical token i to physical token j.
    Includes [None] signal, excludes diagonal.
    """
    T = model.config.block_size  # 256
    n_layers = model.config.n_layer
    n_heads = model.config.n_head

    A_sum = torch.zeros(T, T, device=device)
    A_count = torch.zeros(T, T, device=device)

    n_available = len(data_tokens) // T
    n_images = min(n_images, n_available)

    M_ORDERS = 3  # random orders per image to reduce causal-position bias

    print(f"Extracting attention from {n_images} images x {M_ORDERS} orders...", flush=True)

    for img_idx in range(n_images):
        # Get VQ tokens for this image (physical order from prepare.py)
        start = img_idx * T
        tokens = torch.from_numpy(
            data_tokens[start:start + T].astype(np.int64)
        ).to(device).unsqueeze(0)  # (1, T)

        for _ in range(M_ORDERS):
            # Random reading order
            rand_order = torch.randperm(T, device=device).unsqueeze(0)  # (1, T)

            # Forward with attention
            with torch.no_grad():
                _, _, attn_list = model.forward_fn(
                    tokens, rand_order, return_attentions=True
                )

            # attn_list: list of (B, nh, T+1, T+1) — includes [None] at pos 0
            # Take last 4 layers, all heads
            attn_stack = torch.stack(attn_list[-4:], dim=0)  # (4, B, nh, T+1, T+1)
            attn = attn_stack.mean(dim=[0, 2])[0]  # (T+1, T+1)

            # Remove [None], content-content only
            attn_content = attn[1:, 1:]  # (T, T) in model order

            # Remap to physical order
            inv_order = torch.argsort(rand_order[0])  # model_pos -> physical_pos
            attn_phys = attn_content[inv_order][:, inv_order]

            A_sum += attn_phys
            A_count += 1.0

        if (img_idx + 1) % 100 == 0:
            print(f"  {img_idx + 1}/{n_images}", flush=True)

    A = A_sum / A_count
    A = A.float().cpu().numpy()

    # [None] signal: average attention from each token to [None]
    none_attn = torch.stack(attn_list[-4:], dim=0).mean(dim=[0, 2])[0][1:, 0]
    none_phys = none_attn[inv_order].float().cpu().numpy()
    A += 0.1 * none_phys[:, np.newaxis]

    # Zero diagonal
    np.fill_diagonal(A, 0.0)

    return A.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data", type=str, required=True, help="train.bin path")
    p.add_argument("--n-images", type=int, default=500)
    p.add_argument("--out", type=str, default="probe_results_image_large/imagenet64_vqf4_140k_l8h8e512")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    # Load model
    print(f"Loading model from {args.ckpt}...", flush=True)
    model, model_args = load_model(args.ckpt, device)
    print(f"  n_layer={model_args['n_layer']}, n_head={model_args['n_head']}, "
          f"n_embd={model_args['n_embd']}, block_size={model_args['block_size']}",
          flush=True)

    # Load data
    print(f"Loading data from {args.data}...", flush=True)
    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    n_images_total = len(data) // 256
    print(f"  {n_images_total} images available", flush=True)

    # Extract
    A = extract_A_global(model, data, args.n_images, device)

    # Save
    out_path = out_dir / "A_global.npy"
    np.save(out_path, A)
    print(f"Saved A_global: {out_path} | shape={A.shape} | "
          f"mean={A.mean():.6f} | max={A.max():.6f}", flush=True)

    # Quick diagnostic
    # Check if nearby tokens in 16x16 grid have higher attention
    grid_size = 16
    A_grid = np.zeros((grid_size, grid_size))
    for r in range(grid_size):
        for c in range(grid_size):
            phys_idx = r * grid_size + c
            A_grid[r, c] = A[phys_idx].mean()

    print(f"\nQuick spatial check:", flush=True)
    print(f"  A_grid mean={A_grid.mean():.6f}, std={A_grid.std():.6f}", flush=True)
    # Row and column autocorrelation
    row_means = A_grid.mean(axis=1)
    col_means = A_grid.mean(axis=0)
    print(f"  row means range: [{row_means.min():.6f}, {row_means.max():.6f}]", flush=True)
    print(f"  col means range: [{col_means.min():.6f}, {col_means.max():.6f}]", flush=True)


if __name__ == "__main__":
    main()
