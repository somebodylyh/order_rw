"""Extract A_global from a nanoGPT-style AOGPT discrete-token checkpoint.

E2 adaptation of block_lo_arm_order_network/extract_image_attention.py:
  - Reads tokens_per_image from meta.pkl (no hardcode).
  - Drops the +0.1 * none_phys adjustment (matches toy extract_image_attention.py
    convention so the resulting A_global is apples-to-apples with E0/E1).
  - Averages over ALL layers (not just last 4), matching the toy extractor.

Usage:
    python block_lo_arm_order_network/extract_image_attention_e2.py \\
        --ckpt nanogpt-learned-order/out/image_alignment/e2_.../ckpt.pt \\
        --data block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/val.bin \\
        --meta block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/meta.pkl \\
        --n-images 500 --M-passes 3 --out probe_results_image/e2_imagenet32_vqf4_seq64/attention
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "nanogpt-learned-order"))

from AOGPT import AOGPTConfig, AOGPT


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_args = ckpt.get("model_args", {})
    cfg = ckpt.get("config", {})
    keys = ["block_size", "vocab_size", "n_layer", "n_head", "n_embd",
            "dropout", "bias", "block_order_block_len", "order_impl"]
    model_args = {}
    for k in keys:
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
        raise RuntimeError(f"Missing keys when loading ckpt: {missing[:5]}...")
    if unexpected:
        print(f"  [warn] unexpected keys ignored: {unexpected[:5]}")
    model.crop_block_size(model_args["block_size"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, model_args


def extract_a_global(model, data_tokens: np.ndarray, tokens_per_image: int,
                     n_images: int, m_passes: int, device: str) -> np.ndarray:
    T = model.config.block_size
    if T != tokens_per_image:
        raise ValueError(f"meta tokens_per_image={tokens_per_image} != model.block_size={T}")

    A_sum = torch.zeros(T, T, device=device)
    A_count = 0

    n_available = len(data_tokens) // T
    n_images = min(n_images, n_available)
    print(f"Extracting attention from {n_images} images x M={m_passes} orders, T={T}", flush=True)

    for img_idx in range(n_images):
        start = img_idx * T
        tokens = torch.from_numpy(
            data_tokens[start:start + T].astype(np.int64)
        ).to(device).unsqueeze(0)  # (1, T)

        for _ in range(m_passes):
            rand_order = torch.randperm(T, device=device).unsqueeze(0)  # (1, T)

            with torch.no_grad():
                _, _, attn_list = model.forward_fn(
                    tokens, rand_order, return_attentions=True
                )

            # Average ALL layers and ALL heads, matching toy extractor.
            attn_stack = torch.stack(attn_list, dim=0)         # (L, B, nh, T+1, T+1)
            attn = attn_stack.mean(dim=[0, 2])[0]               # (T+1, T+1)
            attn_content = attn[1:, 1:]                         # (T, T) in model-pos order

            # Remap model-pos → physical-pos
            inv_order = torch.argsort(rand_order[0])
            attn_phys = attn_content[inv_order][:, inv_order]

            A_sum += attn_phys
            A_count += 1

        if (img_idx + 1) % 100 == 0:
            print(f"  {img_idx + 1}/{n_images}", flush=True)

    A = (A_sum / A_count).float().cpu().numpy()
    np.fill_diagonal(A, 0.0)
    return A.astype(np.float32)


def save_heatmap(A: np.ndarray, path: str, title: str, grid: int = 8) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(A, cmap="viridis", aspect="equal")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Key patch index")
    ax.set_ylabel("Query patch index")
    n = A.shape[0]
    for tick in range(0, n + 1, grid):
        ax.axhline(tick - 0.5, color="white", linewidth=0.5, alpha=0.6)
        ax.axvline(tick - 0.5, color="white", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--meta", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n-images", type=int, default=500)
    p.add_argument("--M-passes", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    tokens_per_image = int(meta["tokens_per_image"])

    print(f"Loading model from {args.ckpt}...", flush=True)
    model, model_args = load_model(args.ckpt, args.device)
    print(f"  n_layer={model_args['n_layer']}, n_head={model_args['n_head']}, "
          f"n_embd={model_args['n_embd']}, block_size={model_args['block_size']}", flush=True)

    print(f"Loading data from {args.data} (tokens_per_image={tokens_per_image})", flush=True)
    data = np.memmap(args.data, dtype=np.uint16, mode="r")

    A = extract_a_global(model, data, tokens_per_image, args.n_images,
                         args.M_passes, args.device)
    B = A.T.copy()
    np.fill_diagonal(B, 0.0)

    a_path = out_dir / "A_global.npy"
    b_path = out_dir / "B_global.npy"
    np.save(a_path, A)
    np.save(b_path, B)
    print(f"Saved {a_path}, {b_path}; A mean={A.mean():.6f}, max={A.max():.6f}", flush=True)

    grid = int(round(tokens_per_image ** 0.5))
    save_heatmap(A, str(out_dir / "A_global_heatmap.png"), "A_global", grid=grid)
    save_heatmap(B, str(out_dir / "B_global_heatmap.png"), "B_global", grid=grid)
    print("Saved heatmaps.")


if __name__ == "__main__":
    main()
