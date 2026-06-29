"""E2 data: ImageNet32 → VQ-f4 → 8x8=64 tokens per image.

Source: the bilinear-downsampled ImageNet32 .npy produced by
`image_order/data/imagenet32/build_imagenet32.py` (uint8, (N, 3, 32, 32)).

Output (matches ych nanogpt-learned-order convention):
    train.bin   uint16 (N_train * 64,)
    val.bin     uint16 (N_val   * 64,)
    meta.pkl    {"vocab_size": 8192, "tokens_per_image": 64,
                  "image_size": 32, "vae_latent_grid": 8, ...}

The VQ-f4 VAE has fixed downsample factor 4, so 32x32 → 8x8 = 64 tokens.
That's the native 8x8 grid we need for E2 — no patch aggregation.

Usage:
    python block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64/prepare.py \\
        --image-source image_order/data/imagenet32 \\
        --vae-path /home/admin/.cache/huggingface/hub/models--xvjiarui--ldm-vq-f4/snapshots/<sha> \\
        --output-dir block_lo_arm_order_network/data/Imagenet32VQ_f4_800k_seq64
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np


IMAGE_SIZE = 32
VAE_DOWN_FACTOR = 4
TOKEN_GRID = IMAGE_SIZE // VAE_DOWN_FACTOR    # 8
TOKENS_PER_IMAGE = TOKEN_GRID * TOKEN_GRID    # 64
VOCAB_SIZE = 8192


def _load_vqf4(vae_path: str, device: str):
    """Load CompVis VQ-f4 via diffusers (preferred) or fall back to HF auto-download."""
    from diffusers import VQModel  # type: ignore
    try:
        vae = VQModel.from_pretrained(vae_path)
    except Exception:
        # Fall back to HF repo id; uses the same files already in the local cache.
        vae = VQModel.from_pretrained("xvjiarui/ldm-vq-f4")
    return vae.to(device).eval()


def encode_batch(vae, batch_uint8: np.ndarray, device: str) -> np.ndarray:
    """Encode (B, 3, 32, 32) uint8 → (B, 64) int32 codebook indices."""
    import torch
    with torch.no_grad():
        x = torch.from_numpy(batch_uint8).to(device, dtype=torch.float32)
        x = x.div_(127.5).sub_(1.0)
        h = vae.encoder(x)
        h = vae.quant_conv(h)
        quant, _, info = vae.quantize(h)
        indices = info[2] if isinstance(info, tuple) else info
    indices = indices.view(batch_uint8.shape[0], -1).to(torch.int32).cpu().numpy()
    if indices.shape[1] != TOKENS_PER_IMAGE:
        raise ValueError(f"Expected {TOKENS_PER_IMAGE} tokens/image, got {indices.shape[1]}")
    if (indices < 0).any() or (indices >= VOCAB_SIZE).any():
        raise ValueError(
            f"Indices out of [0, {VOCAB_SIZE}): min={indices.min()} max={indices.max()}"
        )
    return indices


def write_split(
    split_name: str,
    images_uint8: np.ndarray,
    vae,
    out_path: Path,
    batch_size: int,
    device: str,
):
    n = images_uint8.shape[0]
    out = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(n * TOKENS_PER_IMAGE,))
    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = images_uint8[start:end]
        chunk = np.ascontiguousarray(chunk)  # mmap views may be non-contiguous
        tok = encode_batch(vae, chunk, device=device)
        out[start * TOKENS_PER_IMAGE : end * TOKENS_PER_IMAGE] = tok.astype(np.uint16).ravel()
        if start % (batch_size * 50) == 0:
            elapsed = time.time() - t0
            print(f"  [{split_name}] {start:>7d} / {n} ({elapsed:.0f}s)", flush=True)
    out.flush()
    del out
    print(
        f"  [{split_name}] DONE {n} images in {time.time()-t0:.0f}s → "
        f"{out_path} ({out_path.stat().st_size/1024/1024:.1f} MB)",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--image-source",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent.parent / "image_order/data/imagenet32",
        help="Directory containing train.npy / val.npy (uint8 (N, 3, 32, 32))",
    )
    p.add_argument("--vae-path", type=str, default="xvjiarui/ldm-vq-f4",
                   help="Diffusers VQ-f4 model id or local path. Default uses HF cache.")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.image_source / "train.npy"
    val_path = args.image_source / "val.npy"
    if not train_path.exists() or not val_path.exists():
        sys.exit(f"Missing {train_path} or {val_path}. Run image_order/data/imagenet32/build_imagenet32.py first.")

    print(f"Loading ImageNet32 train from {train_path} ...", flush=True)
    train_imgs = np.load(train_path, mmap_mode="r")
    if args.limit_train is not None:
        train_imgs = train_imgs[: args.limit_train]
    print(f"  train: {train_imgs.shape}", flush=True)

    print(f"Loading ImageNet32 val from {val_path} ...", flush=True)
    val_imgs = np.load(val_path, mmap_mode="r")
    if args.limit_val is not None:
        val_imgs = val_imgs[: args.limit_val]
    print(f"  val:   {val_imgs.shape}", flush=True)

    print(f"Loading VQ-f4 from {args.vae_path} ...", flush=True)
    vae = _load_vqf4(args.vae_path, device=args.device)

    write_split("train", train_imgs, vae, args.output_dir / "train.bin",
                batch_size=args.batch_size, device=args.device)
    write_split("val", val_imgs, vae, args.output_dir / "val.bin",
                batch_size=args.batch_size, device=args.device)

    meta = {
        "vocab_size": VOCAB_SIZE,
        "tokens_per_image": TOKENS_PER_IMAGE,
        "image_size": IMAGE_SIZE,
        "vae_latent_grid": TOKEN_GRID,
        "n_train": int(train_imgs.shape[0]),
        "n_val": int(val_imgs.shape[0]),
        "source_image_dir": str(args.image_source),
        "vae_path": args.vae_path,
    }
    with open(args.output_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"Wrote meta.pkl: {meta}")
    print("DONE")


if __name__ == "__main__":
    main()
