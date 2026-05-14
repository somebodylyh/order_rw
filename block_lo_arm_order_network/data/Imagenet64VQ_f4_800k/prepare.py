"""Prepare ImageNet-64 → CompVis VQ-f4 → uint16 .bin token files.

Output layout (matches ych/nanogpt-learned-order/data/<dataset>/ convention):
    train.bin      uint16, shape == (N_train * 256,)
    val.bin        uint16, shape == (N_val   * 256,)
    meta.pkl       {"vocab_size": 8192, "tokens_per_image": 256,
                    "image_size": 64, "vae_latent_grid": 16,
                    "subset_size": 800000, "subset_seed": 42}

Subset rule: deterministic shuffle of train indices with RNG seed=42; take first 800 000.
Validation: full ImageNet-64 val (50 000 images).

Usage:
    python prepare.py \\
        --imagenet-train-dir /path/to/Imagenet64_train \\
        --imagenet-val-dir   /path/to/Imagenet64_val \\
        --vae-path           ~/models/vq-f4 \\
        --output-dir         .

Smoke (no encoding, just verifies shapes / loaders):
    python prepare.py --smoke

External deps (install once):
    pip install diffusers==0.27.0 omegaconf einops pillow
    # CompVis VQ-f4 ckpt: download via scripts/image_partner/download_vq_f4.sh
    # ImageNet-64 .npz files: scripts/image_partner/download_imagenet64.sh

This script is single-GPU; rough cost @ batch=128 on a 4090: ~3-5 min for the full
800k train + 50k val encoding pass.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_SIZE = 64
VAE_DOWN_FACTOR = 4
TOKEN_GRID = IMAGE_SIZE // VAE_DOWN_FACTOR        # 16
TOKENS_PER_IMAGE = TOKEN_GRID * TOKEN_GRID        # 256
VOCAB_SIZE = 8192                                  # CompVis vq-f4 codebook size
SUBSET_SEED = 42
SUBSET_SIZE = 800_000


# ---------------------------------------------------------------------------
# ImageNet-64 loader (official .npz batch format from Chen et al. 2018)
# ---------------------------------------------------------------------------

def _load_imagenet64_npz(npz_path: Path) -> np.ndarray:
    """Load one ImageNet-64 batch file, return (N, 3, 64, 64) uint8.

    The official format stores each batch as a pickled dict with keys
    `'data'` (uint8 (N, 12288) flat C,H,W) and `'labels'` (list of ints).
    Some redistributions use the same .npz wrapper.
    """
    if npz_path.suffix == ".npz":
        with np.load(npz_path, allow_pickle=True) as f:
            data = f["data"] if "data" in f.files else f[f.files[0]]
    else:
        with open(npz_path, "rb") as fh:
            obj = pickle.load(fh, encoding="latin1")
            data = obj["data"] if isinstance(obj, dict) else obj

    if data.ndim == 2 and data.shape[1] == 3 * IMAGE_SIZE * IMAGE_SIZE:
        data = data.reshape(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
    elif data.shape[-3:] != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Unexpected ImageNet-64 batch shape: {data.shape}")
    return data.astype(np.uint8)


def iter_imagenet64_dir(directory: Path) -> Iterable[Tuple[str, np.ndarray]]:
    """Yield `(batch_name, images_uint8)` over every batch file in `directory`.

    Accepts both `.npz` and pickled `train_data_batch_<i>` (no extension).
    Batches are returned in sorted order so subset shuffling is reproducible.
    """
    files = sorted(directory.iterdir())
    if not files:
        raise FileNotFoundError(f"Empty ImageNet-64 directory: {directory}")
    for path in files:
        if not path.is_file():
            continue
        if path.suffix not in {"", ".npz", ".bin"}:
            continue
        yield path.name, _load_imagenet64_npz(path)


def load_full_imagenet64(directory: Path) -> np.ndarray:
    """Concatenate every batch in `directory` into a single (N, 3, 64, 64) array."""
    parts = [imgs for _, imgs in iter_imagenet64_dir(directory)]
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# CompVis vq-f4 VAE wrapper (encode-only)
# ---------------------------------------------------------------------------

def _load_vqf4(vae_path: str, device: str = "cuda"):
    """Load CompVis vq-f4 VAE in encode-only mode.

    Tries the diffusers path first; if the supplied path is a raw CompVis
    checkpoint dir (with `.ckpt` + `config.yaml`), falls back to the
    taming-transformers loader.
    """
    try:
        from diffusers import VQModel  # type: ignore
        vae = VQModel.from_pretrained(vae_path)
        vae = vae.to(device).eval()
        return ("diffusers", vae)
    except Exception as diffusers_err:
        try:
            import yaml
            from omegaconf import OmegaConf
            from taming.models.vqgan import VQModel as TamingVQModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                f"Failed to load via diffusers ({diffusers_err}); "
                f"taming-transformers also unavailable ({e}). Install one of them."
            )

        cfg_path = Path(vae_path) / "config.yaml"
        ckpt_path = Path(vae_path) / "model.ckpt"
        if not cfg_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(
                f"Expected {cfg_path} and {ckpt_path} for taming-style loader"
            )
        config = OmegaConf.load(cfg_path)
        model = TamingVQModel(**config.model.params)
        sd = __import__("torch").load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
        model = model.to(device).eval()
        return ("taming", model)


def encode_batch(loader_kind: str, vae, batch_uint8: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Encode one (B, 3, 64, 64) uint8 batch → (B, 256) int32 indices."""
    import torch
    with torch.no_grad():
        x = torch.from_numpy(batch_uint8).to(device, dtype=torch.float32)
        x = x.div_(127.5).sub_(1.0)        # [-1, 1]
        if loader_kind == "diffusers":
            h = vae.encoder(x)
            h = vae.quant_conv(h)
            quant, _, info = vae.quantize(h)
            indices = info[2] if isinstance(info, tuple) else info
        else:  # taming
            h = vae.encoder(x)
            h = vae.quant_conv(h)
            quant, _, info = vae.quantize(h)
            indices = info[2] if isinstance(info, tuple) else info
    indices = indices.view(batch_uint8.shape[0], -1).to(torch.int32).cpu().numpy()
    if indices.shape[1] != TOKENS_PER_IMAGE:
        raise ValueError(f"Expected {TOKENS_PER_IMAGE} tokens / image, got {indices.shape[1]}")
    if (indices < 0).any() or (indices >= VOCAB_SIZE).any():
        raise ValueError(f"Indices out of range [0, {VOCAB_SIZE}); min={indices.min()} max={indices.max()}")
    return indices


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def select_subset(n_total: int, subset: int, seed: int) -> np.ndarray:
    """Reproducible random shuffle, take first `subset` indices."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_total)
    return order[:subset]


def write_split(
    split_name: str,
    images: np.ndarray,
    indices: np.ndarray,
    loader_kind: str,
    vae,
    out_path: Path,
    batch_size: int,
    device: str,
):
    """Encode `images[indices]` in batches and write tokens to `out_path` as uint16."""
    import torch
    n = len(indices)
    out = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(n * TOKENS_PER_IMAGE,))

    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk_idx = indices[start:end]
        chunk_imgs = images[chunk_idx]
        tok = encode_batch(loader_kind, vae, chunk_imgs, device=device)
        out[start * TOKENS_PER_IMAGE : end * TOKENS_PER_IMAGE] = tok.astype(np.uint16).ravel()
        if start % (batch_size * 50) == 0:
            elapsed = time.time() - t0
            print(f"  [{split_name}] {start:>7d} / {n} ({elapsed:.0f}s)", flush=True)
    out.flush()
    del out
    elapsed = time.time() - t0
    print(f"  [{split_name}] DONE {n} images in {elapsed:.0f}s → {out_path} "
          f"({out_path.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imagenet-train-dir", type=Path, help="dir containing ImageNet-64 train batch .npz files")
    p.add_argument("--imagenet-val-dir", type=Path, help="dir containing ImageNet-64 val batch .npz files")
    p.add_argument("--vae-path", type=str, help="diffusers-style VQModel dir or CompVis ckpt dir")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--subset-size", type=int, default=SUBSET_SIZE)
    p.add_argument("--subset-seed", type=int, default=SUBSET_SEED)
    p.add_argument("--smoke", action="store_true",
                   help="Skip downloads + encoding; verify script structure end-to-end with synthetic 256 imgs")
    args = p.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        # Synthetic data smoke: random uint8 noise, no real VAE.
        print("=== SMOKE MODE: no downloads, no real encoding ===")
        n = 256
        rng = np.random.RandomState(0)
        synthetic_indices = rng.randint(0, VOCAB_SIZE, size=(n, TOKENS_PER_IMAGE), dtype=np.int32)
        train_path = output_dir / "train.bin"
        val_path = output_dir / "val.bin"
        n_tr = int(0.8 * n); n_va = n - n_tr
        np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(n_tr * TOKENS_PER_IMAGE,))[:] = \
            synthetic_indices[:n_tr].astype(np.uint16).ravel()
        np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(n_va * TOKENS_PER_IMAGE,))[:] = \
            synthetic_indices[n_tr:].astype(np.uint16).ravel()
        print(f"  wrote {train_path} ({train_path.stat().st_size} bytes)")
        print(f"  wrote {val_path}   ({val_path.stat().st_size} bytes)")

        meta = {
            "vocab_size": VOCAB_SIZE,
            "tokens_per_image": TOKENS_PER_IMAGE,
            "image_size": IMAGE_SIZE,
            "vae_latent_grid": TOKEN_GRID,
            "subset_size": n_tr,
            "subset_seed": 0,
            "smoke": True,
        }
        with open(output_dir / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        print(f"  wrote meta.pkl: {meta}")
        print("OK smoke")
        return

    # Real pipeline
    if not args.imagenet_train_dir or not args.imagenet_val_dir or not args.vae_path:
        sys.exit("--imagenet-train-dir, --imagenet-val-dir and --vae-path are required (or pass --smoke)")

    print(f"Loading ImageNet-64 train images from {args.imagenet_train_dir} ...")
    train_imgs = load_full_imagenet64(args.imagenet_train_dir)
    print(f"  train images: {train_imgs.shape}")

    print(f"Loading ImageNet-64 val images from {args.imagenet_val_dir} ...")
    val_imgs = load_full_imagenet64(args.imagenet_val_dir)
    print(f"  val images:   {val_imgs.shape}")

    print(f"Loading VQ-f4 VAE from {args.vae_path} ...")
    loader_kind, vae = _load_vqf4(args.vae_path, device=args.device)
    print(f"  loaded via: {loader_kind}")

    train_indices = select_subset(len(train_imgs), args.subset_size, args.subset_seed)
    val_indices = np.arange(len(val_imgs))
    print(f"Subset: train={len(train_indices)} (seed={args.subset_seed}), val={len(val_indices)}")

    write_split("train", train_imgs, train_indices, loader_kind, vae,
                output_dir / "train.bin", batch_size=args.batch_size, device=args.device)
    write_split("val", val_imgs, val_indices, loader_kind, vae,
                output_dir / "val.bin", batch_size=args.batch_size, device=args.device)

    meta = {
        "vocab_size": VOCAB_SIZE,
        "tokens_per_image": TOKENS_PER_IMAGE,
        "image_size": IMAGE_SIZE,
        "vae_latent_grid": TOKEN_GRID,
        "subset_size": len(train_indices),
        "subset_seed": args.subset_seed,
        "vae_loader_kind": loader_kind,
        "smoke": False,
    }
    with open(output_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"Wrote meta.pkl: {meta}")
    print("DONE")


if __name__ == "__main__":
    main()
