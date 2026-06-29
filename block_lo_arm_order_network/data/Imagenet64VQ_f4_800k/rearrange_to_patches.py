"""Rearrange ImageNet64 VQ-f4 tokens: 16×16 row-major → 8×8 patch raster.

Each 2×2 VQ-token patch becomes a contiguous block of 4 tokens (top-left,
top-right, bottom-left, bottom-right).  Patches are ordered in 8×8 raster.
block_order_block_len=4 linear expansion then correctly maps each block to
its 2×2 VQ tokens without any code changes to nanogpt-learned-order.

Usage:
    python rearrange_to_patches.py \
        --input-dir /path/to/nanogpt-learned-order/data/Imagenet64VQ_f4_800k \
        --output-dir /path/to/output/Imagenet64VQ_f4_800k_patch2x2
"""

from __future__ import annotations

import argparse
import numpy as np
import pickle
import shutil
from pathlib import Path


TOKEN_GRID = 16          # original 16×16 VQ token grid
PATCH_SIZE = 2           # 2×2 VQ tokens per patch
PATCH_GRID = TOKEN_GRID // PATCH_SIZE  # 8
TOKENS_PER_IMAGE = 256
BLOCKS_PER_IMAGE = 64    # 8×8 patches
TOKENS_PER_BLOCK = 4     # 2×2


def build_rearrange_indices(patch_size: int = 2):
    """Build forward and inverse permutation indices for 16×16 → 8×8 patches.

    Returns
    -------
    fwd : ndarray (256,) int
        fwd[new_pos] = old_pos   — to rearrange:  new_data = old_data[fwd]
    inv : ndarray (256,) int
        inv[old_pos] = new_pos   — to recover:   old_data = new_data[inv]
    """
    grid_size = 16
    patch_grid = grid_size // patch_size  # 8
    fwd = np.zeros(256, dtype=np.int32)
    inv = np.zeros(256, dtype=np.int32)

    for pr in range(patch_grid):          # patch row (0..7)
        for pc in range(patch_grid):      # patch col (0..7)
            block_idx = pr * patch_grid + pc  # 0..63
            # tokens in this 2×2 patch, row-major within the patch
            local_tokens = [
                (pr * patch_size + 0, pc * patch_size + 0),  # top-left
                (pr * patch_size + 0, pc * patch_size + 1),  # top-right
                (pr * patch_size + 1, pc * patch_size + 0),  # bottom-left
                (pr * patch_size + 1, pc * patch_size + 1),  # bottom-right
            ]
            for k, (r, c) in enumerate(local_tokens):
                old_pos = r * grid_size + c       # row-major in 16×16
                new_pos = block_idx * 4 + k        # position in rearranged sequence
                fwd[new_pos] = old_pos
                inv[old_pos] = new_pos

    return fwd, inv


def rearrange_file(input_path: Path, output_path: Path, fwd: np.ndarray):
    """Rearrange a .bin file using the precomputed fwd permutation."""
    data = np.memmap(input_path, dtype=np.uint16, mode='r')
    n_total = len(data)
    assert n_total % 256 == 0, f"File length {n_total} not multiple of 256"
    n_images = n_total // 256

    out = np.memmap(output_path, dtype=np.uint16, mode='w+', shape=(n_total,))

    # Process image by image to keep memory low
    for i in range(n_images):
        img = data[i * 256 : (i + 1) * 256].copy()
        out[i * 256 : (i + 1) * 256] = img[fwd]

    out.flush()
    del out
    return n_images


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory containing train.bin, val.bin, meta.pkl (row-major)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory for rearranged files")
    p.add_argument("--patch-size", type=int, default=2,
                   help="Patch size (default 2 → 2×2 patches)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build permutation
    fwd, inv = build_rearrange_indices(patch_size=args.patch_size)
    np.save(args.output_dir / "patch_order_indices.npy", fwd)
    np.save(args.output_dir / "inverse_patch_order_indices.npy", inv)

    # Verify reversibility
    recovered = np.arange(256)
    recovered = recovered[fwd][inv]
    assert np.array_equal(recovered, np.arange(256)), "Permutation is not reversible!"

    # Rearrange train.bin and val.bin
    for split in ["train", "val"]:
        src = args.input_dir / f"{split}.bin"
        dst = args.output_dir / f"{split}.bin"
        if not src.exists():
            print(f"  SKIP {src} (not found)")
            continue
        n = rearrange_file(src, dst, fwd)
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"  [{split}] {n} images → {dst} ({size_mb:.1f} MB)")

    # Copy and update meta.pkl
    meta_src = args.input_dir / "meta.pkl"
    if meta_src.exists():
        with open(meta_src, "rb") as f:
            meta = pickle.load(f)
        meta["original_layout"] = "16x16_row_major"
        meta["rearranged_layout"] = f"{PATCH_GRID}x{PATCH_GRID}_patch_raster"
        meta["patch_size"] = args.patch_size
        meta["blocks_per_image"] = BLOCKS_PER_IMAGE
        meta["tokens_per_block"] = TOKENS_PER_BLOCK
        meta["block_order_block_len"] = TOKENS_PER_BLOCK
        with open(args.output_dir / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        print(f"  [meta] updated and saved to {args.output_dir / 'meta.pkl'}")

    # Quick sanity check on first image
    train_out = args.output_dir / "train.bin"
    if train_out.exists():
        data = np.memmap(train_out, dtype=np.uint16, mode='r')
        # Block 0 should contain tokens from 16×16 positions (0,0), (0,1), (1,0), (1,1)
        # which in row-major are: 0, 1, 16, 17
        block0 = data[:4].copy()
        print(f"\n  Sanity check: block 0 tokens = {block0.tolist()}")
        print(f"  (should be original tokens at row-major positions [0, 1, 16, 17])")

        # Verify tokens are in valid range
        first_img_new = data[:256].copy()
        assert (first_img_new >= 0).all() and (first_img_new < 8192).all(), \
            f"First image tokens out of range [0, 8192): min={first_img_new.min()}, max={first_img_new.max()}"

    print("\nDone.")


if __name__ == "__main__":
    main()
