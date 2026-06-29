"""ImageNet32 patch dataset — mirror of data_image_patches.CIFAR10Patches.

Loads the bilinear-downsampled ImageNet32 .npy produced by
`image_order/data/imagenet32/build_imagenet32.py`. Same interface as
CIFAR10Patches so train_image_random_imagenet32.py can swap it in.

Memory model differs from CIFAR10Patches because the train split is 16× larger
(800k images). Materializing 800k × 64 × 48 float32 patches up front would peak
at ~25 GB and OOM on a 30 GB box. We instead keep images as uint8 tensors
(~2.3 GB) and provide `patches_for_indices(idx, device)` which normalizes and
unfolds a batch on-the-fly. For small splits (val, or capped train via
`max_images`) we also expose a pre-materialized `.patches` for compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.utils.data

from data_image_patches import (
    IMG_SIZE,
    N_PATCHES,
    PATCH_DIM,
    image_to_patches,
)


IMAGENET32_DIR = Path(__file__).resolve().parent / "data" / "imagenet32"

# Anything bigger than this gets the lazy path (no .patches materialization).
MATERIALIZE_THRESHOLD = 100_000


def load_imagenet32(split: str, cache_dir: Path = IMAGENET32_DIR) -> np.ndarray:
    """Load ImageNet32 split. Returns (N, 3, 32, 32) uint8 (mmap'd)."""
    if split == "train":
        path = cache_dir / "train.npy"
    elif split == "val":
        path = cache_dir / "val.npy"
    else:
        raise ValueError(f"Unknown split: {split!r} (use 'train' or 'val')")

    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `python image_order/data/imagenet32/build_imagenet32.py` first."
        )
    arr = np.load(path, mmap_mode="r")
    if arr.dtype != np.uint8 or arr.shape[-3:] != (3, IMG_SIZE, IMG_SIZE):
        raise ValueError(f"Unexpected ImageNet32 array: dtype={arr.dtype}, shape={arr.shape}")
    return arr


def _normalize_uint8_to_patches(images_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 (B, 3, 32, 32) → float32 (B, 64, 48), normalized to [-1, 1]."""
    x = images_uint8.float().div_(127.5).sub_(1.0)
    return image_to_patches(x)


class ImageNet32Patches(torch.utils.data.Dataset):
    """ImageNet32 as patch sequences. Drop-in for CIFAR10Patches.

    Two memory modes (auto-selected by image count):
      - Materialized (default for ≤100k images, e.g. val): builds `.patches`
        up front. Same as CIFAR10Patches.
      - Lazy (for the 800k train split): keeps uint8 in RAM (~2.3 GB), exposes
        `patches_for_indices(idx, device)`; `.patches` is None.

    Force a mode via `lazy=True/False`.
    """

    def __init__(
        self,
        split: str = "train",
        normalize: bool = True,
        cache_dir: Path = IMAGENET32_DIR,
        max_images: Optional[int] = None,
        lazy: Optional[bool] = None,
    ):
        images_uint8 = load_imagenet32(split, cache_dir=cache_dir)
        if max_images is not None:
            images_uint8 = images_uint8[:max_images]
        n = images_uint8.shape[0]

        if lazy is None:
            lazy = n > MATERIALIZE_THRESHOLD
        self.lazy = lazy
        self.normalize = normalize

        # Always materialize images as a uint8 torch tensor (2.3 GB for 800k).
        images_u8_np = np.ascontiguousarray(images_uint8)
        self.images_u8 = torch.from_numpy(images_u8_np)
        self.labels = torch.zeros(n, dtype=torch.long)

        if not lazy:
            x = self.images_u8.float()
            if normalize:
                x = x.div_(127.5).sub_(1.0)
            self.images = x
            self.patches = image_to_patches(x)
            self.images_minmax = (float(x.min()), float(x.max()))
        else:
            self.images = None
            self.patches = None
            self.images_minmax = (-1.0, 1.0) if normalize else (0.0, 255.0)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.lazy:
            return self.patches[i], self.labels[i]
        patches = _normalize_uint8_to_patches(self.images_u8[i : i + 1])[0]
        return patches, self.labels[i]

    def patches_for_indices(self, idx, device: str = "cpu") -> torch.Tensor:
        """Return (len(idx), 64, 48) float32 patches on `device` for the requested rows."""
        if isinstance(idx, np.ndarray):
            idx_t = torch.from_numpy(idx.astype(np.int64))
        elif isinstance(idx, torch.Tensor):
            idx_t = idx.to(torch.long)
        else:
            idx_t = torch.as_tensor(idx, dtype=torch.long)
        batch_u8 = self.images_u8.index_select(0, idx_t).to(device, non_blocking=True)
        return _normalize_uint8_to_patches(batch_u8)


if __name__ == "__main__":
    ds_val = ImageNet32Patches("val")
    print(f"Val: {len(ds_val)} images  lazy={ds_val.lazy}  patches={None if ds_val.patches is None else tuple(ds_val.patches.shape)}")

    ds_train = ImageNet32Patches("train")
    print(f"Train: {len(ds_train)} images  lazy={ds_train.lazy}  patches={None if ds_train.patches is None else 'materialized'}")

    # Lazy access
    p = ds_train.patches_for_indices([0, 1, 2])
    assert p.shape == (3, N_PATCHES, PATCH_DIM), p.shape
    assert p.dtype == torch.float32
    print(f"lazy slice: shape={tuple(p.shape)} dtype={p.dtype} range=({p.min().item():.3f}, {p.max().item():.3f})")

    # Round-trip parity vs materialized val
    p_val_lazy = ds_val.patches_for_indices([0, 1])
    p_val_mat = ds_val.patches[:2]
    diff = (p_val_lazy - p_val_mat).abs().max().item()
    assert diff < 1e-5, f"lazy/materialized disagree by {diff}"
    print(f"lazy/materialized parity OK (max diff {diff:.2e})")
    print("OK data_imagenet32_patches")
