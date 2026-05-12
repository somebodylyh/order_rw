"""CIFAR-10 patch dataset and grid utilities."""

import pickle
from typing import Tuple

import numpy as np
import torch
import torch.utils.data

# Constants
IMG_SIZE = 32
PATCH_GRID = 8
PATCH_SIZE = 4
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3  # 48
N_PATCHES = PATCH_GRID * PATCH_GRID  # 64
CIFAR_DIR = "/home/admin/bw/data/cifar-10-batches-py"
RASTER_ORDER = np.arange(N_PATCHES, dtype=np.int64)


def load_cifar10(split: str, cache_dir: str = CIFAR_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 from local pickle files.

    Args:
        split: "train" or "test"
        cache_dir: path to CIFAR-10 pickle directory

    Returns:
        Tuple of (images_uint8 shape (N, 3, 32, 32), labels shape (N,) int64)
    """
    if split == "train":
        batch_paths = [f"{cache_dir}/data_batch_{i}" for i in range(1, 6)]
    elif split == "test":
        batch_paths = [f"{cache_dir}/test_batch"]
    else:
        raise ValueError(f"Unknown split: {split}")

    all_data = []
    all_labels = []

    for path in batch_paths:
        with open(path, "rb") as f:
            batch = pickle.load(f, encoding="latin1")
            all_data.append(batch["data"])
            all_labels.extend(batch["labels"])

    data = np.concatenate(all_data, axis=0)
    labels = np.array(all_labels, dtype=np.int64)

    # Reshape from (N, 3072) to (N, 3, 32, 32)
    images = data.reshape(-1, 3, IMG_SIZE, IMG_SIZE)

    return images, labels


def image_to_patches(x: torch.Tensor) -> torch.Tensor:
    """Convert image batch to patch batch.

    Args:
        x: (B, 3, 32, 32) float or uint8

    Returns:
        (B, 64, 48) float32
    """
    if x.dtype == torch.uint8:
        x = x.float()

    # unfold: dimension, kernel_size, stride
    # unfold(2, 4, 4) unfolds H dim into sliding windows of height 4, stride 4
    # unfold(3, 4, 4) unfolds W dim into sliding windows of width 4, stride 4
    # Result shape: (B, 3, 8, 8, 4, 4)
    patches = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)

    # Permute to (B, 8, 8, 3, 4, 4) so we can flatten the last 3 dims together
    patches = patches.permute(0, 2, 3, 1, 4, 5)

    # Reshape to (B, 64, 48)
    b = patches.shape[0]
    patches = patches.reshape(b, N_PATCHES, PATCH_DIM)

    return patches


def patches_to_image(p: torch.Tensor) -> torch.Tensor:
    """Convert patch batch back to image batch.

    Args:
        p: (B, 64, 48) float32

    Returns:
        (B, 3, 32, 32) float32
    """
    b = p.shape[0]

    # Reshape (B, 64, 48) -> (B, 8, 8, 3, 4, 4)
    patches = p.reshape(b, PATCH_GRID, PATCH_GRID, 3, PATCH_SIZE, PATCH_SIZE)

    # Permute to (B, 3, 8, 4, 8, 4)
    patches = patches.permute(0, 3, 1, 4, 2, 5)

    # Reshape to (B, 3, 32, 32)
    images = patches.reshape(b, 3, IMG_SIZE, IMG_SIZE)

    return images


def grid_pos(idx: int) -> Tuple[int, int]:
    """Convert patch index to (row, col) on 8x8 grid."""
    row = idx // PATCH_GRID
    col = idx % PATCH_GRID
    return (row, col)


def manhattan(a: int, b: int) -> int:
    """Manhattan distance between two patch indices on the 8x8 grid."""
    r_a, c_a = grid_pos(a)
    r_b, c_b = grid_pos(b)
    return abs(r_a - r_b) + abs(c_a - c_b)


class CIFAR10Patches(torch.utils.data.Dataset):
    """CIFAR-10 dataset as patch sequences."""

    def __init__(
        self, split: str = "train", normalize: bool = True, cache_dir: str = CIFAR_DIR
    ):
        """Initialize dataset.

        Args:
            split: "train" or "test"
            normalize: if True, map [0, 255] -> [-1, 1]
            cache_dir: path to CIFAR-10 pickle directory
        """
        images_uint8, labels = load_cifar10(split, cache_dir=cache_dir)

        # Convert to float32
        images = images_uint8.astype(np.float32)

        # Normalize to [-1, 1]
        if normalize:
            images = images / 127.5 - 1.0

        # Convert to torch tensors
        images_tensor = torch.from_numpy(images)
        labels_tensor = torch.from_numpy(labels).long()

        # Precompute patches
        self.patches = image_to_patches(images_tensor)
        self.images = images_tensor
        self.labels = labels_tensor

        # Store min/max for diagnostics
        self.images_minmax = (
            float(images.min()),
            float(images.max()),
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.patches[i], self.labels[i]


if __name__ == "__main__":
    # Load train and test
    ds_train = CIFAR10Patches("train")
    ds_test = CIFAR10Patches("test")

    n_train = len(ds_train)
    n_test = len(ds_test)

    print(f"Train: {n_train} images, {ds_train.patches.shape}")
    print(f"Test: {n_test} images, {ds_test.patches.shape}")
    print(f"Images range: {ds_train.images_minmax}")

    # Test round-trip
    x = torch.stack([ds_train.patches[i] for i in range(4)])
    img = patches_to_image(x)
    x2 = image_to_patches(img)
    assert torch.allclose(x, x2, atol=1e-5), "Round-trip failed"

    # Test grid helpers
    assert manhattan(0, 8) == 1, f"manhattan(0, 8) = {manhattan(0, 8)}"
    assert manhattan(0, 63) == 14, f"manhattan(0, 63) = {manhattan(0, 63)}"
    assert manhattan(0, 9) == 2, f"manhattan(0, 9) = {manhattan(0, 9)}"
    assert grid_pos(0) == (0, 0), f"grid_pos(0) = {grid_pos(0)}"
    assert grid_pos(63) == (7, 7), f"grid_pos(63) = {grid_pos(63)}"

    # Test random image round-trip
    idx = np.random.randint(0, n_train)
    img_orig = ds_train.images[idx]
    patches = image_to_patches(img_orig.unsqueeze(0))
    img_recon = patches_to_image(patches).squeeze(0)
    max_error = (img_orig - img_recon).abs().max().item()
    assert max_error < 1e-5, f"Max reconstruction error: {max_error}"

    print(f"OK data_image_patches: train={n_train}, test={n_test}, patches={tuple(ds_train.patches.shape)}, range={ds_train.images_minmax}")
