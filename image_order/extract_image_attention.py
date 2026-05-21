"""Extract A_global (64x64 patch-patch attention) from an image AOGPT checkpoint.

For each image, M random reveal orders are used. Each forward pass returns
attention tensors of shape (B, n_head, N+1, N+1) per layer (N=64 patches,
+1 for the [None] prefix token).

The extraction pipeline:
  1. Average attention over all n_head heads -> (B, N+1, N+1)
  2. Average over all n_layer layers -> (B, N+1, N+1)
  3. Drop [None] row and col (slice [:, 1:, 1:]) -> (B, N, N) in shuffled order
  4. Un-shuffle to physical patch index order
  5. Accumulate across M passes and images, then normalize by count

Usage:
    python image_order/extract_image_attention.py \\
        --ckpt <ckpt.pt> \\
        --output-dir <dir> \\
        [--num-images 200] [--M-passes 3] [--batch-size 32] \\
        [--device cuda] [--seed 42] [--split test]
"""

from __future__ import annotations

import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Make model and data importable regardless of cwd
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_ORDER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, IMAGE_ORDER_DIR)
sys.path.insert(0, REPO_ROOT)

from data_image_patches import CIFAR10Patches
from data_imagenet32_patches import ImageNet32Patches
from model_image_aogpt import ImageAOGPT, ImageAOGPTConfig


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: str) -> ImageAOGPT:
    """Load ImageAOGPT from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "config" in ckpt and isinstance(ckpt["config"], dict):
        config = ImageAOGPTConfig(**ckpt["config"])
    else:
        config = ImageAOGPTConfig()
        print("  [warn] No 'config' in checkpoint; using default ImageAOGPTConfig.")

    model = ImageAOGPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Un-shuffle helper
# ---------------------------------------------------------------------------

def unshuffle_attention(attn_shuf: torch.Tensor, orders: torch.Tensor) -> np.ndarray:
    """Un-permute attention from shuffled-position order to physical patch order.

    Args:
        attn_shuf : (B, N, N) float32 — attention in shuffled-position space.
                    attn_shuf[b, i, j] = attention from shuffled-pos i to
                    shuffled-pos j.
        orders    : (B, N) int64 — orders[b, i] = physical patch index placed
                    at shuffled position i.

    Returns:
        attn_phys : (B, N, N) float64 numpy array — attention in physical
                    patch-index space.  attn_phys[b, p, q] = attention from
                    physical patch p to physical patch q.
    """
    B, N, _ = attn_shuf.shape
    attn_phys = np.zeros((B, N, N), dtype=np.float64)

    orders_np = orders.cpu().numpy()          # (B, N)
    attn_np = attn_shuf.cpu().numpy()         # (B, N, N)

    for b in range(B):
        ord_b = orders_np[b]                  # (N,) physical indices
        # attn_phys[b, ord_b[i], ord_b[j]] = attn_np[b, i, j]
        # Vectorised: index on both axes simultaneously
        row_idx = ord_b[:, np.newaxis]        # (N, 1)
        col_idx = ord_b[np.newaxis, :]        # (1, N)
        attn_phys[b][row_idx, col_idx] = attn_np[b]

    return attn_phys


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def _check_row_sums(attn_shuf: torch.Tensor, attn_phys: np.ndarray,
                    rtol: float = 1e-4) -> None:
    """Assert that row sums are preserved under the unshuffle permutation."""
    row_sum_shuf = attn_shuf.cpu().numpy().sum(axis=-1)   # (B, N)
    row_sum_phys = attn_phys.sum(axis=-1)                  # (B, N)
    # Row sums of attn_phys (rows permuted, then columns permuted) must equal
    # the row sums of attn_shuf (permuting columns doesn't change row sums, and
    # permuting rows only reorders the row-sum vector).
    sorted_shuf = np.sort(row_sum_shuf, axis=-1)
    sorted_phys = np.sort(row_sum_phys, axis=-1)
    max_err = np.abs(sorted_shuf - sorted_phys).max()
    assert max_err < rtol, (
        f"Unshuffle row-sum sanity failed: max_err={max_err:.2e} (tol={rtol}). "
        "The permutation is incorrect."
    )


# ---------------------------------------------------------------------------
# Heatmap saving
# ---------------------------------------------------------------------------

def save_heatmap(A: np.ndarray, path: str, title: str = "A_global") -> None:
    """Save a heatmap PNG with grid lines every 8 patches (block boundaries)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(A, cmap="viridis", aspect="equal")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Key patch index")
    ax.set_ylabel("Query patch index")
    # Grid lines at every 8 patches
    for tick in range(0, 65, 8):
        ax.axhline(tick - 0.5, color="white", linewidth=0.5, alpha=0.6)
        ax.axvline(tick - 0.5, color="white", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract(args) -> None:
    device = args.device

    # 1. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load model
    print(f"[1/4] Loading checkpoint: {args.ckpt}")
    model = load_model(args.ckpt, device)
    model.eval()
    N = model.config.n_patches  # 64

    # 3. Load dataset
    if args.dataset == "cifar10":
        print(f"[2/4] Loading CIFAR10Patches(split='{args.split}')")
        ds = CIFAR10Patches(args.split)
        patches_all = ds.patches[: args.num_images]
    elif args.dataset == "imagenet32":
        print(f"[2/4] Loading ImageNet32Patches(split='{args.split}')")
        ds = ImageNet32Patches(args.split, max_images=args.num_images)
        if ds.patches is not None:
            patches_all = ds.patches[: args.num_images]
        else:
            patches_all = ds.patches_for_indices(
                np.arange(min(args.num_images, len(ds))), device="cpu"
            )
    else:
        raise ValueError(f"Unknown --dataset {args.dataset!r}")
    num_images = patches_all.shape[0]
    print(f"  Using {num_images} images.")

    # 4. Extract attention
    print(f"[3/4] Extracting attention (M={args.M_passes} passes, "
          f"batch_size={args.batch_size})")
    torch.manual_seed(args.seed)

    A_sum = np.zeros((N, N), dtype=np.float64)
    count = 0

    total_batches = (num_images + args.batch_size - 1) // args.batch_size
    batch_num = 0

    for start in range(0, num_images, args.batch_size):
        batch_num += 1
        end = min(start + args.batch_size, num_images)
        batch = patches_all[start:end].to(device)   # (B, 64, 48)
        B = batch.shape[0]

        # Accumulate over M passes
        pass_sum = np.zeros((B, N, N), dtype=np.float64)

        for m in range(args.M_passes):
            # Generate per-sample random reveal orders
            orders = torch.stack(
                [torch.randperm(N) for _ in range(B)]
            ).to(device)  # (B, N)

            with torch.no_grad():
                _, _, attn_outputs = model.forward_fn(
                    batch, orders, return_attentions=True
                )
                # attn_outputs: list[n_layer] each (B, n_head, N+1, N+1)

            # Stack layers: (n_layer, B, n_head, N+1, N+1)
            attn_stack = torch.stack(attn_outputs, dim=0)

            # Average over heads: (n_layer, B, N+1, N+1)
            attn_avg_heads = attn_stack.mean(dim=2)

            # Average over layers: (B, N+1, N+1)
            attn_avg = attn_avg_heads.mean(dim=0)

            # Drop [None] token (row 0 and col 0): -> (B, N, N)
            attn_patch = attn_avg[:, 1:, 1:]   # (B, N, N)

            # Un-shuffle to physical patch order
            attn_phys = unshuffle_attention(attn_patch, orders)  # (B, N, N) float64

            # Sanity check (row-sum invariance): only for first pass of first batch
            if m == 0 and batch_num == 1:
                _check_row_sums(attn_patch, attn_phys)

            pass_sum += attn_phys

        # Average across M passes, then accumulate
        A_sum += pass_sum.sum(axis=0) / args.M_passes
        count += B

        if batch_num % 5 == 0 or batch_num == total_batches:
            print(f"  batch {batch_num}/{total_batches}, images processed: {count}")

    # 5. Compute global average
    print("[4/4] Finalising A_global")
    A_global = (A_sum / count).astype(np.float32)
    np.fill_diagonal(A_global, 0.0)

    # Sanity checks
    assert not np.any(np.isnan(A_global)), "NaN detected in A_global!"
    assert not np.any(np.isinf(A_global)), "Inf detected in A_global!"

    B_global = A_global.T.copy()
    np.fill_diagonal(B_global, 0.0)

    # 6. Save .npy files
    a_path = os.path.join(args.output_dir, "A_global.npy")
    b_path = os.path.join(args.output_dir, "B_global.npy")
    np.save(a_path, A_global)
    np.save(b_path, B_global)
    print(f"  Saved {a_path}")
    print(f"  Saved {b_path}")

    # 7. Save heatmaps
    a_heatmap = os.path.join(args.output_dir, "A_global_heatmap.png")
    b_heatmap = os.path.join(args.output_dir, "B_global_heatmap.png")
    save_heatmap(A_global, a_heatmap, title="A_global (patch-patch attention)")
    save_heatmap(B_global, b_heatmap, title="B_global (transposed)")
    print(f"  Saved {a_heatmap}")
    print(f"  Saved {b_heatmap}")

    # 8. Verify PNG file sizes
    for png_path in (a_heatmap, b_heatmap):
        sz = os.path.getsize(png_path)
        assert sz > 5 * 1024, (
            f"PNG file too small ({sz} bytes): {png_path}. "
            "matplotlib may not have written the file correctly."
        )

    # 9. Compute summary stats
    A_mean = float(A_global.mean())
    A_max = float(A_global.max())

    # sparsity_topk5: mean over rows of (sum of top-5 entries / row sum)
    row_sums = A_global.sum(axis=1)                        # (N,)
    top5_sums = np.sort(A_global, axis=1)[:, -5:].sum(axis=1)  # (N,)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_per_row = np.where(row_sums > 0, top5_sums / row_sums, 0.0)
    sparsity_topk5 = float(frac_per_row.mean())

    print(
        f"OK extract_image_attention: A_global mean={A_mean:.4f}, "
        f"max={A_max:.4f}, sparsity_topk5={sparsity_topk5:.3f}, "
        f"saved to {args.output_dir}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract A_global from an image AOGPT checkpoint."
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to baseline ckpt .pt")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save A_global.npy / B_global.npy / heatmaps")
    parser.add_argument("--num-images", type=int, default=200,
                        help="Number of images to use (default: 200)")
    parser.add_argument("--M-passes", type=int, default=3,
                        help="Forward passes per image with different random orders (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for forward passes (default: 32)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (default: cuda if available)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use (default: test for cifar10, val for imagenet32)")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "imagenet32"],
                        help="Which dataset's patch sequences to use (default: cifar10)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract(args)
