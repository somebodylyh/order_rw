"""P1: Layer/head-wise locality scan for ImageNet-64 VQ-f4 800k patch8x8 baseline.

For each (layer, head), extract A_block[64,64] and compute locality metrics.
Goal: check whether local spatial structure exists in selected heads but is
washed out by global averaging.

Output:
  layer_head_locality_step{STEP}.tsv
  layer_head_manh_heatmap.png
  layer_head_near_prob_heatmap.png

Usage:
    python block_lo_arm_order_network/layer_head_locality_scan.py \
        --ckpt nanogpt-learned-order/out/image_large/.../checkpoints/ckpt_iter0020000.pt \
        --data nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/train.bin \
        --step 20000 --n-images 200 --device cuda:1
"""

import os, sys, argparse, time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
_NANO = _ROOT.parent / "nanogpt-learned-order"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_NANO))

from AOGPT import AOGPTConfig, AOGPT

N_LAYERS = 8
N_HEADS = 8
T = 256
BLOCK_LEN = 4
NUM_BLOCKS = 64
GRID = 8
M_ORDERS = 3


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_args = ckpt.get("model_args", {})
    cfg = ckpt.get("config", {})
    model_args = {}
    for k in ["block_size", "vocab_size", "n_layer", "n_head", "n_embd",
              "dropout", "bias", "block_order_block_len", "order_impl"]:
        if k in m_args:
            model_args[k] = m_args[k]
        elif k in cfg:
            model_args[k] = cfg[k]
    model_args.setdefault("force_manual_attention", True)
    model = AOGPT(AOGPTConfig(**model_args))
    state_dict = ckpt["model"]
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.crop_block_size(model_args["block_size"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, model_args


def extract_per_layer_head(model, data_tokens, n_images, device):
    """Extract per-layer, per-head A_token[256,256] in physical order."""
    n_available = len(data_tokens) // T
    n_images = min(n_images, n_available)

    # Accumulate per layer/head
    A_sum = torch.zeros(N_LAYERS, N_HEADS, T, T, device=device)
    count = 0

    print(f"Extracting {n_images} images × {M_ORDERS} orders per-layer-head...", flush=True)
    t0 = time.time()

    for img_idx in range(n_images):
        start = img_idx * T
        tokens = torch.from_numpy(
            data_tokens[start:start + T].astype(np.int64)
        ).to(device).unsqueeze(0)

        for _ in range(M_ORDERS):
            rand_order = torch.randperm(T, device=device).unsqueeze(0)

            with torch.no_grad():
                _, _, attn_list = model.forward_fn(
                    tokens, rand_order, return_attentions=True
                )

            # attn_list: list of (1, nh, T+1, T+1)
            inv_order = torch.argsort(rand_order[0])  # model_pos → physical_pos

            for layer_idx, attn in enumerate(attn_list):
                # attn: (1, nh, T+1, T+1) → remove [None] → (nh, T, T)
                a = attn[0, :, 1:, 1:]  # (nh, T, T) in model order
                # Remap to physical order
                a_phys = a[:, inv_order][:, :, inv_order]  # (nh, T, T)
                A_sum[layer_idx] += a_phys

            count += 1

        if (img_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {img_idx+1}/{n_images} ({elapsed:.0f}s)", flush=True)

    A_avg = (A_sum / count).cpu().numpy()  # (n_layers, n_heads, 256, 256)
    print(f"  Done. Total {count} forward passes in {time.time()-t0:.0f}s", flush=True)
    return A_avg


def aggregate_to_blocks(A_token_lh):
    """Aggregate (n_layers, n_heads, 256, 256) → (n_layers, n_heads, 64, 64).

    For patch8x8 data, consecutive 4 tokens form one 2×2 patch block.
    """
    nl, nh, _, _ = A_token_lh.shape
    A_block = np.zeros((nl, nh, NUM_BLOCKS, NUM_BLOCKS), dtype=np.float64)
    for i in range(NUM_BLOCKS):
        for j in range(NUM_BLOCKS):
            block = A_token_lh[:, :, i*BLOCK_LEN:(i+1)*BLOCK_LEN, j*BLOCK_LEN:(j+1)*BLOCK_LEN]
            A_block[:, :, i, j] = block.mean(axis=(-2, -1))
    return A_block


def compute_locality_metrics(A_block_single):
    """Compute locality metrics for one 64×64 attention matrix on 8×8 grid."""
    N = NUM_BLOCKS
    rows = np.arange(N) // GRID
    cols = np.arange(N) % GRID

    # Manhattan distances between all pairs
    manh_matrix = np.abs(rows[:, None] - rows[None, :]) + np.abs(cols[:, None] - cols[None, :])

    A = A_block_single.copy()
    np.fill_diagonal(A, 0)
    A_norm = A / (A.sum() + 1e-12)

    # Weighted mean manhattan
    mean_manh = float((A_norm * manh_matrix).sum())

    # P(d<=1): fraction of attention weight going to manhattan distance <= 1
    mask_d1 = (manh_matrix <= 1) & ~np.eye(N, dtype=bool)
    p_d1 = float(A[mask_d1].sum() / (A.sum() + 1e-12))

    # P(d<=2)
    mask_d2 = (manh_matrix <= 2) & ~np.eye(N, dtype=bool)
    p_d2 = float(A[mask_d2].sum() / (A.sum() + 1e-12))

    # Same quadrant (4×4 quadrants in 8×8 grid)
    quad = (rows // 4) * 2 + (cols // 4)  # 0,1,2,3
    same_q_mask = (quad[:, None] == quad[None, :]) & ~np.eye(N, dtype=bool)
    p_same_q = float(A[same_q_mask].sum() / (A.sum() + 1e-12))

    # Hub score: max row sum / mean row sum
    row_sums = A.sum(axis=1)
    hub_score = float(row_sums.max() / (row_sums.mean() + 1e-12))

    # Readiness signal: std(source) / mean(|B|)
    B = A.T  # simple transpose as graph
    out_deg = B.sum(axis=1)
    in_deg = B.sum(axis=0)
    source = out_deg - 0.5 * in_deg
    readiness = float(np.std(source) / (np.mean(np.abs(B)) + 1e-12))

    # Near/far ratio
    mask_near = manh_matrix <= 2
    mask_far = manh_matrix >= 5
    np.fill_diagonal(A, 0)
    near_mean = A[mask_near & ~np.eye(N, dtype=bool)].mean() if mask_near.sum() > N else 0
    far_mean = A[mask_far].mean() if mask_far.sum() > 0 else 1e-12
    near_far_ratio = float(near_mean / (far_mean + 1e-12))

    return {
        "mean_manh": mean_manh,
        "P(d<=1)": p_d1,
        "P(d<=2)": p_d2,
        "same_q": p_same_q,
        "hub_score": hub_score,
        "readiness_signal": readiness,
        "near_far_ratio": near_far_ratio,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--n-images", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "probe_results_image_large" / "layer_head_scan"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.ckpt}...", flush=True)
    model, model_args = load_model(args.ckpt, args.device)
    print(f"  n_layer={model_args['n_layer']}, n_head={model_args['n_head']}", flush=True)

    # Load data
    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    print(f"  Data: {len(data)//T} images available", flush=True)

    # Extract per-layer per-head attention
    A_token_lh = extract_per_layer_head(model, data, args.n_images, args.device)
    # (8, 8, 256, 256)

    # Aggregate to block level
    print("Aggregating to 64×64 block level...", flush=True)
    A_block_lh = aggregate_to_blocks(A_token_lh)  # (8, 8, 64, 64)

    # Also compute global average for comparison
    A_global = A_block_lh.mean(axis=(0, 1))  # (64, 64)

    # Compute metrics for each layer/head
    print("Computing locality metrics...", flush=True)
    rows_tsv = []
    metrics_manh = np.zeros((N_LAYERS, N_HEADS))
    metrics_pd1 = np.zeros((N_LAYERS, N_HEADS))

    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            m = compute_locality_metrics(A_block_lh[l, h])
            metrics_manh[l, h] = m["mean_manh"]
            metrics_pd1[l, h] = m["P(d<=1)"]
            rows_tsv.append({
                "step": args.step,
                "layer": l,
                "head": h,
                **m,
            })

    # Global metrics for comparison
    m_global = compute_locality_metrics(A_global)
    print(f"\n  Global (all layers/heads avg): mean_manh={m_global['mean_manh']:.3f}, "
          f"P(d<=1)={m_global['P(d<=1)']:.4f}, hub={m_global['hub_score']:.3f}", flush=True)

    # Random reference: uniform attention → expected mean_manh on 8×8 grid
    # E[manh] for uniform random pair on 8×8 grid ≈ 5.33
    print(f"  Random uniform expected mean_manh ≈ 5.33")

    # Best/worst heads
    best_manh_idx = np.unravel_index(np.argmin(metrics_manh), metrics_manh.shape)
    worst_manh_idx = np.unravel_index(np.argmax(metrics_manh), metrics_manh.shape)
    best_pd1_idx = np.unravel_index(np.argmax(metrics_pd1), metrics_pd1.shape)
    print(f"  Most local head (lowest manh): layer={best_manh_idx[0]} head={best_manh_idx[1]} "
          f"mean_manh={metrics_manh[best_manh_idx]:.3f}")
    print(f"  Most global head (highest manh): layer={worst_manh_idx[0]} head={worst_manh_idx[1]} "
          f"mean_manh={metrics_manh[worst_manh_idx]:.3f}")
    print(f"  Best P(d<=1) head: layer={best_pd1_idx[0]} head={best_pd1_idx[1]} "
          f"P(d<=1)={metrics_pd1[best_pd1_idx]:.4f}")

    # Save TSV
    tsv_path = out_dir / f"layer_head_locality_step{args.step}.tsv"
    with open(tsv_path, "w") as f:
        header = "step\tlayer\thead\tmean_manh\tP(d<=1)\tP(d<=2)\tsame_q\thub_score\treadiness_signal\tnear_far_ratio"
        f.write(header + "\n")
        for row in rows_tsv:
            f.write(f"{row['step']}\t{row['layer']}\t{row['head']}\t"
                    f"{row['mean_manh']:.4f}\t{row['P(d<=1)']:.4f}\t{row['P(d<=2)']:.4f}\t"
                    f"{row['same_q']:.4f}\t{row['hub_score']:.3f}\t"
                    f"{row['readiness_signal']:.4f}\t{row['near_far_ratio']:.4f}\n")
    print(f"\n  Saved: {tsv_path}", flush=True)

    # Save raw attention for later use
    np.save(out_dir / f"A_block_lh_step{args.step}.npy", A_block_lh.astype(np.float32))

    # Generate heatmaps
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        im0 = axes[0].imshow(metrics_manh, cmap="RdYlGn_r", aspect="auto")
        axes[0].set_title(f"Mean Manhattan Distance (step {args.step})")
        axes[0].set_xlabel("Head")
        axes[0].set_ylabel("Layer")
        axes[0].set_xticks(range(N_HEADS))
        axes[0].set_yticks(range(N_LAYERS))
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(metrics_pd1, cmap="RdYlGn", aspect="auto")
        axes[1].set_title(f"P(d<=1) Near-neighbor Prob (step {args.step})")
        axes[1].set_xlabel("Head")
        axes[1].set_ylabel("Layer")
        axes[1].set_xticks(range(N_HEADS))
        axes[1].set_yticks(range(N_LAYERS))
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        fig_path = out_dir / f"layer_head_heatmaps_step{args.step}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Saved: {fig_path}", flush=True)
    except Exception as e:
        print(f"  Warning: could not generate plots: {e}", flush=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"LAYER/HEAD LOCALITY SCAN COMPLETE (step {args.step})")
    print(f"{'='*60}")
    print(f"  Global mean_manh: {m_global['mean_manh']:.3f}")
    print(f"  Layer-head range: [{metrics_manh.min():.3f}, {metrics_manh.max():.3f}]")
    print(f"  Std across heads: {metrics_manh.std():.3f}")
    print(f"  Heads with mean_manh < 4.5 (strong locality): {(metrics_manh < 4.5).sum()}")
    print(f"  Heads with P(d<=1) > 0.10: {(metrics_pd1 > 0.10).sum()}")
    if metrics_manh.min() < 4.5:
        print(f"  *** SPATIAL STRUCTURE FOUND in specific heads ***")
    else:
        print(f"  No strong spatial locality in individual heads.")
    print(f"\nOutput: {out_dir}/")


if __name__ == "__main__":
    main()
