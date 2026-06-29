"""P2: Batch-level block graph diagnostic for ImageNet-64 VQ-f4 800k patch8x8.

For each of 100 batches, compute A_batch_block[64,64] and locality metrics.
Goal: check whether spatial structure is batch/context-dependent rather than
globally stable.

Output:
  batch_graph_diagnostics_step{STEP}.tsv
  batch_graph_histograms_step{STEP}.png

Usage:
    python block_lo_arm_order_network/batch_level_graph_diagnostic.py \
        --ckpt nanogpt-learned-order/out/image_large/.../checkpoints/ckpt_iter0020000.pt \
        --data nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/train.bin \
        --step 20000 --n-batches 100 --batch-size 8 --device cuda:1
"""

import os, sys, argparse, time, json
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
_NANO = _ROOT.parent / "nanogpt-learned-order"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_NANO))

from AOGPT import AOGPTConfig, AOGPT

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


def extract_single_image_attention(model, tokens_1d, device):
    """Extract A_token[256,256] for a single image (averaged over M_ORDERS random orders).

    Uses last 4 layers, all heads averaged (same as extract_image_attention.py).
    """
    tokens = torch.from_numpy(tokens_1d.astype(np.int64)).to(device).unsqueeze(0)
    A_sum = torch.zeros(T, T, device=device)

    for _ in range(M_ORDERS):
        rand_order = torch.randperm(T, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, attn_list = model.forward_fn(tokens, rand_order, return_attentions=True)

        # Average last 4 layers, all heads
        attn_stack = torch.stack(attn_list[-4:], dim=0)  # (4, 1, nh, T+1, T+1)
        attn = attn_stack.mean(dim=[0, 2])[0]  # (T+1, T+1)
        attn_content = attn[1:, 1:]  # (T, T) in model order

        inv_order = torch.argsort(rand_order[0])
        attn_phys = attn_content[inv_order][:, inv_order]
        A_sum += attn_phys

    A_avg = A_sum / M_ORDERS
    return A_avg.cpu().numpy()


def aggregate_token_to_block(A_token):
    """Aggregate A_token[256,256] → A_block[64,64] by consecutive 4-token blocks."""
    A_block = np.zeros((NUM_BLOCKS, NUM_BLOCKS), dtype=np.float64)
    for i in range(NUM_BLOCKS):
        for j in range(NUM_BLOCKS):
            block = A_token[i*BLOCK_LEN:(i+1)*BLOCK_LEN, j*BLOCK_LEN:(j+1)*BLOCK_LEN]
            A_block[i, j] = block.mean()
    return A_block


def compute_locality_metrics(A_block_single):
    """Compute locality metrics for one 64×64 attention matrix on 8×8 grid."""
    N = NUM_BLOCKS
    rows = np.arange(N) // GRID
    cols = np.arange(N) % GRID
    manh_matrix = np.abs(rows[:, None] - rows[None, :]) + np.abs(cols[:, None] - cols[None, :])

    A = A_block_single.copy()
    np.fill_diagonal(A, 0)
    total = A.sum() + 1e-12

    # Weighted mean manhattan
    A_norm = A / total
    mean_manh = float((A_norm * manh_matrix).sum())

    # P(d<=1)
    mask_d1 = (manh_matrix <= 1) & ~np.eye(N, dtype=bool)
    p_d1 = float(A[mask_d1].sum() / total)

    # P(d<=2)
    mask_d2 = (manh_matrix <= 2) & ~np.eye(N, dtype=bool)
    p_d2 = float(A[mask_d2].sum() / total)

    # Same quadrant
    quad = (rows // 4) * 2 + (cols // 4)
    same_q_mask = (quad[:, None] == quad[None, :]) & ~np.eye(N, dtype=bool)
    p_same_q = float(A[same_q_mask].sum() / total)

    # Same 2×2 super-region (rows//2, cols//2)
    sr = (rows // 2) * 4 + (cols // 2)
    same_s_mask = (sr[:, None] == sr[None, :]) & ~np.eye(N, dtype=bool)
    p_same_s = float(A[same_s_mask].sum() / total)

    # Hub score
    row_sums = A.sum(axis=1)
    hub_score = float(row_sums.max() / (row_sums.mean() + 1e-12))

    # Readiness signal
    B = A.T
    out_deg = B.sum(axis=1)
    in_deg = B.sum(axis=0)
    source = out_deg - 0.5 * in_deg
    readiness = float(np.std(source) / (np.mean(np.abs(B)) + 1e-12))

    return {
        "mean_manh": mean_manh,
        "P(d<=1)": p_d1,
        "P(d<=2)": p_d2,
        "same_q": p_same_q,
        "same_s": p_same_s,
        "hub_score": hub_score,
        "readiness_signal": readiness,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--n-batches", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "probe_results_image_large" / "batch_level_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.ckpt}...", flush=True)
    model, model_args = load_model(args.ckpt, args.device)
    print(f"  n_layer={model_args['n_layer']}, n_head={model_args['n_head']}", flush=True)

    # Load data
    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    n_total_images = len(data) // T
    n_needed = args.n_batches * args.batch_size
    print(f"  Data: {n_total_images} images available, need {n_needed}", flush=True)
    assert n_total_images >= n_needed, f"Not enough images: {n_total_images} < {n_needed}"

    # Shuffle image indices for batch sampling
    rng = np.random.RandomState(42)
    img_indices = rng.permutation(n_total_images)[:n_needed]

    # Process batch by batch
    print(f"\nProcessing {args.n_batches} batches × {args.batch_size} images...", flush=True)
    t0 = time.time()

    batch_metrics = []
    A_global_sum = np.zeros((NUM_BLOCKS, NUM_BLOCKS), dtype=np.float64)

    for batch_idx in range(args.n_batches):
        batch_start = batch_idx * args.batch_size
        batch_img_ids = img_indices[batch_start:batch_start + args.batch_size]

        # Compute A_batch_block[64,64]
        A_batch_sum = np.zeros((NUM_BLOCKS, NUM_BLOCKS), dtype=np.float64)
        for img_id in batch_img_ids:
            tokens_1d = data[img_id * T:(img_id + 1) * T]
            A_token = extract_single_image_attention(model, tokens_1d, args.device)
            A_block = aggregate_token_to_block(A_token)
            A_batch_sum += A_block

        A_batch_block = A_batch_sum / args.batch_size
        A_global_sum += A_batch_block

        # Compute metrics for this batch
        m = compute_locality_metrics(A_batch_block)
        m["batch_id"] = batch_idx
        batch_metrics.append(m)

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Batch {batch_idx+1}/{args.n_batches} ({elapsed:.0f}s)", flush=True)

    # Global average
    A_global_block = A_global_sum / args.n_batches
    m_global = compute_locality_metrics(A_global_block)

    elapsed = time.time() - t0
    print(f"\n  Total extraction time: {elapsed:.0f}s", flush=True)

    # Compute distribution stats
    metrics_keys = ["mean_manh", "P(d<=1)", "P(d<=2)", "same_q", "same_s", "hub_score", "readiness_signal"]
    print(f"\n{'='*70}")
    print(f"BATCH-LEVEL BLOCK GRAPH DIAGNOSTIC (step {args.step})")
    print(f"  {args.n_batches} batches × {args.batch_size} images")
    print(f"{'='*70}")
    print(f"\n{'metric':<20s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s} {'global':>10s}")
    print("-" * 72)

    dist_stats = {}
    for key in metrics_keys:
        vals = np.array([m[key] for m in batch_metrics])
        dist_stats[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "global": m_global[key],
        }
        print(f"  {key:<18s} {vals.mean():10.4f} {vals.std():10.4f} "
              f"{vals.min():10.4f} {vals.max():10.4f} {m_global[key]:10.4f}")

    # Key interpretation
    print(f"\n--- Interpretation ---")
    manh_vals = np.array([m["mean_manh"] for m in batch_metrics])
    pd1_vals = np.array([m["P(d<=1)"] for m in batch_metrics])

    n_local_batches = (manh_vals < 4.5).sum()
    n_strong_pd1 = (pd1_vals > 0.10).sum()
    print(f"  Batches with mean_manh < 4.5 (local structure): {n_local_batches}/{args.n_batches}")
    print(f"  Batches with P(d<=1) > 0.10: {n_strong_pd1}/{args.n_batches}")
    print(f"  Random uniform reference: mean_manh ≈ 5.33, P(d<=1) ≈ 0.063")

    if n_local_batches > 10 and m_global["mean_manh"] > 4.8:
        print(f"  *** BATCH-DEPENDENT locality: some batches have it, global doesn't ***")
        print(f"  → Next step: context-dependent / batch-dependent Graph-RW")
    elif n_local_batches > 10:
        print(f"  *** Both batch-level and global show locality ***")
    else:
        print(f"  No strong batch-level spatial locality detected.")
        print(f"  → Random baseline does not naturally form spatial attention in this setting.")

    # Save TSV
    tsv_path = out_dir / f"batch_graph_diagnostics_step{args.step}.tsv"
    with open(tsv_path, "w") as f:
        header = "batch_id\tmean_manh\tP(d<=1)\tP(d<=2)\tsame_q\tsame_s\thub_score\treadiness_signal"
        f.write(header + "\n")
        for m in batch_metrics:
            f.write(f"{m['batch_id']}\t{m['mean_manh']:.4f}\t{m['P(d<=1)']:.4f}\t"
                    f"{m['P(d<=2)']:.4f}\t{m['same_q']:.4f}\t{m['same_s']:.4f}\t"
                    f"{m['hub_score']:.3f}\t{m['readiness_signal']:.4f}\n")
    print(f"\n  Saved: {tsv_path}", flush=True)

    # Save summary JSON
    summary = {
        "step": args.step,
        "n_batches": args.n_batches,
        "batch_size": args.batch_size,
        "n_total_images": n_needed,
        "global_metrics": m_global,
        "distribution_stats": dist_stats,
        "n_local_batches_manh45": int(n_local_batches),
        "n_strong_pd1_010": int(n_strong_pd1),
        "elapsed_seconds": elapsed,
    }
    with open(out_dir / f"batch_diagnostic_summary_step{args.step}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save global A_block for reference
    np.save(out_dir / f"A_global_block_step{args.step}.npy", A_global_block.astype(np.float32))

    # Generate histograms
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].hist(manh_vals, bins=20, edgecolor='black', alpha=0.7)
        axes[0].axvline(m_global["mean_manh"], color='red', linestyle='--', label='global avg')
        axes[0].axvline(5.33, color='gray', linestyle=':', label='random ref')
        axes[0].set_xlabel("Mean Manhattan")
        axes[0].set_title(f"Batch-level Mean Manhattan (step {args.step})")
        axes[0].legend()

        axes[1].hist(pd1_vals, bins=20, edgecolor='black', alpha=0.7)
        axes[1].axvline(m_global["P(d<=1)"], color='red', linestyle='--', label='global avg')
        axes[1].axvline(0.063, color='gray', linestyle=':', label='random ref')
        axes[1].set_xlabel("P(d<=1)")
        axes[1].set_title(f"Batch-level P(d<=1) (step {args.step})")
        axes[1].legend()

        hub_vals = np.array([m["hub_score"] for m in batch_metrics])
        axes[2].hist(hub_vals, bins=20, edgecolor='black', alpha=0.7)
        axes[2].axvline(m_global["hub_score"], color='red', linestyle='--', label='global avg')
        axes[2].set_xlabel("Hub Score")
        axes[2].set_title(f"Batch-level Hub Score (step {args.step})")
        axes[2].legend()

        plt.tight_layout()
        fig_path = out_dir / f"batch_graph_histograms_step{args.step}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Saved: {fig_path}", flush=True)
    except Exception as e:
        print(f"  Warning: could not generate plots: {e}", flush=True)

    print(f"\nDone. Output: {out_dir}/")


if __name__ == "__main__":
    main()
