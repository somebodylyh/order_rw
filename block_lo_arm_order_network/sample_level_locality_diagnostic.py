"""P1: Sample-level locality diagnostic for ImageNet-64 VQ-f4 800k patch8x8.

For each of N individual images, compute A_x_block[64,64] and locality metrics.
Goal: judge whether spatial structure is sample-dependent (visible at single-image
level but washed out by batch/global averaging).

Output:
  sample_metrics_step{STEP}.tsv
  sample_histograms_step{STEP}.png
  sample_vs_batch_vs_global_step{STEP}.json

Usage:
    python block_lo_arm_order_network/sample_level_locality_diagnostic.py \
        --ckpt nanogpt-learned-order/out/image_large/.../checkpoints/ckpt_iter0020000.pt \
        --data nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/train.bin \
        --step 20000 --n-images 500 --device cuda:1
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
    """Return A_token[256,256] for one image, averaged over M_ORDERS random orders.
    Uses last 4 layers, all heads averaged (same as batch_level_graph_diagnostic)."""
    tokens = torch.from_numpy(tokens_1d.astype(np.int64)).to(device).unsqueeze(0)
    A_sum = torch.zeros(T, T, device=device)

    for _ in range(M_ORDERS):
        rand_order = torch.randperm(T, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, attn_list = model.forward_fn(tokens, rand_order, return_attentions=True)
        attn_stack = torch.stack(attn_list[-4:], dim=0)  # (4, 1, nh, T+1, T+1)
        attn = attn_stack.mean(dim=[0, 2])[0]  # (T+1, T+1)
        attn_content = attn[1:, 1:]  # (T, T) in model order
        inv_order = torch.argsort(rand_order[0])
        attn_phys = attn_content[inv_order][:, inv_order]
        A_sum += attn_phys

    return (A_sum / M_ORDERS).cpu().numpy()


def aggregate_token_to_block(A_token):
    """A_token[256,256] -> A_block[64,64] via consecutive 4-token mean pooling."""
    A_block = np.zeros((NUM_BLOCKS, NUM_BLOCKS), dtype=np.float64)
    for i in range(NUM_BLOCKS):
        for j in range(NUM_BLOCKS):
            block = A_token[i*BLOCK_LEN:(i+1)*BLOCK_LEN, j*BLOCK_LEN:(j+1)*BLOCK_LEN]
            A_block[i, j] = block.mean()
    return A_block


def compute_locality_metrics(A_block_single):
    """Locality metrics on a single 64x64 block matrix over an 8x8 grid."""
    N = NUM_BLOCKS
    rows = np.arange(N) // GRID
    cols = np.arange(N) % GRID
    manh_matrix = np.abs(rows[:, None] - rows[None, :]) + np.abs(cols[:, None] - cols[None, :])

    A = A_block_single.copy()
    np.fill_diagonal(A, 0)
    total = A.sum() + 1e-12

    mean_manh = float((A / total * manh_matrix).sum())
    mask_d1 = (manh_matrix <= 1) & ~np.eye(N, dtype=bool)
    p_d1 = float(A[mask_d1].sum() / total)
    mask_d2 = (manh_matrix <= 2) & ~np.eye(N, dtype=bool)
    p_d2 = float(A[mask_d2].sum() / total)

    quad = (rows // 4) * 2 + (cols // 4)
    same_q_mask = (quad[:, None] == quad[None, :]) & ~np.eye(N, dtype=bool)
    p_same_q = float(A[same_q_mask].sum() / total)

    sr = (rows // 2) * 4 + (cols // 2)
    same_s_mask = (sr[:, None] == sr[None, :]) & ~np.eye(N, dtype=bool)
    p_same_s = float(A[same_s_mask].sum() / total)

    row_sums = A.sum(axis=1)
    hub_score = float(row_sums.max() / (row_sums.mean() + 1e-12))

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
    p.add_argument("--n-images", type=int, default=500)
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--batch-size-ref", type=int, default=8,
                   help="Batch size used for the batch-level reference comparison")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "probe_results_image_large" / "sample_level_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.ckpt}...", flush=True)
    model, model_args = load_model(args.ckpt, args.device)
    print(f"  n_layer={model_args['n_layer']}, n_head={model_args['n_head']}", flush=True)

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    n_total_images = len(data) // T
    print(f"  Data: {n_total_images} images available, sampling {args.n_images}", flush=True)
    assert n_total_images >= args.n_images, f"Not enough images: {n_total_images} < {args.n_images}"

    rng = np.random.RandomState(args.seed)
    img_indices = rng.permutation(n_total_images)[:args.n_images]

    print(f"\nProcessing {args.n_images} individual images (M_ORDERS={M_ORDERS})...", flush=True)
    t0 = time.time()

    sample_metrics = []
    A_block_all_sum = np.zeros((NUM_BLOCKS, NUM_BLOCKS), dtype=np.float64)
    # Keep per-sample A_block in memory for batch-level rebuild comparison
    # (n_images * 64*64 * 8B = ~16MB for 500 imgs — fine)
    A_block_store = np.zeros((args.n_images, NUM_BLOCKS, NUM_BLOCKS), dtype=np.float32)

    for i, img_id in enumerate(img_indices):
        tokens_1d = data[img_id * T:(img_id + 1) * T]
        A_token = extract_single_image_attention(model, tokens_1d, args.device)
        A_block = aggregate_token_to_block(A_token)
        A_block_store[i] = A_block.astype(np.float32)
        A_block_all_sum += A_block

        m = compute_locality_metrics(A_block)
        m["sample_idx"] = int(i)
        m["img_id"] = int(img_id)
        sample_metrics.append(m)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.n_images} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s", flush=True)

    # Global average from sample-level
    A_global_block = A_block_all_sum / args.n_images
    m_global = compute_locality_metrics(A_global_block)

    # Reconstruct batch-level (batch_size_ref) averages from same samples for fair comparison
    bs = args.batch_size_ref
    n_batches = args.n_images // bs
    print(f"\n  Rebuilding {n_batches} reference batches (size={bs}) from same samples...", flush=True)
    batch_metrics_recon = []
    for b in range(n_batches):
        A_b = A_block_store[b*bs:(b+1)*bs].astype(np.float64).mean(axis=0)
        m_b = compute_locality_metrics(A_b)
        m_b["batch_id"] = int(b)
        batch_metrics_recon.append(m_b)

    # Print summary
    keys = ["mean_manh", "P(d<=1)", "P(d<=2)", "same_q", "same_s", "hub_score", "readiness_signal"]
    print(f"\n{'='*100}")
    print(f"SAMPLE-LEVEL vs BATCH-LEVEL vs GLOBAL  (step {args.step}, N={args.n_images} samples)")
    print(f"{'='*100}")
    header = f"\n{'metric':<20s}"
    for label in ["sample mean", "sample std", "sample min", "sample max",
                  "batch mean", "batch std", "global"]:
        header += f" {label:>12s}"
    print(header)
    print("-" * 110)

    dist_stats = {"sample": {}, "batch_recon": {}, "global": {}}
    for k in keys:
        s_vals = np.array([m[k] for m in sample_metrics])
        b_vals = np.array([m[k] for m in batch_metrics_recon])
        dist_stats["sample"][k] = {
            "mean": float(s_vals.mean()), "std": float(s_vals.std()),
            "min": float(s_vals.min()), "max": float(s_vals.max()),
        }
        dist_stats["batch_recon"][k] = {
            "mean": float(b_vals.mean()), "std": float(b_vals.std()),
            "min": float(b_vals.min()), "max": float(b_vals.max()),
        }
        dist_stats["global"][k] = m_global[k]
        print(f"  {k:<18s} "
              f"{s_vals.mean():12.4f} {s_vals.std():12.4f} {s_vals.min():12.4f} {s_vals.max():12.4f} "
              f"{b_vals.mean():12.4f} {b_vals.std():12.4f} {m_global[k]:12.4f}")

    # Interpretation
    print(f"\n--- Interpretation ---")
    s_manh = np.array([m["mean_manh"] for m in sample_metrics])
    s_pd1 = np.array([m["P(d<=1)"] for m in sample_metrics])
    n_local_samples = int((s_manh < 4.5).sum())
    n_strong_pd1 = int((s_pd1 > 0.10).sum())
    print(f"  Random uniform reference: mean_manh ≈ 5.33, P(d<=1) ≈ 0.063")
    print(f"  Samples with mean_manh < 4.5: {n_local_samples}/{args.n_images}")
    print(f"  Samples with P(d<=1) > 0.10: {n_strong_pd1}/{args.n_images}")

    if n_local_samples > args.n_images * 0.1 and m_global["mean_manh"] > 4.8:
        print(f"  *** SAMPLE-DEPENDENT locality: present per-sample, washed by averaging ***")
        print(f"  → Next step: sample-conditioned / context-dependent Graph-RW")
    elif n_local_samples > args.n_images * 0.1:
        print(f"  *** Locality present at both sample and global level ***")
    else:
        print(f"  No strong sample-level spatial locality detected.")
        print(f"  → Confirms: current random-order VQ image baseline has no spatial attention.")

    # Save TSV
    tsv_path = out_dir / f"sample_metrics_step{args.step}.tsv"
    with open(tsv_path, "w") as f:
        f.write("sample_idx\timg_id\tmean_manh\tP(d<=1)\tP(d<=2)\tsame_q\tsame_s\thub_score\treadiness_signal\n")
        for m in sample_metrics:
            f.write(f"{m['sample_idx']}\t{m['img_id']}\t{m['mean_manh']:.4f}\t"
                    f"{m['P(d<=1)']:.4f}\t{m['P(d<=2)']:.4f}\t{m['same_q']:.4f}\t"
                    f"{m['same_s']:.4f}\t{m['hub_score']:.3f}\t{m['readiness_signal']:.4f}\n")
    print(f"\n  Saved: {tsv_path}")

    # Save batch-recon TSV
    btsv = out_dir / f"batch_recon_metrics_step{args.step}.tsv"
    with open(btsv, "w") as f:
        f.write("batch_id\tmean_manh\tP(d<=1)\tP(d<=2)\tsame_q\tsame_s\thub_score\treadiness_signal\n")
        for m in batch_metrics_recon:
            f.write(f"{m['batch_id']}\t{m['mean_manh']:.4f}\t{m['P(d<=1)']:.4f}\t"
                    f"{m['P(d<=2)']:.4f}\t{m['same_q']:.4f}\t{m['same_s']:.4f}\t"
                    f"{m['hub_score']:.3f}\t{m['readiness_signal']:.4f}\n")
    print(f"  Saved: {btsv}")

    # Save summary JSON
    summary = {
        "step": args.step,
        "n_samples": args.n_images,
        "batch_size_ref": bs,
        "n_batches_recon": n_batches,
        "global_metrics": m_global,
        "distribution_stats": dist_stats,
        "n_local_samples_manh45": n_local_samples,
        "n_strong_pd1_010": n_strong_pd1,
        "elapsed_seconds": elapsed,
    }
    with open(out_dir / f"sample_vs_batch_vs_global_step{args.step}.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.save(out_dir / f"A_global_block_step{args.step}.npy", A_global_block.astype(np.float32))

    # Histograms
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes = axes.flatten()

        for ax, key, ref, label in [
            (axes[0], "mean_manh", 5.33, "random ref 5.33"),
            (axes[1], "P(d<=1)", 0.063, "random ref 0.063"),
            (axes[2], "readiness_signal", None, None),
            (axes[3], "hub_score", None, None),
        ]:
            vals = np.array([m[key] for m in sample_metrics])
            ax.hist(vals, bins=30, edgecolor='black', alpha=0.6, label=f"sample (n={args.n_images})")
            b_vals = np.array([m[key] for m in batch_metrics_recon])
            ax.hist(b_vals, bins=20, edgecolor='black', alpha=0.6,
                    color='orange', label=f"batch_recon (n={n_batches})")
            ax.axvline(m_global[key], color='red', linestyle='--', label='global')
            if ref is not None:
                ax.axvline(ref, color='gray', linestyle=':', label=label)
            ax.set_xlabel(key)
            ax.set_title(f"{key} distribution (step {args.step})")
            ax.legend(fontsize=8)

        plt.tight_layout()
        fig_path = out_dir / f"sample_histograms_step{args.step}.png"
        plt.savefig(fig_path, dpi=130)
        plt.close()
        print(f"  Saved: {fig_path}")
    except Exception as e:
        print(f"  Warning: could not generate plots: {e}", flush=True)

    print(f"\nDone. Output: {out_dir}/")


if __name__ == "__main__":
    main()
