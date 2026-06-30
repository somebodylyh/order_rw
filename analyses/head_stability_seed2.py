#!/usr/bin/env python3
"""Quick cross-ckpt head stability scan for seed2 (M=50, all 32 heads, 5 checkpoints).
Small M → fast (~10-15 min). Purpose: does the best head change across steps?
"""
import sys, os, json, time, pathlib
import numpy as np
import torch
from scipy.stats import kendalltau

_ROOT = pathlib.Path(__file__).resolve().parent.parent / "block_lo_arm_order_network"
sys.path.insert(0, str(_ROOT))

from training_utils import SEQ_LEN, N, BLOCK_LEN
from clean_training_protocol import expand_model_blocks_to_token_order
from neural_readout.extract_b import _load_model_and_chunks
from neural_readout.teacher_labels import generate_teacher_label
from per_head_order_scan import extract_per_head_and_heavy_A, _batch_mean_B
from batch_readout.diversity_batch import teacher_diversity_stats

BASE = pathlib.Path("/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/probe_results")
CKPTS = [
    BASE / "random_baseline_continuous_jun08_seed2/ckpt_step10000.pt",
    BASE / "random_baseline_continuous_jun08_seed2/ckpt_step20000.pt",
    BASE / "random_baseline_continuous_jun08_seed2/ckpt_step30000.pt",
    BASE / "random_baseline_continuous_jun08_seed2/ckpt_step40000.pt",
    BASE / "random_baseline_continuous_jun08_seed2/ckpt_step50000.pt",
]
M = 50
BATCH_SIZE = 32
DEVICE = "cuda:1"

def scan_one(ckpt_path, label):
    total = M * BATCH_SIZE
    model, chunks, clean_perm, dev, _ci = _load_model_and_chunks(
        ckpt_path, total, seed=42, device=DEVICE, split="train"
    )
    A_lh, A_heavy = extract_per_head_and_heavy_A(
        model, chunks, clean_perm, dev, seed=42, fwd_batch=32, none_mode="b0"
    )
    L, H = A_lh.shape[1], A_lh.shape[2]
    l2r = np.arange(N)
    results = []
    for l in range(L):
        for h in range(H):
            B = _batch_mean_B(A_lh[:, l, h], M, BATCH_SIZE)
            sigmas = np.empty((M, N), dtype=np.int64)
            for m in range(M):
                sigmas[m], _, _ = generate_teacher_label(B[m], alpha_dep=0.5)
            taus = []
            for s in sigmas:
                t, _ = kendalltau(s, l2r)
                if not np.isnan(t): taus.append(t)
            tau = float(np.mean(taus)) if taus else float("nan")
            div = teacher_diversity_stats(sigmas)
            results.append({
                "layer": l, "head": h,
                "tau_vs_l2r": tau,
                "mean_pairwise_tau": div["mean_pairwise_tau"],
            })
    results.sort(key=lambda d: abs(d["tau_vs_l2r"]), reverse=True)
    return results


def main():
    all_data = {}
    for ckpt in CKPTS:
        step = int(ckpt.stem.replace("ckpt_step", ""))
        label = f"{step//1000}k"
        print(f"\n[{label}] scanning...", flush=True)
        t0 = time.time()
        results = scan_one(ckpt, label)
        elapsed = time.time() - t0
        all_data[label] = results
        # Top-5 per checkpoint
        print(f"  Top-5 (|τ|):")
        for r in results[:5]:
            print(f"    L{r['layer']}H{r['head']}: τ={r['tau_vs_l2r']:+.4f}  pw={r['mean_pairwise_tau']:.4f}")
        print(f"  elapsed: {elapsed:.0f}s")

    # ── Stability summary: track top heads across checkpoints ──
    print("\n" + "=" * 80)
    print("CROSS-CKPT HEAD STABILITY (seed2, M=50, CDL C−D+L)")
    print("=" * 80)
    # Collect all unique heads that ever appear in top-3
    top_heads = set()
    for label, results in all_data.items():
        for r in results[:3]:
            top_heads.add((r["layer"], r["head"]))

    # Build table
    labels = list(all_data.keys())
    header = f"{'Head':<8}" + "".join(f"{l:>10}" for l in labels)
    print(header)
    print("-" * (8 + 10 * len(labels)))
    for lh in sorted(top_heads):
        row = f"L{lh[0]}H{lh[1]:<5}"
        for label in labels:
            for r in all_data[label]:
                if r["layer"] == lh[0] and r["head"] == lh[1]:
                    tau = r["tau_vs_l2r"]
                    marker = " ★" if any(r2["layer"] == lh[0] and r2["head"] == lh[1] for r2 in all_data[label][:1]) else ""
                    row += f" {tau:+.3f}{marker:<2}"
                    break
            else:
                row += f" {'—':>8}  "
        print(row)

    # ── Rank stability of the canonical head (L0H2) ──
    print(f"\nL0H2 rank across checkpoints:")
    for label in labels:
        for i, r in enumerate(all_data[label]):
            if r["layer"] == 0 and r["head"] == 2:
                print(f"  {label}: rank={i+1}/32  τ={r['tau_vs_l2r']:+.4f}")
                break

    # Save
    out = BASE / "head_stability_seed2.json"
    with open(out, "w") as f:
        json.dump({l: r for l, r in all_data.items()}, f, indent=1, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
