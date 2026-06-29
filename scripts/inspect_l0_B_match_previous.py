#!/usr/bin/env python3
"""Replicate previous B heatmap pipeline faithfully.

Matches: analyses/plot_b_heatmaps.py and analyses/plot_B_diagnostics.py
Pipeline: load model + chunks → extract_per_head_and_heavy_A (physical coords)
         → _batch_mean_B → B = A^T (diag=0) → grand-mean → imshow raw.
"""
from __future__ import annotations

import argparse, json, math, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))

from per_head_order_scan import (
    extract_per_head_and_heavy_A,
    _batch_mean_B,
)
from neural_readout.extract_b import _load_model_and_chunks
from training_utils import N


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--M", type=int, default=100,
                   help="number of batch-mean graphs")
    p.add_argument("--batch-size", type=int, default=32,
                   help="samples per batch-mean graph")
    p.add_argument("--fwd-batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layer", type=int, default=0,
                   help="layer to inspect (default 0)")
    args = p.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load ──
    print(f"Loading {args.ckpt} ...")
    model, chunks, clean_perm, dev, _ci = _load_model_and_chunks(
        ckpt_path=args.ckpt, M=args.M * args.batch_size,
        seed=args.seed, device=args.device, split="train",
    )
    n_heads = model.config.n_head
    n_layers = len(model.transformer.h)
    print(f"  layers={n_layers}  heads={n_heads}  M={args.M}  batch_sz={args.batch_size}")

    # ── Extract (physical coords via clean_perm.inv_perm) ──
    print("Extracting per-head A (loss_aligned / B0) ...")
    A_lh, _ = extract_per_head_and_heavy_A(
        model, chunks, clean_perm, dev, seed=args.seed,
        fwd_batch=args.fwd_batch, none_mode="loss_aligned", head=None,
    )
    # A_lh: (n_chunks, L, H, N, N) in PHYSICAL coordinates
    print(f"  A_lh shape: {A_lh.shape}")

    L = args.layer
    H_all = list(range(n_heads))

    # ── Per-head B matrices ──
    B_heads = {}  # {h: (M, N, N)}
    for h in H_all:
        A_h = A_lh[:, L, h]  # (n_chunks, N, N)
        B_h = _batch_mean_B(A_h, args.M, args.batch_size)  # (M, N, N) B=A^T, diag=0
        B_heads[h] = B_h
    B_grand = {h: B_heads[h].mean(axis=0) for h in H_all}  # (N, N)

    # ── Stats ──
    print("\nPer-head B summary (grand-mean, physical coords):")
    head_stats = {}
    for h in H_all:
        B = B_grand[h]
        off = B[~np.eye(N, dtype=bool)]
        upper = np.triu(B, k=1).mean()
        lower = np.tril(B, k=-1).mean()
        diag1 = np.diag(B, k=1).mean()
        diag_1 = np.diag(B, k=-1).mean()
        # near-diag mass d≤3
        nd3 = sum(np.diag(B, k=d).sum() + np.diag(B, k=-d).sum() for d in [1, 2, 3])
        nd3 /= B.sum()
        head_stats[f"H{h:02d}"] = {
            "mean": float(B.mean()),
            "max": float(B.max()),
            "upper_mean": float(upper),
            "lower_mean": float(lower),
            "upper_lower_ratio": float(upper / max(lower, 1e-12)),
            "diag_k1_mean": float(diag1),
            "diag_km1_mean": float(diag_1),
            "near_diag_d3_mass": float(nd3),
            "sparsity_lt_1e5": float((B < 1e-5).mean()),
        }
        print(f"  H{h:02d}: mean={B.mean():.5f}  max={B.max():.4f}  "
              f"diag+1={diag1:.4f}  upper={upper:.4f}  lower={lower:.4f}  "
              f"nd3={nd3:.3f}")

    # ── Shuffle baseline ──
    print("\nShuffle baseline (row-shuffled B):")
    for h in H_all:
        B = B_grand[h].copy()
        B_shuf = B.copy()
        for i in range(N):
            np.random.seed(i)
            np.random.shuffle(B_shuf[i])
        nd3_shuf = sum(np.diag(B_shuf, k=d).sum() + np.diag(B_shuf, k=-d).sum()
                       for d in [1, 2, 3]) / B_shuf.sum()
        nd3_real = head_stats[f"H{h:02d}"]["near_diag_d3_mass"]
        head_stats[f"H{h:02d}"]["nd3_shuffled"] = float(nd3_shuf)
        head_stats[f"H{h:02d}"]["nd3_ratio"] = float(nd3_real / max(nd3_shuf, 1e-12))
        print(f"  H{h:02d}: nd3_real={nd3_real:.4f}  nd3_shuf={nd3_shuf:.4f}  "
              f"ratio={nd3_real/max(nd3_shuf,1e-12):.1f}x")

    # ── Save stats ──
    (out / "summary.json").write_text(json.dumps({
        "ckpt": str(args.ckpt),
        "pipeline": "extract_per_head_and_heavy_A(loss_aligned) → _batch_mean_B → phys coords",
        "M": args.M, "batch_size": args.batch_size, "layer": L,
        "n_heads": n_heads, "N": N, "heads": head_stats,
    }, indent=2))

    # ── Plot: match previous style (raw B, YlOrRd, no clipping) ──
    print("\nPlotting (YlOrRd, raw B, previous style) ...")

    # Grid
    ncols = 4
    nrows = math.ceil(len(H_all) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
    axes = np.atleast_2d(axes)
    for i, h in enumerate(H_all):
        ax = axes[i // ncols, i % ncols]
        B = B_grand[h]
        im = ax.imshow(B, cmap="YlOrRd", aspect="equal", interpolation="nearest")
        ax.set_title(f"L{L} H{h}", fontsize=10, fontweight="bold")
        ax.set_xlabel("target (phys)"); ax.set_ylabel("source (phys)")
        plt.colorbar(im, ax=ax, shrink=0.8)
    for j in range(len(H_all), nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    fig.suptitle(f"L{L} B = A^T (diag=0)  —  physical coords, grand-mean over {args.M} batch-mean graphs\n"
                 f"{pathlib.Path(args.ckpt).parent.name}/{pathlib.Path(args.ckpt).name}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "l0_B_grid_previous_style.png", dpi=150, facecolor="white")
    plt.close(fig)

    # Individual heads
    for h in H_all:
        fig, ax = plt.subplots(figsize=(8, 7))
        B = B_grand[h]
        im = ax.imshow(B, cmap="YlOrRd", aspect="equal", interpolation="nearest")
        ax.set_title(f"L{L} H{h}  B (phys coords, M={args.M})", fontsize=12, fontweight="bold")
        ax.set_xlabel("target block (physical)"); ax.set_ylabel("source block (physical)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out / f"l{L}_H{h:02d}_previous_style.png", dpi=150, facecolor="white")
        plt.close(fig)

    print(f"\nDone → {out}/")
    print(f"  l0_B_grid_previous_style.png")
    for h in H_all:
        print(f"  l{L}_H{h:02d}_previous_style.png")
    print(f"  summary.json")


if __name__ == "__main__":
    main()
