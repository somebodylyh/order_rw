#!/usr/bin/env python3
"""Visualize attention B matrices for key heads across checkpoints (seed2)."""
import sys, pathlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = pathlib.Path(__file__).resolve().parent.parent / "block_lo_arm_order_network"
sys.path.insert(0, str(_ROOT))

from training_utils import N
from neural_readout.extract_b import _load_model_and_chunks
from per_head_order_scan import extract_per_head_and_heavy_A, _batch_mean_B

BASE = pathlib.Path("/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/probe_results")
CKPTS = [
    (BASE / "random_baseline_continuous_jun08_seed2/ckpt_step10000.pt", "10k"),
    (BASE / "random_baseline_continuous_jun08_seed2/ckpt_step30000.pt", "30k"),
    (BASE / "random_baseline_continuous_jun08_seed2/ckpt_step50000.pt", "50k"),
]
HEADS = [(0, 2, "L0H2 (canonical)"), (0, 7, "L0H7"), (0, 6, "L0H6"), (0, 0, "L0H0 (anti-L2R)")]
M, BATCH_SIZE, DEVICE = 100, 32, "cuda:1"

fig, axes = plt.subplots(len(HEADS), len(CKPTS), figsize=(4*len(CKPTS), 4*len(HEADS)))

for col, (ckpt_path, label) in enumerate(CKPTS):
    print(f"[{label}] extracting...", flush=True)
    total = M * BATCH_SIZE
    model, chunks, clean_perm, dev, _ci = _load_model_and_chunks(
        ckpt_path, total, seed=42, device=DEVICE, split="train")
    A_lh, _ = extract_per_head_and_heavy_A(
        model, chunks, clean_perm, dev, seed=42, fwd_batch=32,
        none_mode="b0", head=None)

    for row, (l, h, hname) in enumerate(HEADS):
        B = _batch_mean_B(A_lh[:, l, h], M, BATCH_SIZE)  # (M, N, N)
        B_grand = B.mean(axis=0)  # grand mean over M graphs
        # B is A^T in _batch_mean_B, and already zero-diagonal
        ax = axes[row, col]
        im = ax.imshow(B_grand, cmap="RdBu_r", vmin=-0.02, vmax=0.04,
                        aspect="equal", interpolation="nearest")
        ax.set_title(f"{hname} @ {label}", fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"{hname}", fontsize=10)
        if row == len(HEADS) - 1:
            ax.set_xlabel("physical block index (target)", fontsize=9)
        if col == 0:
            ax.set_ylabel("physical block index (source)", fontsize=9)

plt.colorbar(im, ax=axes, shrink=0.6, label="attention weight (B = A^T)")
plt.suptitle("seed2 — Grand-Mean B Matrix (100 batch-mean graphs, B0 convention)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
out = pathlib.Path("/home/admin/lyuyuhuan/order_lyu/analyses/figures/seed2_attention_heatmaps.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.close()
