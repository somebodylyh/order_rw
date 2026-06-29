#!/usr/bin/env python3
"""Diagnostic attention visualization: B matrix + row profiles + CDL trace."""
import sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy

_ROOT = pathlib.Path(__file__).resolve().parent.parent / "block_lo_arm_order_network"
sys.path.insert(0, str(_ROOT))

from training_utils import N
from neural_readout.extract_b import _load_model_and_chunks
from neural_readout.teacher_labels import generate_teacher_label
from per_head_order_scan import extract_per_head_and_heavy_A, _batch_mean_B

BASE = pathlib.Path("/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/probe_results")
CKPT = BASE / "random_baseline_continuous_jun08_seed2/ckpt_step10000.pt"
HEADS = [(0, 2, "L0H2"), (0, 7, "L0H7"), (0, 0, "L0H0")]
M, BATCH_SIZE = 100, 32

print("Loading...", flush=True)
model, chunks, clean_perm, dev, _ci = _load_model_and_chunks(
    CKPT, M * BATCH_SIZE, seed=42, device="cuda:1", split="train")
A_lh, _ = extract_per_head_and_heavy_A(
    model, chunks, clean_perm, dev, seed=42, fwd_batch=32, none_mode="b0", head=None)

fig, axes = plt.subplots(len(HEADS), 4, figsize=(20, 5 * len(HEADS)))

for row, (l, h, hname) in enumerate(HEADS):
    B_all = _batch_mean_B(A_lh[:, l, h], M, BATCH_SIZE)  # (M, N, N)
    B_grand = B_all.mean(axis=0)

    # Col 0: B heatmap
    ax = axes[row, 0]
    im = ax.imshow(B_grand, cmap="RdBu_r", aspect="equal",
                    interpolation="nearest")
    ax.set_title(f"{hname} — grand-mean B", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Col 1: row-concentration (negentropy per row)
    ax = axes[row, 1]
    row_ents = []
    for i in range(N):
        p = np.abs(B_grand[i]) + 1e-10
        p = p / p.sum()
        row_ents.append(entropy(p))
    colors = plt.cm.viridis(np.array(row_ents) / max(row_ents))
    ax.bar(range(N), row_ents, color=colors, width=0.8)
    ax.set_title(f"{hname} — row entropy (↓ = concentrated)", fontsize=10)
    ax.set_xlabel("physical block")
    ax.set_ylabel("entropy (nats)")
    ax.axhline(y=np.log(N), color="gray", linestyle="--", alpha=0.5, label=f"uniform ({np.log(N):.1f})")
    ax.legend(fontsize=7)

    # Col 2: mean row pattern (mean over rows → which columns get attention)
    ax = axes[row, 2]
    col_mean = B_grand.mean(axis=0)
    ax.bar(range(N), col_mean, color="steelblue", width=0.8)
    ax.set_title(f"{hname} — column-mean attention", fontsize=10)
    ax.set_xlabel("physical block (target)")
    ax.set_ylabel("mean B weight")

    # Col 3: CDL teacher variance — how many unique σ_T across M graphs?
    ax = axes[row, 3]
    sigmas = []
    for m in range(min(M, 20)):  # first 20 graphs
        s, _, _ = generate_teacher_label(B_all[m], alpha_dep=0.5)
        sigmas.append(s)
    sigmas = np.array(sigmas)
    l2r = np.arange(N)
    # Show first 5 CDL orders vs L2R
    for m in range(min(5, len(sigmas))):
        match = np.mean(sigmas[m] == l2r)
        ax.plot(range(N), sigmas[m], 'o-', markersize=1.5, linewidth=0.5,
                alpha=0.7, label=f"σ_{m} (L2R-match={match:.0%})" if m == 0 else "")
    ax.plot(range(N), l2r, 'k--', linewidth=2, label="L2R")
    ax.set_title(f"{hname} — CDL orders (first 5 graphs)", fontsize=10)
    ax.set_xlabel("reveal step")
    ax.set_ylabel("physical block index")
    if row == 1:
        ax.legend(fontsize=6, loc="lower right")

plt.suptitle("seed2 @10k — B Matrix Diagnostics (M=100 batch-mean graphs, B0 convention)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out = pathlib.Path("/home/admin/lyuyuhuan/order_lyu/analyses/figures/seed2_B_diagnostics.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.close()

# Print textual summary
print("\n=== B matrix structure summary (seed2 @10k) ===")
for l, h, hname in HEADS:
    B_all = _batch_mean_B(A_lh[:, l, h], M, BATCH_SIZE)
    B_grand = B_all.mean(axis=0)
    # Off-diagonal mass (|i-j|=1)
    diag1 = np.diag(B_grand, k=1).mean()
    diag_1 = np.diag(B_grand, k=-1).mean()
    # Upper vs lower triangle
    upper = np.triu(B_grand, k=1).mean()
    lower = np.tril(B_grand, k=-1).mean()
    # Row concentration
    row_conc = np.mean([-entropy(np.abs(B_grand[i]) + 1e-10) + np.log(N) for i in range(N)])
    # Check: how many unique σ_T across all M?
    sigmas = set()
    for m in range(M):
        s, _, _ = generate_teacher_label(B_all[m], alpha_dep=0.5)
        sigmas.add(tuple(s.tolist()))
    print(f"\n{hname}:")
    print(f"  superdiag (i→i+1): {diag1:.4f}   subdiag (i+1→i): {diag_1:.4f}")
    print(f"  upper-tri mean: {upper:.4f}   lower-tri mean: {lower:.4f}")
    print(f"  row-concentration: {row_conc:.3f} (max={np.log(N):.1f} = uniform)")
    print(f"  unique σ_T / M: {len(sigmas)}/{M}")
    # Is B upper-triangular? (L2R = future blocks get more attention)
    print(f"  upper/lower ratio: {upper/lower:.2f}" if lower > 0 else "  upper-dominant")
