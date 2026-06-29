#!/usr/bin/env python3
"""Visualize the actual generation orders sigma produced by each readout policy.

Faithfully reproduces (in pure numpy) the deterministic coverage sampler from
block_lo_arm_order_network/train_imagelarge_round2.py:sample_coverage_batched_torch:
    score_t(v) = gamma_B * minmax_U(B[last,v]) - gamma_d * minmax_U(manh(last,v))
    next = argmax(score over unvisited)
Policies:
    raster          : 0..63 row-major (geometry reference)
    distance_only   : gamma_B=0.01, gamma_d=1   (real B, B-weight ~0)
    Bcov_balanced   : gamma_B=1,    gamma_d=1   (real B)
    shuffled_Bcov   : gamma_B=1,    gamma_d=1   on row-shuffled B (seed=42+424242=424284)
B = A_block.T with diagonal zeroed (build_directed_graph). A is read from the small
16KB A_block_8x8.npy (NOT a checkpoint). Start block fixed to 0 (top-left) for display.
Does NOT train, does NOT touch ckpt.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/home/admin/lyuyuhuan/order_lyu")
A_PATH = ROOT / "probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy"
OUTDIR = ROOT / "probe_results_image_large/structure_adaptive/image_story_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

N, G = 64, 8
START = 0  # top-left block, fixed for a clean display trajectory

# coords (row-major): block i -> (row=i//8, col=i%8)
idx = np.arange(N)
ROW, COL = idx // G, idx % G
D = (np.abs(ROW[:, None] - ROW[None, :]) + np.abs(COL[:, None] - COL[None, :])).astype(float)


def build_B(A):
    B = A.T.astype(np.float64).copy()
    np.fill_diagonal(B, 0.0)
    return B


def make_shuffled(B, seed):
    rng = np.random.default_rng(seed)
    out = np.empty_like(B)
    for i in range(B.shape[0]):
        out[i] = B[i, rng.permutation(B.shape[1])]
    np.fill_diagonal(out, 0.0)
    return out


def coverage_order(B, start, gamma_B, gamma_d, eps=1e-12):
    order = [start]
    selected = np.zeros(N, bool); selected[start] = True
    last = start
    for _ in range(1, N):
        bs, ds = B[last], D[last]
        bmin = np.where(selected, np.inf, bs).min()
        bmax = np.where(selected, -np.inf, bs).max()
        dmin = np.where(selected, np.inf, ds).min()
        dmax = np.where(selected, -np.inf, ds).max()
        bn = (bs - bmin) / (bmax - bmin + eps)
        dn = (ds - dmin) / (dmax - dmin + eps)
        scores = gamma_B * bn - gamma_d * dn
        scores[selected] = -np.inf
        nxt = int(np.argmax(scores))
        order.append(nxt); selected[nxt] = True; last = nxt
    return np.array(order)


def mean_manh(order):
    return float(np.mean([D[order[t], order[t + 1]] for t in range(N - 1)]))


def plot_trajectory(ax, order, title):
    # color each cell by its visit step
    visit = np.empty(N, int)
    visit[order] = np.arange(N)
    grid = visit.reshape(G, G)
    im = ax.imshow(grid, cmap="viridis", origin="upper")
    # trajectory polyline (col=x, row=y)
    xs = [order[t] % G for t in range(N)]
    ys = [order[t] // G for t in range(N)]
    ax.plot(xs, ys, "-", color="white", lw=0.8, alpha=0.7)
    ax.plot(xs[0], ys[0], "o", color="red", ms=8)          # start
    ax.text(xs[0], ys[0], "S", color="white", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_title(f"{title}\nmean_manh={mean_manh(order):.2f}", fontsize=10)
    ax.set_xticks(range(G)); ax.set_yticks(range(G))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_xlabel("col"); ax.set_ylabel("row")
    return im


def main():
    A = np.load(A_PATH).astype(np.float32)
    assert A.shape == (64, 64), A.shape
    B = build_B(A)
    B_shuf = make_shuffled(B, seed=42 + 424242)  # 424284, matches committed config

    orders = {
        "raster (0..63)": np.arange(N),
        "distance_only (gB=0.01)": coverage_order(B, START, 0.01, 1.0),
        "Bcov_balanced (gB=1)": coverage_order(B, START, 1.0, 1.0),
        "shuffled_Bcov (gB=1, B shuffled)": coverage_order(B_shuf, START, 1.0, 1.0),
    }

    # ---- print first steps ----
    print(f"A_block: {A_PATH}")
    print(f"start block = {START} (row {START//G}, col {START%G}); B = A.T diag-zeroed\n")
    for name, o in orders.items():
        print(f"{name:34s} mean_manh={mean_manh(o):.3f}")
        print(f"    first 16 blocks: {o[:16].tolist()}")
        print(f"    as (row,col):    {[(int(i//G), int(i%G)) for i in o[:16]]}")
    print()

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5))
    last_im = None
    for ax, (name, o) in zip(axes.ravel(), orders.items()):
        last_im = plot_trajectory(ax, o, name)
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("visit step t  (0 = first revealed, 63 = last)")
    fig.suptitle("Generation orders sigma on the 8x8 block grid (start=block 0, seed=42)\n"
                 "color = when a block is revealed; white line = traversal path",
                 fontsize=12, fontweight="bold")
    path = OUTDIR / "fig7_readout_order_trajectories.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
