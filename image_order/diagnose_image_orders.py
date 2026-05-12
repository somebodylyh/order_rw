"""Image-specific Graph-RW order diagnostics.

Research question: does AOGPT's internal attention on image-patch data encode
dataset-level shared structure (spatial locality, region grouping, center vs edge
tendency)?  Graph-RW reveal orders should expose this.

Raster scan is reported as a diagnostic baseline — NOT as an oracle / label.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path setup so we can import from image_order/ (may be run from repo root)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from graph_rw_image import (
    make_B_from_attention,
    sample_orders_for_eval,
    random_orders,
    raster_orders,
    IMAGE_RW_PARAMS_DEFAULT,
)


# ===========================================================================
# Core analysis functions
# ===========================================================================

def locality_stats(orders: np.ndarray, grid: int = 8) -> dict:
    """Consecutive-step Manhattan distance statistics.

    Returns dict: {mean, median, per_step (N-1,)} over (K, N) orders.
    """
    orders = np.asarray(orders, dtype=np.int64)
    rows = orders // grid          # (K, N)
    cols = orders % grid           # (K, N)
    d = np.abs(np.diff(rows, axis=1)) + np.abs(np.diff(cols, axis=1))  # (K, N-1)
    return {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "per_step": d.mean(axis=0),  # (N-1,)
    }


def center_distance_curve(orders: np.ndarray, grid: int = 8) -> np.ndarray:
    """Mean Euclidean distance from image center at each reveal step. Returns (N,)."""
    orders = np.asarray(orders, dtype=np.int64)
    center = (grid - 1) / 2.0
    rows = (orders // grid).astype(np.float64)   # (K, N)
    cols = (orders % grid).astype(np.float64)    # (K, N)
    dist = np.sqrt((rows - center) ** 2 + (cols - center) ** 2)  # (K, N)
    return dist.mean(axis=0)  # (N,)


# ===========================================================================
# Plotting functions
# ===========================================================================

def plot_order_path(order: np.ndarray, save_path: str, title: str = None, grid: int = 8):
    """Draw 8×8 grid colored by reveal step with consecutive-patch arrows. Save PNG."""
    N = len(order)
    fig, ax = plt.subplots(figsize=(5, 5))

    # Color each cell by its reveal step
    step_map = np.empty(N, dtype=np.float64)
    for step, patch_idx in enumerate(order):
        step_map[patch_idx] = step
    grid_img = step_map.reshape(grid, grid)

    im = ax.imshow(grid_img, cmap="viridis", vmin=0, vmax=N - 1, origin="upper")
    plt.colorbar(im, ax=ax, label="Reveal step")

    # Draw arrows between consecutive patches
    for t in range(N - 1):
        r0, c0 = order[t] // grid, order[t] % grid
        r1, c1 = order[t + 1] // grid, order[t + 1] % grid
        ax.annotate(
            "",
            xy=(c1, r1),
            xytext=(c0, r0),
            arrowprops=dict(arrowstyle="->", color="white", lw=0.6),
        )

    # Grid lines
    for i in range(grid + 1):
        ax.axhline(i - 0.5, color="gray", lw=0.4)
        ax.axvline(i - 0.5, color="gray", lw=0.4)

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_avg_step_heatmap(orders: np.ndarray, save_path: str, grid: int = 8):
    """Average reveal step per patch across K orders as 8×8 heatmap. Save PNG."""
    orders = np.asarray(orders, dtype=np.int64)
    K, N = orders.shape
    # Build inverse: avg_step[patch] = mean step at which patch is revealed
    step_idx = np.arange(N, dtype=np.float64)
    avg_step = np.zeros(N, dtype=np.float64)
    counts = np.zeros(N, dtype=np.int64)
    for k in range(K):
        avg_step[orders[k]] += step_idx
        counts[orders[k]] += 1
    avg_step = avg_step / np.maximum(counts, 1)

    grid_img = avg_step.reshape(grid, grid)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(grid_img, cmap="viridis", origin="upper")
    plt.colorbar(im, ax=ax, label="Avg reveal step")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Average reveal step per patch")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_transition_heatmap(orders: np.ndarray, save_path: str):
    """64×64 transition count (T[u,v]+=1 per consecutive pair), normalized. Save PNG."""
    orders = np.asarray(orders, dtype=np.int64)
    N = orders.shape[1]
    T = np.zeros((N, N), dtype=np.float64)
    src = orders[:, :-1].ravel()   # (K*(N-1),)
    dst = orders[:, 1:].ravel()    # (K*(N-1),)
    np.add.at(T, (src, dst), 1)
    T_max = T.max()
    if T_max > 0:
        T /= T_max

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(T, cmap="viridis", aspect="auto", origin="upper")
    plt.colorbar(im, ax=ax, label="Normalized count")
    ax.set_xlabel("Destination patch")
    ax.set_ylabel("Source patch")
    ax.set_title("Transition heatmap (normalized)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


# ===========================================================================
# Kendall tau vs raster
# ===========================================================================

def rank_correlation_vs_raster(orders: np.ndarray) -> float:
    """Mean Kendall tau vs raster (np.arange(N)). Uses scipy when available, else O(N^2)."""
    orders = np.asarray(orders, dtype=np.int64)
    K, N = orders.shape
    raster = np.arange(N)

    try:
        from scipy.stats import kendalltau as _kendalltau

        taus = np.array([_kendalltau(orders[k], raster).statistic for k in range(K)])
        return float(taus.mean())
    except (ImportError, AttributeError):
        # Fallback: inline O(N^2) concordant/discordant count
        taus = []
        for k in range(K):
            o = orders[k]
            # Rank of each element in raster is just its value (raster = arange)
            # tau = (C - D) / (N*(N-1)/2)
            C = 0
            D = 0
            for i in range(N):
                for j in range(i + 1, N):
                    diff_o = o[i] - o[j]
                    diff_r = raster[i] - raster[j]
                    if diff_o * diff_r > 0:
                        C += 1
                    elif diff_o * diff_r < 0:
                        D += 1
            taus.append((C - D) / (N * (N - 1) / 2))
        return float(np.mean(taus))


# ===========================================================================
# CLI driver
# ===========================================================================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_diagnostics(B: np.ndarray, output_dir: str, K: int, seed_base: int, example_idx: int):
    """Full diagnostic pipeline for one B matrix."""
    assert B.shape == (64, 64), f"Expected B shape (64, 64), got {B.shape}"

    configs = [
        ("random",          random_orders(K, 64, seed_base)),
        ("raster",          raster_orders(K, 64)),
        ("graph_rw_top4",   sample_orders_for_eval(B, {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 4}, K=K, seed_base=seed_base)["orders"]),
        ("graph_rw_top8",   sample_orders_for_eval(B, {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 8}, K=K, seed_base=seed_base)["orders"]),
        ("graph_rw_eps015", sample_orders_for_eval(B, {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 0, "epsilon_uniform": 0.15}, K=K, seed_base=seed_base)["orders"]),
    ]

    _ensure_dir(output_dir)

    # TSV header
    tsv_rows = ["name\tmean_manhattan\tmedian_manhattan\tmean_center_distance\ttau_vs_raster\tnotes"]

    # Combined center-distance curve figure
    fig_cd, ax_cd = plt.subplots(figsize=(8, 4))
    step_x = np.arange(64)

    summary_lines = []

    for name, orders in configs:
        config_dir = os.path.join(output_dir, name)
        _ensure_dir(config_dir)

        # --- locality stats ---
        loc = locality_stats(orders)
        mean_m = loc["mean"]
        med_m = loc["median"]

        # --- center distance curve ---
        cd_curve = center_distance_curve(orders)
        mean_cd = float(cd_curve.mean())
        np.save(os.path.join(config_dir, "center_distance_curve.npy"), cd_curve)

        # --- tau vs raster ---
        tau = rank_correlation_vs_raster(orders)

        # --- plots ---
        eg_idx = min(example_idx, len(orders) - 1)
        plot_order_path(
            orders[eg_idx],
            save_path=os.path.join(config_dir, "order_path_example.png"),
            title=f"{name} — example {eg_idx}",
        )
        plot_avg_step_heatmap(orders, save_path=os.path.join(config_dir, "avg_step_heatmap.png"))
        plot_transition_heatmap(orders, save_path=os.path.join(config_dir, "transition_heatmap.png"))

        # TSV row
        notes = "raster_baseline" if name == "raster" else ""
        tsv_rows.append(f"{name}\t{mean_m:.4f}\t{med_m:.4f}\t{mean_cd:.4f}\t{tau:.4f}\t{notes}")

        # Combined curve
        ax_cd.plot(step_x, cd_curve, label=name)

        summary_lines.append((name, mean_m, tau))

    # Write TSV
    with open(os.path.join(output_dir, "summary.tsv"), "w") as f:
        f.write("\n".join(tsv_rows) + "\n")

    # Save combined center-distance comparison
    ax_cd.set_xlabel("Reveal step")
    ax_cd.set_ylabel("Mean Euclidean distance to center")
    ax_cd.set_title("Center distance curve by config")
    ax_cd.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "center_distance_comparison.png"), dpi=120)
    plt.close(fig_cd)

    # Print summary
    print(f"OK diagnose_image_orders: K={K}, B={output_dir}")
    for name, mean_m, tau in summary_lines:
        print(f"  {name:<20} mean_manhattan={mean_m:.4f}  tau_vs_raster={tau:.4f}")
    print(f"  outputs: {output_dir}")


def _run_self_test():
    """Self-test with synthetic random B (no real data needed)."""
    print("WARNING: B not provided or not found — running self-test with synthetic B")
    rng = np.random.RandomState(0)
    B = (rng.rand(64, 64).astype(np.float32)) ** 4  # concentrated random
    output_dir = "probe_results_image/diagnostics/self_test"
    run_diagnostics(B, output_dir=output_dir, K=200, seed_base=42, example_idx=0)


def main():
    parser = argparse.ArgumentParser(description="Image-specific Graph-RW order diagnostics")
    parser.add_argument("--B", type=str, default=None, help="Path to B_global.npy")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--K", type=int, default=200, help="Number of orders to sample")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--example-idx", type=int, default=0, help="Which order to render as example path")
    args = parser.parse_args()

    # Self-test path
    if args.B is None or not os.path.isfile(args.B):
        _run_self_test()
        return

    B = np.load(args.B)
    assert B.shape == (64, 64), f"Expected (64, 64), got {B.shape}"

    output_dir = args.output_dir or "probe_results_image/diagnostics/default"
    run_diagnostics(
        B,
        output_dir=output_dir,
        K=args.K,
        seed_base=args.seed_base,
        example_idx=args.example_idx,
    )


if __name__ == "__main__":
    main()
