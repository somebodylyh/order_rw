"""Image-specific Graph-RW order diagnostics — structural-discovery oriented.

Research question: does AOGPT's internal attention on image-patch data encode
dataset-level shared structure (spatial locality, region grouping, center vs
edge tendency, appearance similarity)?  Graph-RW reveal orders should expose
this.

Raster scan is reported ONLY as a weak sanity-check reference, not an oracle:
the goal is NOT high `tau_vs_raster`. Image structure is genuinely 2D and may
be center-out, region-block, or appearance-driven rather than raster-aligned.

Primary success criterion: Graph-RW top_k=4 should show stronger spatial
locality / region grouping / center-boundary trend / appearance grouping
than random, AND high-noise variants (epsilon=0.15, larger top_k) should
weaken those structures back toward random.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
# A. Spatial locality
# ===========================================================================

def locality_stats(orders: np.ndarray, grid: int = 8) -> dict:
    """Consecutive Manhattan distance: mean / median / per-step / P(d<=1) / P(d<=2)."""
    orders = np.asarray(orders, dtype=np.int64)
    rows = orders // grid
    cols = orders % grid
    d = np.abs(np.diff(rows, axis=1)) + np.abs(np.diff(cols, axis=1))  # (K, N-1)
    flat = d.ravel()
    return {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "per_step": d.mean(axis=0),
        "p_le1": float((flat <= 1).mean()),
        "p_le2": float((flat <= 2).mean()),
        "all_distances": flat,
    }


# ===========================================================================
# B. Region grouping
# ===========================================================================

def _region_indices(orders: np.ndarray, grid: int, region_size: int) -> np.ndarray:
    """Map patch indices in `orders` to region ids on a grid//region_size super-grid."""
    n_per_side = grid // region_size
    rows = (orders // grid) // region_size
    cols = (orders % grid) // region_size
    return rows * n_per_side + cols


def same_region_rate(orders: np.ndarray, region_size: int, grid: int = 8) -> float:
    """Fraction of consecutive transitions that stay in the same region.

    region_size=4 → 2x2 quadrants (each 4x4 patches).
    region_size=2 → 4x4 super-grid (each 2x2 patches).
    """
    region = _region_indices(np.asarray(orders, dtype=np.int64), grid, region_size)
    return float((region[:, 1:] == region[:, :-1]).mean())


def mean_region_run_length(orders: np.ndarray, region_size: int, grid: int = 8) -> float:
    """Average length of consecutive-same-region runs across K orders."""
    region = _region_indices(np.asarray(orders, dtype=np.int64), grid, region_size)
    K, N = region.shape
    boundaries = region[:, 1:] != region[:, :-1]  # (K, N-1)
    num_runs = boundaries.sum(axis=1) + 1          # (K,)
    return float((N / num_runs).mean())


def per_step_same_region(orders: np.ndarray, region_size: int, grid: int = 8) -> np.ndarray:
    """At each step t, fraction of orders where transition (t-1)->t stays in same region. Returns (N-1,)."""
    region = _region_indices(np.asarray(orders, dtype=np.int64), grid, region_size)
    same = (region[:, 1:] == region[:, :-1]).astype(np.float64)
    return same.mean(axis=0)


# ===========================================================================
# C. Center / boundary tendency
# ===========================================================================

def center_distance_curve(orders: np.ndarray, grid: int = 8) -> np.ndarray:
    """Mean Euclidean distance from image center at each reveal step. (N,)."""
    orders = np.asarray(orders, dtype=np.int64)
    center = (grid - 1) / 2.0
    rows = (orders // grid).astype(np.float64)
    cols = (orders % grid).astype(np.float64)
    dist = np.sqrt((rows - center) ** 2 + (cols - center) ** 2)
    return dist.mean(axis=0)


def center_step_correlation(orders: np.ndarray, grid: int = 8) -> float:
    """Mean Pearson corr between reveal step and patch-to-center distance, across K orders.

    Positive → center-to-boundary trend (later steps farther from center).
    Negative → boundary-to-center trend.
    Near zero → no global radial trend.
    """
    orders = np.asarray(orders, dtype=np.int64)
    K, N = orders.shape
    center = (grid - 1) / 2.0
    rows = (orders // grid).astype(np.float64)
    cols = (orders % grid).astype(np.float64)
    dist = np.sqrt((rows - center) ** 2 + (cols - center) ** 2)
    step = np.arange(N, dtype=np.float64)
    step_z = step - step.mean()
    step_norm = np.linalg.norm(step_z)
    if step_norm < 1e-12:
        return 0.0

    corrs = []
    for k in range(K):
        d = dist[k] - dist[k].mean()
        denom = np.linalg.norm(d) * step_norm
        if denom < 1e-12:
            continue
        corrs.append(float((d * step_z).sum() / denom))
    return float(np.mean(corrs)) if corrs else 0.0


# ===========================================================================
# D. Appearance grouping
# ===========================================================================

def compute_patch_features(images: np.ndarray, patch_size: int = 4) -> np.ndarray:
    """Per-patch appearance features.

    images: (N, 3, H, W) float (any normalization).
    Returns: (N, n_patches, 7) — [R_mean, G_mean, B_mean, R_std, G_std, B_std, edge_mag].
    edge_mag is the mean Sobel-style gradient magnitude over the patch on grayscale.
    """
    N, C, H, W = images.shape
    grid = H // patch_size
    n_patches = grid * grid
    p = images.reshape(N, C, grid, patch_size, grid, patch_size)
    p = p.transpose(0, 2, 4, 1, 3, 5).reshape(N, n_patches, C, patch_size, patch_size)

    rgb_mean = p.mean(axis=(-1, -2))                 # (N, n_patches, 3)
    rgb_std = p.std(axis=(-1, -2))                   # (N, n_patches, 3)

    gray = p.mean(axis=2)                            # (N, n_patches, ps, ps)
    gx = np.diff(gray, axis=-1, prepend=gray[..., :, :1])
    gy = np.diff(gray, axis=-2, prepend=gray[..., :1, :])
    edge_mag = np.sqrt(gx ** 2 + gy ** 2).mean(axis=(-1, -2))[..., None]  # (N, n_patches, 1)

    return np.concatenate([rgb_mean, rgb_std, edge_mag], axis=-1).astype(np.float32)


def consecutive_feature_distance(
    orders: np.ndarray, features: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """For each transition in each order, mean L2 feature distance averaged over images.

    Returns flat array of length K*(N-1).
    If `normalize` is True, features are z-scored per dim across all (image, patch) entries
    so that channels with larger raw scale do not dominate.
    """
    orders = np.asarray(orders, dtype=np.int64)
    K, N = orders.shape
    feats = features
    if normalize:
        flat = feats.reshape(-1, feats.shape[-1])
        mu = flat.mean(axis=0)
        sd = flat.std(axis=0) + 1e-8
        feats = (feats - mu) / sd

    out = np.empty((K, N - 1), dtype=np.float64)
    for k in range(K):
        o = orders[k]
        diffs = feats[:, o[1:], :] - feats[:, o[:-1], :]   # (n_img, N-1, F)
        out[k] = np.linalg.norm(diffs, axis=-1).mean(axis=0)
    return out.ravel()


# ===========================================================================
# Secondary: Kendall tau vs raster (sanity check only)
# ===========================================================================

def rank_correlation_vs_raster(orders: np.ndarray) -> float:
    """Mean Kendall tau against raster scan. Reported as sanity check, NOT a target."""
    orders = np.asarray(orders, dtype=np.int64)
    K, N = orders.shape
    raster = np.arange(N)
    try:
        from scipy.stats import kendalltau as _kendalltau
        taus = np.array([_kendalltau(orders[k], raster).statistic for k in range(K)])
        return float(taus.mean())
    except (ImportError, AttributeError):
        taus = []
        for k in range(K):
            o = orders[k]
            C = D = 0
            for i in range(N):
                for j in range(i + 1, N):
                    s = (o[i] - o[j]) * (raster[i] - raster[j])
                    if s > 0:
                        C += 1
                    elif s < 0:
                        D += 1
            taus.append((C - D) / (N * (N - 1) / 2))
        return float(np.mean(taus))


# ===========================================================================
# Plots
# ===========================================================================

def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_order_path(order: np.ndarray, save_path: str, title: str = None, grid: int = 8):
    """8×8 grid colored by reveal step + arrows between consecutive patches."""
    N = len(order)
    fig, ax = plt.subplots(figsize=(5, 5))
    step_map = np.empty(N, dtype=np.float64)
    for step, patch_idx in enumerate(order):
        step_map[patch_idx] = step
    im = ax.imshow(step_map.reshape(grid, grid), cmap="viridis", vmin=0, vmax=N - 1, origin="upper")
    plt.colorbar(im, ax=ax, label="Reveal step")
    for t in range(N - 1):
        r0, c0 = order[t] // grid, order[t] % grid
        r1, c1 = order[t + 1] // grid, order[t + 1] % grid
        ax.annotate("", xy=(c1, r1), xytext=(c0, r0),
                    arrowprops=dict(arrowstyle="->", color="white", lw=0.6))
    for i in range(grid + 1):
        ax.axhline(i - 0.5, color="gray", lw=0.4)
        ax.axvline(i - 0.5, color="gray", lw=0.4)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    plt.tight_layout()
    _ensure_parent(save_path)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_avg_step_heatmap(orders: np.ndarray, save_path: str, grid: int = 8):
    """8×8 heatmap of average reveal step per patch across K orders."""
    orders = np.asarray(orders, dtype=np.int64)
    K, N = orders.shape
    step_idx = np.arange(N, dtype=np.float64)
    avg_step = np.zeros(N, dtype=np.float64)
    counts = np.zeros(N, dtype=np.int64)
    for k in range(K):
        avg_step[orders[k]] += step_idx
        counts[orders[k]] += 1
    avg_step = avg_step / np.maximum(counts, 1)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(avg_step.reshape(grid, grid), cmap="viridis", origin="upper")
    plt.colorbar(im, ax=ax, label="Avg reveal step")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Average reveal step per patch")
    plt.tight_layout()
    _ensure_parent(save_path)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_transition_heatmap(orders: np.ndarray, save_path: str):
    """64×64 transition count heatmap (consecutive pairs only)."""
    orders = np.asarray(orders, dtype=np.int64)
    K, N = orders.shape
    T = np.zeros((N, N), dtype=np.float64)
    np.add.at(T, (orders[:, :-1].ravel(), orders[:, 1:].ravel()), 1)
    if T.max() > 0:
        T /= T.max()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(T, cmap="viridis", aspect="auto", origin="upper")
    plt.colorbar(im, ax=ax, label="Normalized count")
    ax.set_xlabel("Destination patch"); ax.set_ylabel("Source patch")
    ax.set_title("Transition heatmap (normalized)")
    plt.tight_layout()
    _ensure_parent(save_path)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_transition_distance_histogram(
    orders: np.ndarray, save_path: str, grid: int = 8,
    ref_orders: np.ndarray = None, ref_label: str = "random",
    main_label: str = "this", title: str = None,
):
    """Histogram of consecutive Manhattan distances, optionally overlaying a reference."""
    orders = np.asarray(orders, dtype=np.int64)
    d = (np.abs(np.diff(orders // grid, axis=1)) + np.abs(np.diff(orders % grid, axis=1))).ravel()
    max_d = int(d.max())
    if ref_orders is not None:
        ref = np.asarray(ref_orders, dtype=np.int64)
        ref_d = (np.abs(np.diff(ref // grid, axis=1)) + np.abs(np.diff(ref % grid, axis=1))).ravel()
        max_d = max(max_d, int(ref_d.max()))
    bins = np.arange(max_d + 2) - 0.5
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d, bins=bins, alpha=0.55, density=True, label=main_label, color="C0")
    if ref_orders is not None:
        ax.hist(ref_d, bins=bins, alpha=0.45, density=True, label=ref_label, color="C1")
    ax.set_xlabel("Consecutive Manhattan distance")
    ax.set_ylabel("Density")
    ax.set_title(title or "Transition distance histogram")
    ax.legend()
    plt.tight_layout()
    _ensure_parent(save_path)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_region_stay_curve(
    orders: np.ndarray, save_path: str, region_size: int, grid: int = 8,
    ref_orders: np.ndarray = None, main_label: str = "this", ref_label: str = "random",
):
    """Per-step same-region rate, optionally overlaying a reference."""
    curve = per_step_same_region(orders, region_size, grid)
    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(len(curve))
    ax.plot(x, curve, label=main_label)
    if ref_orders is not None:
        ref_curve = per_step_same_region(ref_orders, region_size, grid)
        ax.plot(x, ref_curve, label=ref_label, linestyle="--")
    ax.set_xlabel("Reveal step")
    ax.set_ylabel(f"P(same region @ step t), region={region_size}×{region_size}")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    _ensure_parent(save_path)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_feature_distance_histogram(
    orders: np.ndarray, features: np.ndarray, save_path: str,
    ref_orders: np.ndarray = None, main_label: str = "this", ref_label: str = "random",
    title: str = None,
):
    """Histogram of consecutive z-scored feature distances; optional reference overlay."""
    d = consecutive_feature_distance(orders, features)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d, bins=40, alpha=0.55, density=True, label=main_label, color="C0")
    if ref_orders is not None:
        rd = consecutive_feature_distance(ref_orders, features)
        ax.hist(rd, bins=40, alpha=0.45, density=True, label=ref_label, color="C1")
    ax.set_xlabel("Consecutive feature distance (z-scored)")
    ax.set_ylabel("Density")
    ax.set_title(title or "Feature distance histogram")
    ax.legend()
    plt.tight_layout()
    _ensure_parent(save_path)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


# ===========================================================================
# CLI driver
# ===========================================================================

def _try_load_cifar_features(num_images: int) -> np.ndarray:
    """Load CIFAR-10 test images and compute per-patch features. Returns None on failure."""
    try:
        from data_image_patches import CIFAR10Patches
    except Exception as e:
        print(f"  appearance: failed to import CIFAR10Patches ({e}); skipping.")
        return None
    try:
        ds = CIFAR10Patches("test")
        imgs = ds.images[:num_images].cpu().numpy()
        feats = compute_patch_features(imgs, patch_size=4)
        return feats
    except Exception as e:
        print(f"  appearance: failed to compute features ({e}); skipping.")
        return None


def run_diagnostics(
    B: np.ndarray, output_dir: str, K: int, seed_base: int, example_idx: int,
    num_images_for_appearance: int = 500, use_appearance: bool = True,
):
    """Full structural-discovery diagnostic pipeline."""
    assert B.shape == (64, 64), f"Expected B shape (64, 64), got {B.shape}"

    rng_seed_for_random = seed_base
    configs = [
        ("random",          random_orders(K, 64, rng_seed_for_random)),
        ("raster",          raster_orders(K, 64)),
        ("graph_rw_top4",   sample_orders_for_eval(B, {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 4}, K=K, seed_base=seed_base)["orders"]),
        ("graph_rw_top8",   sample_orders_for_eval(B, {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 8}, K=K, seed_base=seed_base)["orders"]),
        ("graph_rw_eps015", sample_orders_for_eval(B, {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 0, "epsilon_uniform": 0.15}, K=K, seed_base=seed_base)["orders"]),
    ]
    name_to_orders = {n: o for n, o in configs}
    random_ref = name_to_orders["random"]

    os.makedirs(output_dir, exist_ok=True)

    features = None
    if use_appearance:
        print(f"  appearance: computing patch features on {num_images_for_appearance} CIFAR-10 test images ...")
        features = _try_load_cifar_features(num_images_for_appearance)
        if features is not None:
            print(f"  appearance: features shape {features.shape}")

    cols = [
        "name",
        "mean_manhattan", "median_manhattan", "p_dist_le1", "p_dist_le2",
        "same_region_2x2", "same_region_4x4",
        "mean_run_len_2x2", "mean_run_len_4x4",
        "center_step_corr",
        "mean_feature_dist",
        "tau_vs_raster",  # sanity check only
        "notes",
    ]
    tsv_rows = ["\t".join(cols)]

    fig_cd, ax_cd = plt.subplots(figsize=(8, 4))
    fig_td, ax_td = plt.subplots(figsize=(6, 4))
    fig_rs2, ax_rs2 = plt.subplots(figsize=(8, 3))
    fig_rs4, ax_rs4 = plt.subplots(figsize=(8, 3))
    step_x = np.arange(64)

    summary_rows = []

    for name, orders in configs:
        config_dir = os.path.join(output_dir, name)
        os.makedirs(config_dir, exist_ok=True)

        loc = locality_stats(orders)
        cd_curve = center_distance_curve(orders)
        np.save(os.path.join(config_dir, "center_distance_curve.npy"), cd_curve)

        same_q = same_region_rate(orders, region_size=4)
        same_s = same_region_rate(orders, region_size=2)
        run_q = mean_region_run_length(orders, region_size=4)
        run_s = mean_region_run_length(orders, region_size=2)
        cs_corr = center_step_correlation(orders)
        tau = rank_correlation_vs_raster(orders)

        feat_dist_mean = float("nan")
        if features is not None:
            feat_d = consecutive_feature_distance(orders, features)
            feat_dist_mean = float(feat_d.mean())

        # Per-config plots
        eg_idx = min(example_idx, len(orders) - 1)
        plot_order_path(orders[eg_idx], os.path.join(config_dir, "order_path_example.png"),
                        title=f"{name} — example {eg_idx}")
        plot_avg_step_heatmap(orders, os.path.join(config_dir, "avg_step_heatmap.png"))
        plot_transition_heatmap(orders, os.path.join(config_dir, "transition_heatmap.png"))
        plot_transition_distance_histogram(
            orders, os.path.join(config_dir, "transition_distance_histogram.png"),
            ref_orders=random_ref if name != "random" else None,
            main_label=name, ref_label="random",
            title=f"Transition distances — {name}",
        )
        plot_region_stay_curve(
            orders, os.path.join(config_dir, "region_stay_curve_4x4.png"),
            region_size=4,
            ref_orders=random_ref if name != "random" else None,
            main_label=name, ref_label="random",
        )
        plot_region_stay_curve(
            orders, os.path.join(config_dir, "region_stay_curve_2x2.png"),
            region_size=2,
            ref_orders=random_ref if name != "random" else None,
            main_label=name, ref_label="random",
        )
        if features is not None:
            plot_feature_distance_histogram(
                orders, features, os.path.join(config_dir, "feature_distance_histogram.png"),
                ref_orders=random_ref if name != "random" else None,
                main_label=name, ref_label="random",
                title=f"Appearance feature distance — {name}",
            )

        # Combined plots: one line/hist per config
        ax_cd.plot(step_x, cd_curve, label=name)
        td = (np.abs(np.diff(orders // 8, axis=1)) + np.abs(np.diff(orders % 8, axis=1))).ravel()
        ax_td.hist(td, bins=np.arange(15) - 0.5, alpha=0.35, density=True, label=name)
        ax_rs2.plot(np.arange(63), per_step_same_region(orders, region_size=2),
                    label=name, alpha=0.85)
        ax_rs4.plot(np.arange(63), per_step_same_region(orders, region_size=4),
                    label=name, alpha=0.85)

        # TSV row
        notes = "raster_baseline" if name == "raster" else ""
        tsv_rows.append("\t".join([
            name,
            f"{loc['mean']:.4f}", f"{loc['median']:.4f}",
            f"{loc['p_le1']:.4f}", f"{loc['p_le2']:.4f}",
            f"{same_q:.4f}", f"{same_s:.4f}",
            f"{run_q:.4f}", f"{run_s:.4f}",
            f"{cs_corr:.4f}",
            f"{feat_dist_mean:.4f}",
            f"{tau:.4f}",
            notes,
        ]))

        summary_rows.append((name, loc["mean"], loc["p_le1"], same_q, cs_corr, feat_dist_mean, tau))

    # Combined-figure styling + save
    ax_cd.set_xlabel("Reveal step"); ax_cd.set_ylabel("Mean Euclidean distance to center")
    ax_cd.set_title("Center-distance curve by config"); ax_cd.legend()
    fig_cd.tight_layout()
    fig_cd.savefig(os.path.join(output_dir, "center_distance_comparison.png"), dpi=120)
    plt.close(fig_cd)

    ax_td.set_xlabel("Consecutive Manhattan distance"); ax_td.set_ylabel("Density")
    ax_td.set_title("Transition-distance distribution by config"); ax_td.legend()
    fig_td.tight_layout()
    fig_td.savefig(os.path.join(output_dir, "transition_distance_comparison.png"), dpi=120)
    plt.close(fig_td)

    ax_rs2.set_xlabel("Reveal step"); ax_rs2.set_ylabel("P(same 2x2 region)")
    ax_rs2.set_title("Region-stay curve (2x2 super-grid)"); ax_rs2.set_ylim(0, 1); ax_rs2.legend()
    fig_rs2.tight_layout()
    fig_rs2.savefig(os.path.join(output_dir, "region_stay_curve_2x2_comparison.png"), dpi=120)
    plt.close(fig_rs2)

    ax_rs4.set_xlabel("Reveal step"); ax_rs4.set_ylabel("P(same 4x4 quadrant)")
    ax_rs4.set_title("Region-stay curve (2x2 quadrants)"); ax_rs4.set_ylim(0, 1); ax_rs4.legend()
    fig_rs4.tight_layout()
    fig_rs4.savefig(os.path.join(output_dir, "region_stay_curve_4x4_comparison.png"), dpi=120)
    plt.close(fig_rs4)

    # Write TSV
    with open(os.path.join(output_dir, "summary.tsv"), "w") as f:
        f.write("\n".join(tsv_rows) + "\n")

    # Print structural summary
    print(f"\nOK diagnose_image_orders: K={K}, outputs={output_dir}")
    print("  Primary structural metrics:")
    print(f"  {'config':<20} {'mean_manh':>10} {'P(d<=1)':>9} {'same_q':>8} {'cs_corr':>8} {'feat_d':>9}")
    for name, mean_m, p1, sq, cs, fd in [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in summary_rows]:
        fd_str = f"{fd:.4f}" if not np.isnan(fd) else "    n/a"
        print(f"  {name:<20} {mean_m:>10.4f} {p1:>9.4f} {sq:>8.4f} {cs:>8.4f} {fd_str:>9}")
    print("  (sanity check, NOT a target): tau_vs_raster per row:")
    for name, *_, tau in summary_rows:
        print(f"    {name:<20} {tau:+.4f}")


def _run_self_test():
    print("WARNING: B not provided or not found — running self-test with synthetic B")
    rng = np.random.RandomState(0)
    B = (rng.rand(64, 64).astype(np.float32)) ** 4
    run_diagnostics(B, output_dir="probe_results_image/diagnostics/self_test",
                    K=200, seed_base=42, example_idx=0,
                    num_images_for_appearance=100, use_appearance=True)


def main():
    parser = argparse.ArgumentParser(description="Image-order structural diagnostics")
    parser.add_argument("--B", type=str, default=None, help="Path to B_global.npy")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--K", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--example-idx", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=500,
                        help="Number of CIFAR-10 test images for appearance features")
    parser.add_argument("--no-appearance", action="store_true",
                        help="Skip appearance / feature-distance metrics")
    args = parser.parse_args()

    if args.B is None or not os.path.isfile(args.B):
        _run_self_test()
        return

    B = np.load(args.B)
    assert B.shape == (64, 64), f"Expected (64, 64), got {B.shape}"

    output_dir = args.output_dir or "probe_results_image/diagnostics/default"
    run_diagnostics(
        B, output_dir=output_dir, K=args.K, seed_base=args.seed_base,
        example_idx=args.example_idx,
        num_images_for_appearance=args.num_images,
        use_appearance=not args.no_appearance,
    )


if __name__ == "__main__":
    main()
