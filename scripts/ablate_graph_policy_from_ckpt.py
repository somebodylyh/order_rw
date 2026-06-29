#!/usr/bin/env python3
"""
Frozen graph-policy ablation: analyse graph policy variants on a FIXED checkpoint.
No model update — purely graph-structure + policy-component analysis.

Usage (text):
  python scripts/ablate_graph_policy_from_ckpt.py \
    --modality text --b-path probe_results/A_train_n64_10k.npy \
    --output-dir probe_results/graph_policy_ablation/text

Usage (image):
  python scripts/ablate_graph_policy_from_ckpt.py \
    --modality image --b-path probe_results_image/attention/baseline10k/B_global.npy \
    --output-dir probe_results_image/graph_policy_ablation/baseline10k
"""

import argparse, json, os, sys, time
import numpy as np
from typing import Dict, Tuple, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_REPO, "block_lo_arm_order_network"))

from directed_graph_policy import build_directed_graph, compute_source, _softmax
from order_diagnostics import _kendall_tau

# ═══════════════════════════════════════════════════════════════════════════════════════
# Policy variant definitions
# ═══════════════════════════════════════════════════════════════════════════════════════

# Component-based variants: score(v) = support + eta*local - lam*future + rho*r
# Components is a dict with bool flags; eta/lam/rho are scalar weights
COMPONENT_VARIANTS = {
    "full_v3":         {"support": True,  "local": True,  "future": True,  "readiness": True},
    "no_readiness":    {"support": True,  "local": True,  "future": True,  "readiness": False},
    "readiness_only":  {"support": False, "local": False, "future": False, "readiness": True},
    "no_dependency":   {"support": True,  "local": True,  "future": False, "readiness": True},
    "no_local":        {"support": True,  "local": False, "future": True,  "readiness": True},
    "support_only":    {"support": True,  "local": False, "future": False, "readiness": False},
    "local_only":      {"support": False, "local": True,  "future": False, "readiness": False},
    "graph_guided":   {"support": True,  "local": True,  "future": True,  "readiness": True},
}

# B-transformed variants: run full_v3 on a transformed B matrix
B_TRANSFORM_VARIANTS = ["shuffled_B_full", "sym_B_full", "random_B_full"]

ALL_VARIANTS = list(COMPONENT_VARIANTS.keys()) + B_TRANSFORM_VARIANTS

DEFAULTS = dict(
    lam=0.75, rho=0.2, tau_start=0.10, tau_step=0.10,
    alpha_dep=0.5, top_k=4, epsilon=0.0,
)


# ═══════════════════════════════════════════════════════════════════════════════════════
# Core: per-step ablation score with explicit component control
# ═══════════════════════════════════════════════════════════════════════════════════════

def graph_asymmetry(B: np.ndarray) -> float:
    """a(B) = ||B - B^T||_1 / ||B||_1. 0=symmetric, 1=purely antisymmetric."""
    B = np.asarray(B, dtype=np.float64)
    norm_B = float(np.abs(B).sum())
    if norm_B == 0:
        return 0.0
    return float(np.abs(B - B.T).sum()) / norm_B


def graph_readiness_signal(B: np.ndarray, alpha_dep: float = 0.5) -> float:
    """Readiness signal strength: std(r) / mean(|B|).

    Text ≈ 12, Image ≈ 5.5. High values indicate readiness dominates.
    """
    B = np.asarray(B, dtype=np.float64)
    r, _, _ = compute_source(B, alpha_dep)
    edge_mass = float(np.abs(B[B != 0]).mean()) if np.any(B != 0) else 1e-10
    return float(r.std() / edge_mass)


def graph_guided_params(B: np.ndarray) -> Tuple[Dict[str, bool], Dict[str, float]]:
    """Graph-statistic-guided hyperparameter prediction (diagnostic, not a new algorithm).

    Uses the SAME full_v3 components (support + local + future + readiness).
    Only predicts ρ, η, λ from graph statistics — demonstrating that coefficient
    sensitivity is explainable from graph structure, not arbitrary.

    Key graph statistic: readiness signal s = std(r) / mean(|B|).
      - Text s≈12.3 → s↑ ⇒ ρ↑ (readiness matters), η↓ (local less important)
      - Image s≈5.5 → s↓ ⇒ ρ↓ (readiness hurts), η↑ (local matters more)
      - λ≈0 always: future dependency insensitive for both modalities.
    """
    s = graph_readiness_signal(B)
    t = min(max((s - 4.0) / 8.0, 0.0), 1.0)  # normalised [0,1]

    components = {"support": True, "local": True, "future": True, "readiness": True}  # full_v3
    params = {
        "lam": 0.0,                         # insensitive for both modalities
        "rho": 0.8 * t,                     # text≈0.8, image≈0.15
        "eta": 1.0 - 0.6 * t,              # text≈0.4, image≈0.89
        "readiness_signal": s,
    }
    return components, params


def _ablation_score(
    B: np.ndarray, S: np.ndarray, U: np.ndarray, last: int,
    components: Dict[str, bool],
    lam: float, rho: float, readiness: np.ndarray,
    eta: float = 1.0,
) -> np.ndarray:
    """Compute per-step score from explicit component flags.

    score(v) = I_s * support + η * I_l * local - I_f * lam * future + I_r * rho * readiness[v]
    """
    B = np.asarray(B, dtype=np.float64)
    U = np.asarray(U, dtype=np.int64)

    score = np.zeros(len(U), dtype=np.float64)

    if components["support"] and len(S) > 0:
        score += B[np.asarray(S, dtype=np.int64)].sum(axis=0)[U]

    if components["local"] and last >= 0:
        score += eta * B[last, U]

    if components["future"] and len(U) > 0:
        score -= lam * B[U].sum(axis=0)[U]

    if components["readiness"]:
        score += rho * readiness[U]

    return score


def sample_order_ablated(
    B: np.ndarray,
    components: Dict[str, bool],
    params: Dict[str, float],
    seed: int,
    eta: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Sample one order with explicit component flags."""
    rng = np.random.default_rng(seed)
    B = np.asarray(B, dtype=np.float64)
    N = B.shape[0]

    lam = float(params.get("lam", DEFAULTS["lam"]))
    rho = float(params.get("rho", DEFAULTS["rho"]))
    tau_start = float(params.get("tau_start", DEFAULTS["tau_start"]))
    tau_step = float(params.get("tau_step", DEFAULTS["tau_step"]))
    alpha_dep = float(params.get("alpha_dep", DEFAULTS["alpha_dep"]))
    top_k = int(params.get("top_k", DEFAULTS["top_k"]) or 0)
    eps = float(params.get("epsilon", DEFAULTS["epsilon"]))

    readiness, _, _ = compute_source(B, alpha_dep)

    # Step 0: always use readiness for the first node
    p0 = _softmax(readiness, tau_start, rng, top_k=top_k)
    if eps > 0:
        p0 = (1.0 - eps) * p0 + eps / N
    idx0 = int(rng.choice(N, p=p0))

    order = np.zeros(N, dtype=np.int64)
    order[0] = idx0
    logprob = float(np.log(max(p0[idx0], 1e-300)))

    S = np.array([idx0], dtype=np.int64)
    U = np.setdiff1d(np.arange(N, dtype=np.int64), S)
    last = idx0

    for t in range(1, N):
        scores = _ablation_score(B, S, U, last, components, lam, rho, readiness, eta=eta)
        p_t = _softmax(scores, tau_step, rng, top_k=top_k)
        if eps > 0:
            p_t = (1.0 - eps) * p_t + eps / len(U)
            p_t = p_t / p_t.sum()

        idx_t = int(rng.choice(len(U), p=p_t))
        node_t = int(U[idx_t])

        order[t] = node_t
        logprob += float(np.log(max(p_t[idx_t], 1e-300)))

        S = np.append(S, node_t)
        U = U[U != node_t]
        last = node_t

    return order, logprob


def transform_B(B: np.ndarray, variant: str, seed: int = 42) -> np.ndarray:
    """Apply B-space transformation for ablation variant."""
    B = np.asarray(B, dtype=np.float64).copy()
    N = B.shape[0]
    rng = np.random.default_rng(seed)

    if variant == "shuffled_B_full":
        perm = rng.permutation(N)
        B = B[perm][:, perm]
    elif variant == "sym_B_full":
        B = 0.5 * (B + B.T)
    elif variant == "random_B_full":
        scale = float(np.abs(B[B != 0]).mean()) if np.any(B != 0) else 1e-3
        B = np.abs(rng.normal(loc=scale, scale=scale * 0.5, size=(N, N)))
        np.fill_diagonal(B, 0.0)
    else:
        raise ValueError(f"Unknown B transform: {variant}")

    return B


# ═══════════════════════════════════════════════════════════════════════════════════════
# Sampling
# ═══════════════════════════════════════════════════════════════════════════════════════

def sample_K_orders(B: np.ndarray, components: Dict[str, bool],
                    params: Dict[str, float], K: int, seed_base: int,
                    eta: float = 1.0) -> Dict:
    """Sample K orders and compute text-side diagnostics."""
    N = B.shape[0]
    orders = np.zeros((K, N), dtype=np.int64)
    logprobs = np.zeros(K, dtype=np.float64)

    for k in range(K):
        seed = seed_base * 10000 + k
        orders[k], logprobs[k] = sample_order_ablated(B, components, params, seed, eta=eta)

    # legality
    valid = sum(1 for k in range(K) if sorted(orders[k].tolist()) == list(range(N)))
    legal_rate = float(valid / K)

    # first_node_entropy
    first_nodes = orders[:, 0]
    counts = np.bincount(first_nodes, minlength=N).astype(np.float64)
    probs = counts / K
    probs_nz = probs[probs > 0]
    first_node_entropy = float(-np.sum(probs_nz * np.log(probs_nz)))

    # pairwise_tau
    n_pairs = min(500, K * (K - 1) // 2)
    pair_rng = np.random.default_rng(seed_base * 10000 + 99999)
    if K >= 2:
        all_i, all_j = np.triu_indices(K, k=1)
        if len(all_i) > n_pairs:
            idx = pair_rng.choice(len(all_i), size=n_pairs, replace=False)
            all_i, all_j = all_i[idx], all_j[idx]
        pair_taus = [_kendall_tau(orders[i], orders[j]) for i, j in zip(all_i, all_j)]
        pairwise_tau = float(np.mean(pair_taus)) if pair_taus else 0.0
    else:
        pairwise_tau = 0.0

    # tau_vs_L2R
    L2R = np.arange(N, dtype=np.int64)
    taus_l2r = np.array([_kendall_tau(orders[i], L2R) for i in range(K)], dtype=np.float64)
    tau_mean = float(taus_l2r.mean())
    tau_std = float(taus_l2r.std())

    # avg rank by position
    avg_rank = np.zeros(N, dtype=np.float64)
    for pos in range(N):
        avg_rank[pos] = orders[:, pos].mean()
    # Kendall tau between avg_rank ordering and L2R
    avg_rank_order = np.argsort(avg_rank)
    rank_vs_l2r_tau = float(_kendall_tau(avg_rank_order, L2R))

    # policy_step_entropy (replay first min(K,100) orders)
    n_replay = min(100, K)
    H_all = np.zeros((n_replay, N), dtype=np.float64)
    for i in range(n_replay):
        H_all[i] = _replay_entropy(B, components, params, orders[i],
                                    seed_base * 10000 + i, eta=eta)
    H_mean_per_step = H_all.mean(axis=0)
    third = max(1, N // 3)
    H_early = float(H_mean_per_step[:third].mean())
    H_mid = float(H_mean_per_step[third:2 * third].mean())
    H_late = float(H_mean_per_step[2 * third:].mean())
    H_overall = float(H_mean_per_step.mean())

    # logprob stats
    lp_mean = float(logprobs.mean())
    lp_std = float(logprobs.std())

    return {
        "orders": orders,
        "legal_rate": legal_rate,
        "tau_vs_l2r_mean": tau_mean,
        "tau_vs_l2r_std": tau_std,
        "first_node_entropy": first_node_entropy,
        "pairwise_tau_mean": pairwise_tau,
        "avg_rank_vs_l2r_tau": rank_vs_l2r_tau,
        "avg_rank_by_position": avg_rank.tolist(),
        "H_mean": H_overall,
        "H_early": H_early,
        "H_mid": H_mid,
        "H_late": H_late,
        "logprob_mean": lp_mean,
        "logprob_std": lp_std,
    }


def _replay_entropy(B: np.ndarray, components: Dict[str, bool],
                    params: Dict[str, float], order: np.ndarray, seed: int,
                    eta: float = 1.0) -> np.ndarray:
    """Replay one order step-by-step; return per-step entropies H_t (N,)."""
    rng = np.random.default_rng(seed)
    B = np.asarray(B, dtype=np.float64)
    N = B.shape[0]
    order = np.asarray(order, dtype=np.int64)

    tau_start = float(params.get("tau_start", DEFAULTS["tau_start"]))
    tau_step = float(params.get("tau_step", DEFAULTS["tau_step"]))
    alpha_dep = float(params.get("alpha_dep", DEFAULTS["alpha_dep"]))
    top_k = int(params.get("top_k", DEFAULTS["top_k"]) or 0)
    lam = float(params.get("lam", DEFAULTS["lam"]))
    rho = float(params.get("rho", DEFAULTS["rho"]))

    readiness, _, _ = compute_source(B, alpha_dep)
    H = np.zeros(N, dtype=np.float64)

    # Step 0
    p0 = _softmax(readiness, tau_start, rng, top_k=top_k)
    eps_val = 1e-300
    H[0] = float(-np.sum(p0 * np.log(np.maximum(p0, eps_val))))

    first = int(order[0])
    S = np.array([first], dtype=np.int64)
    U = np.setdiff1d(np.arange(N, dtype=np.int64), S)
    last = first

    for t in range(1, N):
        scores = _ablation_score(B, S, U, last, components, lam, rho, readiness, eta=eta)
        p_t = _softmax(scores, tau_step, rng, top_k=top_k)
        H[t] = float(-np.sum(p_t * np.log(np.maximum(p_t, eps_val))))

        chosen = int(order[t])
        if chosen not in U:
            break
        S = np.append(S, chosen)
        U = U[U != chosen]
        last = chosen

    return H


# ═══════════════════════════════════════════════════════════════════════════════════════
# Image spatial metrics
# ═══════════════════════════════════════════════════════════════════════════════════════

def compute_image_metrics(orders: np.ndarray, grid: int = 8) -> Dict:
    """Compute image-specific spatial/locality diagnostics.

    orders: (K, N) int array, N = grid*grid.
    """
    K, N = orders.shape
    assert N == grid * grid, f"Expected N={grid*grid}, got {N}"

    rows = orders // grid   # (K, N)
    cols = orders % grid    # (K, N)

    # -- mean Manhattan step --
    if N >= 2:
        d_row = np.abs(np.diff(rows, axis=1))
        d_col = np.abs(np.diff(cols, axis=1))
        mean_manh = float((d_row + d_col).mean())
    else:
        mean_manh = 0.0

    # -- P(d<=1), P(d<=2) --
    d = d_row + d_col  # (K, N-1)
    P_d_le_1 = float((d <= 1).mean())
    P_d_le_2 = float((d <= 2).mean())

    # -- same quadrant --
    half = grid // 2
    quad = (rows // half) * 2 + (cols // half)  # (K, N), values 0-3
    same_q = float((np.diff(quad, axis=1) == 0).mean())

    # -- same super-region (half-grid) --
    sreg_row = rows // half  # (K, N) 0 or 1
    sreg_col = cols // half  # (K, N) 0 or 1
    same_s_row = float((np.diff(sreg_row, axis=1) == 0).mean())
    same_s_col = float((np.diff(sreg_col, axis=1) == 0).mean())

    # -- run length (consecutive same-quadrant) --
    run_lengths = []
    for k in range(K):
        run = 1
        for t in range(1, N):
            if quad[k, t] == quad[k, t - 1]:
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)
    run_len_q = float(np.mean(run_lengths))

    # -- center step correlation --
    center = (grid - 1) / 2.0
    dist_to_center = np.sqrt((rows - center) ** 2 + (cols - center) ** 2)  # (K, N)
    steps = np.arange(N)[None, :].repeat(K, axis=0)  # (K, N)
    center_corr = float(np.corrcoef(steps.ravel(), dist_to_center.ravel())[0, 1])

    # -- transition distance histogram --
    d_flat = d.ravel()
    hist_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14]
    hist_counts = []
    for i in range(len(hist_bins) - 1):
        hist_counts.append(int(((d_flat > hist_bins[i]) & (d_flat <= hist_bins[i + 1])).sum()))
    hist_counts.append(int((d_flat > hist_bins[-1]).sum()))

    return {
        "mean_manhattan": mean_manh,
        "P_d_le_1": P_d_le_1,
        "P_d_le_2": P_d_le_2,
        "same_quadrant": same_q,
        "same_super_region_row": same_s_row,
        "same_super_region_col": same_s_col,
        "run_length_quadrant": run_len_q,
        "center_step_corr": center_corr,
        "transition_hist_bins": hist_bins,
        "transition_hist_counts": hist_counts,
    }


def raster_orders(K: int, N: int, seed_base: int = 42) -> np.ndarray:
    """Generate K raster-scan orders for reference."""
    grid = int(np.sqrt(N))
    order = np.arange(N, dtype=np.int64).reshape(grid, grid)
    order = order.ravel()
    return np.tile(order, (K, 1))


def random_orders_fn(K: int, N: int, seed_base: int = 42) -> np.ndarray:
    """Generate K random permutations."""
    orders = np.zeros((K, N), dtype=np.int64)
    rng = np.random.default_rng(seed_base)
    for k in range(K):
        orders[k] = rng.permutation(N)
    return orders


# ═══════════════════════════════════════════════════════════════════════════════════════
# Main ablation loop
# ═══════════════════════════════════════════════════════════════════════════════════════

def run_ablation(B_orig: np.ndarray, params: Dict[str, float],
                 K: int, seed_base: int, modality: str,
                 variants=None) -> Dict[str, Dict]:
    """Run all policy variants and return results dict keyed by variant name."""
    if variants is None:
        variants = ALL_VARIANTS

    results = {}
    for variant_name in variants:
        t0 = time.time()

        eta = 1.0
        use_params = params.copy()

        if variant_name in COMPONENT_VARIANTS:
            components = COMPONENT_VARIANTS[variant_name]
            B_use = B_orig.copy()
            if variant_name == "graph_guided":
                components, aw = graph_guided_params(B_use)
                use_params["lam"] = aw["lam"]
                use_params["rho"] = aw["rho"]
                eta = aw["eta"]
                s = aw["readiness_signal"]
                print(f"  graph_guided: s={s:.2f} → ρ={aw['rho']:.3f}, η={aw['eta']:.3f}, λ={aw['lam']:.1f}")
        elif variant_name in B_TRANSFORM_VARIANTS:
            components = COMPONENT_VARIANTS["full_v3"]
            B_use = transform_B(B_orig, variant_name, seed=seed_base)
        else:
            raise ValueError(f"Unknown variant: {variant_name}")

        diag = sample_K_orders(B_use, components, use_params, K, seed_base, eta=eta)

        if modality == "image":
            img_metrics = compute_image_metrics(diag["orders"], grid=8)
            diag.update(img_metrics)

        elapsed = time.time() - t0
        diag["wall_time_s"] = round(elapsed, 1)
        results[variant_name] = diag

        # Quick status
        if modality == "text":
            print(f"  {variant_name:22s}  tau={diag['tau_vs_l2r_mean']:+.4f}±{diag['tau_vs_l2r_std']:.4f}  "
                  f"H={diag['H_mean']:.3f}  firstH={diag['first_node_entropy']:.3f}  {elapsed:.0f}s")
        else:
            print(f"  {variant_name:22s}  manh={diag['mean_manhattan']:.3f}  "
                  f"P(d≤1)={diag['P_d_le_1']:.4f}  same_q={diag['same_quadrant']:.4f}  {elapsed:.0f}s")

    return results


# ═══════════════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════════════

def save_results(results: Dict, output_dir: str, modality: str, params: Dict,
                 B_orig: np.ndarray, K: int):
    """Save results: summary.tsv, summary.md, per-variant JSON, and plots."""
    os.makedirs(output_dir, exist_ok=True)

    # -- config.json --
    config = {
        "modality": modality,
        "K": K,
        "params": params,
        "variants": list(results.keys()),
        "B_shape": list(B_orig.shape),
        "B_mean": float(B_orig.mean()),
        "B_std": float(B_orig.std()),
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # -- summary.tsv --
    if modality == "text":
        columns = ["variant", "tau_vs_l2r_mean", "tau_vs_l2r_std", "avg_rank_vs_l2r_tau",
                   "first_node_entropy", "pairwise_tau_mean",
                   "H_mean", "H_early", "H_mid", "H_late",
                   "logprob_mean", "logprob_std", "legal_rate", "wall_time_s"]
    else:
        columns = ["variant", "mean_manhattan", "P_d_le_1", "P_d_le_2",
                   "same_quadrant", "same_super_region_row", "same_super_region_col",
                   "run_length_quadrant", "center_step_corr",
                   "first_node_entropy", "pairwise_tau_mean",
                   "H_mean", "H_early", "H_mid", "H_late",
                   "logprob_mean", "legal_rate", "wall_time_s"]

    tsv_path = os.path.join(output_dir, "summary.tsv")
    real_variants = [v for v in results if not v.startswith("_")]
    with open(tsv_path, "w") as f:
        f.write("\t".join(columns) + "\n")
        for vname in real_variants:
            r = results[vname]
            row = [vname]
            for col in columns[1:]:
                val = r.get(col, "")
                if isinstance(val, float):
                    row.append(f"{val:.6f}")
                else:
                    row.append(str(val))
            f.write("\t".join(row) + "\n")

    # -- summary.md --
    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write(f"# Graph Policy Ablation ({modality})\n\n")
        f.write(f"K={K}, params: {json.dumps(params)}\n\n")
        f.write(f"B shape: {B_orig.shape}, mean={B_orig.mean():.6f}\n\n")

        if modality == "text":
            f.write("| Variant | τ vs L2R | avg rank τ | first H | pairwise τ | H mean | H early | H mid | H late | logprob |\n")
            f.write("|---------|----------|------------|---------|------------|--------|---------|-------|--------|--------|\n")
            for vname in real_variants:
                r = results[vname]
                f.write(f"| {vname} | {r['tau_vs_l2r_mean']:+.4f}±{r['tau_vs_l2r_std']:.3f} | "
                        f"{r['avg_rank_vs_l2r_tau']:+.4f} | {r['first_node_entropy']:.3f} | "
                        f"{r['pairwise_tau_mean']:+.4f} | {r['H_mean']:.3f} | "
                        f"{r['H_early']:.3f} | {r['H_mid']:.3f} | {r['H_late']:.3f} | "
                        f"{r['logprob_mean']:.1f} |\n")
        else:
            f.write("| Variant | manh | P(d≤1) | P(d≤2) | same_q | same_s_row | same_s_col | run_len_q | center_corr | first_H | H_mean |\n")
            f.write("|---------|------|--------|--------|--------|------------|------------|-----------|-------------|---------|--------|\n")
            for vname in real_variants:
                r = results[vname]
                f.write(f"| {vname} | {r['mean_manhattan']:.3f} | {r['P_d_le_1']:.4f} | "
                        f"{r['P_d_le_2']:.4f} | {r['same_quadrant']:.4f} | "
                        f"{r['same_super_region_row']:.4f} | {r['same_super_region_col']:.4f} | "
                        f"{r['run_length_quadrant']:.2f} | {r['center_step_corr']:+.4f} | "
                        f"{r['first_node_entropy']:.3f} | {r['H_mean']:.3f} |\n")

        # Add reference lines
        if modality == "image":
            N = B_orig.shape[0]
            rand_orders = random_orders_fn(500, N)
            rand_img = compute_image_metrics(rand_orders, grid=8)
            f.write(f"\nReference:\n")
            f.write(f"- random: manh={rand_img['mean_manhattan']:.3f}, "
                    f"P(d≤1)={rand_img['P_d_le_1']:.4f}, same_q={rand_img['same_quadrant']:.4f}\n")

            rast_orders = raster_orders(500, N)
            rast_img = compute_image_metrics(rast_orders, grid=8)
            f.write(f"- raster: manh={rast_img['mean_manhattan']:.3f}, "
                    f"P(d≤1)={rast_img['P_d_le_1']:.4f}, same_q={rast_img['same_quadrant']:.4f}\n")

    # -- per-variant JSON --
    for vname in real_variants:
        r_serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in r.items() if k != "orders"}
        with open(os.path.join(output_dir, f"{vname}.json"), "w") as f:
            json.dump(r_serializable, f, indent=2)

    # -- orders (sampled permutations) saved separately --
    orders_dir = os.path.join(output_dir, "orders")
    os.makedirs(orders_dir, exist_ok=True)
    for vname in real_variants:
        np.save(os.path.join(orders_dir, f"{vname}_orders.npy"), results[vname]["orders"])

    print(f"\nSaved to {output_dir}/")
    print(f"  summary.tsv, summary.md, config.json")
    print(f"  per-variant JSONs, orders/*.npy")


# ═══════════════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════════════

def make_plots(results: Dict, output_dir: str, modality: str, B_orig: np.ndarray):
    """Generate diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    orders_dir = os.path.join(output_dir, "orders")

    if modality == "text":
        _plot_text_avg_rank(results, plots_dir)
        _plot_text_tau_bar(results, plots_dir)
        _plot_text_entropy_curve(results, plots_dir)
    else:
        _plot_image_transition_hist(results, plots_dir)
        _plot_image_metric_bars(results, plots_dir)
        _plot_image_heatmaps(results, B_orig, plots_dir, orders_dir)


def _plot_text_avg_rank(results, plots_dir):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8))
    key_variants = ["full_v3", "no_readiness", "readiness_only",
                    "shuffled_B_full", "sym_B_full", "random_B_full"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(key_variants)))
    for vi, vname in enumerate(key_variants):
        if vname not in results:
            continue
        avg_rank = np.array(results[vname]["avg_rank_by_position"])
        ax.plot(avg_rank, label=vname, color=colors[vi], linewidth=1.5)
    N = len(avg_rank)
    ax.plot(np.arange(N), "k--", alpha=0.3, label="L2R")
    ax.set_xlabel("Position in sampled order")
    ax.set_ylabel("Mean L2R rank")
    ax.set_title("Avg physical rank by order position")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "avg_rank_vs_position.png"), dpi=150)
    plt.close(fig)


def _plot_text_tau_bar(results, plots_dir):
    import matplotlib.pyplot as plt
    variants = list(results.keys())
    taus = [results[v]["tau_vs_l2r_mean"] for v in variants]
    errs = [results[v]["tau_vs_l2r_std"] for v in variants]
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#2ecc71" if t > 0.5 else "#e74c3c" if t < 0.2 else "#f39c12" for t in taus]
    ax.bar(variants, taus, yerr=errs, color=colors, capsize=3)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.set_ylabel("τ vs L2R")
    ax.set_title("Graph policy component ablation: τ vs L2R")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "tau_vs_l2r_bars.png"), dpi=150)
    plt.close(fig)


def _plot_text_entropy_curve(results, plots_dir):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    key_variants = ["full_v3", "no_readiness", "readiness_only",
                    "support_only", "local_only"]
    for vname in key_variants:
        if vname not in results:
            continue
        lo = max(results[vname]["H_mean"] - results[vname]["H_early"], 0)
        hi = max(results[vname]["H_late"] - results[vname]["H_mean"], 0)
        ax.barh(vname, results[vname]["H_mean"],
                xerr=[[lo], [hi]], capsize=3)
    ax.set_xlabel("Policy step entropy (bits)")
    ax.set_title("Entropy by policy variant (error bar: early→late)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "entropy_bars.png"), dpi=150)
    plt.close(fig)


def _plot_image_transition_hist(results, plots_dir):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.ravel()
    key_variants = list(results.keys())
    for idx, vname in enumerate(key_variants[:10]):
        ax = axes[idx]
        r = results[vname]
        bins = r["transition_hist_bins"]
        counts = r["transition_hist_counts"]
        labels = [str(b) for b in bins[:-1]] + [f">{bins[-1]}"]
        ax.bar(range(len(counts)), counts, color="steelblue")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(vname, fontsize=8)
    fig.suptitle("Transition distance histograms")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "transition_distance_comparison.png"), dpi=150)
    plt.close(fig)


def _plot_image_metric_bars(results, plots_dir):
    import matplotlib.pyplot as plt
    variants = list(results.keys())
    metrics = ["mean_manhattan", "same_quadrant", "P_d_le_1", "run_length_quadrant"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()
    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        vals = [results[v][metric] for v in variants]
        ax.bar(variants, vals, color="steelblue")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
    fig.suptitle("Image spatial metrics by policy variant")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "image_metric_bars.png"), dpi=150)
    plt.close(fig)


def _plot_image_heatmaps(results, B_orig, plots_dir, orders_dir=None):
    import matplotlib.pyplot as plt
    key_variants = ["full_v3", "no_readiness", "readiness_only", "random_B_full"]
    fig, axes = plt.subplots(1, len(key_variants) + 1, figsize=(20, 5))

    # B heatmap
    ax = axes[0]
    im = ax.imshow(B_orig, cmap="viridis", aspect="auto")
    ax.set_title("B_global")
    plt.colorbar(im, ax=ax)

    for vi, vname in enumerate(key_variants):
        if vname not in results:
            continue
        ax = axes[vi + 1]
        # Try to get orders from results dict, or load from npy
        orders = results[vname].get("orders")
        if orders is None and orders_dir is not None:
            orders_path = os.path.join(orders_dir, f"{vname}_orders.npy")
            if os.path.exists(orders_path):
                orders = np.load(orders_path)
        if orders is None:
            ax.set_title(f"{vname}\n(no data)")
            continue
        orders = np.asarray(orders)
        avg_step = np.zeros(B_orig.shape[0])
        for t in range(B_orig.shape[0]):
            avg_step[t] = orders[:, t].mean()
        grid = int(np.sqrt(B_orig.shape[0]))
        step_map = avg_step.reshape(grid, grid)
        im = ax.imshow(step_map, cmap="plasma", aspect="auto")
        ax.set_title(f"{vname}\navg step")
        plt.colorbar(im, ax=ax)

    fig.suptitle("B matrix and avg step position heatmaps")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "avg_step_heatmaps.png"), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Frozen graph-policy ablation from checkpoint")
    p.add_argument("--modality", choices=["text", "image"], required=True)
    p.add_argument("--b-path", required=True,
                   help="Path to A .npy (text: A_train_n64_10k.npy) or B .npy (image: B_global.npy)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--K", type=int, default=1000,
                   help="Number of orders to sample per variant (text default 1000, image 500)")
    p.add_argument("--lam", type=float, default=DEFAULTS["lam"])
    p.add_argument("--rho", type=float, default=DEFAULTS["rho"])
    p.add_argument("--tau-start", type=float, default=DEFAULTS["tau_start"])
    p.add_argument("--tau-step", type=float, default=DEFAULTS["tau_step"])
    p.add_argument("--alpha-dep", type=float, default=DEFAULTS["alpha_dep"])
    p.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    p.add_argument("--epsilon", type=float, default=DEFAULTS["epsilon"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", nargs="*", default=None,
                   help="Specific variants to run (default: all 10)")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--sweep", choices=["rho", "lambda"], default=None,
                   help="Run hparam sweep instead of variant ablation")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════════════
# Sweep mode
# ═══════════════════════════════════════════════════════════════════════════════════════

def run_sweep(B: np.ndarray, base_params: Dict, K: int, seed: int, modality: str,
              sweep_param: str, output_dir: str):
    """Run lambda or rho sweep and save hparam_sweep.tsv + plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    components = COMPONENT_VARIANTS["full_v3"]

    if sweep_param == "rho":
        sweep_name = "ρ (readiness strength)"
        sweep_key = "rho"
        sweep_values = [0, 0.05, 0.1, 0.2, 0.4, 0.8] if modality == "text" else [0, 0.05, 0.1, 0.2, 0.4]
    else:
        sweep_name = "λ (dependency penalty)"
        sweep_key = "lam"
        sweep_values = [0, 0.25, 0.5, 0.75, 1.0, 1.5] if modality == "text" else [0, 0.25, 0.5, 0.75, 1.0]

    print(f"\n{'='*60}")
    print(f"Sweep: {sweep_name}")
    print(f"Values: {sweep_values}")
    print(f"Base params: lam={base_params['lam']}, rho={base_params['rho']}")
    print(f"{'='*60}")

    rows = []
    for val in sweep_values:
        params = base_params.copy()
        params[sweep_key] = val
        t0 = time.time()

        diag = sample_K_orders(B, components, params, K, seed)
        if modality == "image":
            img = compute_image_metrics(diag["orders"], grid=8)
            diag.update(img)

        elapsed = time.time() - t0
        row = {"sweep_param": sweep_key, "value": val, **diag, "wall_time_s": round(elapsed, 1)}
        rows.append(row)

        if modality == "text":
            print(f"  {sweep_key}={val:.2f}  τ={diag['tau_vs_l2r_mean']:+.4f}±{diag['tau_vs_l2r_std']:.4f}  "
                  f"H={diag['H_mean']:.3f}  firstH={diag['first_node_entropy']:.3f}  {elapsed:.0f}s")
        else:
            print(f"  {sweep_key}={val:.2f}  manh={diag['mean_manhattan']:.3f}  "
                  f"P(d≤1)={diag['P_d_le_1']:.4f}  same_q={diag['same_quadrant']:.4f}  {elapsed:.0f}s")

    # Save sweep TSV
    os.makedirs(output_dir, exist_ok=True)
    tsv_path = os.path.join(output_dir, f"hparam_sweep_{sweep_key}.tsv")

    if modality == "text":
        cols = ["sweep_param", "value", "tau_vs_l2r_mean", "tau_vs_l2r_std",
                "avg_rank_vs_l2r_tau", "first_node_entropy", "pairwise_tau_mean",
                "H_mean", "H_early", "H_mid", "H_late", "logprob_mean", "wall_time_s"]
    else:
        cols = ["sweep_param", "value", "mean_manhattan", "P_d_le_1", "P_d_le_2",
                "same_quadrant", "same_super_region_row", "same_super_region_col",
                "run_length_quadrant", "center_step_corr",
                "first_node_entropy", "pairwise_tau_mean",
                "H_mean", "H_early", "H_mid", "H_late", "logprob_mean", "wall_time_s"]

    with open(tsv_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")

    # Save per-value JSONs
    for row in rows:
        val = row["value"]
        r_ser = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in row.items() if k != "orders"}
        with open(os.path.join(output_dir, f"sweep_{sweep_key}_{val}.json"), "w") as f:
            json.dump(r_ser, f, indent=2)

    # Save orders
    orders_dir = os.path.join(output_dir, "orders")
    os.makedirs(orders_dir, exist_ok=True)
    for row in rows:
        np.save(os.path.join(orders_dir, f"sweep_{sweep_key}_{row['value']}_orders.npy"),
                row["orders"])

    # Plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    vals = [r["value"] for r in rows]

    if modality == "text":
        _plot_sweep_line(rows, vals, sweep_key, "tau_vs_l2r_mean", "τ vs L2R",
                         f"Text sensitivity: {sweep_name}", plots_dir,
                         f"tau_vs_{sweep_key}.png", y_err="tau_vs_l2r_std")
        _plot_sweep_line(rows, vals, sweep_key, "H_mean", "H mean (bits)",
                         f"Text entropy: {sweep_name}", plots_dir,
                         f"H_vs_{sweep_key}.png")
        _plot_sweep_line(rows, vals, sweep_key, "pairwise_tau_mean", "Pairwise τ",
                         f"Text pairwise τ: {sweep_name}", plots_dir,
                         f"pairwise_tau_vs_{sweep_key}.png")
    else:
        _plot_sweep_line(rows, vals, sweep_key, "mean_manhattan", "Mean Manhattan",
                         f"Image spatial locality: {sweep_name}", plots_dir,
                         f"manh_vs_{sweep_key}.png")
        _plot_sweep_line(rows, vals, sweep_key, "P_d_le_1", "P(d≤1)",
                         f"Image near-neighbor: {sweep_name}", plots_dir,
                         f"Pd1_vs_{sweep_key}.png")
        _plot_sweep_line(rows, vals, sweep_key, "same_quadrant", "Same quadrant",
                         f"Image region grouping: {sweep_name}", plots_dir,
                         f"sameq_vs_{sweep_key}.png")

    print(f"\nSweep saved to {output_dir}/")
    print(f"  {tsv_path}")
    return rows


def _plot_sweep_line(rows, vals, sweep_key, metric, metric_label, title, plots_dir,
                     filename, y_err=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    y = [r[metric] for r in rows]
    if y_err:
        err = [r[y_err] for r in rows]
        ax.errorbar(vals, y, yerr=err, marker="o", capsize=4, linewidth=2, markersize=8)
    else:
        ax.plot(vals, y, marker="o", linewidth=2, markersize=8)
    ax.set_xlabel(sweep_key, fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def main():
    args = parse_args()

    # Auto-adjust K for image
    K = args.K
    if args.modality == "image" and args.K == 1000:
        K = 500  # image default

    # Load B
    A_or_B = np.load(args.b_path)
    if A_or_B.ndim == 3:
        print(f"A_all shape: {A_or_B.shape}, averaging over axis 0")
        A_global = A_or_B.mean(axis=0).astype(np.float64)
        np.fill_diagonal(A_global, 0.0)
        B = build_directed_graph(A_global)
    elif A_or_B.ndim == 2:
        B = build_directed_graph(A_or_B)
    else:
        raise ValueError(f"Unexpected array shape: {A_or_B.shape}")

    if args.modality == "image":
        B = np.asarray(A_or_B, dtype=np.float64).copy()
        np.fill_diagonal(B, 0.0)
        if "A_global" in args.b_path:
            B = build_directed_graph(A_or_B)

    print(f"B shape: {B.shape}, mean={B.mean():.6f}, std={B.std():.6f}")
    print(f"Modality: {args.modality}, K={K}")

    base_params = {
        "lam": args.lam, "rho": args.rho,
        "tau_start": args.tau_start, "tau_step": args.tau_step,
        "alpha_dep": args.alpha_dep, "top_k": args.top_k,
        "epsilon": args.epsilon,
    }

    # ── Sweep mode ──
    if args.sweep:
        run_sweep(B, base_params, K, args.seed, args.modality, args.sweep, args.output_dir)
        print("Done.")
        return

    # ── Variant ablation mode ──
    print(f"Params: lam={args.lam}, rho={args.rho}, tau_start={args.tau_start}, "
          f"tau_step={args.tau_step}, top_k={args.top_k}, eps={args.epsilon}")

    variants = args.variants if args.variants else ALL_VARIANTS
    print(f"\nVariants ({len(variants)}): {variants}")
    print(f"{'='*80}")

    results = run_ablation(B, base_params, K, args.seed, args.modality, variants)

    if args.modality == "image":
        N = B.shape[0]
        print("\nReferences:")
        rand_orders = random_orders_fn(K, N, args.seed)
        rand_img = compute_image_metrics(rand_orders, grid=8)
        print(f"  {'random':22s}  manh={rand_img['mean_manhattan']:.3f}  "
              f"P(d≤1)={rand_img['P_d_le_1']:.4f}  same_q={rand_img['same_quadrant']:.4f}")
        rast_orders = raster_orders(K, N, args.seed)
        rast_img = compute_image_metrics(rast_orders, grid=8)
        print(f"  {'raster':22s}  manh={rast_img['mean_manhattan']:.3f}  "
              f"P(d≤1)={rast_img['P_d_le_1']:.4f}  same_q={rast_img['same_quadrant']:.4f}")
        results["_reference_random"] = rand_img
        results["_reference_raster"] = rast_img

    print(f"\n{'='*80}")
    print(f"Saving results...")
    save_results(results, args.output_dir, args.modality, base_params, B, K)

    if not args.no_plots:
        print("Generating plots...")
        make_plots(results, args.output_dir, args.modality, B)

    print("Done.")


if __name__ == "__main__":
    main()
