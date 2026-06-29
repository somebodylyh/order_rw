#!/usr/bin/env python3
"""Build clustered pairwise teachers from top-k label-free candidates.

Reads an A_with_none_lh_mean.npy tensor and candidate JSON, rolls out orders,
clusters them by pairwise Kendall tau, and writes per-cluster (and naive
ensemble) soft pairwise teacher .npz files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

# Reuse existing B-graph and rollout infrastructure.
from block_lo_arm_order_network.none_separated_block_graph import (
    build_none_separated_B,
    rollout_by_method,
)


# ---------------------------------------------------------------------------
# Candidate → order
# ---------------------------------------------------------------------------

def candidate_order(A_lh: np.ndarray, candidate: dict) -> np.ndarray:
    """Return the 64-block order for a single candidate (layer, head, method)."""
    layer = int(candidate["layer"])
    head = int(candidate["head"])
    method = str(candidate["method"])
    B = build_none_separated_B(A_lh[layer, head])
    return rollout_by_method(B, method=method)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_orders_by_tau(
    orders: np.ndarray,
    threshold: float = 0.0,
) -> list[list[int]]:
    """Greedy connected-components clustering over tau(orders[i], orders[j]) >= threshold.

    Args:
        orders: shape (K, N) — K integer permutations of N items.
        threshold: minimum pairwise Kendall tau to join the same cluster.

    Returns:
        list of clusters; each cluster is a list of indices into ``orders``.
    """
    K = len(orders)
    if K == 0:
        return []

    # Build adjacency via pairwise tau
    adj = np.zeros((K, K), dtype=bool)
    for i in range(K):
        for j in range(i + 1, K):
            tau, _ = kendalltau(orders[i], orders[j])
            if tau is not None and tau >= threshold:
                adj[i, j] = adj[j, i] = True

    # Connected components (DFS)
    visited = set()
    clusters: list[list[int]] = []
    for i in range(K):
        if i in visited:
            continue
        stack = [i]
        comp: list[int] = []
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            comp.append(v)
            for u in range(K):
                if adj[v, u] and u not in visited:
                    stack.append(u)
        clusters.append(sorted(comp))
    return clusters


# ---------------------------------------------------------------------------
# Soft pairwise teacher
# ---------------------------------------------------------------------------

def build_soft_pairwise_from_orders(
    orders: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build soft pairwise teacher from a set of orders.

    Args:
        orders: shape (K, N) integer permutations.
        weights: shape (K,) non-negative weights (sum to 1).

    Returns:
        Y_pair: shape (N, N) — Y[a, b] = sum_i w_i * 1[rank_i[a] < rank_i[b]].
        confidence: shape (N, N) — 2 * |Y[a, b] - 0.5| (scaled to [0, 1]).
    """
    K, N = orders.shape
    if K == 0:
        raise ValueError("Must have at least one order")
    if len(weights) != K:
        raise ValueError(f"weights length {len(weights)} != K {K}")
    weights = np.asarray(weights, dtype="float32")
    weights /= weights.sum()
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contain non-finite values")

    # Start from 0 (no prior).  Y[a,b] = weighted fraction of orders where a < b.
    Y = np.zeros((N, N), dtype="float32")
    for k in range(K):
        rank = np.argsort(orders[k])
        w = float(weights[k])
        # Vectorised: for all (a,b) with a!=b, add w if rank[a] < rank[b]
        r = rank[:, None]  # (N, 1)
        less = (r < r.T).astype("float32")  # (N, N): 1 if rank[a] < rank[b]
        Y += w * less
    # Diagonal stays at 0 (i before i meaningless).  Leave at 0 for interpretation
    # but report confidence=0 there.
    confidence = 2.0 * np.abs(Y - 0.5)  # 0 = no consensus, 1 = perfect consensus
    return Y.astype("float32"), confidence.astype("float32")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_clustered_teachers(
    a_npy: str,
    candidates: list[dict],
    out_dir: str,
    cluster_threshold: float = 0.0,
    score_temperature: float = 0.5,
) -> dict:
    """Build naive + per-cluster soft pairwise teacher .npz files.

    Returns a report dict suitable for JSON serialisation.
    """
    A_lh = np.load(a_npy)  # shape: (L, H, 65, 65)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    K = len(candidates)
    orders = np.zeros((K, 64), dtype=np.int64)
    candidate_labels: list[str] = []
    for i, cand in enumerate(candidates):
        orders[i] = candidate_order(A_lh, cand)
        candidate_labels.append(f"L{cand['layer']}H{cand['head']}_{cand['method']}")

    # Uniform weights
    weights = np.full(K, 1.0 / K, dtype="float32")

    # --- Naive teacher (all candidates together) ---
    Y_naive, C_naive = build_soft_pairwise_from_orders(orders, weights)
    np.savez_compressed(
        str(out / "teacher_naive.npz"),
        B_heads=np.stack([build_none_separated_B(A_lh[c["layer"], c["head"]]) for c in candidates]),
        orders=orders,
        weights=weights,
        Y_pair=Y_naive,
        confidence=C_naive,
        candidate_labels=np.array(candidate_labels),
    )

    # --- Cluster teachers ---
    clusters = cluster_orders_by_tau(orders, threshold=cluster_threshold)
    cluster_info: list[dict] = []
    for ci, cidx in enumerate(clusters):
        cidx_arr = np.array(cidx, dtype=np.int64)
        cluster_orders_sub = orders[cidx_arr]
        cluster_weights = np.full(len(cidx), 1.0 / len(cidx), dtype="float32")
        Y_c, C_c = build_soft_pairwise_from_orders(cluster_orders_sub, cluster_weights)
        labels = [candidate_labels[i] for i in cidx]
        fname = f"teacher_cluster{ci}.npz"
        np.savez_compressed(
            str(out / fname),
            B_heads=np.stack([build_none_separated_B(A_lh[candidates[i]["layer"], candidates[i]["head"]]) for i in cidx]),
            orders=cluster_orders_sub,
            weights=cluster_weights,
            Y_pair=Y_c,
            confidence=C_c,
            candidate_labels=np.array(labels),
        )
        cluster_info.append({
            "cluster_id": ci,
            "size": len(cidx),
            "indices": cidx,
            "labels": labels,
        })

    # --- Report ---
    report = {
        "a_npy": a_npy,
        "n_candidates": K,
        "candidate_labels": candidate_labels,
        "cluster_threshold": cluster_threshold,
        "n_clusters": len(clusters),
        "clusters": cluster_info,
    }
    (out / "cluster_report.json").write_text(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a-npy", required=True, help="Path to A_with_none_lh_mean.npy")
    p.add_argument("--candidate-json", required=True, help="Path to candidate-set JSON")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cluster-threshold", type=float, default=0.0)
    p.add_argument("--score-temperature", type=float, default=0.5)
    args = p.parse_args()

    candidate_data = json.loads(Path(args.candidate_json).read_text())
    candidates = candidate_data.get("candidates", candidate_data)
    if isinstance(candidates, dict):
        candidates = [candidates]

    report = build_clustered_teachers(
        a_npy=args.a_npy,
        candidates=candidates,
        out_dir=args.out_dir,
        cluster_threshold=args.cluster_threshold,
        score_temperature=args.score_temperature,
    )
    print(f"Built {report['n_clusters']} cluster(s) + naive teacher from {report['n_candidates']} candidates → {args.out_dir}")


if __name__ == "__main__":
    main()
