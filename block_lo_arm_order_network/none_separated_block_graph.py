"""65-node None-separated block graph for block-level order discovery.

Node convention:
  node 0      = None / BOS start node
  node 1 + i  = physical content block i = x_{4i}..x_{4i+3}

The token-level AR model only provides observations.  This module consumes the
loss-aligned block aggregate A[target_block, source_node] and builds the
block-level graph used by CDL / diagnostics.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

from attn_order_teacher import MODES
from attn_order_teacher import teacher_scores


def build_none_separated_B(A_with_none: np.ndarray) -> np.ndarray:
    """Build B[source_node, target_node] from A[target_block, source_node].

    Args:
        A_with_none: (N, N+1). Columns are [None], source block0..blockN-1.

    Returns:
        B65: (N+1, N+1). Row/col 0 is None. Edges into None are zero.
    """
    A = np.asarray(A_with_none, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] != A.shape[0] + 1:
        raise ValueError(f"A_with_none must have shape (N, N+1), got {A.shape}")
    N = A.shape[0]
    B = np.zeros((N + 1, N + 1), dtype=np.float64)
    B[0, 1:] = A[:, 0]
    B[1:, 1:] = A[:, 1:].T
    np.fill_diagonal(B, 0.0)
    return B


def rollout_from_none(B65: np.ndarray, mode: str = "C-D+L") -> np.ndarray:
    """Greedy CDL rollout with None fixed as the start node.

    Returns:
        content_order: (N,) content block ids in physical block coordinates.
    """
    B = np.asarray(B65, dtype=np.float64)
    if B.ndim != 2 or B.shape[0] != B.shape[1] or B.shape[0] < 2:
        raise ValueError(f"B65 must be square with at least 2 nodes, got {B.shape}")
    if not np.all(np.diag(B) == 0.0):
        raise ValueError("B65 must have zero diagonal")

    N = B.shape[0] - 1
    selected = [0]
    unselected = list(range(1, N + 1))
    last = 0
    node_order = []
    while unselected:
        scores, candidates = teacher_scores(B, selected, unselected, last, mode=mode)
        node = int(candidates[int(np.argmax(scores))])
        node_order.append(node)
        selected.append(node)
        unselected.remove(node)
        last = node
    return (np.asarray(node_order, dtype=np.int64) - 1)


def rollout_by_method(B65: np.ndarray, method: str) -> np.ndarray:
    """Return a content-block order from one of several label-free readouts."""
    B = np.asarray(B65, dtype=np.float64)
    if method in MODES:
        return rollout_from_none(B, mode=method)
    if B.ndim != 2 or B.shape[0] != B.shape[1] or B.shape[0] < 2:
        raise ValueError(f"B65 must be square with at least 2 nodes, got {B.shape}")
    if not np.all(np.diag(B) == 0.0):
        raise ValueError("B65 must have zero diagonal")
    N = B.shape[0] - 1
    if method == "none_edge":
        scores = B[0, 1:]
    elif method == "content_out_degree":
        scores = B[1:, 1:].sum(axis=1)
    elif method == "content_in_degree_low":
        scores = -B[1:, 1:].sum(axis=0)
    else:
        raise ValueError(f"unknown method {method!r}")
    return np.argsort(-scores).astype(np.int64)[:N]


def discovery_metrics(content_order: np.ndarray) -> dict:
    """Metrics for a 64-content-block order discovered from the 65-node graph."""
    sigma = np.asarray(content_order, dtype=np.int64)
    if sigma.ndim != 1:
        raise ValueError(f"content_order must be 1D, got {sigma.shape}")
    N = sigma.shape[0]
    l2r = np.arange(N, dtype=np.int64)
    tau, _ = kendalltau(sigma, l2r)
    phys0_pos = np.where(sigma == 0)[0]
    phys0_rank = int(phys0_pos[0]) if phys0_pos.size else -1
    return {
        "first_block": int(sigma[0]) if N else -1,
        "first_is_phys0": bool(N > 0 and sigma[0] == 0),
        "phys0_rank": phys0_rank,
        "tau_vs_l2r": float(tau) if not np.isnan(tau) else float("nan"),
        "prefix4_exact": bool(N >= 4 and np.array_equal(sigma[:4], l2r[:4])),
        "prefix8_exact": bool(N >= 8 and np.array_equal(sigma[:8], l2r[:8])),
        "prefix4_overlap": int(np.intersect1d(sigma[: min(4, N)], l2r[: min(4, N)]).size),
        "prefix8_overlap": int(np.intersect1d(sigma[: min(8, N)], l2r[: min(8, N)]).size),
        "order_first16": sigma[: min(16, N)].tolist(),
    }


def classify_gate_status(
    metrics: dict,
    destroyed_abs_tau_mean: float,
    strong_tau: float = 0.7,
    weak_tau: float = 0.5,
    destroyed_tau_max: float = 0.1,
) -> str:
    """Classify strict 65-node block-level discovery gate status."""
    destroyed_ok = float(destroyed_abs_tau_mean) <= float(destroyed_tau_max)
    tau = float(metrics["tau_vs_l2r"])
    if (
        destroyed_ok
        and bool(metrics["first_is_phys0"])
        and tau >= float(strong_tau)
        and int(metrics["prefix4_overlap"]) >= 3
    ):
        return "strong_pass"
    if (
        destroyed_ok
        and int(metrics["phys0_rank"]) <= 3
        and tau >= float(weak_tau)
    ):
        return "weak_pass"
    return "fail"


def combined_discovery_score(metrics: dict, destroyed_abs_tau_mean: float) -> float:
    """Rank all-head sweep candidates by start quality, tau gap, and prefix."""
    tau_gap = float(metrics["tau_vs_l2r"]) - float(destroyed_abs_tau_mean)
    phys0_rank = max(int(metrics["phys0_rank"]), 0)
    first_bonus = 0.25 if bool(metrics["first_is_phys0"]) else 0.0
    rank_score = 0.25 * (1.0 / (1.0 + phys0_rank))
    prefix_score = 0.125 * (int(metrics["prefix4_overlap"]) / 4.0)
    prefix_score += 0.125 * (int(metrics["prefix8_overlap"]) / 8.0)
    return float(tau_gap + first_bonus + rank_score + prefix_score)


def entry_shuffled_control(B65: np.ndarray, seed: int = 0) -> np.ndarray:
    """Shuffle allowed graph entries while keeping None as node 0 and diag zero.

    Allowed entries are source nodes {None, blocks} to content targets.  Edges
    into None stay zero; content self-diagonals stay zero.
    """
    B = np.asarray(B65, dtype=np.float64)
    if B.ndim != 2 or B.shape[0] != B.shape[1] or B.shape[0] < 2:
        raise ValueError(f"B65 must be square with at least 2 nodes, got {B.shape}")
    rng = np.random.default_rng(seed)
    out = np.zeros_like(B)
    mask = np.zeros_like(B, dtype=bool)
    mask[:, 1:] = True
    np.fill_diagonal(mask, False)
    vals = B[mask].copy()
    rng.shuffle(vals)
    out[mask] = vals
    np.fill_diagonal(out, 0.0)
    return out


def content_label_permutation_control(B65: np.ndarray, seed: int = 0) -> np.ndarray:
    """Apply a random content-label permutation while keeping None fixed."""
    B = np.asarray(B65, dtype=np.float64)
    N = B.shape[0] - 1
    rng = np.random.default_rng(seed)
    perm = np.concatenate([[0], 1 + rng.permutation(N)])
    out = B[np.ix_(perm, perm)].copy()
    np.fill_diagonal(out, 0.0)
    return out
