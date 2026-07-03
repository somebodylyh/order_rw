"""65-node None-separated block graph for block-level order discovery (ported).

Node convention:
  node 0      = None / BOS start node
  node 1 + i  = model-frame content block i

Consumes the loss-aligned block aggregate A[target_block, source_node] and builds
the block-level graph used by CDL / diagnostics.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

from orderhead_v3.attn_order_teacher import MODES
from orderhead_v3.attn_order_teacher import teacher_scores


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
        content_order: (N,) content block ids in model-frame block coordinates.
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
