"""Attention-only curriculum order generators.

Hard rule: do not implement any L2R anchor, index-based prior, or
original-position-based score. L2R is only an evaluation reference.

Block ids are only row/column identifiers for attention matrices. They are not
used as positional features in any score below.
"""

from typing import Optional

import numpy as np


def _as_square_attention(A: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(A, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2D attention matrix")
    arr = arr.copy()
    np.fill_diagonal(arr, 0.0)
    return arr


def _check_same_shape(A: np.ndarray, A_global: np.ndarray) -> None:
    if A.shape != A_global.shape:
        raise ValueError(
            f"A and A_global must have the same shape, got {A.shape} and {A_global.shape}"
        )


def _symmetrized_proximity(A: np.ndarray) -> np.ndarray:
    W = 0.5 * (A + A.T)
    np.fill_diagonal(W, 0.0)
    return W


def _source_scores(A: np.ndarray, alpha_dep: float) -> np.ndarray:
    return A.sum(axis=0) - alpha_dep * A.sum(axis=1)


def _path_weight(W: np.ndarray, order: np.ndarray) -> float:
    if len(order) < 2:
        return 0.0
    return float(sum(W[order[i], order[i + 1]] for i in range(len(order) - 1)))


def _old_nn_from_proximity(W: np.ndarray, start: int) -> np.ndarray:
    n = W.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    order = [int(start)]
    cur = int(start)
    for _ in range(n - 1):
        candidates = np.flatnonzero(~visited)
        scores = W[cur, candidates]
        nxt = int(candidates[int(np.argmax(scores))])
        visited[nxt] = True
        order.append(nxt)
        cur = nxt
    return np.asarray(order, dtype=np.int64)


def old_nn_greedy(A: np.ndarray, n_candidates: int = 2) -> np.ndarray:
    """Legacy nearest-neighbor greedy on W = 0.5 * (A + A.T)."""
    A = _as_square_attention(A, "A")
    W = _symmetrized_proximity(A)
    n = W.shape[0]
    k = min(max(int(n_candidates), 1), n)
    candidates = np.argsort(W.sum(axis=1))[:k]

    best_order = None
    best_weight = -np.inf
    for start in candidates:
        order = _old_nn_from_proximity(W, int(start))
        weight = _path_weight(W, order)
        if weight > best_weight:
            best_weight = weight
            best_order = order
    return best_order


def _attention_order(
    A: np.ndarray,
    *,
    alpha_dep: float = 0.5,
    beta_seen: float = 1.0,
    beta_unseen: float = 0.5,
    beta_local: float = 0.5,
    beta_source: float = 0.2,
    beta_consensus: float = 0.0,
    rank_global: Optional[np.ndarray] = None,
) -> np.ndarray:
    n = A.shape[0]
    W = _symmetrized_proximity(A)
    source = _source_scores(A, alpha_dep)

    order = []
    visited = np.zeros(n, dtype=bool)
    if beta_consensus:
        if rank_global is None:
            raise ValueError("rank_global is required when beta_consensus is nonzero")
        start_scores = beta_source * source + beta_consensus * (-rank_global.astype(np.float64) / n)
        start = int(np.argmax(start_scores))
    else:
        start = int(np.argmax(source))
    order.append(start)
    visited[start] = True

    for t in range(1, n):
        candidates = np.flatnonzero(~visited)
        seen_nodes = np.asarray(order, dtype=np.int64)
        scores = []
        for v in candidates:
            unseen_nodes = candidates[candidates != v]
            seen = float(A[v, seen_nodes].mean()) if len(seen_nodes) else 0.0
            unseen = float(A[v, unseen_nodes].mean()) if len(unseen_nodes) else 0.0
            local = float(W[order[-1], v])
            score = (
                beta_seen * seen
                - beta_unseen * unseen
                + beta_local * local
                + beta_source * float(source[v])
            )
            if beta_consensus:
                if rank_global is None:
                    raise ValueError("rank_global is required when beta_consensus is nonzero")
                # rank_global is the inverse permutation of an attention-derived
                # sigma_global. Raw block id magnitude is deliberately not used.
                score += beta_consensus * (-abs(float(rank_global[v]) - float(t)) / n)
            scores.append(score)
        nxt = int(candidates[int(np.argmax(np.asarray(scores)))])
        order.append(nxt)
        visited[nxt] = True

    return np.asarray(order, dtype=np.int64)


def global_a_greedy(
    A: np.ndarray,
    A_global: np.ndarray,
    *,
    alpha_dep: float = 0.5,
) -> np.ndarray:
    """Generate an order from train-derived global attention only."""
    A = _as_square_attention(A, "A")
    A_global = _as_square_attention(A_global, "A_global")
    _check_same_shape(A, A_global)
    return old_nn_greedy(A_global)


def mixed_a_greedy(
    A: np.ndarray,
    A_global: np.ndarray,
    *,
    lambda_mix: float = 0.5,
    alpha_dep: float = 0.5,
) -> np.ndarray:
    """Generate from lambda*A + (1-lambda)*A_global."""
    A = _as_square_attention(A, "A")
    A_global = _as_square_attention(A_global, "A_global")
    _check_same_shape(A, A_global)
    if not 0.0 <= lambda_mix <= 1.0:
        raise ValueError("lambda_mix must be in [0, 1]")
    A_mix = lambda_mix * A + (1.0 - lambda_mix) * A_global
    return old_nn_greedy(A_mix)


def set_aware_mixed_a_greedy(
    A: np.ndarray,
    A_global: np.ndarray,
    *,
    lambda_mix: float = 0.5,
    alpha_dep: float = 0.5,
    beta_seen: float = 1.0,
    beta_unseen: float = 0.5,
    beta_local: float = 0.5,
    beta_source: float = 0.2,
) -> np.ndarray:
    """Set-aware mixed-A greedy using only seen/unseen/local/source scores."""
    A = _as_square_attention(A, "A")
    A_global = _as_square_attention(A_global, "A_global")
    _check_same_shape(A, A_global)
    if not 0.0 <= lambda_mix <= 1.0:
        raise ValueError("lambda_mix must be in [0, 1]")
    A_mix = lambda_mix * A + (1.0 - lambda_mix) * A_global
    return _attention_order(
        A_mix,
        alpha_dep=alpha_dep,
        beta_seen=beta_seen,
        beta_unseen=beta_unseen,
        beta_local=beta_local,
        beta_source=beta_source,
    )


def inverse_permutation(order: np.ndarray) -> np.ndarray:
    order = np.asarray(order, dtype=np.int64)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order), dtype=np.int64)
    return inv


def global_consensus_greedy(
    A: np.ndarray,
    A_global: np.ndarray,
    *,
    lambda_mix: float = 0.5,
    alpha_dep: float = 0.5,
    beta_seen: float = 1.0,
    beta_unseen: float = 0.5,
    beta_local: float = 0.5,
    beta_source: float = 0.2,
    beta_consensus: float = 0.1,
) -> np.ndarray:
    """Set-aware mixed-A greedy with attention-derived global consensus."""
    A = _as_square_attention(A, "A")
    A_global = _as_square_attention(A_global, "A_global")
    _check_same_shape(A, A_global)
    if not 0.0 <= lambda_mix <= 1.0:
        raise ValueError("lambda_mix must be in [0, 1]")

    sigma_global = _attention_order(A_global, alpha_dep=alpha_dep)
    rank_global = inverse_permutation(sigma_global)
    A_mix = lambda_mix * A + (1.0 - lambda_mix) * A_global
    return _attention_order(
        A_mix,
        alpha_dep=alpha_dep,
        beta_seen=beta_seen,
        beta_unseen=beta_unseen,
        beta_local=beta_local,
        beta_source=beta_source,
        beta_consensus=beta_consensus,
        rank_global=rank_global,
    )


def generate_order_from_attention(
    A: np.ndarray,
    *,
    method: str,
    A_global: Optional[np.ndarray] = None,
    lambda_mix: float = 0.5,
) -> np.ndarray:
    """Dispatch one attention-only order policy by name."""
    if method == "old_nn":
        return old_nn_greedy(A)
    if A_global is None:
        raise ValueError(f"A_global is required for method={method}")
    if method == "global":
        return global_a_greedy(A, A_global)
    if method == "mixed":
        return mixed_a_greedy(A, A_global, lambda_mix=lambda_mix)
    if method == "set_aware":
        return set_aware_mixed_a_greedy(A, A_global, lambda_mix=lambda_mix)
    if method == "consensus":
        return global_consensus_greedy(A, A_global, lambda_mix=lambda_mix)
    raise ValueError(f"Unknown attention order method: {method}")
