"""Cheap teacher variants: refresh-K / chunked CDL and prefix-K CDL.

Places full sequential CDL and one-shot initial CDL on a continuous
cost spectrum by controlling the score-recomputation interval K:

  K=1  → full sequential CDL (64 score recomputations)
  K=64 → initial one-shot CDL (1 score recomputation)
  Intermediate K → cheaper than full, better than one-shot

Also provides prefix-K teachers that only generate the first K blocks
sequentially and fill the remainder with a simple strategy.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from attn_order_teacher import teacher_scores, MAIN_MODE


# ---------------------------------------------------------------------------
# Refresh-K / Chunked CDL
# ---------------------------------------------------------------------------

def refresh_k_cdl_rollout(
    B65: np.ndarray,
    K: int = 8,
    mode: str = MAIN_MODE,
) -> dict:
    """Greedy CDL rollout that recomputes scores every K steps.

    At each iteration, compute ``teacher_scores`` once, take the top-K
    (or fewer if |U| < K) remaining candidates, update S_t / U_t / last,
    and repeat until all blocks are placed.

    K=1  → full sequential CDL (identical to cdl_rollout_with_standardized_margin)
    K=64 → initial one-shot CDL

    Args:
        B65: (65, 65) strict-65 block graph.  Row/col 0 is None.
        K: number of blocks to select per score recomputation.
        mode: teacher mode (default "C-D+L").

    Returns:
        dict with:
          content_order: (64,) int64 — block ids 0..63 in reveal order.
          rank:          (64,) int64 — rank[block] = reveal position.
          n_refreshes:   int — number of score recomputations.
          wall_time:     float — seconds for this rollout.
    """
    B = np.asarray(B65, dtype=np.float64)
    if B.shape != (65, 65):
        raise ValueError(f"B65 must be (65, 65), got {B.shape}")
    if not np.all(np.diag(B) == 0.0):
        raise ValueError("B65 must have zero diagonal")

    K = max(1, int(K))
    N = 64
    selected = [0]                # None node fixed as start
    unselected = list(range(1, N + 1))
    last = 0
    node_order = []
    n_refreshes = 0

    t0 = time.perf_counter()

    while unselected:
        q, candidates = teacher_scores(B, selected, unselected, last, mode=mode)
        q = np.asarray(q, dtype=np.float64)
        n_refreshes += 1

        # Take top min(K, |U|) candidates
        take = min(K, len(candidates))
        top_indices = np.argsort(-q)[:take]
        chunk_nodes = [int(candidates[i]) for i in top_indices]

        node_order.extend(chunk_nodes)
        selected.extend(chunk_nodes)
        for node in chunk_nodes:
            unselected.remove(node)
        last = chunk_nodes[-1]

    elapsed = time.perf_counter() - t0

    content_order = np.asarray(node_order, dtype=np.int64) - 1
    rank = order_to_rank(content_order)

    return {
        "content_order": content_order,
        "rank": rank,
        "n_refreshes": n_refreshes,
        "wall_time": elapsed,
    }


# ---------------------------------------------------------------------------
# Prefix-K CDL
# ---------------------------------------------------------------------------

def cdl_prefix_k_rollout(
    B65: np.ndarray,
    K: int = 8,
    suffix: str = "random",
    mode: str = MAIN_MODE,
    seed: int = 42,
) -> dict:
    """CDL rollout that only generates the first K blocks sequentially,
    then fills the remainder with a cheap strategy.

    Args:
        B65: (65, 65) strict-65 block graph.
        K: number of blocks to generate via full sequential CDL.
        suffix: fill strategy for remaining blocks.
            "random"     — random permutation of remaining blocks
            "score"      — sort remaining by current one-shot CDL score
            "layout"     — identity order 0..63 (model-frame layout path)
        mode: teacher mode (default "C-D+L").
        seed: random seed for "random" suffix.

    Returns:
        dict with content_order, rank, n_refreshes (= K), wall_time, suffix.
    """
    B = np.asarray(B65, dtype=np.float64)
    if B.shape != (65, 65):
        raise ValueError(f"B65 must be (65, 65), got {B.shape}")

    N = 64
    K = max(0, min(int(K), N))

    t0 = time.perf_counter()

    # ── Sequential prefix (K steps) ──
    selected = [0]
    unselected = list(range(1, N + 1))
    last = 0
    node_order = []
    n_refreshes = 0

    for _ in range(K):
        if not unselected:
            break
        q, candidates = teacher_scores(B, selected, unselected, last, mode=mode)
        q = np.asarray(q, dtype=np.float64)
        n_refreshes += 1
        best_idx = int(np.argmax(q))
        node = int(candidates[best_idx])
        node_order.append(node)
        selected.append(node)
        unselected.remove(node)
        last = node

    # ── Suffix ──
    remaining = list(unselected)  # 1-indexed nodes still unplaced
    if suffix == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(remaining)
    elif suffix == "score":
        # One-shot CDL score over remaining
        q, candidates = teacher_scores(B, selected, remaining, last, mode=mode)
        q = np.asarray(q, dtype=np.float64)
        order_desc = np.argsort(-q)
        remaining = [int(candidates[i]) for i in order_desc]
    elif suffix == "layout":
        remaining.sort()  # 1..64 natural order = model-frame identity
    else:
        raise ValueError(f"unknown suffix={suffix!r}")

    node_order.extend(remaining)

    elapsed = time.perf_counter() - t0

    content_order = np.asarray(node_order, dtype=np.int64) - 1
    rank = order_to_rank(content_order)

    return {
        "content_order": content_order,
        "rank": rank,
        "n_refreshes": n_refreshes,
        "wall_time": elapsed,
        "suffix": suffix,
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def order_to_rank(order: np.ndarray) -> np.ndarray:
    """Convert reveal order (0..63) to rank[block] = position."""
    rank = np.empty(64, dtype=np.int64)
    rank[order] = np.arange(64, dtype=np.int64)
    return rank


def kendall_tau(order_a: np.ndarray, order_b: np.ndarray) -> float:
    """Kendall tau between two orders (0..63 permutations)."""
    a = np.asarray(order_a, dtype=np.int64)
    b = np.asarray(order_b, dtype=np.int64)
    if a.shape != (64,) or b.shape != (64,):
        raise ValueError(f"Expected (64,), got {a.shape}, {b.shape}")
    rank_a = order_to_rank(a)
    rank_b = order_to_rank(b)
    n = 64
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_a = rank_a[j] - rank_a[i]
            d_b = rank_b[j] - rank_b[i]
            if (d_a > 0 and d_b > 0) or (d_a < 0 and d_b < 0):
                concordant += 1
            elif (d_a > 0 and d_b < 0) or (d_a < 0 and d_b > 0):
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def pairwise_accuracy(order_pred: np.ndarray, order_ref: np.ndarray) -> float:
    """Fraction of pairs where both orders agree on relative ordering."""
    rank_pred = order_to_rank(order_pred)
    rank_ref = order_to_rank(order_ref)
    n = 64
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_pred = rank_pred[j] - rank_pred[i]
            d_ref = rank_ref[j] - rank_ref[i]
            if (d_pred > 0 and d_ref > 0) or (d_pred < 0 and d_ref < 0):
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def prefix_at_k(order_pred: np.ndarray, order_ref: np.ndarray, k: int = 8) -> float:
    """Fraction of top-K blocks shared between the two orders."""
    return len(set(order_pred[:k]) & set(order_ref[:k])) / k


def evaluate_teacher_variants(
    B_batch: np.ndarray,
    teacher_ref: np.ndarray,
    variants: List[dict],
) -> List[dict]:
    """Evaluate cheap teacher variants against a reference teacher.

    Args:
        B_batch: (M, 65, 65) strict65 graphs.
        teacher_ref: (M, 64) int64 reference orders.
        variants: list of variant dicts with keys "name" and "fn".

    Returns:
        list of result dicts with aggregated metrics.
    """
    M = B_batch.shape[0]
    results = []

    for variant in variants:
        name = variant["name"]
        fn = variant["fn"]
        print(f"  {name} ...", end=" ", flush=True)
        t0 = time.perf_counter()

        orders = np.empty((M, 64), dtype=np.int64)
        extras = []
        for i in range(M):
            res = fn(B_batch[i])
            orders[i] = res["content_order"]
            extras.append(res)

        total_time = time.perf_counter() - t0

        taus = np.array([kendall_tau(orders[i], teacher_ref[i]) for i in range(M)])
        pws = np.array([pairwise_accuracy(orders[i], teacher_ref[i]) for i in range(M)])
        p8s = np.array([prefix_at_k(orders[i], teacher_ref[i], 8) for i in range(M)])
        p16s = np.array([prefix_at_k(orders[i], teacher_ref[i], 16) for i in range(M)])
        n_refreshes = [e.get("n_refreshes", 0) for e in extras]

        results.append({
            "name": name,
            "tau_mean": float(taus.mean()),
            "tau_std": float(taus.std()),
            "pairwise_acc": float(pws.mean()),
            "prefix8": float(p8s.mean()),
            "prefix16": float(p16s.mean()),
            "total_time": total_time,
            "time_per_graph": total_time / M,
            "n_refreshes_mean": float(np.mean(n_refreshes)),
            "speedup_vs_full": 64.0 / max(float(np.mean(n_refreshes)), 1.0),
        })

        print(f"τ={taus.mean():.4f}  p8={p8s.mean():.3f}  "
              f"refreshes={np.mean(n_refreshes):.1f}  "
              f"{total_time:.2f}s", flush=True)

    return results
