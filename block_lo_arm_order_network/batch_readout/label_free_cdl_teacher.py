"""Label-free dynamic CDL consensus teacher for strict-65 block graphs.

Given per-sample all-head model-frame strict-65 B matrices (Task 1 output),
this module produces:

  - per-head CDL orders and inverse-rank vectors
  - per-head quality scores (standardized margin, destroyed gap, agreement)
  - per-sample dynamic teacher weights (softmax + smoothing)
  - per-sample soft pairwise teacher Y (antisymmetric around 0.5)

All computation is sample-local and head-relative.  No physical order,
inverse_perm, block_perm, clean_perm, or L2R labels are used anywhere.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

from attn_order_teacher import teacher_scores


# ---------------------------------------------------------------------------
# Rank conversion
# ---------------------------------------------------------------------------

def order_to_rank(order: np.ndarray) -> np.ndarray:
    """Convert a reveal order to its inverse rank vector.

    rank[block] = step at which block was revealed (0 = earliest).
    """
    order = np.asarray(order, dtype=np.int64)
    rank = np.empty(order.shape[0], dtype=np.int64)
    rank[order] = np.arange(order.shape[0], dtype=np.int64)
    return rank


# ---------------------------------------------------------------------------
# CDL rollout with per-step standardized margins (on 65-node B)
# ---------------------------------------------------------------------------

def cdl_rollout_with_standardized_margin(
    B65: np.ndarray,
    mode: str = "C-D+L",
) -> dict:
    """Greedy None-fixed CDL rollout on a strict-65 graph.

    Node 0 is the fixed start (None).  At each step, utilities over the
    remaining candidates are z-scored; the margin is top1-minus-top2.
    If candidate utilities have zero variance, margin is 0.

    Args:
        B65: (65, 65) strict-65 block graph.  Row/col 0 is None.
        mode: teacher mode, passed to ``teacher_scores``.

    Returns:
        dict with keys:
          content_order: (64,) int64 — content block ids 0..63 in reveal order.
          rank:          (64,) int64 — rank[block] = reveal position.
          margins:       list of float — per-step standardized margins.
          avg_margin:    float — mean margin over steps with ≥2 candidates.
    """
    B = np.asarray(B65, dtype=np.float64)
    if B.shape != (65, 65):
        raise ValueError(f"B65 must be (65, 65), got {B.shape}")
    if not np.all(np.diag(B) == 0.0):
        raise ValueError("B65 must have zero diagonal")

    N = 64
    selected = [0]              # None node fixed as start
    unselected = list(range(1, N + 1))  # content nodes 1..64
    last = 0
    node_order = []             # 1-indexed node order
    margins = []

    while unselected:
        q, candidates = teacher_scores(B, selected, unselected, last, mode=mode)
        # q: raw C-D+L (or variant) for each candidate
        q = np.asarray(q, dtype=np.float64)

        if len(candidates) >= 2:
            q_std = float(np.std(q))
            if q_std < 1e-9:
                margin = 0.0
            else:
                q_z = (q - q.mean()) / q_std
                order_desc = np.argsort(-q_z)
                margin = float(q_z[order_desc[0]] - q_z[order_desc[1]])
            margins.append(margin)
        # else: 1 candidate left — no margin to compute

        node = int(candidates[int(np.argmax(q))])
        node_order.append(node)
        selected.append(node)
        unselected.remove(node)
        last = node

    # Convert 1-indexed nodes → 0-indexed content blocks
    content_order = np.asarray(node_order, dtype=np.int64) - 1  # → 0..63
    rank = order_to_rank(content_order)

    avg_margin = float(np.mean(margins)) if margins else 0.0

    return {
        "content_order": content_order,
        "rank": rank,
        "margins": margins,
        "avg_margin": float(avg_margin),
    }


# ---------------------------------------------------------------------------
# Structure-preserving destroy
# ---------------------------------------------------------------------------

def destroy_strict65(B65: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Structure-preserving destruction of a strict-65 block graph.

    Preserves:
      - content-to-None column (B[1:, 0]) — bit-identical
      - row value multisets (each row's values, excluding the preserved column
        and the diagonal, are independently shuffled)
      - zero diagonal

    Destroys:
      - block identity (which value belongs to which target)
      - order topology (correlations across rows)

    Args:
        B65: (65, 65) strict-65 graph.
        rng: seeded NumPy Generator for reproducibility.

    Returns:
        B_destroyed: (65, 65), same shape and row distributions as input.
    """
    B = np.asarray(B65, dtype=np.float64)
    if B.shape != (65, 65):
        raise ValueError(f"B65 must be (65, 65), got {B.shape}")

    out = np.zeros((65, 65), dtype=np.float64)

    # Preserve content-to-None column bit-identically.
    out[1:, 0] = B[1:, 0].copy()

    # Shuffle None-to-content row (row 0, columns 1..65).
    none_row_vals = B[0, 1:].copy()
    rng.shuffle(none_row_vals)
    out[0, 1:] = none_row_vals

    # Shuffle each content-to-content row independently.
    for i in range(1, 65):
        row_vals = B[i, 1:].copy()
        # Exclude the diagonal entry (i-1 is the content block's self-index).
        diag_idx = i - 1
        diag_val = row_vals[diag_idx]
        # Shuffle the other 63 values.
        mask = np.ones(64, dtype=bool)
        mask[diag_idx] = False
        shuffled = row_vals[mask].copy()
        rng.shuffle(shuffled)
        out[i, 1:][mask] = shuffled
        out[i, 1:][diag_idx] = 0.0  # zero diagonal
        # diag_val is discarded (original diagonal was zero, and we want
        # the destroyed diagonal also zero — no information leak).

    # Ensure diagonal is zero everywhere (row 0,0 is also zero).
    np.fill_diagonal(out, 0.0)

    return out


# ---------------------------------------------------------------------------
# Destroyed gap
# ---------------------------------------------------------------------------

def destroyed_gap(
    B65: np.ndarray,
    base_seed: int,
    n_replicas: int = 3,
    mode: str = "C-D+L",
) -> float:
    """Standardized-margin gap between real and destroyed graphs.

    Creates ``n_replicas`` independently destroyed copies, computes the
    standardized rollout margin for each, and returns:

        gap = real_avg_margin - mean(destroyed_margins)

    Args:
        B65: (65, 65) strict-65 graph.
        base_seed: integer seed for the RNG.
        n_replicas: number of destroyed replicas (default 3).
        mode: CDL teacher mode.

    Returns:
        gap: float.  Positive → real margin exceeds destroyed.
    """
    real = cdl_rollout_with_standardized_margin(B65, mode=mode)
    real_margin = real["avg_margin"]

    destroyed_margins = []
    for k in range(n_replicas):
        rng = np.random.default_rng(base_seed * 100_000 + k)
        Bd = destroy_strict65(B65, rng)
        res = cdl_rollout_with_standardized_margin(Bd, mode=mode)
        destroyed_margins.append(res["avg_margin"])

    return float(real_margin - np.mean(destroyed_margins))


# ---------------------------------------------------------------------------
# Rank agreement (per-head, within-sample)
# ---------------------------------------------------------------------------

def rank_agreement(ranks: np.ndarray) -> np.ndarray:
    """Per-head agreement: mean Kendall tau of each head against all others.

    IMPORTANT: operates on RANK vectors (rank[block] = reveal position),
    NOT on order lists.  Kendall tau on rank vectors correctly measures
    whether heads agree on the relative ordering of blocks.

    Args:
        ranks: (H, 64) int64 — one rank vector per head.

    Returns:
        agreement: (H,) float64 — mean tau vs all other heads.
    """
    ranks = np.asarray(ranks, dtype=np.int64)
    H = ranks.shape[0]
    if ranks.shape != (H, 64):
        raise ValueError(f"ranks must be (H, 64), got {ranks.shape}")

    agreements = np.zeros(H, dtype=np.float64)
    for h in range(H):
        taus = []
        for other in range(H):
            if other == h:
                continue
            tau, _ = kendalltau(ranks[h], ranks[other])
            if not np.isnan(tau):
                taus.append(tau)
        agreements[h] = float(np.mean(taus)) if taus else 0.0
    return agreements


# ---------------------------------------------------------------------------
# Within-sample headwise z-scoring
# ---------------------------------------------------------------------------

def headwise_zscore(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score *values* across heads within a single sample.

    Args:
        values: (H,) float array.
        eps: floor on standard deviation.

    Returns:
        z: (H,) float64.
    """
    values = np.asarray(values, dtype=np.float64)
    std = float(np.std(values))
    if std < eps:
        return np.zeros_like(values)
    return (values - values.mean()) / std


# ---------------------------------------------------------------------------
# Label-free quality scores
# ---------------------------------------------------------------------------

def compute_label_free_quality(
    B_heads: np.ndarray,
    destroy_seed: int = 0,
    n_destroy_replicas: int = 3,
    mode: str = "C-D+L",
) -> dict:
    """Compute per-head quality scores for one sample's all-head B matrices.

    For each head, three signals are computed:
      1. standardized_rollout_margin — confidence of the CDL rollout
      2. destroyed_gap — real margin minus mean destroyed margin
      3. rank_agreement — mean Kendall tau vs other heads' rank vectors

    Each signal is z-scored across the H=8 heads within this sample,
    then summed to produce the head's total quality.

    Args:
        B_heads: (H, 65, 65) float64 — one strict-65 graph per head.
        destroy_seed: base seed for destroy RNG.
        n_destroy_replicas: destroyed replicas per head (default 3).
        mode: CDL teacher mode.

    Returns:
        dict with keys:
          orders:        (H, 64) int64 — content reveal orders
          ranks:         (H, 64) int64 — inverse rank vectors
          margin:        (H,) float64 — standardized rollout margins
          destroyed_gap: (H,) float64 — real − destroyed margin gaps
          agreement:     (H,) float64 — per-head rank agreements
          quality:       (H,) float64 — summed z-scored quality
    """
    B_heads = np.asarray(B_heads, dtype=np.float64)
    H = B_heads.shape[0]
    if B_heads.shape != (H, 65, 65):
        raise ValueError(f"B_heads must be (H, 65, 65), got {B_heads.shape}")

    orders = np.zeros((H, 64), dtype=np.int64)
    ranks = np.zeros((H, 64), dtype=np.int64)
    margins = np.zeros(H, dtype=np.float64)
    gaps = np.zeros(H, dtype=np.float64)

    for h in range(H):
        res = cdl_rollout_with_standardized_margin(B_heads[h], mode=mode)
        orders[h] = res["content_order"]
        ranks[h] = res["rank"]
        margins[h] = res["avg_margin"]
        gaps[h] = destroyed_gap(
            B_heads[h],
            base_seed=destroy_seed * 100 + h,
            n_replicas=n_destroy_replicas,
            mode=mode,
        )

    agreement = rank_agreement(ranks)

    # Within-sample headwise z-score, then sum.
    quality = (
        headwise_zscore(margins)
        + headwise_zscore(gaps)
        + headwise_zscore(agreement)
    )

    return {
        "orders": orders,
        "ranks": ranks,
        "margin": margins,
        "destroyed_gap": gaps,
        "agreement": agreement,
        "quality": quality,
    }


# ---------------------------------------------------------------------------
# Dynamic teacher weights
# ---------------------------------------------------------------------------

def dynamic_teacher_weights(
    quality: np.ndarray,
    temperature: float = 1.0,
    smoothing: float = 0.05,
) -> np.ndarray:
    """Convert per-head quality to sample-dependent teacher weights.

        w_h = (1 - smoothing) * softmax(quality_h / temperature) + smoothing / H

    Args:
        quality: (H,) float — per-head quality scores.
        temperature: softmax temperature (default 1.0).
        smoothing: weight reserved for uniform mixture (default 0.05).

    Returns:
        weights: (H,) float64 — sum to 1.0.
    """
    quality = np.asarray(quality, dtype=np.float64)
    H = quality.shape[0]

    q = quality / max(temperature, 1e-9)
    q = q - q.max()  # numerical stability
    exp_q = np.exp(q)
    soft = exp_q / exp_q.sum()

    weights = (1.0 - smoothing) * soft + smoothing / H
    return weights


# ---------------------------------------------------------------------------
# Soft pairwise teacher
# ---------------------------------------------------------------------------

def soft_pairwise_teacher(
    ranks: np.ndarray,
    teacher_weights: np.ndarray,
) -> np.ndarray:
    """Build soft pairwise teacher Y from head ranks and dynamic weights.

        Y[i, j] = sum_h w_h * indicator(rank_h[i] < rank_h[j])

    Properties:
        Y[i, j] + Y[j, i] = 1  for i ≠ j
        Y[i, i] = 0.5

    Args:
        ranks: (H, 64) int64 — one rank vector per head.
        teacher_weights: (H,) float64 — per-head teacher weights.

    Returns:
        Y: (64, 64) float64 — soft pairwise precedence matrix.
    """
    ranks = np.asarray(ranks, dtype=np.int64)
    teacher_weights = np.asarray(teacher_weights, dtype=np.float64)
    H, N = ranks.shape
    if N != 64:
        raise ValueError(f"ranks must have 64 columns, got {N}")
    if teacher_weights.shape != (H,):
        raise ValueError(f"teacher_weights must be ({H},), got {teacher_weights.shape}")

    # Absolute value of weights matters; sum should be ≈1.
    Y = np.zeros((N, N), dtype=np.float64)
    for h in range(H):
        # rank_h[i] < rank_h[j] iff block i is revealed before block j
        Y += teacher_weights[h] * (ranks[h, :, None] < ranks[h, None, :])

    # Explicitly set diagonal to 0.5 (indicator gives 0 since rank < rank is
    # always False).
    diag = np.arange(N)
    Y[diag, diag] = 0.5

    return Y


# ---------------------------------------------------------------------------
# Convenience: full teacher from B_heads
# ---------------------------------------------------------------------------

def build_dynamic_teacher(
    B_heads: np.ndarray,
    destroy_seed: int = 0,
    n_destroy_replicas: int = 3,
    teacher_temperature: float = 1.0,
    teacher_smoothing: float = 0.05,
    mode: str = "C-D+L",
) -> dict:
    """Full dynamic teacher pipeline for one sample.

    Args:
        B_heads: (H, 65, 65) — per-head strict-65 B matrices.
        destroy_seed: base seed for destroy RNG.
        n_destroy_replicas: destroyed replicas per head.
        teacher_temperature: softmax temperature for weights.
        teacher_smoothing: uniform mixture weight.
        mode: CDL teacher mode.

    Returns:
        dict with all intermediate and final teacher products.
    """
    quality_result = compute_label_free_quality(
        B_heads,
        destroy_seed=destroy_seed,
        n_destroy_replicas=n_destroy_replicas,
        mode=mode,
    )

    weights = dynamic_teacher_weights(
        quality_result["quality"],
        temperature=teacher_temperature,
        smoothing=teacher_smoothing,
    )

    pairwise = soft_pairwise_teacher(
        quality_result["ranks"],
        weights,
    )

    return {
        **quality_result,
        "weights": weights,
        "pairwise": pairwise,
    }
