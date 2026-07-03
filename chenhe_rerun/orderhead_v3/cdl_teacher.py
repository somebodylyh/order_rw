"""Label-free dynamic CDL consensus teacher for strict-65 block graphs (ported).

Per-sample, per-head, model-frame ONLY. No physical order, inverse_perm,
block_perm, clean_perm, or L2R labels are used anywhere. Output block indices
are model-frame content blocks 0..63.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

from orderhead_v3.attn_order_teacher import teacher_scores


def order_to_rank(order: np.ndarray) -> np.ndarray:
    """rank[block] = step at which block was revealed (0 = earliest)."""
    order = np.asarray(order, dtype=np.int64)
    rank = np.empty(order.shape[0], dtype=np.int64)
    rank[order] = np.arange(order.shape[0], dtype=np.int64)
    return rank


def cdl_rollout_with_standardized_margin(B65: np.ndarray, mode: str = "C-D+L") -> dict:
    """Greedy None-fixed CDL rollout on a strict-65 graph (per-step z-scored margin)."""
    B = np.asarray(B65, dtype=np.float64)
    if B.shape != (65, 65):
        raise ValueError(f"B65 must be (65, 65), got {B.shape}")
    if not np.all(np.diag(B) == 0.0):
        raise ValueError("B65 must have zero diagonal")

    N = 64
    selected = [0]
    unselected = list(range(1, N + 1))
    last = 0
    node_order = []
    margins = []

    while unselected:
        q, candidates = teacher_scores(B, selected, unselected, last, mode=mode)
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
        node = int(candidates[int(np.argmax(q))])
        node_order.append(node)
        selected.append(node)
        unselected.remove(node)
        last = node

    content_order = np.asarray(node_order, dtype=np.int64) - 1  # → 0..63
    rank = order_to_rank(content_order)
    avg_margin = float(np.mean(margins)) if margins else 0.0
    return {"content_order": content_order, "rank": rank, "margins": margins,
            "avg_margin": float(avg_margin)}


def destroy_strict65(B65: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Structure-preserving destruction (row multisets kept, topology destroyed)."""
    B = np.asarray(B65, dtype=np.float64)
    if B.shape != (65, 65):
        raise ValueError(f"B65 must be (65, 65), got {B.shape}")
    out = np.zeros((65, 65), dtype=np.float64)
    out[1:, 0] = B[1:, 0].copy()
    none_row_vals = B[0, 1:].copy()
    rng.shuffle(none_row_vals)
    out[0, 1:] = none_row_vals
    for i in range(1, 65):
        row_vals = B[i, 1:].copy()
        diag_idx = i - 1
        mask = np.ones(64, dtype=bool)
        mask[diag_idx] = False
        shuffled = row_vals[mask].copy()
        rng.shuffle(shuffled)
        out[i, 1:][mask] = shuffled
        out[i, 1:][diag_idx] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


def destroyed_gap(B65: np.ndarray, base_seed: int, n_replicas: int = 3,
                  mode: str = "C-D+L") -> float:
    """gap = real_avg_margin - mean(destroyed_avg_margins)."""
    real = cdl_rollout_with_standardized_margin(B65, mode=mode)
    real_margin = real["avg_margin"]
    destroyed_margins = []
    for k in range(n_replicas):
        rng = np.random.default_rng(base_seed * 100_000 + k)
        Bd = destroy_strict65(B65, rng)
        res = cdl_rollout_with_standardized_margin(Bd, mode=mode)
        destroyed_margins.append(res["avg_margin"])
    return float(real_margin - np.mean(destroyed_margins))


def rank_agreement(ranks: np.ndarray) -> np.ndarray:
    """Per-head mean Kendall tau of rank vectors vs all other heads."""
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


def headwise_zscore(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = float(np.std(values))
    if std < eps:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def compute_label_free_quality(B_heads: np.ndarray, destroy_seed: int = 0,
                               n_destroy_replicas: int = 3, mode: str = "C-D+L") -> dict:
    """Per-head quality (margin, destroyed_gap, agreement) for one sample's B_heads."""
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
        gaps[h] = destroyed_gap(B_heads[h], base_seed=destroy_seed * 100 + h,
                                n_replicas=n_destroy_replicas, mode=mode)
    agreement = rank_agreement(ranks)
    quality = (headwise_zscore(margins) + headwise_zscore(gaps)
               + headwise_zscore(agreement))
    return {"orders": orders, "ranks": ranks, "margin": margins,
            "destroyed_gap": gaps, "agreement": agreement, "quality": quality}


def dynamic_teacher_weights(quality: np.ndarray, temperature: float = 1.0,
                            smoothing: float = 0.05) -> np.ndarray:
    """w_h = (1-smoothing)*softmax(quality/temp) + smoothing/H."""
    quality = np.asarray(quality, dtype=np.float64)
    H = quality.shape[0]
    q = quality / max(temperature, 1e-9)
    q = q - q.max()
    exp_q = np.exp(q)
    soft = exp_q / exp_q.sum()
    return (1.0 - smoothing) * soft + smoothing / H


def soft_pairwise_teacher(ranks: np.ndarray, teacher_weights: np.ndarray) -> np.ndarray:
    """Y[i,j] = sum_h w_h * 1[rank_h[i] < rank_h[j]]; Y[i,i]=0.5, Y[i,j]+Y[j,i]=1."""
    ranks = np.asarray(ranks, dtype=np.int64)
    teacher_weights = np.asarray(teacher_weights, dtype=np.float64)
    H, N = ranks.shape
    if N != 64:
        raise ValueError(f"ranks must have 64 columns, got {N}")
    if teacher_weights.shape != (H,):
        raise ValueError(f"teacher_weights must be ({H},), got {teacher_weights.shape}")
    Y = np.zeros((N, N), dtype=np.float64)
    for h in range(H):
        Y += teacher_weights[h] * (ranks[h, :, None] < ranks[h, None, :])
    diag = np.arange(N)
    Y[diag, diag] = 0.5
    return Y


def build_dynamic_teacher(B_heads: np.ndarray, destroy_seed: int = 0,
                          n_destroy_replicas: int = 3, teacher_temperature: float = 1.0,
                          teacher_smoothing: float = 0.05, mode: str = "C-D+L") -> dict:
    """Full dynamic teacher for ONE sample's (H, 65, 65). Returns dict incl.
    'pairwise' (64,64), 'weights' (H,), 'ranks'/'orders' (H,64) — model-frame [0,63]."""
    quality_result = compute_label_free_quality(
        B_heads, destroy_seed=destroy_seed, n_destroy_replicas=n_destroy_replicas, mode=mode)
    weights = dynamic_teacher_weights(
        quality_result["quality"], temperature=teacher_temperature, smoothing=teacher_smoothing)
    pairwise = soft_pairwise_teacher(quality_result["ranks"], weights)
    return {**quality_result, "weights": weights, "pairwise": pairwise}


def consensus_order_from_pairwise(pairwise: np.ndarray) -> np.ndarray:
    """Derive a single consensus reveal order (model-frame 0..63) from the soft
    pairwise teacher: blocks preferred-earlier (higher row-sum of Y) reveal first."""
    Y = np.asarray(pairwise, dtype=np.float64)
    score = Y.sum(axis=1)          # how often i precedes others
    return np.argsort(-score).astype(np.int64)
