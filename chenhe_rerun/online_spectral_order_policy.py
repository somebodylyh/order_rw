from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass
class FixedHeadSpectralPolicyConfig:
    num_components: int = 4
    component_pairs: str = "1-2"
    num_angles: int = 16
    k_values: str = "8,10"
    group_methods: str = "gap"
    threshold_percentile: float = 60.0
    transform: str = "relu"
    temperature: float = 1.0
    direction_lambdas: str = "0,0.1,0.25"
    directed_score_weight: float = 0.25
    band_quality_weight: float = 0.05
    score_adjacency_sym: str = "max"


def recover_fixed_head_spectral_order(matrix, config: FixedHeadSpectralPolicyConfig | None = None):
    """Recover the best current-frame block order from one fixed-head attention matrix.

    This is the online fixed-head version of the attention-spectral teacher:
    robust attention normalization, symmetric/directed graph features, spectral
    coordinates, angle/band candidate search, and model-side attention scoring.
    """

    top_candidates = recover_fixed_head_spectral_candidates(matrix, config=config, top_m=1)
    best = top_candidates[0]
    return best["order"], best


def recover_fixed_head_spectral_candidates(
    matrix,
    config: FixedHeadSpectralPolicyConfig | None = None,
    top_m: int = 32,
):
    """Recover top-M current-frame spectral order candidates from one attention matrix."""

    config = config or FixedHeadSpectralPolicyConfig()
    top_m = int(top_m)
    if top_m <= 0:
        raise ValueError(f"top_m must be positive, got {top_m}")
    values = _to_numpy_matrix(matrix)
    num_blocks = int(values.shape[0])
    np.fill_diagonal(values, np.nan)

    z = robust_z(values)
    score_adjacency = robust_z(sym(z, config.score_adjacency_sym))
    directed_variants = {
        "attn_query_key": anti(z),
        "attn_key_query": anti(z.T),
    }
    affinity = affinity_from_adjacency(
        values,
        threshold_percentile=float(config.threshold_percentile),
        transform=str(config.transform),
        temperature=float(config.temperature),
    )
    coords, eigvals = spectral_coordinates(affinity, num_components=int(config.num_components))
    axes = list(candidate_axes(coords, parse_pairs(config.component_pairs), int(config.num_angles)))
    if not axes:
        raise ValueError("fixed-head spectral policy produced no candidate axes")

    candidates = []
    seen = set()
    k_values = parse_ints(config.k_values)
    group_methods = [item.strip() for item in str(config.group_methods).split(",") if item.strip()]
    direction_lambdas = parse_floats(config.direction_lambdas)
    for primary_name, primary_axis, primary_meta in axes:
        for secondary_name, secondary_axis, secondary_meta in axes:
            for k in k_values:
                for group_method in group_methods:
                    for primary_reverse in (False, True):
                        for secondary_reverse in (False, True):
                            for group_order_reverse in (False, True):
                                order, bands, band_score = grouped_order(
                                    primary_axis,
                                    secondary_axis,
                                    k=int(k),
                                    group_method=str(group_method),
                                    primary_reverse=bool(primary_reverse),
                                    secondary_reverse=bool(secondary_reverse),
                                    group_order_reverse=bool(group_order_reverse),
                                )
                                key = tuple(order)
                                if key in seen:
                                    continue
                                seen.add(key)
                                score, score_meta = score_order(
                                    order,
                                    score_adjacency,
                                    directed_variants,
                                    direction_lambdas,
                                    directed_weight=float(config.directed_score_weight),
                                    band_score=float(band_score),
                                    band_weight=float(config.band_quality_weight),
                                )
                                name = (
                                    f"{primary_name}_x_{secondary_name}_k{int(k):02d}_{group_method}"
                                    f"_p{'d' if primary_reverse else 'a'}"
                                    f"_s{'d' if secondary_reverse else 'a'}"
                                    f"_g{'d' if group_order_reverse else 'a'}"
                                )
                                candidate = {
                                    "name": name,
                                    "order": [int(value) for value in order],
                                    "ordered_bands": bands,
                                    "score": float(score),
                                    "meta": {
                                        "primary_axis": primary_meta,
                                        "secondary_axis": secondary_meta,
                                        "k": int(k),
                                        "group_method": str(group_method),
                                        "primary_reverse": bool(primary_reverse),
                                        "secondary_reverse": bool(secondary_reverse),
                                        "group_order_reverse": bool(group_order_reverse),
                                        "band_quality": float(band_score),
                                        "group_sizes": [int(len(group)) for group in bands],
                                        **score_meta,
                                    },
                                }
                                candidates.append(candidate)
    if not candidates:
        raise ValueError("fixed-head spectral policy produced no candidates")

    common_meta = {
        "eigvals": [float(value) for value in eigvals.tolist()],
        "num_unique_candidates": int(len(seen)),
        "num_blocks": int(num_blocks),
        "selection_rule": (
            "fixed-head attention-spectral score over current-frame attention only; "
            "no original-frame or evaluation-only signal is used"
        ),
    }
    for candidate in candidates:
        if sorted(candidate["order"]) != list(range(num_blocks)):
            raise ValueError("fixed-head spectral policy produced an invalid permutation")
        candidate["meta"].update(common_meta)

    candidates.sort(key=lambda item: (-float(item["score"]), str(item["name"])))
    return candidates[:top_m]


def candidates_to_priority_vector(
    candidates: Sequence[dict],
    num_blocks: int,
    teacher_temperature: float = 1.0,
    score_normalization: str = "zscore",
):
    """Convert scored order candidates into a soft current-frame priority vector."""

    num_blocks = int(num_blocks)
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")
    if not candidates:
        raise ValueError("candidates_to_priority_vector requires at least one candidate")

    positions = []
    scores = []
    expected = list(range(num_blocks))
    denom = float(max(1, num_blocks - 1))
    for candidate in candidates:
        order = [int(value) for value in candidate.get("order", [])]
        if len(order) != num_blocks or sorted(order) != expected:
            raise ValueError("candidate order is not a valid current-frame block permutation")
        pos = torch.empty(num_blocks, dtype=torch.float32)
        pos[torch.tensor(order, dtype=torch.long)] = torch.arange(num_blocks, dtype=torch.float32)
        priority = 1.0 - pos / denom
        positions.append(priority)
        scores.append(float(candidate.get("score", 0.0)))

    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    finite_scores = scores_tensor[torch.isfinite(scores_tensor)]
    if finite_scores.numel() == 0:
        scores_norm = torch.zeros_like(scores_tensor)
        score_mean = float("nan")
        score_std = float("nan")
        use_uniform = True
    else:
        score_mean = float(finite_scores.mean().item())
        score_std_value = float(finite_scores.std(unbiased=False).item())
        safe_scores = torch.where(torch.isfinite(scores_tensor), scores_tensor, torch.tensor(score_mean))
        normalization = str(score_normalization)
        if normalization == "zscore":
            use_uniform = bool(score_std_value < 1e-8)
            scores_norm = torch.zeros_like(safe_scores) if use_uniform else (safe_scores - score_mean) / score_std_value
        elif normalization in {"none", "raw"}:
            use_uniform = False
            scores_norm = safe_scores
        else:
            raise ValueError(f"unknown score_normalization={score_normalization!r}")
        score_std = score_std_value

    if use_uniform:
        weights = torch.full((len(candidates),), 1.0 / float(len(candidates)), dtype=torch.float32)
    else:
        temperature = max(float(teacher_temperature), 1e-6)
        weights = torch.softmax(scores_norm / temperature, dim=0).to(dtype=torch.float32)

    stacked_priorities = torch.stack(positions, dim=0)
    r_update = torch.sum(weights.view(-1, 1) * stacked_priorities, dim=0).to(dtype=torch.float32)
    entropy = float((-(weights * weights.clamp_min(1e-12).log()).sum()).item())
    top_weight = float(weights.max().item()) if weights.numel() > 0 else float("nan")
    priority_std = float(r_update.std(unbiased=False).item()) if r_update.numel() > 0 else float("nan")
    priority_min = float(r_update.min().item()) if r_update.numel() > 0 else float("nan")
    priority_max = float(r_update.max().item()) if r_update.numel() > 0 else float("nan")
    priority_range = float(priority_max - priority_min) if math.isfinite(priority_min) and math.isfinite(priority_max) else float("nan")
    order_to_weight = {}
    for candidate, weight in zip(candidates, weights.tolist()):
        order_to_weight[tuple(int(value) for value in candidate["order"])] = float(weight)
    reverse_pair_weight_mass = 0.0
    reverse_pair_count = 0
    for order, weight in order_to_weight.items():
        reverse_order = tuple(reversed(order))
        if reverse_order in order_to_weight:
            reverse_pair_count += 1
            reverse_pair_weight_mass += float(weight)
    meta = {
        "top_m_used": int(len(candidates)),
        "score_mean": float(score_mean),
        "score_std": float(score_std),
        "score_normalization": str(score_normalization),
        "teacher_temperature": float(teacher_temperature),
        "weight_entropy": entropy,
        "top_weight": top_weight,
        "effective_num_candidates": float(math.exp(entropy)) if math.isfinite(entropy) else float("nan"),
        "uniform_weights": bool(use_uniform),
        "priority_mean": float(r_update.mean().item()) if r_update.numel() > 0 else float("nan"),
        "priority_std": priority_std,
        "priority_min": priority_min,
        "priority_max": priority_max,
        "priority_range": priority_range,
        "priority_effectively_constant": bool(
            math.isfinite(priority_std) and math.isfinite(priority_range) and priority_std < 1e-6 and priority_range < 1e-6
        ),
        "reverse_pair_count": int(reverse_pair_count),
        "reverse_pair_weight_mass": float(reverse_pair_weight_mass),
    }
    return r_update, weights, meta


def sample_order_from_priority(priority_logits: torch.Tensor, temperature: float = 0.7):
    """Sample a full current-frame block order with Gumbel-top-k priority sampling."""

    if not torch.is_tensor(priority_logits):
        priority_logits = torch.as_tensor(priority_logits, dtype=torch.float32)
    if priority_logits.ndim != 1:
        raise ValueError(f"priority_logits must be 1D, got shape={tuple(priority_logits.shape)}")
    logits = priority_logits.float()
    if not torch.isfinite(logits).all():
        raise ValueError("priority_logits contains non-finite values")
    eps = 1e-6
    u = torch.rand_like(logits).clamp(eps, 1.0 - eps)
    gumbel = -torch.log(-torch.log(u))
    temperature = max(float(temperature), 1e-6)
    return torch.argsort(logits / temperature + gumbel, descending=True)


def parse_ints(text: str):
    return [int(item) for item in str(text).split(",") if str(item).strip()]


def parse_floats(text: str):
    return [float(item) for item in str(text).split(",") if str(item).strip()]


def parse_pairs(text: str):
    pairs = []
    for item in str(text).split(","):
        if not item.strip():
            continue
        left, right = item.split("-")
        pairs.append((int(left) - 1, int(right) - 1))
    return pairs


def robust_z(matrix):
    values = np.asarray(matrix, dtype=np.float64)
    finite = values[np.isfinite(values)]
    out = np.full_like(values, np.nan, dtype=np.float64)
    if finite.size == 0:
        return out
    median = float(np.median(finite))
    q25, q75 = np.percentile(finite, [25.0, 75.0])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    out[np.isfinite(values)] = (finite - median) / scale
    return out


def sym(matrix, mode):
    values = np.asarray(matrix, dtype=np.float64)
    if str(mode) == "max":
        out = np.maximum(values, values.T)
    elif str(mode) == "mean":
        out = 0.5 * (values + values.T)
    else:
        raise ValueError(f"unknown score_adjacency_sym={mode!r}")
    np.fill_diagonal(out, np.nan)
    return out


def anti(matrix):
    values = np.asarray(matrix, dtype=np.float64)
    out = 0.5 * (values - values.T)
    out[~np.isfinite(out)] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


def affinity_from_adjacency(adjacency, *, threshold_percentile, transform, temperature):
    z = robust_z(adjacency)
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        raise ValueError("fixed-head spectral attention adjacency has no finite entries")
    threshold = float(np.percentile(finite, float(threshold_percentile)))
    shifted = z - threshold
    if str(transform) == "relu":
        affinity = np.maximum(shifted, 0.0)
    elif str(transform) == "exp":
        affinity = np.exp(np.clip(shifted / max(float(temperature), 1e-6), -20.0, 20.0))
        affinity[shifted < 0.0] = 0.0
    elif str(transform) == "softplus":
        affinity = np.log1p(np.exp(np.clip(shifted / max(float(temperature), 1e-6), -20.0, 20.0)))
        affinity[shifted < 0.0] *= 0.1
    else:
        raise ValueError(f"unknown transform={transform!r}")
    affinity = np.where(np.isfinite(affinity), affinity, 0.0)
    affinity = 0.5 * (affinity + affinity.T)
    np.fill_diagonal(affinity, 0.0)
    return affinity


def spectral_coordinates(affinity, *, num_components):
    affinity = np.asarray(affinity, dtype=np.float64)
    degree = affinity.sum(axis=1)
    if float(degree.max()) <= 0.0:
        raise ValueError("fixed-head spectral affinity graph is empty")
    degree = np.maximum(degree, 1e-8)
    inv_sqrt = 1.0 / np.sqrt(degree)
    norm_adj = affinity * inv_sqrt[:, None] * inv_sqrt[None, :]
    eigvals, eigvecs = np.linalg.eigh(norm_adj)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    start = 1 if eigvecs.shape[1] > 1 else 0
    end = min(eigvecs.shape[1], start + int(num_components))
    coords = eigvecs[:, start:end] * eigvals[start:end][None, :]
    if coords.shape[1] < int(num_components):
        coords = np.pad(coords, ((0, 0), (0, int(num_components) - coords.shape[1])), mode="constant")
    coords = coords - coords.mean(axis=0, keepdims=True)
    scale = coords.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return coords / scale, eigvals[: min(8, eigvals.shape[0])]


def candidate_axes(coords, component_pairs, num_angles):
    for a, b in component_pairs:
        if a < 0 or b < 0 or a >= coords.shape[1] or b >= coords.shape[1] or a == b:
            continue
        plane = coords[:, [a, b]]
        for angle_idx, theta in enumerate(np.linspace(0.0, math.pi, int(num_angles), endpoint=False)):
            direction = np.asarray([math.cos(float(theta)), math.sin(float(theta))], dtype=np.float64)
            axis = plane @ direction
            yield (
                f"c{a + 1}{b + 1}_ang{angle_idx:03d}",
                axis,
                {
                    "component_pair": [int(a + 1), int(b + 1)],
                    "angle_index": int(angle_idx),
                    "angle_radians": float(theta),
                },
            )


def equal_count_groups(primary, k, reverse):
    sign = -1.0 if bool(reverse) else 1.0
    order = sorted(range(primary.shape[0]), key=lambda idx: (sign * float(primary[idx]), int(idx)))
    groups = []
    n = len(order)
    for group_idx in range(int(k)):
        start = (group_idx * n) // int(k)
        end = ((group_idx + 1) * n) // int(k)
        if end > start:
            groups.append(order[start:end])
    return groups


def largest_gap_groups(primary, k, reverse):
    sign = -1.0 if bool(reverse) else 1.0
    order = sorted(range(primary.shape[0]), key=lambda idx: (sign * float(primary[idx]), int(idx)))
    if int(k) <= 1:
        return [order]
    vals = np.asarray([float(primary[idx]) for idx in order], dtype=np.float64)
    gaps = np.abs(np.diff(vals))
    if gaps.shape[0] < int(k) - 1:
        return equal_count_groups(primary, k, reverse)
    cut_positions = sorted(int(value) + 1 for value in np.argsort(-gaps)[: int(k) - 1])
    groups = []
    start = 0
    for cut in cut_positions + [len(order)]:
        if cut > start:
            groups.append(order[start:cut])
        start = cut
    return groups


def band_quality(primary, groups):
    vals = np.asarray(primary, dtype=np.float64)
    vals = (vals - float(vals.mean())) / max(float(vals.std()), 1e-8)
    centers = []
    within = 0.0
    count = 0
    for group in groups:
        if not group:
            continue
        g = vals[group]
        center = float(g.mean())
        centers.append(center)
        within += float(np.square(g - center).sum())
        count += int(len(group))
    if len(centers) < 2:
        return 0.0
    sep = float(np.mean(np.abs(np.diff(sorted(centers)))))
    spread = math.sqrt(max(0.0, within / max(1, count)))
    return float(sep / (0.05 + spread))


def grouped_order(
    primary,
    secondary,
    *,
    k,
    group_method,
    primary_reverse,
    secondary_reverse,
    group_order_reverse,
):
    if str(group_method) == "equal":
        groups = equal_count_groups(primary, int(k), primary_reverse)
    elif str(group_method) == "gap":
        groups = largest_gap_groups(primary, int(k), primary_reverse)
    else:
        raise ValueError(f"unknown group_method={group_method!r}")
    if bool(group_order_reverse):
        groups = list(reversed(groups))
    sign = -1.0 if bool(secondary_reverse) else 1.0
    ordered_groups = []
    order = []
    for group in groups:
        ordered = sorted(group, key=lambda idx: (sign * float(secondary[idx]), int(idx)))
        ordered_groups.append([int(value) for value in ordered])
        order.extend(ordered)
    if sorted(order) != list(range(primary.shape[0])):
        raise ValueError("generated order is not a complete block permutation")
    return [int(value) for value in order], ordered_groups, band_quality(primary, groups)


def path_score(order, adjacency, directed, direction_lambda):
    values = []
    for first, second in zip(order[:-1], order[1:]):
        adj_value = float(adjacency[int(first), int(second)])
        if not np.isfinite(adj_value):
            adj_value = 0.0
        dir_value = float(directed[int(first), int(second)])
        if not np.isfinite(dir_value):
            dir_value = 0.0
        values.append(adj_value + float(direction_lambda) * dir_value)
    return float(np.mean(values)) if values else 0.0


def score_order(order, adjacency, directed_variants, direction_lambdas, *, directed_weight, band_score, band_weight):
    adj_part = path_score(order, adjacency, np.zeros_like(adjacency), 0.0)
    best_directed = float("-inf")
    best_name = ""
    for directed_name, directed in directed_variants.items():
        for direction_lambda in direction_lambdas:
            value = path_score(order, adjacency, directed, float(direction_lambda))
            if value > best_directed:
                best_directed = float(value)
                best_name = f"{directed_name}_dl{float(direction_lambda):g}"
    if not np.isfinite(best_directed):
        best_directed = 0.0
    score = float(adj_part + float(directed_weight) * best_directed + float(band_weight) * float(band_score))
    return score, {
        "adjacency_path_score": float(adj_part),
        "best_directed_path_score": float(best_directed),
        "best_directed_name": best_name,
    }


def _to_numpy_matrix(matrix):
    if torch.is_tensor(matrix):
        values = matrix.detach().float().cpu().numpy()
    else:
        values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"expected a square attention matrix, got shape={tuple(values.shape)}")
    return np.asarray(values, dtype=np.float64).copy()
