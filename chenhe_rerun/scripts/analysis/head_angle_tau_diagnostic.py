"""
Diagnose head/angle/sample instability for online spectral order recovery.

This script evaluates a trained AO-GPT checkpoint without changing training.
It mines current-frame block attention matrices from probe batches, recovers
attention-spectral candidate orders, selects top-1 by current-frame attention
score, and reports original-frame L2R Kendall tau only as a diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from statistics import median

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AOGPT import AOGPT, AOGPTConfig  # noqa: E402
from online_spectral_order_policy import (  # noqa: E402
    FixedHeadSpectralPolicyConfig,
    affinity_from_adjacency,
    anti,
    grouped_order,
    parse_floats,
    parse_ints,
    parse_pairs,
    robust_z,
    score_order,
    spectral_coordinates,
    sym,
)
from order_utils import (  # noqa: E402
    block_permutation_to_token_permutation,
    build_fixed_block_permutation,
    expand_block_orders_to_token_orders,
    invert_permutation,
)


DEFAULT_CKPT = (
    "out/base/permute/seq256/block64/"
    "out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt"
)
DEFAULT_OUT = "Report/analysis/wikitext103_perm_b64_head_angle_tau_diagnostic"


def parse_csv_ints(text: str) -> list[int]:
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            out.extend(range(int(left), int(right) + 1))
        else:
            out.append(int(item))
    return out


def parse_heads(text: str, num_layers: int, num_heads: int, max_heads: int = 0) -> list[tuple[int, int]]:
    if str(text).strip().lower() in {"", "all"}:
        pairs = [(layer, head) for layer in range(num_layers) for head in range(num_heads)]
    else:
        pairs = []
        for item in str(text).split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                layer_text, head_text = item.split(":", 1)
                pairs.append((int(layer_text), int(head_text)))
            else:
                head = int(item)
                for layer in range(num_layers):
                    pairs.append((layer, head))
    cleaned = []
    seen = set()
    for layer, head in pairs:
        if layer < 0 or layer >= num_layers or head < 0 or head >= num_heads:
            raise ValueError(f"head spec {(layer, head)} outside model shape {num_layers}x{num_heads}")
        key = (int(layer), int(head))
        if key not in seen:
            seen.add(key)
            cleaned.append(key)
    if int(max_heads) > 0:
        cleaned = cleaned[: int(max_heads)]
    return cleaned


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu")


def build_model(checkpoint: dict, device: str):
    model = AOGPT(AOGPTConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def infer_data_record_mode(data_dir: Path, fallback: str = "stream") -> str:
    meta_path = data_dir / "meta.pkl"
    if not meta_path.exists():
        return str(fallback)
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)
    return str(meta.get("data_record_mode", fallback))


class TokenArray:
    def __init__(self, values, data_record_mode: str):
        self.values = values
        self.data_record_mode = str(data_record_mode)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return self.values[item]


def resolve_data_dir(args, checkpoint: dict) -> Path:
    if args.data_dir is not None:
        return Path(args.data_dir)
    dataset = args.dataset or checkpoint.get("config", {}).get("dataset")
    if dataset is None:
        raise ValueError("Could not infer dataset. Pass --dataset or --data_dir.")
    return REPO_ROOT / "data" / str(dataset)


def load_tokens(data_dir: Path, split: str, checkpoint: dict):
    split_path = data_dir / f"{split}.bin"
    if not split_path.exists():
        raise FileNotFoundError(f"Could not find split file: {split_path}")
    mode = str(checkpoint.get("config", {}).get("data_record_mode") or infer_data_record_mode(data_dir))
    return TokenArray(np.memmap(split_path, dtype=np.uint16, mode="r"), data_record_mode=mode)


def sample_batch(tokens, batch_size: int, block_size: int, rng, device: str, token_perm=None):
    data_record_mode = str(getattr(tokens, "data_record_mode", "stream"))
    if data_record_mode == "fixed":
        num_records = len(tokens) // int(block_size)
        if num_records <= 0:
            raise ValueError("Dataset split is shorter than one fixed record.")
        starts = (rng.integers(0, num_records, size=int(batch_size)) * int(block_size)).tolist()
    elif data_record_mode == "stream":
        max_start = len(tokens) - int(block_size)
        if max_start <= 0:
            raise ValueError("Dataset split is shorter than block_size.")
        starts = rng.integers(0, max_start, size=int(batch_size)).tolist()
    else:
        raise ValueError(f"Unsupported data_record_mode={data_record_mode!r}")
    batch = torch.stack(
        [torch.from_numpy(np.asarray(tokens[start : start + int(block_size)], dtype=np.int64)) for start in starts]
    )
    batch = batch.to(device)
    if token_perm is not None:
        batch = batch.index_select(1, token_perm.to(device=device, dtype=torch.long))
    return batch


def get_autocast_context(device: str, dtype: str):
    if "cuda" not in str(device) or str(dtype) == "float32":
        return nullcontext()
    return torch.amp.autocast(
        device_type="cuda",
        dtype={"float16": torch.float16, "bfloat16": torch.bfloat16}[str(dtype)],
    )


def load_permutation_state(checkpoint: dict, model):
    config = checkpoint.get("config", {})
    if not bool(config.get("permute_data", False)):
        return None
    perm_state = checkpoint.get("data_permutation") or {}
    if perm_state.get("block_perm") is not None:
        block_perm = torch.tensor(perm_state["block_perm"], dtype=torch.long)
    else:
        block_perm = build_fixed_block_permutation(model.num_blocks, int(config.get("permute_seed", 42)))
    inverse_block_perm = invert_permutation(block_perm)
    token_perm = block_permutation_to_token_permutation(
        block_perm,
        block_len=int(model.block_order_block_len),
        block_order_layout=str(getattr(model, "block_order_layout", "contiguous")),
        image_size=int(getattr(model, "image_size", 0)),
        image_block_size=int(getattr(model, "image_block_size", 0)),
        image_block_height=int(getattr(model, "image_block_height", 0)),
        image_block_width=int(getattr(model, "image_block_width", 0)),
    )
    return {
        "block_perm": block_perm,
        "inverse_block_perm": inverse_block_perm,
        "token_perm": token_perm,
    }


def build_probe_orders(model, batch_size: int, device: str, mode: str, generator: torch.Generator):
    if str(mode) == "AR":
        block_orders = torch.arange(model.num_blocks, device=device).unsqueeze(0).expand(int(batch_size), -1)
    elif str(mode) == "Random":
        block_orders = torch.stack(
            [
                torch.randperm(model.num_blocks, generator=generator, device=device)
                for _ in range(int(batch_size))
            ],
            dim=0,
        )
    else:
        raise ValueError(f"Unsupported reveal mode={mode!r}")
    token_orders = expand_block_orders_to_token_orders(
        block_orders,
        block_len=int(model.block_order_block_len),
        block_order_layout=str(getattr(model, "block_order_layout", "contiguous")),
        image_size=int(getattr(model, "image_size", 0)),
        image_block_size=int(getattr(model, "image_block_size", 0)),
        image_block_height=int(getattr(model, "image_block_height", 0)),
        image_block_width=int(getattr(model, "image_block_width", 0)),
    )
    return token_orders, block_orders


def extract_attentions_from_outputs(outputs):
    if not isinstance(outputs, (tuple, list)):
        raise RuntimeError("Expected model(..., return_attentions=True) to return a tuple/list.")
    for candidate in reversed(outputs[2:]):
        if isinstance(candidate, (list, tuple)) and candidate:
            if all(torch.is_tensor(item) and item.ndim == 4 for item in candidate):
                return candidate
    raise RuntimeError("Could not find attention tensors in model outputs.")


def aggregate_layer_attention_to_block(layer_attn: torch.Tensor, block_len: int, export_type: str) -> torch.Tensor:
    if str(export_type) == "with_none":
        shifted = layer_attn[:, :, :-1, :-1]
    elif str(export_type) == "without_none":
        shifted = layer_attn[:, :, 1:, 1:]
    else:
        raise ValueError(f"unknown export_type={export_type!r}")
    batch, heads, seq_a, seq_b = shifted.shape
    if seq_a != seq_b or seq_a % int(block_len) != 0:
        raise ValueError(f"attention shape {tuple(shifted.shape)} is incompatible with block_len={block_len}")
    num_blocks = seq_a // int(block_len)
    return shifted.view(batch, heads, num_blocks, int(block_len), num_blocks, int(block_len)).mean(dim=(3, 5))


@torch.no_grad()
def mine_attention_matrices(
    model,
    tokens,
    *,
    sample_count: int,
    batch_size: int,
    split_seed: int,
    order_seed: int,
    reveal_mode: str,
    token_perm,
    device: str,
    ctx,
    export_type: str,
):
    np_rng = np.random.default_rng(int(split_seed))
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(int(order_seed))
    matrices_sum = None
    total_samples = 0
    while total_samples < int(sample_count):
        local_batch_size = min(int(batch_size), int(sample_count) - int(total_samples))
        idx = sample_batch(
            tokens,
            local_batch_size,
            int(model.config.block_size),
            np_rng,
            device,
            token_perm=token_perm,
        )
        token_orders, block_orders = build_probe_orders(
            model,
            int(idx.size(0)),
            device,
            reveal_mode,
            torch_gen,
        )
        with ctx:
            outputs = model(
                idx,
                mode=None,
                orders=token_orders,
                return_attentions=True,
                return_logits=False,
            )
        attn_outputs = extract_attentions_from_outputs(outputs)
        if matrices_sum is None:
            matrices_sum = torch.zeros(
                (len(attn_outputs), int(attn_outputs[0].size(1)), model.num_blocks, model.num_blocks),
                dtype=torch.float64,
                device=device,
            )
        for layer_idx, layer_attn in enumerate(attn_outputs):
            block_batch = aggregate_layer_attention_to_block(
                layer_attn.detach().float(),
                block_len=int(model.block_order_block_len),
                export_type=export_type,
            )
            for sample_idx in range(block_batch.size(0)):
                inverse = invert_permutation(block_orders[sample_idx].detach()).to(device=block_batch.device)
                matrices_sum[layer_idx] += block_batch[sample_idx][:, inverse, :][:, :, inverse].double()
        total_samples += int(idx.size(0))
    if matrices_sum is None or total_samples <= 0:
        raise RuntimeError("No attention matrices were mined.")
    matrices = (matrices_sum / float(total_samples)).detach().cpu().numpy()
    for layer_idx in range(matrices.shape[0]):
        for head_idx in range(matrices.shape[1]):
            np.fill_diagonal(matrices[layer_idx, head_idx], np.nan)
    return matrices


def candidate_axes_for_indices(coords: np.ndarray, component_pairs, num_angles: int, angle_indices: list[int]):
    for a, b in component_pairs:
        if a < 0 or b < 0 or a >= coords.shape[1] or b >= coords.shape[1] or a == b:
            continue
        plane = coords[:, [a, b]]
        for angle_idx in angle_indices:
            theta = float(angle_idx) * math.pi / float(num_angles)
            direction = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
            axis = plane @ direction
            yield (
                f"c{a + 1}{b + 1}_ang{int(angle_idx):03d}",
                axis,
                {
                    "component_pair": [int(a + 1), int(b + 1)],
                    "angle_index": int(angle_idx),
                    "angle_radians": float(theta),
                },
            )


def recover_candidates_for_angles(
    matrix,
    config: FixedHeadSpectralPolicyConfig,
    *,
    num_angles: int,
    angle_indices: list[int],
    top_m: int,
):
    values = np.asarray(matrix, dtype=np.float64).copy()
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
    axes = list(candidate_axes_for_indices(coords, parse_pairs(config.component_pairs), int(num_angles), angle_indices))
    if not axes:
        raise ValueError("no spectral axes were produced")
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
                                candidates.append(
                                    {
                                        "name": (
                                            f"{primary_name}_x_{secondary_name}_k{int(k):02d}_{group_method}"
                                            f"_p{'d' if primary_reverse else 'a'}"
                                            f"_s{'d' if secondary_reverse else 'a'}"
                                            f"_g{'d' if group_order_reverse else 'a'}"
                                        ),
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
                                            "eigvals": [float(value) for value in eigvals.tolist()],
                                            "num_unique_candidates": int(len(seen)),
                                            **score_meta,
                                        },
                                    }
                                )
    if not candidates:
        raise ValueError("no candidates were produced")
    candidates.sort(key=lambda item: (-float(item["score"]), str(item["name"])))
    for candidate in candidates:
        candidate["meta"]["num_unique_candidates"] = int(len(seen))
    return candidates[: int(top_m)], eigvals


def inversions(values: list[int]) -> int:
    vals = [int(value) for value in values]
    if len(vals) < 2:
        return 0
    ranks = {value: idx + 1 for idx, value in enumerate(sorted(set(vals)))}
    tree = [0] * (len(ranks) + 1)

    def add(index: int, amount: int) -> None:
        while index < len(tree):
            tree[index] += amount
            index += index & -index

    def prefix_sum(index: int) -> int:
        total = 0
        while index > 0:
            total += tree[index]
            index -= index & -index
        return total

    count = 0
    seen = 0
    for value in vals:
        rank = ranks[value]
        count += seen - prefix_sum(rank)
        add(rank, 1)
        seen += 1
    return count


def tau_to_l2r(order: list[int]) -> float:
    n = len(order)
    total = n * (n - 1) // 2
    if total <= 0:
        return 1.0
    return float(1.0 - 2.0 * float(inversions(order)) / float(total))


def kendall_between_orders(order_a: list[int], order_b: list[int]) -> float:
    if len(order_a) != len(order_b) or sorted(order_a) != sorted(order_b):
        raise ValueError("orders must contain the same items")
    rank_b = {int(value): idx for idx, value in enumerate(order_b)}
    projected = [rank_b[int(value)] for value in order_a]
    return tau_to_l2r(projected)


def order_to_original(order_current: list[int], block_perm):
    if block_perm is None:
        return [int(value) for value in order_current]
    return [int(block_perm[int(value)].item()) for value in order_current]


def matrix_stats(matrix: np.ndarray) -> dict:
    finite = np.asarray(matrix, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "attention_mean": float("nan"),
            "attention_std": float("nan"),
            "attention_fro_norm": float("nan"),
            "attention_entropy": float("nan"),
        }
    positive = np.maximum(finite, 0.0)
    total = float(positive.sum())
    if total <= 0.0:
        entropy = float("nan")
    else:
        probs = positive / total
        entropy = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())
    return {
        "attention_mean": float(np.mean(finite)),
        "attention_std": float(np.std(finite)),
        "attention_fro_norm": float(np.linalg.norm(np.nan_to_num(matrix, nan=0.0))),
        "attention_entropy": float(entropy),
    }


def entropy_from_counts(values) -> float:
    counts = Counter(values)
    total = float(sum(counts.values()))
    if total <= 0.0:
        return float("nan")
    probs = [count / total for count in counts.values()]
    return float(-sum(p * math.log(max(p, 1e-12)) for p in probs))


def row_base(
    *,
    split: str,
    probe_seed: int,
    order_seed: int,
    repeat_idx: int,
    probe_budget: int,
    layer: int,
    head: int,
    angle_mode: str,
    fixed_angle_idx: int,
):
    return {
        "split": split,
        "probe_seed": int(probe_seed),
        "order_seed": int(order_seed),
        "repeat_idx": int(repeat_idx),
        "probe_budget": int(probe_budget),
        "layer": int(layer),
        "head": int(head),
        "angle_mode": str(angle_mode),
        "fixed_angle_idx": int(fixed_angle_idx),
    }


def summarize_candidate_set(candidates, block_perm, top_k: int):
    candidate_rows = []
    taus = []
    best_tau = float("-inf")
    best_tau_rank = -1
    for rank, candidate in enumerate(candidates[: int(top_k)], start=1):
        order_current = [int(value) for value in candidate["order"]]
        order_original = order_to_original(order_current, block_perm)
        tau_original = tau_to_l2r(order_original)
        tau_current = tau_to_l2r(order_current)
        taus.append(float(tau_original))
        if float(tau_original) > best_tau:
            best_tau = float(tau_original)
            best_tau_rank = int(rank)
        meta = candidate.get("meta", {})
        candidate_rows.append(
            {
                "candidate_rank": int(rank),
                "candidate_name": str(candidate.get("name", "")),
                "candidate_score": float(candidate.get("score", float("nan"))),
                "candidate_tau_origin_l2r": float(tau_original),
                "candidate_tau_current_l2r": float(tau_current),
                "candidate_primary_angle_idx": int(
                    (meta.get("primary_axis") or {}).get("angle_index", -1)
                ),
                "candidate_secondary_angle_idx": int(
                    (meta.get("secondary_axis") or {}).get("angle_index", -1)
                ),
                "candidate_k": int(meta.get("k", -1)),
                "candidate_primary_reverse": bool(meta.get("primary_reverse", False)),
                "candidate_secondary_reverse": bool(meta.get("secondary_reverse", False)),
                "candidate_group_order_reverse": bool(meta.get("group_order_reverse", False)),
                "candidate_adjacency_path_score": float(meta.get("adjacency_path_score", float("nan"))),
                "candidate_directed_path_score": float(meta.get("best_directed_path_score", float("nan"))),
                "candidate_band_quality": float(meta.get("band_quality", float("nan"))),
                "candidate_order_current": json.dumps(order_current, separators=(",", ":")),
                "candidate_order_original": json.dumps(order_original, separators=(",", ":")),
            }
        )
    tau_arr = np.asarray(taus, dtype=np.float64)
    if tau_arr.size == 0:
        tau_mean = tau_std = tau_min = tau_max = float("nan")
    else:
        tau_mean = float(np.mean(tau_arr))
        tau_std = float(np.std(tau_arr))
        tau_min = float(np.min(tau_arr))
        tau_max = float(np.max(tau_arr))
    return candidate_rows, {
        "best_tau_in_topk": float(best_tau),
        "best_tau_rank_by_score": int(best_tau_rank),
        "topk_tau_mean": float(tau_mean),
        "topk_tau_std": float(tau_std),
        "topk_tau_min": float(tau_min),
        "topk_tau_max": float(tau_max),
    }


def evaluate_one_mode(
    *,
    matrix,
    config,
    num_angles: int,
    angle_indices: list[int],
    top_m: int,
    block_perm,
):
    candidates, eigvals = recover_candidates_for_angles(
        matrix,
        config,
        num_angles=int(num_angles),
        angle_indices=angle_indices,
        top_m=int(top_m),
    )
    top = candidates[0]
    top_meta = top.get("meta", {})
    top_order_current = [int(value) for value in top["order"]]
    top_order_original = order_to_original(top_order_current, block_perm)
    score_margin = float("nan")
    if len(candidates) > 1:
        score_margin = float(float(candidates[0]["score"]) - float(candidates[1]["score"]))
    candidate_rows, candidate_summary = summarize_candidate_set(candidates, block_perm, int(top_m))
    eigvals_list = [float(value) for value in np.asarray(eigvals).tolist()]
    def eig_gap(i, j):
        if len(eigvals_list) <= max(i, j):
            return float("nan")
        return float(eigvals_list[i] - eigvals_list[j])
    result = {
        "top1_name": str(top.get("name", "")),
        "top1_score": float(top.get("score", float("nan"))),
        "score_margin_top1_top2": float(score_margin),
        "top1_tau_origin_l2r": float(tau_to_l2r(top_order_original)),
        "top1_tau_current_l2r": float(tau_to_l2r(top_order_current)),
        "selected_primary_angle_idx": int((top_meta.get("primary_axis") or {}).get("angle_index", -1)),
        "selected_secondary_angle_idx": int((top_meta.get("secondary_axis") or {}).get("angle_index", -1)),
        "selected_k": int(top_meta.get("k", -1)),
        "selected_group_method": str(top_meta.get("group_method", "")),
        "selected_primary_reverse": bool(top_meta.get("primary_reverse", False)),
        "selected_secondary_reverse": bool(top_meta.get("secondary_reverse", False)),
        "selected_group_order_reverse": bool(top_meta.get("group_order_reverse", False)),
        "selected_adjacency_path_score": float(top_meta.get("adjacency_path_score", float("nan"))),
        "selected_directed_path_score": float(top_meta.get("best_directed_path_score", float("nan"))),
        "selected_band_quality": float(top_meta.get("band_quality", float("nan"))),
        "num_unique_candidates": int(top_meta.get("num_unique_candidates", len(candidates))),
        "eig_gap_0_1": eig_gap(0, 1),
        "eig_gap_1_2": eig_gap(1, 2),
        "eig_gap_2_3": eig_gap(2, 3),
        "eigvals_first8": json.dumps(eigvals_list, separators=(",", ":")),
        "top1_order_current": json.dumps(top_order_current, separators=(",", ":")),
        "top1_order_original": json.dumps(top_order_original, separators=(",", ":")),
    }
    result.update(candidate_summary)
    return result, candidate_rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    arr = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))


def build_group_summaries(rows: list[dict], tau_threshold: float):
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["layer"],
                row["head"],
                row["angle_mode"],
                row["probe_budget"],
                row["fixed_angle_idx"],
            )
        ].append(row)
    summary = []
    for key, items in sorted(groups.items()):
        layer, head, angle_mode, probe_budget, fixed_angle_idx = key
        tau_values = [float(row["top1_tau_origin_l2r"]) for row in items]
        best_values = [float(row["best_tau_in_topk"]) for row in items]
        margin_values = [float(row["score_margin_top1_top2"]) for row in items]
        tau_mean, tau_std, tau_min, tau_max = mean_std(tau_values)
        best_mean, best_std, _, _ = mean_std(best_values)
        finite_margins = [value for value in margin_values if math.isfinite(value)]
        selected_primary = [int(row["selected_primary_angle_idx"]) for row in items]
        selected_secondary = [int(row["selected_secondary_angle_idx"]) for row in items]
        unique_orders = len(set(str(row["top1_order_current"]) for row in items))
        high_tau_rate = float(np.mean([float(value) >= float(tau_threshold) for value in tau_values])) if tau_values else float("nan")
        summary.append(
            {
                "layer": int(layer),
                "head": int(head),
                "angle_mode": str(angle_mode),
                "probe_budget": int(probe_budget),
                "fixed_angle_idx": int(fixed_angle_idx),
                "num_repeats": int(len(items)),
                "top1_tau_mean": float(tau_mean),
                "top1_tau_std": float(tau_std),
                "top1_tau_min": float(tau_min),
                "top1_tau_max": float(tau_max),
                "top1_tau_range": float(tau_max - tau_min) if math.isfinite(tau_max) and math.isfinite(tau_min) else float("nan"),
                "best_tau_in_topk_mean": float(best_mean),
                "best_tau_in_topk_std": float(best_std),
                "score_margin_median": float(median(finite_margins)) if finite_margins else float("nan"),
                "selected_primary_angle_entropy": float(entropy_from_counts(selected_primary)),
                "selected_secondary_angle_entropy": float(entropy_from_counts(selected_secondary)),
                "top1_order_unique_count": int(unique_orders),
                "stable_high_tau_rate": float(high_tau_rate),
            }
        )
    return summary


def build_budget_summary(rows: list[dict]):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["angle_mode"], row["probe_budget"])].append(row)
    summary = []
    for (angle_mode, probe_budget), items in sorted(groups.items()):
        tau_mean, tau_std, tau_min, tau_max = mean_std([row["top1_tau_origin_l2r"] for row in items])
        best_mean, best_std, _, _ = mean_std([row["best_tau_in_topk"] for row in items])
        summary.append(
            {
                "angle_mode": str(angle_mode),
                "probe_budget": int(probe_budget),
                "num_rows": int(len(items)),
                "top1_tau_mean": float(tau_mean),
                "top1_tau_std": float(tau_std),
                "top1_tau_min": float(tau_min),
                "top1_tau_max": float(tau_max),
                "best_tau_in_topk_mean": float(best_mean),
                "best_tau_in_topk_std": float(best_std),
            }
        )
    return summary


def build_angle_summary(rows: list[dict]):
    groups = defaultdict(list)
    for row in rows:
        if str(row["angle_mode"]) not in {"fixed", "calibrated"}:
            continue
        groups[(row["layer"], row["head"], row["probe_budget"], row["fixed_angle_idx"], row["angle_mode"])].append(row)
    summary = []
    for key, items in sorted(groups.items()):
        layer, head, probe_budget, fixed_angle_idx, angle_mode = key
        tau_mean, tau_std, tau_min, tau_max = mean_std([row["top1_tau_origin_l2r"] for row in items])
        summary.append(
            {
                "layer": int(layer),
                "head": int(head),
                "angle_mode": str(angle_mode),
                "probe_budget": int(probe_budget),
                "fixed_angle_idx": int(fixed_angle_idx),
                "num_repeats": int(len(items)),
                "top1_tau_mean": float(tau_mean),
                "top1_tau_std": float(tau_std),
                "top1_tau_min": float(tau_min),
                "top1_tau_max": float(tau_max),
            }
        )
    return summary


def build_selected_angle_histograms(rows: list[dict]):
    groups = defaultdict(Counter)
    for row in rows:
        groups[(row["layer"], row["head"], row["angle_mode"], row["probe_budget"], "primary")][
            int(row["selected_primary_angle_idx"])
        ] += 1
        groups[(row["layer"], row["head"], row["angle_mode"], row["probe_budget"], "secondary")][
            int(row["selected_secondary_angle_idx"])
        ] += 1
    out = []
    for key, counts in sorted(groups.items()):
        layer, head, angle_mode, probe_budget, axis_role = key
        total = sum(counts.values())
        for angle_idx, count in sorted(counts.items()):
            out.append(
                {
                    "layer": int(layer),
                    "head": int(head),
                    "angle_mode": str(angle_mode),
                    "probe_budget": int(probe_budget),
                    "axis_role": str(axis_role),
                    "angle_idx": int(angle_idx),
                    "count": int(count),
                    "prob": float(count / max(1, total)),
                }
            )
    return out


def build_repeat_stability(rows: list[dict]):
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["layer"],
                row["head"],
                row["angle_mode"],
                row["probe_budget"],
                row["fixed_angle_idx"],
            )
        ].append(row)
    out = []
    for key, items in sorted(groups.items()):
        layer, head, angle_mode, probe_budget, fixed_angle_idx = key
        if len(items) < 2:
            continue
        current_taus = []
        original_taus = []
        for i in range(len(items)):
            order_i_current = json.loads(items[i]["top1_order_current"])
            order_i_original = json.loads(items[i]["top1_order_original"])
            for j in range(i + 1, len(items)):
                order_j_current = json.loads(items[j]["top1_order_current"])
                order_j_original = json.loads(items[j]["top1_order_original"])
                current_taus.append(kendall_between_orders(order_i_current, order_j_current))
                original_taus.append(kendall_between_orders(order_i_original, order_j_original))
        cur_mean, cur_std, cur_min, cur_max = mean_std(current_taus)
        ori_mean, ori_std, ori_min, ori_max = mean_std(original_taus)
        out.append(
            {
                "layer": int(layer),
                "head": int(head),
                "angle_mode": str(angle_mode),
                "probe_budget": int(probe_budget),
                "fixed_angle_idx": int(fixed_angle_idx),
                "num_repeats": int(len(items)),
                "num_pairs": int(len(current_taus)),
                "pairwise_order_tau_current_mean": float(cur_mean),
                "pairwise_order_tau_current_std": float(cur_std),
                "pairwise_order_tau_current_min": float(cur_min),
                "pairwise_order_tau_current_max": float(cur_max),
                "pairwise_order_tau_original_mean": float(ori_mean),
                "pairwise_order_tau_original_std": float(ori_std),
                "pairwise_order_tau_original_min": float(ori_min),
                "pairwise_order_tau_original_max": float(ori_max),
            }
        )
    return out


def maybe_make_plots(out_dir: Path, rows: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[head-angle-diagnostic] plot import failed: {exc}")
        return
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows = [row for row in rows if str(row["angle_mode"]) == "sweep16"]
    if sweep_rows:
        points = defaultdict(list)
        for row in sweep_rows:
            points[(int(row["layer"]), int(row["head"]))].append(float(row["top1_tau_origin_l2r"]))
        layers = sorted(set(layer for layer, _ in points))
        heads = sorted(set(head for _, head in points))
        heat = np.full((len(layers), len(heads)), np.nan)
        for i, layer in enumerate(layers):
            for j, head in enumerate(heads):
                vals = points.get((layer, head), [])
                if vals:
                    heat[i, j] = float(np.mean(vals))
        fig, ax = plt.subplots(figsize=(max(6, len(heads) * 0.6), max(4, len(layers) * 0.6)))
        im = ax.imshow(heat, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(heads)), labels=[str(h) for h in heads])
        ax.set_yticks(range(len(layers)), labels=[str(l) for l in layers])
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title("Sweep16 top1 tau mean vs original L2R")
        fig.colorbar(im, ax=ax)
        fig.savefig(plots_dir / "head_top1_tau_mean_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            [float(row["score_margin_top1_top2"]) for row in sweep_rows],
            [float(row["top1_tau_origin_l2r"]) for row in sweep_rows],
            s=14,
            alpha=0.7,
        )
        ax.set_xlabel("score margin top1-top2")
        ax.set_ylabel("top1 tau vs original L2R")
        ax.set_title("Sweep16 tau vs score margin")
        ax.grid(alpha=0.25)
        fig.savefig(plots_dir / "tau_vs_score_margin.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            [float(row["eig_gap_1_2"]) for row in sweep_rows],
            [float(row["top1_tau_origin_l2r"]) for row in sweep_rows],
            s=14,
            alpha=0.7,
        )
        ax.set_xlabel("eig gap 1-2")
        ax.set_ylabel("top1 tau vs original L2R")
        ax.set_title("Sweep16 tau vs spectral gap")
        ax.grid(alpha=0.25)
        fig.savefig(plots_dir / "tau_vs_spectral_gap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def choose_calibrated_angles(
    *,
    model,
    tokens,
    args,
    ctx,
    token_perm,
    heads,
    policy_config,
    block_perm,
):
    print("[head-angle-diagnostic] mining calibration attention")
    matrices = mine_attention_matrices(
        model,
        tokens,
        sample_count=int(args.calibration_samples),
        batch_size=int(args.batch_size),
        split_seed=int(args.calibration_seed),
        order_seed=int(args.calibration_order_seed),
        reveal_mode=str(args.reveal_mode),
        token_perm=token_perm,
        device=str(args.device),
        ctx=ctx,
        export_type=str(args.attn_export_type),
    )
    calibrated = {}
    rows = []
    for layer, head in heads:
        best_angle = None
        best_score = float("-inf")
        best_tau = float("nan")
        for angle_idx in parse_csv_ints(args.fixed_angle_indices):
            result, _ = evaluate_one_mode(
                matrix=matrices[layer, head],
                config=policy_config,
                num_angles=int(args.num_angles),
                angle_indices=[int(angle_idx)],
                top_m=int(args.top_m),
                block_perm=block_perm,
            )
            score = float(result["top1_score"])
            if score > best_score:
                best_score = score
                best_angle = int(angle_idx)
                best_tau = float(result["top1_tau_origin_l2r"])
        calibrated[(int(layer), int(head))] = int(best_angle)
        rows.append(
            {
                "layer": int(layer),
                "head": int(head),
                "calibrated_angle_idx": int(best_angle),
                "calibration_top1_score": float(best_score),
                "calibration_top1_tau_origin_l2r_diagnostic": float(best_tau),
                "selection_rule": "max current-frame attention-side top1 score over fixed angles; tau is diagnostic only",
            }
        )
    return calibrated, rows


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(Path(args.ckpt_path))
    model = build_model(checkpoint, str(args.device))
    data_dir = resolve_data_dir(args, checkpoint)
    tokens = load_tokens(data_dir, str(args.split), checkpoint)
    permutation_state = load_permutation_state(checkpoint, model)
    token_perm = None if permutation_state is None else permutation_state["token_perm"]
    block_perm = None if permutation_state is None else permutation_state["block_perm"]
    ctx = get_autocast_context(str(args.device), str(args.dtype))

    heads = parse_heads(str(args.heads), int(model.config.n_layer), int(model.config.n_head), int(args.max_heads))
    budgets = parse_csv_ints(args.probe_budgets)
    fixed_angle_indices = parse_csv_ints(args.fixed_angle_indices)
    angle_modes = [item.strip() for item in str(args.angle_modes).split(",") if item.strip()]
    valid_modes = {"sweep16", "fixed", "calibrated"}
    bad_modes = [mode for mode in angle_modes if mode not in valid_modes]
    if bad_modes:
        raise ValueError(f"Unsupported angle modes: {bad_modes}. Choices: {sorted(valid_modes)}")

    policy_config = FixedHeadSpectralPolicyConfig(
        num_components=int(args.num_components),
        component_pairs=str(args.component_pairs),
        num_angles=int(args.num_angles),
        k_values=str(args.k_values),
        group_methods=str(args.group_methods),
        threshold_percentile=float(args.threshold_percentile),
        transform=str(args.transform),
        temperature=float(args.temperature),
        direction_lambdas=str(args.direction_lambdas),
        directed_score_weight=float(args.directed_score_weight),
        band_quality_weight=float(args.band_quality_weight),
        score_adjacency_sym=str(args.score_adjacency_sym),
    )

    config_payload = {
        "ckpt_path": str(args.ckpt_path),
        "checkpoint_iter": int(checkpoint.get("iter_num", -1)),
        "out_dir": str(out_dir),
        "dataset": str(args.dataset or checkpoint.get("config", {}).get("dataset")),
        "split": str(args.split),
        "data_dir": str(data_dir),
        "permute_data": bool(checkpoint.get("config", {}).get("permute_data", False)),
        "num_blocks": int(model.num_blocks),
        "block_order_block_len": int(model.block_order_block_len),
        "heads": [{"layer": int(layer), "head": int(head)} for layer, head in heads],
        "probe_budgets": [int(value) for value in budgets],
        "repeats": int(args.repeats),
        "angle_modes": angle_modes,
        "fixed_angle_indices": [int(value) for value in fixed_angle_indices],
        "num_angles": int(args.num_angles),
        "top_m": int(args.top_m),
        "batch_size": int(args.batch_size),
        "reveal_mode": str(args.reveal_mode),
        "attn_export_type": str(args.attn_export_type),
        "selection_rule": (
            "top1 is selected only by current-frame attention-spectral score; "
            "origin L2R tau and *_original orders are diagnostics only"
        ),
        "policy_config": vars(policy_config),
    }
    (out_dir / "config.json").write_text(json.dumps(json_safe(config_payload), indent=2), encoding="utf-8")

    calibrated_angles = {}
    calibration_rows = []
    if "calibrated" in angle_modes:
        calibrated_angles, calibration_rows = choose_calibrated_angles(
            model=model,
            tokens=tokens,
            args=args,
            ctx=ctx,
            token_perm=token_perm,
            heads=heads,
            policy_config=policy_config,
            block_perm=block_perm,
        )
        write_csv(out_dir / "calibration_angles.csv", calibration_rows)

    per_probe_rows = []
    per_candidate_rows = []
    for budget in budgets:
        for repeat_idx in range(int(args.repeats)):
            probe_seed = int(args.seed) + int(args.seed_stride) * int(repeat_idx) + int(budget) * 17
            order_seed = int(args.order_seed) + int(args.seed_stride) * int(repeat_idx) + int(budget) * 31
            print(
                "[head-angle-diagnostic] mining "
                f"budget={int(budget)} repeat={repeat_idx + 1}/{int(args.repeats)} "
                f"probe_seed={probe_seed} order_seed={order_seed}"
            )
            matrices = mine_attention_matrices(
                model,
                tokens,
                sample_count=int(budget),
                batch_size=int(args.batch_size),
                split_seed=probe_seed,
                order_seed=order_seed,
                reveal_mode=str(args.reveal_mode),
                token_perm=token_perm,
                device=str(args.device),
                ctx=ctx,
                export_type=str(args.attn_export_type),
            )
            for layer, head in heads:
                matrix = matrices[int(layer), int(head)]
                stats = matrix_stats(matrix)
                mode_specs = []
                if "sweep16" in angle_modes:
                    mode_specs.append(("sweep16", -1, list(range(int(args.num_angles)))))
                if "fixed" in angle_modes:
                    for angle_idx in fixed_angle_indices:
                        mode_specs.append(("fixed", int(angle_idx), [int(angle_idx)]))
                if "calibrated" in angle_modes:
                    angle_idx = calibrated_angles[(int(layer), int(head))]
                    mode_specs.append(("calibrated", int(angle_idx), [int(angle_idx)]))
                for angle_mode, fixed_angle_idx, angle_indices in mode_specs:
                    base = row_base(
                        split=str(args.split),
                        probe_seed=probe_seed,
                        order_seed=order_seed,
                        repeat_idx=repeat_idx,
                        probe_budget=int(budget),
                        layer=int(layer),
                        head=int(head),
                        angle_mode=str(angle_mode),
                        fixed_angle_idx=int(fixed_angle_idx),
                    )
                    try:
                        result, candidate_rows = evaluate_one_mode(
                            matrix=matrix,
                            config=policy_config,
                            num_angles=int(args.num_angles),
                            angle_indices=angle_indices,
                            top_m=int(args.top_m),
                            block_perm=block_perm,
                        )
                        row = dict(base)
                        row.update(stats)
                        row.update(result)
                        per_probe_rows.append(row)
                        if not bool(args.no_write_topk):
                            for candidate_row in candidate_rows:
                                c_row = dict(base)
                                c_row.update(candidate_row)
                                per_candidate_rows.append(c_row)
                    except Exception as exc:
                        row = dict(base)
                        row.update(stats)
                        row["error"] = str(exc)
                        per_probe_rows.append(row)
                        print(
                            "[head-angle-diagnostic] recovery failed "
                            f"layer={layer} head={head} mode={angle_mode} angle={fixed_angle_idx}: {exc}"
                        )
            write_csv(out_dir / "per_probe_results.csv", per_probe_rows)
            if not bool(args.no_write_topk):
                write_csv(out_dir / "per_candidate_topk.csv", per_candidate_rows)

    ok_rows = [row for row in per_probe_rows if "top1_tau_origin_l2r" in row]
    write_csv(out_dir / "per_probe_results.csv", per_probe_rows)
    if not bool(args.no_write_topk):
        write_csv(out_dir / "per_candidate_topk.csv", per_candidate_rows)
    write_csv(out_dir / "head_summary.csv", build_group_summaries(ok_rows, float(args.high_tau_threshold)))
    write_csv(out_dir / "budget_summary.csv", build_budget_summary(ok_rows))
    write_csv(out_dir / "angle_summary.csv", build_angle_summary(ok_rows))
    write_csv(out_dir / "selected_angle_histograms.csv", build_selected_angle_histograms(ok_rows))
    write_csv(out_dir / "head_repeat_stability.csv", build_repeat_stability(ok_rows))
    if not bool(args.no_plots):
        maybe_make_plots(out_dir, ok_rows)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "num_probe_rows": len(per_probe_rows),
                "num_candidate_rows": len(per_candidate_rows),
                "num_ok_rows": len(ok_rows),
                "per_probe_results": str(out_dir / "per_probe_results.csv"),
                "head_summary": str(out_dir / "head_summary.csv"),
            },
            indent=2,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", type=Path, default=Path(DEFAULT_CKPT))
    parser.add_argument("--out_dir", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
        choices=("float32", "float16", "bfloat16"),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--probe_budgets", type=str, default="64,256,1024,4096")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--order_seed", type=int, default=22345)
    parser.add_argument("--seed_stride", type=int, default=1009)
    parser.add_argument("--reveal_mode", type=str, default="Random", choices=("Random", "AR"))
    parser.add_argument("--attn_export_type", type=str, default="with_none", choices=("with_none", "without_none"))
    parser.add_argument("--heads", type=str, default="all")
    parser.add_argument("--max_heads", type=int, default=0)
    parser.add_argument("--angle_modes", type=str, default="sweep16,fixed")
    parser.add_argument("--fixed_angle_indices", type=str, default="0-15")
    parser.add_argument("--num_angles", type=int, default=16)
    parser.add_argument("--top_m", type=int, default=32)
    parser.add_argument("--no_write_topk", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--high_tau_threshold", type=float, default=0.5)
    parser.add_argument("--calibration_samples", type=int, default=4096)
    parser.add_argument("--calibration_seed", type=int, default=99123)
    parser.add_argument("--calibration_order_seed", type=int, default=99223)

    parser.add_argument("--num_components", type=int, default=4)
    parser.add_argument("--component_pairs", type=str, default="1-2")
    parser.add_argument("--k_values", type=str, default="8,10")
    parser.add_argument("--group_methods", type=str, default="gap")
    parser.add_argument("--threshold_percentile", type=float, default=60.0)
    parser.add_argument("--transform", type=str, default="relu", choices=("relu", "exp", "softplus"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--direction_lambdas", type=str, default="0,0.1,0.25")
    parser.add_argument("--directed_score_weight", type=float, default=0.25)
    parser.add_argument("--band_quality_weight", type=float, default=0.05)
    parser.add_argument("--score_adjacency_sym", type=str, default="max", choices=("max", "mean"))
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
