#!/usr/bin/env python3
"""Test asymmetry-filtered direct-asym-eig head selection on a 10k checkpoint.

This is an offline probe for the no-prior head-selection methodology:

1. collect current-frame random-order block attention for every layer/head;
2. rank heads by current-frame attention asymmetry;
3. for each head, recover one order with the latest direct_asym_eig rule used
   by the training probe;
4. orient raw vs reverse by current-model loss profile only;
5. select the lowest-loss oriented order among the top asymmetric heads.

Original-frame tau/distance fields are written only as diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from order_utils import invert_permutation, token_losses_to_block_losses  # noqa: E402
from scripts.analysis.directed_head_order_probe import (  # noqa: E402
    attention_block_orders,
    batched,
    current_batch,
    expand_block_orders,
    extract_attentions,
    infer_record_mode,
    iter_starts,
    kendall_tau,
    load_checkpoint,
    load_model,
    load_tokens,
    permutation_state,
    to_original_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="10k all-head asymmetry + direct_asym_eig methodology test."
    )
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--attention_split", type=str, default="train")
    parser.add_argument("--loss_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--export_type",
        choices=("with_none", "without_none", "target_to_observed"),
        default="without_none",
    )
    parser.add_argument("--attention_order_mode", choices=("random", "current_ar", "original_l2r"), default="random")
    parser.add_argument("--attention_samples", type=int, default=512)
    parser.add_argument("--attention_batch_size", type=int, default=16)
    parser.add_argument("--attention_seed", type=int, default=24681357)
    parser.add_argument("--loss_samples", type=int, default=64)
    parser.add_argument("--loss_batch_size", type=int, default=16)
    parser.add_argument("--loss_candidate_batch_size", type=int, default=8)
    parser.add_argument("--loss_seed", type=int, default=97531)
    parser.add_argument("--val_loss_samples", type=int, default=128)
    parser.add_argument("--val_loss_batch_size", type=int, default=32)
    parser.add_argument("--val_loss_seed", type=int, default=86420)
    parser.add_argument("--prefix_k", type=int, default=16)
    parser.add_argument("--loss_score", choices=("linear_profile", "exp_profile", "prefix", "full"), default="linear_profile")
    parser.add_argument("--exp_tau", type=float, default=16.0)
    parser.add_argument("--asym_top_k", type=int, default=8)
    parser.add_argument("--asym_percentile", type=float, default=75.0)
    parser.add_argument("--direct_asym_eig_mode", type=str, default="raw_right_largest_real_real")
    parser.add_argument("--write_matrices", action="store_true")
    return parser.parse_args()


def autocast_context(args: argparse.Namespace):
    if "cuda" not in str(args.device) or str(args.dtype) == "float32":
        return nullcontext()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[str(args.dtype)]
    return torch.amp.autocast("cuda", dtype=dtype)


def aggregate_layerhead_attention_sums(layer_attn, block_orders, block_len: int, export_type: str):
    export_type = str(export_type)
    if export_type == "with_none":
        shifted = layer_attn[:, :, :-1, :-1]
        exposure_correct = False
    elif export_type == "without_none":
        shifted = layer_attn[:, :, 1:, 1:]
        exposure_correct = False
    elif export_type == "target_to_observed":
        shifted = layer_attn[:, :, :-1, 1:]
        exposure_correct = True
    else:
        raise ValueError(f"Unsupported export_type={export_type!r}")
    batch, heads, seq_a, seq_b = shifted.shape
    if seq_a != seq_b or seq_a % int(block_len) != 0:
        raise ValueError(f"attention shape={tuple(shifted.shape)} incompatible with block_len={block_len}")
    num_blocks = seq_a // int(block_len)
    block_batch = shifted.float().view(batch, heads, num_blocks, block_len, num_blocks, block_len).mean(dim=(3, 5))
    total = torch.zeros((heads, num_blocks, num_blocks), dtype=torch.float64, device=block_batch.device)
    counts = torch.zeros((num_blocks, num_blocks), dtype=torch.float64, device=block_batch.device)
    exposure_mask_reveal = torch.tril(
        torch.ones((num_blocks, num_blocks), dtype=torch.float64, device=block_batch.device),
        diagonal=-1,
    )
    for sample_idx in range(batch):
        inverse = invert_permutation(block_orders[sample_idx].detach()).to(device=block_batch.device)
        current = block_batch[sample_idx][:, inverse, :][:, :, inverse].double()
        if exposure_correct:
            mask = exposure_mask_reveal[inverse, :][:, inverse]
            total += current * mask.unsqueeze(0)
            counts += mask
        else:
            total += current
    if not exposure_correct:
        counts += float(max(1, batch))
    return total, counts


def aggregate_layerhead_attention(layer_attn, block_orders, block_len: int, export_type: str) -> torch.Tensor:
    total, counts = aggregate_layerhead_attention_sums(layer_attn, block_orders, block_len, export_type)
    return total / counts.clamp_min(1.0).unsqueeze(0)


@torch.no_grad()
def collect_all_head_matrices(
    args: argparse.Namespace,
    model,
    tokens,
    record_mode: str,
    perm_state,
    ctx,
) -> Tuple[np.ndarray, int]:
    block_size = int(model.config.block_size)
    num_blocks = int(model.num_blocks)
    starts = iter_starts(
        len(tokens),
        block_size,
        record_mode,
        int(args.attention_samples),
        int(args.attention_seed),
    )
    generator_device = "cuda" if "cuda" in str(args.device) else "cpu"
    rng = torch.Generator(device=generator_device)
    rng.manual_seed(int(args.attention_seed) + 17)
    matrix_sum = None
    count_sum = None
    total = 0
    for batch_idx, batch_starts in enumerate(batched(starts, int(args.attention_batch_size)), start=1):
        batch = current_batch(tokens, batch_starts, block_size, args.device, perm_state)
        block_orders = attention_block_orders(
            args.attention_order_mode,
            int(batch.size(0)),
            num_blocks,
            args.device,
            rng,
            perm_state,
        )
        token_orders = expand_block_orders(model, block_orders)
        with ctx:
            outputs = model(
                batch,
                mode=None,
                orders=token_orders,
                return_attentions=True,
                return_logits=False,
            )
        attentions = extract_attentions(outputs)
        if not attentions:
            raise RuntimeError("Model did not return attentions.")
        layer_sums = []
        layer_counts = []
        for layer_attn in attentions:
            sums, counts = aggregate_layerhead_attention_sums(
                layer_attn.detach(),
                block_orders,
                int(model.block_order_block_len),
                str(args.export_type),
            )
            layer_sums.append(sums.detach())
            layer_counts.append(counts.detach())
        sums = torch.stack(layer_sums, dim=0)
        counts = torch.stack(layer_counts, dim=0)[:, None, :, :].expand_as(sums)
        samples = int(batch.size(0))
        matrix_sum = sums if matrix_sum is None else matrix_sum + sums
        count_sum = counts if count_sum is None else count_sum + counts
        total += samples
        if batch_idx == 1 or total >= int(args.attention_samples) or batch_idx % 8 == 0:
            print(f"[10k-method] attention batches={batch_idx} samples={total}")
    if matrix_sum is None or total <= 0:
        raise RuntimeError("No attention matrices were collected.")
    matrices = (matrix_sum / count_sum.clamp_min(1.0)).detach().cpu().numpy()
    for layer_idx in range(matrices.shape[0]):
        for head_idx in range(matrices.shape[1]):
            np.fill_diagonal(matrices[layer_idx, head_idx], 0.0)
    return matrices, total


def matrix_asymmetry_metrics(matrix: np.ndarray) -> Dict[str, float]:
    values = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    offdiag = ~np.eye(values.shape[0], dtype=bool)
    antisym = values - values.T
    asym_sum = float(np.abs(antisym[offdiag]).sum())
    attn_sum = float(np.abs(values[offdiag]).sum())
    upper = np.triu(np.ones_like(values, dtype=bool), k=1)
    lower = np.tril(np.ones_like(values, dtype=bool), k=-1)
    upper_sum = float(values[upper].sum())
    lower_sum = float(values[lower].sum())
    return {
        "asym_score": asym_sum / max(attn_sum, 1e-12),
        "mean_abs_asym": asym_sum / float(max(1, int(offdiag.sum()))),
        "sum_abs_asym": asym_sum,
        "sum_abs_attn": attn_sum,
        "max_attn": float(values.max()),
        "upper_frac": upper_sum / max(upper_sum + lower_sum, 1e-12),
    }


def original_frame_matrix(matrix: np.ndarray, perm_state) -> np.ndarray:
    if perm_state is None:
        return np.asarray(matrix, dtype=np.float64)
    mapper = perm_state["block_perm"].to(dtype=torch.long, device="cpu")
    original_to_current = invert_permutation(mapper).cpu().numpy()
    return np.asarray(matrix, dtype=np.float64)[np.ix_(original_to_current, original_to_current)]


def direct_asym_eig_order(matrix: np.ndarray, mode: str) -> Tuple[List[int], Dict[str, float | str]]:
    values = np.nan_to_num(np.asarray(matrix, dtype=np.float64).copy(), nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"direct_asym_eig expects a square matrix, got shape={tuple(values.shape)}")
    np.fill_diagonal(values, 0.0)
    num_blocks = int(values.shape[0])

    mode_name = str(mode).strip().lower() or "raw_right_largest_real_real"
    if mode_name == "direct_asym_eig":
        mode_name = "raw_right_largest_real_real"
    parts = {item for item in mode_name.split("_") if item}
    eig_input = values
    input_name = "raw"
    if "center" in parts or "centered" in parts:
        eig_input = values - values.mean(axis=1, keepdims=True)
        input_name = "row_centered"
    if "left" in parts:
        eig_input = eig_input.T
        vector_side = "left"
    else:
        vector_side = "right"

    eigvals, eigvecs = np.linalg.eig(eig_input)
    if "smallest" in parts and "real" in parts:
        eig_idx = int(np.argmin(np.real(eigvals)))
        eigen_selector = "smallest_real"
    elif "abs" in parts or "magnitude" in parts:
        eig_idx = int(np.argmax(np.abs(eigvals)))
        eigen_selector = "largest_abs"
    else:
        eig_idx = int(np.argmax(np.real(eigvals)))
        eigen_selector = "largest_real"

    vector_complex = eigvecs[:, eig_idx]
    if "imag" in parts or "imaginary" in parts:
        vector = np.imag(vector_complex)
        vector_part = "imag"
    elif "absvec" in parts or "vectorabs" in parts:
        vector = np.abs(vector_complex)
        vector_part = "abs"
    else:
        vector = np.real(vector_complex)
        vector_part = "real"
    vector = np.asarray(vector, dtype=np.float64)
    if not np.isfinite(vector).all() or float(np.nanstd(vector)) <= 1e-12:
        fallback = values.sum(axis=1) - values.sum(axis=0)
        if np.isfinite(fallback).all() and float(np.nanstd(fallback)) > 1e-12:
            vector = np.asarray(fallback, dtype=np.float64)
            vector_part = f"{vector_part}_fallback_row_minus_col_sum"
    if not np.isfinite(vector).all():
        raise ValueError("direct_asym_eig produced a non-finite ordering vector")
    if float(np.std(vector)) <= 1e-12:
        raise ValueError("direct_asym_eig produced a degenerate ordering vector")

    order = [
        int(idx)
        for idx in sorted(
            range(num_blocks),
            key=lambda item: (float(vector[item]), int(item)),
        )
    ]
    return order, {
        "candidate_source": "direct_asym_eig",
        "mode": str(mode_name),
        "input": str(input_name),
        "vector_side": str(vector_side),
        "eigen_selector": str(eigen_selector),
        "vector_part": str(vector_part),
        "eigval_real": float(np.real(eigvals[eig_idx])),
        "eigval_imag": float(np.imag(eigvals[eig_idx])),
        "eigval_abs": float(abs(eigvals[eig_idx])),
        "vector_min": float(np.min(vector)),
        "vector_max": float(np.max(vector)),
        "vector_std": float(np.std(vector)),
        "num_blocks": int(num_blocks),
    }


def loss_score(loss_item: Dict[str, float], score_name: str) -> float:
    score_name = str(score_name)
    if score_name in {"prefix", "prefix_loss"}:
        return float(loss_item["prefix_loss"])
    if score_name in {"full", "full_loss"}:
        return float(loss_item["full_loss"])
    if score_name in {"exp", "exp_profile", "exp_profile_loss"}:
        return float(loss_item["exp_profile_loss"])
    return float(loss_item["linear_profile_loss"])


@torch.no_grad()
def evaluate_loss_profiles(
    args: argparse.Namespace,
    model,
    tokens,
    record_mode: str,
    perm_state,
    orders: Sequence[Sequence[int]],
    *,
    split_name: str,
    num_samples: int,
    batch_size: int,
    seed: int,
    ctx,
) -> Dict[Tuple[int, ...], Dict[str, float | List[float] | int]]:
    unique_orders: List[List[int]] = []
    seen = set()
    num_blocks = int(model.num_blocks)
    for order in orders:
        key = tuple(int(v) for v in order)
        if len(key) != num_blocks or sorted(key) != list(range(num_blocks)):
            raise ValueError(f"Invalid order: {key}")
        if key not in seen:
            seen.add(key)
            unique_orders.append(list(key))
    if not unique_orders:
        return {}

    starts = iter_starts(
        len(tokens),
        int(model.config.block_size),
        record_mode,
        int(num_samples),
        int(seed),
    )
    order_tensor = torch.tensor(unique_orders, dtype=torch.long, device=args.device)
    candidate_batch_size = max(1, int(args.loss_candidate_batch_size))
    stats = {
        tuple(order): {
            "full_sum": 0.0,
            "prefix_sum": 0.0,
            "profile_sum": np.zeros(num_blocks, dtype=np.float64),
            "count": 0,
        }
        for order in unique_orders
    }
    prefix_k = max(1, min(int(args.prefix_k), num_blocks))
    for batch_idx, batch_starts in enumerate(batched(starts, int(batch_size)), start=1):
        batch = current_batch(tokens, batch_starts, int(model.config.block_size), args.device, perm_state)
        batch_size_local = int(batch.size(0))
        for start in range(0, len(unique_orders), candidate_batch_size):
            end = min(len(unique_orders), start + candidate_batch_size)
            chunk = order_tensor[start:end]
            chunk_size = int(chunk.size(0))
            block_orders = (
                chunk[:, None, :]
                .expand(chunk_size, batch_size_local, num_blocks)
                .reshape(chunk_size * batch_size_local, num_blocks)
            )
            x_expanded = (
                batch[None, :, :]
                .expand(chunk_size, batch_size_local, int(batch.size(1)))
                .reshape(chunk_size * batch_size_local, int(batch.size(1)))
            )
            token_orders = expand_block_orders(model, block_orders)
            with ctx:
                outputs = model(
                    x_expanded,
                    mode=None,
                    orders=token_orders,
                    return_token_loss=True,
                    return_logits=False,
                )
            token_losses = None
            for value in outputs[2:]:
                if torch.is_tensor(value) and value.ndim == 2:
                    token_losses = value
                    break
            if token_losses is None:
                raise RuntimeError("Could not find token losses in model output.")
            block_losses = token_losses_to_block_losses(
                token_losses.detach(),
                block_len=int(model.block_order_block_len),
            ).float().view(chunk_size, batch_size_local, num_blocks)
            prefix_loss = block_losses[:, :, :prefix_k].mean(dim=2)
            full_loss = block_losses.mean(dim=2)
            for local_idx, order_idx in enumerate(range(start, end)):
                key = tuple(unique_orders[order_idx])
                stats[key]["prefix_sum"] += float(prefix_loss[local_idx].double().sum().item())
                stats[key]["full_sum"] += float(full_loss[local_idx].double().sum().item())
                stats[key]["profile_sum"] += block_losses[local_idx].double().sum(dim=0).detach().cpu().numpy()
                stats[key]["count"] += int(batch_size_local)
        if batch_idx == 1 or batch_idx % 4 == 0 or batch_idx == math.ceil(len(starts) / max(1, int(batch_size))):
            print(f"[10k-method] {split_name} loss batches={batch_idx} samples={min(batch_idx * int(batch_size), len(starts))}")

    linear_weights = np.linspace(1.0, 0.1, num_blocks, dtype=np.float64)
    linear_weights /= float(max(linear_weights.sum(), 1e-12))
    exp_weights = np.exp(-np.arange(num_blocks, dtype=np.float64) / max(float(args.exp_tau), 1e-6))
    exp_weights /= float(max(exp_weights.sum(), 1e-12))
    out: Dict[Tuple[int, ...], Dict[str, float | List[float] | int]] = {}
    for key, item in stats.items():
        count = max(1, int(item["count"]))
        profile = np.asarray(item["profile_sum"], dtype=np.float64) / float(count)
        out[key] = {
            "prefix_loss": float(item["prefix_sum"] / count),
            "full_loss": float(item["full_sum"] / count),
            "linear_profile_loss": float((linear_weights * profile).sum()),
            "exp_profile_loss": float((exp_weights * profile).sum()),
            "loss_profile": [float(value) for value in profile.tolist()],
            "count": int(item["count"]),
        }
    return out


def head_label(row: Dict) -> str:
    return f"L{int(row['layer'])}H{int(row['head'])}"


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = load_checkpoint(args.ckpt_path)
    model = load_model(ckpt, args.device)
    attention_tokens, data_dir = load_tokens(ckpt, args.attention_split, args.data_dir)
    loss_tokens, _ = load_tokens(ckpt, args.loss_split, data_dir)
    val_tokens, _ = load_tokens(ckpt, args.val_split, data_dir)
    record_mode = infer_record_mode(ckpt, data_dir)
    perm_state = permutation_state(ckpt, model, int(model.config.block_size))
    ctx = autocast_context(args)

    matrices, attention_total = collect_all_head_matrices(
        args,
        model,
        attention_tokens,
        record_mode,
        perm_state,
        ctx,
    )
    if bool(args.write_matrices):
        np.savez_compressed(args.out_dir / "all_head_attention_matrices_current.npz", matrices=matrices.astype(np.float32))

    rows: List[Dict] = []
    order_requests: List[List[int]] = []
    num_layers, num_heads, num_blocks, _ = matrices.shape
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            matrix = matrices[layer_idx, head_idx]
            current_metrics = matrix_asymmetry_metrics(matrix)
            original_metrics = matrix_asymmetry_metrics(original_frame_matrix(matrix, perm_state))
            row = {
                "layer": int(layer_idx),
                "head": int(head_idx),
                **current_metrics,
                "original_upper_frac_diagnostic": float(original_metrics["upper_frac"]),
            }
            try:
                raw_order, eig_meta = direct_asym_eig_order(matrix, str(args.direct_asym_eig_mode))
                raw_original = to_original_order(raw_order, perm_state)
                raw_tau = kendall_tau(raw_original) if raw_original is not None else kendall_tau(raw_order)
                reverse_order = list(reversed(raw_order))
                row.update(
                    {
                        "raw_order_current": raw_order,
                        "raw_order_original_diagnostic": raw_original,
                        "raw_tau_original_diagnostic": float(raw_tau),
                        "raw_norm_kendall_distance_original_diagnostic": float((1.0 - raw_tau) / 2.0),
                        **{f"eig_{key}": value for key, value in eig_meta.items()},
                    }
                )
                order_requests.extend([raw_order, reverse_order])
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)

    loss_by_order = evaluate_loss_profiles(
        args,
        model,
        loss_tokens,
        record_mode,
        perm_state,
        order_requests,
        split_name=str(args.loss_split),
        num_samples=int(args.loss_samples),
        batch_size=int(args.loss_batch_size),
        seed=int(args.loss_seed),
        ctx=ctx,
    )

    selected_orders = []
    for row in rows:
        raw_order = row.get("raw_order_current")
        if not raw_order:
            continue
        raw_key = tuple(int(v) for v in raw_order)
        rev_order = list(reversed(raw_order))
        rev_key = tuple(int(v) for v in rev_order)
        raw_loss = loss_by_order.get(raw_key)
        rev_loss = loss_by_order.get(rev_key)
        if raw_loss is None or rev_loss is None:
            row["error"] = "missing_loss"
            continue
        raw_score = loss_score(raw_loss, str(args.loss_score))
        rev_score = loss_score(rev_loss, str(args.loss_score))
        if rev_score < raw_score:
            selected_order = rev_order
            selected_loss = rev_loss
            other_score = raw_score
            selected_reverse = True
        else:
            selected_order = list(raw_order)
            selected_loss = raw_loss
            other_score = rev_score
            selected_reverse = False
        selected_original = to_original_order(selected_order, perm_state)
        selected_tau = kendall_tau(selected_original) if selected_original is not None else kendall_tau(selected_order)
        row.update(
            {
                "raw_linear_profile_loss": float(raw_loss["linear_profile_loss"]),
                "reverse_linear_profile_loss": float(rev_loss["linear_profile_loss"]),
                "raw_prefix_loss": float(raw_loss["prefix_loss"]),
                "reverse_prefix_loss": float(rev_loss["prefix_loss"]),
                "raw_full_loss": float(raw_loss["full_loss"]),
                "reverse_full_loss": float(rev_loss["full_loss"]),
                "selected_reverse": bool(selected_reverse),
                "selected_score": float(loss_score(selected_loss, str(args.loss_score))),
                "selected_score_gap": float(abs(other_score - loss_score(selected_loss, str(args.loss_score)))),
                "selected_prefix_loss": float(selected_loss["prefix_loss"]),
                "selected_full_loss": float(selected_loss["full_loss"]),
                "selected_linear_profile_loss": float(selected_loss["linear_profile_loss"]),
                "selected_exp_profile_loss": float(selected_loss["exp_profile_loss"]),
                "selected_tau_original_diagnostic": float(selected_tau),
                "selected_abs_tau_original_diagnostic": abs(float(selected_tau)),
                "selected_norm_kendall_distance_original_diagnostic": float((1.0 - selected_tau) / 2.0),
                "selected_order_current": selected_order,
                "selected_order_original_diagnostic": selected_original,
            }
        )
        selected_orders.append(selected_order)

    valid_rows = [row for row in rows if "selected_score" in row]
    valid_rows.sort(key=lambda row: (-float(row["asym_score"]), int(row["layer"]), int(row["head"])))
    for rank, row in enumerate(valid_rows, start=1):
        row["asym_rank"] = int(rank)
    by_loss = sorted(valid_rows, key=lambda row: (float(row["selected_score"]), -float(row["asym_score"]), int(row["layer"]), int(row["head"])))
    for rank, row in enumerate(by_loss, start=1):
        row["selected_loss_rank"] = int(rank)

    if selected_orders:
        val_loss_by_order = evaluate_loss_profiles(
            args,
            model,
            val_tokens,
            record_mode,
            perm_state,
            selected_orders,
            split_name=str(args.val_split),
            num_samples=int(args.val_loss_samples),
            batch_size=int(args.val_loss_batch_size),
            seed=int(args.val_loss_seed),
            ctx=ctx,
        )
        for row in valid_rows:
            selected_key = tuple(int(v) for v in row["selected_order_current"])
            val_loss = val_loss_by_order.get(selected_key)
            if val_loss is not None:
                row["selected_val_prefix_loss"] = float(val_loss["prefix_loss"])
                row["selected_val_full_loss"] = float(val_loss["full_loss"])
                row["selected_val_linear_profile_loss"] = float(val_loss["linear_profile_loss"])
                row["selected_val_exp_profile_loss"] = float(val_loss["exp_profile_loss"])

    baselines = {
        "current_ar": list(range(num_blocks)),
        "current_r2l": list(reversed(range(num_blocks))),
    }
    if perm_state is not None:
        original_l2r = [int(v) for v in perm_state["inverse_block_perm"].to(dtype=torch.long, device="cpu").tolist()]
        baselines["original_l2r_diagnostic"] = original_l2r
        baselines["original_r2l_diagnostic"] = list(reversed(original_l2r))
    baseline_loss = evaluate_loss_profiles(
        args,
        model,
        val_tokens,
        record_mode,
        perm_state,
        list(baselines.values()),
        split_name=f"{args.val_split}_baselines",
        num_samples=int(args.val_loss_samples),
        batch_size=int(args.val_loss_batch_size),
        seed=int(args.val_loss_seed) + 101,
        ctx=ctx,
    )
    baseline_rows = []
    for name, order in baselines.items():
        key = tuple(order)
        item = baseline_loss[key]
        order_original = to_original_order(order, perm_state)
        tau = kendall_tau(order_original) if order_original is not None else kendall_tau(order)
        baseline_rows.append(
            {
                "name": name,
                "val_prefix_loss": float(item["prefix_loss"]),
                "val_full_loss": float(item["full_loss"]),
                "val_linear_profile_loss": float(item["linear_profile_loss"]),
                "tau_original_diagnostic": float(tau),
                "norm_kendall_distance_original_diagnostic": float((1.0 - tau) / 2.0),
                "order_current": " ".join(str(v) for v in order),
                "order_original_diagnostic": " ".join(str(v) for v in (order_original or [])),
            }
        )

    asym_top_k = max(1, min(int(args.asym_top_k), len(valid_rows)))
    asym_top_rows = [row for row in valid_rows if int(row["asym_rank"]) <= asym_top_k]
    asym_threshold = float(np.percentile([float(row["asym_score"]) for row in valid_rows], float(args.asym_percentile))) if valid_rows else float("nan")
    asym_percentile_rows = [row for row in valid_rows if float(row["asym_score"]) >= asym_threshold]

    selections = {}
    if valid_rows:
        selections["asym_argmax_only"] = valid_rows[0]
        selections[f"asym_top{asym_top_k}_loss_select"] = sorted(
            asym_top_rows,
            key=lambda row: (float(row["selected_score"]), -float(row["asym_score"]), int(row["layer"]), int(row["head"])),
        )[0]
        if asym_percentile_rows:
            selections[f"asym_p{float(args.asym_percentile):g}_loss_select"] = sorted(
                asym_percentile_rows,
                key=lambda row: (float(row["selected_score"]), -float(row["asym_score"]), int(row["layer"]), int(row["head"])),
            )[0]
        selections["global_loss_select_diagnostic"] = by_loss[0]

    def compact_row(row: Dict) -> Dict:
        keys = [
            "layer",
            "head",
            "asym_rank",
            "selected_loss_rank",
            "asym_score",
            "selected_score",
            "selected_score_gap",
            "selected_reverse",
            "selected_linear_profile_loss",
            "selected_val_linear_profile_loss",
            "selected_tau_original_diagnostic",
            "selected_norm_kendall_distance_original_diagnostic",
            "raw_tau_original_diagnostic",
            "upper_frac",
            "original_upper_frac_diagnostic",
        ]
        return {key: row.get(key) for key in keys}

    fieldnames = [
        "layer",
        "head",
        "asym_rank",
        "selected_loss_rank",
        "asym_score",
        "mean_abs_asym",
        "sum_abs_asym",
        "sum_abs_attn",
        "upper_frac",
        "original_upper_frac_diagnostic",
        "eig_eigval_real",
        "eig_eigval_imag",
        "eig_eigval_abs",
        "eig_vector_std",
        "raw_tau_original_diagnostic",
        "raw_norm_kendall_distance_original_diagnostic",
        "raw_linear_profile_loss",
        "reverse_linear_profile_loss",
        "selected_reverse",
        "selected_score",
        "selected_score_gap",
        "selected_prefix_loss",
        "selected_full_loss",
        "selected_linear_profile_loss",
        "selected_exp_profile_loss",
        "selected_val_prefix_loss",
        "selected_val_full_loss",
        "selected_val_linear_profile_loss",
        "selected_val_exp_profile_loss",
        "selected_tau_original_diagnostic",
        "selected_abs_tau_original_diagnostic",
        "selected_norm_kendall_distance_original_diagnostic",
        "raw_order_current",
        "selected_order_current",
        "selected_order_original_diagnostic",
        "error",
    ]
    csv_rows = []
    for row in sorted(valid_rows, key=lambda item: int(item["asym_rank"])):
        payload = dict(row)
        for key in ("raw_order_current", "selected_order_current", "selected_order_original_diagnostic"):
            if key in payload and isinstance(payload[key], list):
                payload[key] = " ".join(str(v) for v in payload[key])
        csv_rows.append(payload)
    write_csv(args.out_dir / "all_head_direct_asym_eig_results.csv", csv_rows, fieldnames)
    write_csv(args.out_dir / "baseline_val_losses.csv", baseline_rows, list(baseline_rows[0].keys()))

    summary = {
        "ckpt_path": str(args.ckpt_path),
        "checkpoint_iter": int(ckpt.get("iter_num", -1)),
        "data_dir": str(data_dir),
        "data_record_mode": str(record_mode),
        "attention": {
            "split": str(args.attention_split),
            "samples": int(attention_total),
            "batch_size": int(args.attention_batch_size),
            "seed": int(args.attention_seed),
            "order_mode": str(args.attention_order_mode),
            "export_type": str(args.export_type),
        },
        "loss_selection": {
            "split": str(args.loss_split),
            "samples": int(args.loss_samples),
            "batch_size": int(args.loss_batch_size),
            "candidate_batch_size": int(args.loss_candidate_batch_size),
            "seed": int(args.loss_seed),
            "score": str(args.loss_score),
            "prefix_k": int(args.prefix_k),
            "exp_tau": float(args.exp_tau),
        },
        "val_diagnostic": {
            "split": str(args.val_split),
            "samples": int(args.val_loss_samples),
            "batch_size": int(args.val_loss_batch_size),
            "seed": int(args.val_loss_seed),
        },
        "selection_rules": {name: compact_row(row) for name, row in selections.items()},
        "top_by_asym": [compact_row(row) for row in valid_rows[: min(10, len(valid_rows))]],
        "top_by_selected_loss": [compact_row(row) for row in by_loss[: min(10, len(by_loss))]],
        "baseline_val_losses": baseline_rows,
        "no_prior_note": (
            "Selections use current-frame attention asymmetry plus current-model loss profile only. "
            "Original-frame tau/order/distance and original_l2r baselines are diagnostic-only."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    lines = [
        "# 10k Direct-Asym-Eig Head Method Test",
        "",
        f"- checkpoint: `{args.ckpt_path}`",
        f"- checkpoint iter: `{int(ckpt.get('iter_num', -1))}`",
        f"- attention: `{args.attention_split}`, `{args.attention_order_mode}`, `{args.export_type}`, samples={attention_total}",
        f"- selection loss: `{args.loss_split}`, samples={int(args.loss_samples)}, score=`{args.loss_score}`",
        f"- validation diagnostic: `{args.val_split}`, samples={int(args.val_loss_samples)}",
        "",
        "Selection uses only current-frame asymmetry and current-model loss. Original tau/distance below is diagnostic-only.",
        "",
        "## Selection Rules",
        "",
        "| rule | head | asym rank | loss rank | asym | train score | val linear | selected reverse | tau diag | dist diag |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for name, row in selections.items():
        lines.append(
            f"| {name} | {head_label(row)} | {int(row.get('asym_rank', -1))} | "
            f"{int(row.get('selected_loss_rank', -1))} | {float(row['asym_score']):.4f} | "
            f"{float(row['selected_score']):.6f} | {float(row.get('selected_val_linear_profile_loss', float('nan'))):.6f} | "
            f"{bool(row.get('selected_reverse', False))} | {float(row['selected_tau_original_diagnostic']):+.4f} | "
            f"{float(row['selected_norm_kendall_distance_original_diagnostic']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Top Heads By Asymmetry",
            "",
            "| rank | head | asym | train score | val linear | tau diag | dist diag | reverse |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in valid_rows[: min(10, len(valid_rows))]:
        lines.append(
            f"| {int(row['asym_rank'])} | {head_label(row)} | {float(row['asym_score']):.4f} | "
            f"{float(row['selected_score']):.6f} | {float(row.get('selected_val_linear_profile_loss', float('nan'))):.6f} | "
            f"{float(row['selected_tau_original_diagnostic']):+.4f} | "
            f"{float(row['selected_norm_kendall_distance_original_diagnostic']):.4f} | "
            f"{bool(row.get('selected_reverse', False))} |"
        )
    lines.extend(
        [
            "",
            "## Top Heads By Selected Loss",
            "",
            "| rank | head | asym rank | asym | train score | val linear | tau diag | dist diag |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in by_loss[: min(10, len(by_loss))]:
        lines.append(
            f"| {int(row['selected_loss_rank'])} | {head_label(row)} | {int(row['asym_rank'])} | "
            f"{float(row['asym_score']):.4f} | {float(row['selected_score']):.6f} | "
            f"{float(row.get('selected_val_linear_profile_loss', float('nan'))):.6f} | "
            f"{float(row['selected_tau_original_diagnostic']):+.4f} | "
            f"{float(row['selected_norm_kendall_distance_original_diagnostic']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic Baselines",
            "",
            "| name | val linear | val full | tau diag | dist diag |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in baseline_rows:
        lines.append(
            f"| {row['name']} | {float(row['val_linear_profile_loss']):.6f} | "
            f"{float(row['val_full_loss']):.6f} | {float(row['tau_original_diagnostic']):+.4f} | "
            f"{float(row['norm_kendall_distance_original_diagnostic']):.4f} |"
        )
    (args.out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "selection_rules": summary["selection_rules"]}, indent=2))


if __name__ == "__main__":
    main()
