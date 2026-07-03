#!/usr/bin/env python3
"""Select stable attention heads from a checkpoint without original-order priors.

The selection score uses only current-frame attention matrices:
direction strength, directed-ribbon self-consistency, and repeat stability.
Original-frame tau is computed only after selection as a diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from online_spectral_order_policy import robust_z  # noqa: E402
from scripts.analysis.export_all_head_attention_maps import (  # noqa: E402
    EXPORT_TYPES,
    align_batch_to_current,
    block_attention,
)
from scripts.analysis.export_block_attention_heatmap import (  # noqa: E402
    build_model,
    build_orders,
    extract_logits_loss_attentions,
    get_autocast_context,
    infer_data_record_mode,
    load_checkpoint,
    load_tokens,
    maybe_apply_data_permutation,
    resolve_data_dir,
    resolve_data_permutation,
    sample_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-prior stable-head selector for Attn-MLP warmup.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_repeats", type=int, default=8)
    parser.add_argument("--batches_per_repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=24681357)
    parser.add_argument("--mode", type=str, default="Random", choices=["AR", "Random"])
    parser.add_argument("--export_types", type=str, default="with_none,without_none")
    parser.add_argument("--band_width", type=int, default=8)
    parser.add_argument("--ribbon_direction_weight", type=float, default=0.7)
    parser.add_argument("--ribbon_band_weight", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--force_manual_attention", action="store_true")
    parser.add_argument("--save_matrices", action="store_true")
    return parser.parse_args()


def offdiag_mask(n: int) -> np.ndarray:
    return ~np.eye(n, dtype=bool)


def flatten_offdiag(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)[offdiag_mask(matrix.shape[0])]


def cosine_offdiag(a: np.ndarray, b: np.ndarray) -> float:
    av = flatten_offdiag(a)
    bv = flatten_offdiag(b)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if not np.isfinite(denom) or denom <= 1e-12:
        return 0.0
    return float(np.dot(av, bv) / denom)


def kendall_tau(order: list[int]) -> float:
    inv = {int(v): i for i, v in enumerate(order)}
    n = len(order)
    total = n * (n - 1) / 2.0
    if total <= 0:
        return 1.0
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if inv[i] > inv[j]:
                inversions += 1
    return float(1.0 - 2.0 * inversions / total)


def pairwise_order_tau(order_a: list[int], order_b: list[int]) -> float:
    pos_a = {int(v): i for i, v in enumerate(order_a)}
    mapped = [pos_a[int(v)] for v in order_b]
    return kendall_tau(mapped)


def order_to_original(order_current: list[int], data_permutation) -> list[int]:
    if data_permutation is None:
        return [int(v) for v in order_current]
    mapper = data_permutation["block_perm"].to(dtype=torch.long, device="cpu")
    current = torch.as_tensor(order_current, dtype=torch.long)
    return [int(v) for v in mapper[current].tolist()]


def direction_objects(matrix: np.ndarray, flow_sign: float = -1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    values = np.asarray(matrix, dtype=np.float64).copy()
    np.fill_diagonal(values, np.nan)
    z = robust_z(values)
    z = np.where(np.isfinite(z), z, 0.0)
    np.fill_diagonal(z, 0.0)
    q = z - z.T
    w = np.maximum(float(flow_sign) * q, 0.0)
    np.fill_diagonal(w, 0.0)
    priority = w.sum(axis=0) - w.sum(axis=1)
    order = [int(v) for v in np.argsort(priority, kind="mergesort")[::-1].tolist()]
    return z, q, priority, order


def score_one_matrix(matrix: np.ndarray, band_width: int, ribbon_direction_weight: float, ribbon_band_weight: float) -> dict:
    z, q, priority, order = direction_objects(matrix, flow_sign=-1.0)
    n = int(z.shape[0])
    mask = offdiag_mask(n)
    strength = float(np.abs(q[mask]).sum() / max(np.abs(z[mask]).sum(), 1e-12))
    w = np.maximum(-q, 0.0)
    np.fill_diagonal(w, 0.0)
    total_w = float(w.sum())
    if not np.isfinite(total_w) or total_w <= 1e-12:
        direction_fit = 0.0
        band_fit = 0.0
        active_edge_frac = 0.0
    else:
        weights = w / total_w
        delta_ok = priority[None, :] > priority[:, None]
        direction_fit = float((weights * delta_ok).sum())
        rank = np.empty(n, dtype=np.int64)
        rank[np.asarray(order, dtype=np.int64)] = np.arange(n, dtype=np.int64)
        dist = rank[:, None] - rank[None, :]
        band_ok = (dist >= 1) & (dist <= int(band_width))
        band_fit = float((weights * band_ok).sum())
        active_edge_frac = float((w[mask] > 0.0).mean())
    ribbon_quality = float(float(ribbon_direction_weight) * direction_fit + float(ribbon_band_weight) * band_fit)
    return {
        "strength": strength,
        "direction_fit": direction_fit,
        "band_fit": band_fit,
        "ribbon_quality": ribbon_quality,
        "active_edge_frac": active_edge_frac,
        "q": q,
        "priority": priority,
        "order": order,
    }


def collect_repeat_matrices(args: argparse.Namespace, export_types: list[str]):
    checkpoint = load_checkpoint(args.ckpt_path, args.device)
    model = build_model(checkpoint, args.device, force_manual_attention=args.force_manual_attention)
    data_dir = resolve_data_dir(args, checkpoint)
    data_record_mode = str(checkpoint.get("config", {}).get("data_record_mode", infer_data_record_mode(data_dir)))
    tokens = load_tokens(data_dir, args.split, data_record_mode=data_record_mode)
    rng = np.random.default_rng(int(args.seed))
    ctx = get_autocast_context(args.device, args.dtype)
    data_permutation = resolve_data_permutation(
        checkpoint,
        num_blocks=model.num_blocks,
        block_len=model.block_order_block_len,
        model=model,
    )

    n_layer = int(model.config.n_layer)
    n_head = int(model.config.n_head)
    num_blocks = int(model.num_blocks)
    matrices = {
        export_type: np.zeros(
            (int(args.num_repeats), n_layer, n_head, num_blocks, num_blocks),
            dtype=np.float32,
        )
        for export_type in export_types
    }
    repeat_losses: list[float] = []
    repeat_samples: list[int] = []

    for repeat_idx in range(int(args.num_repeats)):
        sums = {
            export_type: torch.zeros((n_layer, n_head, num_blocks, num_blocks), dtype=torch.float64, device="cpu")
            for export_type in export_types
        }
        total_samples = 0
        loss_sum = 0.0
        for _ in range(int(args.batches_per_repeat)):
            idx, _ = sample_batch(tokens, int(args.batch_size), model.config.block_size, rng, args.device)
            idx = maybe_apply_data_permutation(idx, data_permutation)
            token_orders, block_orders = build_orders(model, idx, args.mode)
            with torch.no_grad():
                with ctx:
                    outputs = model.forward_fn(idx, token_orders, return_attentions=True)
            _, loss, attentions = extract_logits_loss_attentions(outputs)
            for layer_idx, layer_attn in enumerate(attentions):
                for export_type in export_types:
                    block = block_attention(layer_attn.detach(), model.block_order_block_len, export_type)
                    current = align_batch_to_current(block, block_orders)
                    sums[export_type][layer_idx] += current.double().sum(dim=0).cpu()
            batch_samples = int(idx.size(0))
            total_samples += batch_samples
            loss_sum += float(loss.detach().item()) * batch_samples
        for export_type in export_types:
            matrices[export_type][repeat_idx] = (sums[export_type] / float(max(1, total_samples))).numpy().astype(np.float32)
        repeat_losses.append(float(loss_sum / max(1, total_samples)))
        repeat_samples.append(int(total_samples))
        print(
            f"repeat {repeat_idx + 1}/{int(args.num_repeats)}: "
            f"samples={int(total_samples)} loss={repeat_losses[-1]:.4f}"
        )

    return checkpoint, model, data_permutation, matrices, repeat_losses, repeat_samples


def summarize_scores(args: argparse.Namespace, export_type: str, matrices: np.ndarray, data_permutation) -> list[dict]:
    repeats, layers, heads = matrices.shape[:3]
    rows = []
    for layer_idx in range(layers):
        for head_idx in range(heads):
            per_repeat = [
                score_one_matrix(
                    matrices[r, layer_idx, head_idx],
                    band_width=int(args.band_width),
                    ribbon_direction_weight=float(args.ribbon_direction_weight),
                    ribbon_band_weight=float(args.ribbon_band_weight),
                )
                for r in range(repeats)
            ]
            mean_matrix = matrices[:, layer_idx, head_idx].mean(axis=0)
            mean_score = score_one_matrix(
                mean_matrix,
                band_width=int(args.band_width),
                ribbon_direction_weight=float(args.ribbon_direction_weight),
                ribbon_band_weight=float(args.ribbon_band_weight),
            )
            q_cosines = []
            order_taus = []
            for r in range(1, repeats):
                q_cosines.append(cosine_offdiag(per_repeat[r]["q"], per_repeat[r - 1]["q"]))
                order_taus.append(pairwise_order_tau(per_repeat[r - 1]["order"], per_repeat[r]["order"]))
            q_stability = float(np.mean(q_cosines)) if q_cosines else 0.0
            order_stability = float(np.mean(order_taus)) if order_taus else 0.0
            strength = float(np.mean([item["strength"] for item in per_repeat]))
            ribbon = float(np.mean([item["ribbon_quality"] for item in per_repeat]))
            direction_fit = float(np.mean([item["direction_fit"] for item in per_repeat]))
            band_fit = float(np.mean([item["band_fit"] for item in per_repeat]))
            active_edge_frac = float(np.mean([item["active_edge_frac"] for item in per_repeat]))
            score_q_weighted = float(strength * ribbon * max(0.0, q_stability))
            score_order_weighted = float(strength * ribbon * max(0.0, order_stability))
            score = float(
                strength
                * ribbon
                * max(0.0, q_stability)
                * max(0.0, order_stability)
            )
            order_current = mean_score["order"]
            order_original = order_to_original(order_current, data_permutation)
            rows.append(
                {
                    "export_type": export_type,
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "label": f"L{layer_idx}H{head_idx}",
                    "score": score,
                    "score_q_stability_variant": score_q_weighted,
                    "score_order_stability_variant": score_order_weighted,
                    "strength_mean": strength,
                    "ribbon_quality_mean": ribbon,
                    "direction_fit_mean": direction_fit,
                    "band_fit_mean": band_fit,
                    "q_stability_adjacent_mean": q_stability,
                    "order_stability_adjacent_mean": order_stability,
                    "active_edge_frac_mean": active_edge_frac,
                    "mean_matrix_strength": float(mean_score["strength"]),
                    "mean_matrix_ribbon_quality": float(mean_score["ribbon_quality"]),
                    "tau_current_l2r_diagnostic": kendall_tau(order_current),
                    "tau_original_l2r_diagnostic": kendall_tau(order_original),
                    "abs_tau_original_l2r_diagnostic": abs(kendall_tau(order_original)),
                    "order_current_first16": json.dumps(order_current[:16], separators=(",", ":")),
                    "order_original_first16_diagnostic": json.dumps(order_original[:16], separators=(",", ":")),
                }
            )
    rows.sort(key=lambda row: (-float(row["score"]), -float(row["score_order_stability_variant"]), row["label"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args: argparse.Namespace, checkpoint: dict, all_rows: dict[str, list[dict]], repeat_losses: list[float]) -> None:
    lines = [
        "# Stable Attention Head Selection",
        "",
        "Selection score uses only current-frame attention: direction strength, ribbon self-consistency, and repeat stability.",
        "Original-frame tau is shown only as a post-hoc diagnostic.",
        "",
        f"- checkpoint: `{args.ckpt_path}`",
        f"- iter: `{int(checkpoint.get('iter_num', -1))}`",
        f"- mode: `{args.mode}`",
        f"- repeats: `{int(args.num_repeats)}`",
        f"- batches per repeat: `{int(args.batches_per_repeat)}`",
        f"- batch size: `{int(args.batch_size)}`",
        f"- mean repeat loss: `{float(np.mean(repeat_losses)):.6f}`",
        "",
    ]
    for export_type, rows in all_rows.items():
        lines.extend(
            [
                f"## {export_type}",
                "",
                "| rank | head | score | strength | ribbon | q stability | order stability | tau orig diag | first16 current |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in rows[:8]:
            lines.append(
                f"| {int(row['rank'])} | {row['label']} | {float(row['score']):.6f} "
                f"| {float(row['strength_mean']):.4f} | {float(row['ribbon_quality_mean']):.4f} "
                f"| {float(row['q_stability_adjacent_mean']):+.4f} "
                f"| {float(row['order_stability_adjacent_mean']):+.4f} "
                f"| {float(row['tau_original_l2r_diagnostic']):+.4f} "
                f"| `{row['order_current_first16']}` |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    export_types = [item.strip() for item in str(args.export_types).split(",") if item.strip()]
    invalid = sorted(set(export_types) - set(EXPORT_TYPES))
    if invalid:
        raise ValueError(f"Unsupported export_types={invalid}; expected subset of {EXPORT_TYPES}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, model, data_permutation, matrices, repeat_losses, repeat_samples = collect_repeat_matrices(args, export_types)
    all_rows = {
        export_type: summarize_scores(args, export_type, matrix, data_permutation)
        for export_type, matrix in matrices.items()
    }
    for export_type, rows in all_rows.items():
        write_csv(args.out_dir / f"head_selection_scores_{export_type}.csv", rows)
    combined = []
    for rows in all_rows.values():
        combined.extend(rows)
    combined.sort(key=lambda row: (-float(row["score"]), row["export_type"], row["label"]))
    write_csv(args.out_dir / "head_selection_scores_all.csv", combined)
    write_report(args.out_dir / "results.md", args, checkpoint, all_rows, repeat_losses)

    if bool(args.save_matrices):
        np.savez_compressed(
            args.out_dir / "repeat_current_attention_matrices.npz",
            **{f"{key}_current_l2r": value for key, value in matrices.items()},
        )
    summary = {
        "ckpt_path": str(args.ckpt_path),
        "iter_num": int(checkpoint.get("iter_num", -1)),
        "best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "mode": str(args.mode),
        "split": str(args.split),
        "batch_size": int(args.batch_size),
        "num_repeats": int(args.num_repeats),
        "batches_per_repeat": int(args.batches_per_repeat),
        "repeat_samples": repeat_samples,
        "repeat_losses": repeat_losses,
        "selection_uses_original_order": False,
        "selection_uses_original_tau": False,
        "selection_uses_validation_loss": False,
        "selection_score": "strength_mean * ribbon_quality_mean * max(0, q_stability_adjacent_mean) * max(0, order_stability_adjacent_mean)",
        "posthoc_tau_is_diagnostic_only": True,
        "top_by_export_type": {
            export_type: rows[0] if rows else None
            for export_type, rows in all_rows.items()
        },
        "top_overall": combined[0] if combined else None,
        "outputs": {
            "results": "results.md",
            "combined_csv": "head_selection_scores_all.csv",
            "per_export_csv": "head_selection_scores_<export_type>.csv",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved stable-head selection results to {args.out_dir}")


if __name__ == "__main__":
    main()
