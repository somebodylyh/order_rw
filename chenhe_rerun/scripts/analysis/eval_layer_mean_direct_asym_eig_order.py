#!/usr/bin/env python3
"""Evaluate layer-mean attention direct-asym-eig orders from a checkpoint.

This is a narrow offline diagnostic for the distribution-style order discovery
path. It aggregates all heads in selected layers, recovers one direct_asym_eig
order per layer-mean matrix, and reports original-frame Kendall tau. Original
tau is diagnostic only; optional loss-profile orientation uses current-model
loss to choose order vs reverse.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.test_10k_direct_asym_eig_head_method import (  # noqa: E402
    autocast_context,
    collect_all_head_matrices,
    direct_asym_eig_order,
    evaluate_loss_profiles,
    infer_record_mode,
    kendall_tau,
    load_checkpoint,
    load_model,
    load_tokens,
    loss_score,
    matrix_asymmetry_metrics,
    permutation_state,
    to_original_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layer-mean direct_asym_eig original-tau diagnostic."
    )
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--attention_split", type=str, default="train", choices=("train", "val"))
    parser.add_argument("--loss_split", type=str, default="train", choices=("train", "val"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--export_type", choices=("with_none", "without_none"), default="without_none")
    parser.add_argument(
        "--attention_order_mode",
        choices=("random", "current_ar", "original_l2r"),
        default="random",
    )
    parser.add_argument("--attention_samples", type=int, default=1024)
    parser.add_argument("--attention_batch_size", type=int, default=16)
    parser.add_argument("--attention_seed", type=int, default=24681357)
    parser.add_argument("--loss_samples", type=int, default=1024)
    parser.add_argument("--loss_batch_size", type=int, default=16)
    parser.add_argument("--loss_candidate_batch_size", type=int, default=4)
    parser.add_argument("--loss_seed", type=int, default=97531)
    parser.add_argument("--prefix_k", type=int, default=16)
    parser.add_argument(
        "--loss_score",
        choices=("linear_profile", "exp_profile", "prefix", "full"),
        default="linear_profile",
    )
    parser.add_argument("--exp_tau", type=float, default=16.0)
    parser.add_argument("--layers", type=str, default="0,1")
    parser.add_argument("--direct_asym_eig_mode", type=str, default="raw_right_largest_real_real")
    parser.add_argument("--write_matrices", action="store_true")
    return parser.parse_args()


def parse_layers(text: str, max_layers: int) -> List[int]:
    layers: List[int] = []
    seen = set()
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        layer = int(item)
        if layer < 0 or layer >= int(max_layers):
            raise ValueError(f"layer {layer} outside 0..{int(max_layers) - 1}")
        if layer not in seen:
            seen.add(layer)
            layers.append(layer)
    if not layers:
        raise ValueError("No layers requested.")
    return layers


def order_tau(order_current: Sequence[int], perm_state) -> Tuple[List[int], float]:
    original = to_original_order(list(order_current), perm_state)
    tau = kendall_tau(original) if original is not None else kendall_tau(list(order_current))
    return [int(v) for v in (original if original is not None else order_current)], float(tau)


def select_by_loss(
    args: argparse.Namespace,
    raw_order: Sequence[int],
    loss_by_order: Dict[Tuple[int, ...], Dict],
) -> Tuple[List[int], Dict, Dict, bool, float]:
    raw_order = [int(v) for v in raw_order]
    reverse_order = list(reversed(raw_order))
    raw_loss = loss_by_order[tuple(raw_order)]
    reverse_loss = loss_by_order[tuple(reverse_order)]
    raw_score = loss_score(raw_loss, str(args.loss_score))
    reverse_score = loss_score(reverse_loss, str(args.loss_score))
    if reverse_score < raw_score:
        return reverse_order, reverse_loss, raw_loss, True, float(raw_score - reverse_score)
    return raw_order, raw_loss, reverse_loss, False, float(reverse_score - raw_score)


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(args.ckpt_path)
    model = load_model(ckpt, args.device)
    attention_tokens, data_dir = load_tokens(ckpt, args.attention_split, args.data_dir)
    loss_tokens, _ = load_tokens(ckpt, args.loss_split, data_dir)
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
    layers = parse_layers(args.layers, int(matrices.shape[0]))
    if bool(args.write_matrices):
        np.savez_compressed(
            args.out_dir / "all_head_attention_matrices_current.npz",
            matrices=matrices.astype(np.float32),
        )

    layer_payloads: List[Dict] = []
    order_requests: List[List[int]] = []
    for layer_idx in layers:
        matrix = np.asarray(matrices[layer_idx], dtype=np.float64).mean(axis=0)
        np.fill_diagonal(matrix, 0.0)
        if bool(args.write_matrices):
            np.save(args.out_dir / f"L{layer_idx}_headmean_attention_current.npy", matrix.astype(np.float32))
        raw_order, eig_meta = direct_asym_eig_order(matrix, str(args.direct_asym_eig_mode))
        reverse_order = list(reversed(raw_order))
        raw_original, raw_tau = order_tau(raw_order, perm_state)
        reverse_original, reverse_tau = order_tau(reverse_order, perm_state)
        metrics = matrix_asymmetry_metrics(matrix)
        layer_payloads.append(
            {
                "layer": int(layer_idx),
                "source": f"L{layer_idx}_head_mean",
                "matrix": matrix,
                "asym_score": float(metrics["asym_score"]),
                "upper_frac": float(metrics["upper_frac"]),
                "raw_order_current": raw_order,
                "raw_order_original": raw_original,
                "raw_original_tau": float(raw_tau),
                "raw_original_abs_tau": abs(float(raw_tau)),
                "raw_original_distance": float((1.0 - raw_tau) / 2.0),
                "reverse_original_tau": float(reverse_tau),
                "reverse_original_distance": float((1.0 - reverse_tau) / 2.0),
                "eig": eig_meta,
            }
        )
        order_requests.extend([raw_order, reverse_order])

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

    rows: List[Dict] = []
    for payload in layer_payloads:
        selected_order, selected_loss, other_loss, selected_reverse, score_gap = select_by_loss(
            args,
            payload["raw_order_current"],
            loss_by_order,
        )
        selected_original, selected_tau = order_tau(selected_order, perm_state)
        row = {
            "layer": int(payload["layer"]),
            "source": str(payload["source"]),
            "attention_samples": int(attention_total),
            "loss_samples": int(args.loss_samples),
            "export_type": str(args.export_type),
            "attention_order_mode": str(args.attention_order_mode),
            "asym_score": float(payload["asym_score"]),
            "upper_frac": float(payload["upper_frac"]),
            "eigval_real": float(payload["eig"]["eigval_real"]),
            "eigval_imag": float(payload["eig"]["eigval_imag"]),
            "eigval_abs": float(payload["eig"]["eigval_abs"]),
            "vector_std": float(payload["eig"]["vector_std"]),
            "raw_original_tau": float(payload["raw_original_tau"]),
            "raw_original_abs_tau": float(payload["raw_original_abs_tau"]),
            "raw_original_distance": float(payload["raw_original_distance"]),
            "reverse_original_tau": float(payload["reverse_original_tau"]),
            "reverse_original_distance": float(payload["reverse_original_distance"]),
            "loss_oriented_selected_reverse": bool(selected_reverse),
            "loss_oriented_score_gap": float(score_gap),
            "loss_oriented_original_tau": float(selected_tau),
            "loss_oriented_original_abs_tau": abs(float(selected_tau)),
            "loss_oriented_original_distance": float((1.0 - selected_tau) / 2.0),
            "selected_prefix_loss": float(selected_loss["prefix_loss"]),
            "selected_full_loss": float(selected_loss["full_loss"]),
            "selected_linear_profile_loss": float(selected_loss["linear_profile_loss"]),
            "other_linear_profile_loss": float(other_loss["linear_profile_loss"]),
            "raw_order_current": " ".join(str(v) for v in payload["raw_order_current"]),
            "raw_order_original": " ".join(str(v) for v in payload["raw_order_original"]),
            "loss_oriented_order_current": " ".join(str(v) for v in selected_order),
            "loss_oriented_order_original": " ".join(str(v) for v in selected_original),
        }
        rows.append(row)

    write_csv(args.out_dir / "layer_mean_direct_asym_eig_tau.csv", rows)
    summary = {
        "ckpt_path": str(args.ckpt_path),
        "checkpoint_iter": int(ckpt.get("iter_num", -1)),
        "checkpoint_best_val_loss": float(ckpt.get("best_val_loss", float("nan"))),
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
        "loss_orientation": {
            "split": str(args.loss_split),
            "samples": int(args.loss_samples),
            "batch_size": int(args.loss_batch_size),
            "candidate_batch_size": int(args.loss_candidate_batch_size),
            "seed": int(args.loss_seed),
            "score": str(args.loss_score),
            "prefix_k": int(args.prefix_k),
        },
        "rows": rows,
        "note": (
            "raw_original_tau is the sign-arbitrary eig output. "
            "loss_oriented_original_tau chooses raw vs reverse by current-model loss_profile only. "
            "Original tau is diagnostic-only."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Random 50k Layer-Mean Direct-Asym-Eig Diagnostic",
        "",
        f"- checkpoint: `{args.ckpt_path}`",
        f"- checkpoint iter: `{int(ckpt.get('iter_num', -1))}`",
        f"- attention samples: `{attention_total}`",
        f"- attention split/order/export: `{args.attention_split}` / `{args.attention_order_mode}` / `{args.export_type}`",
        f"- loss-orientation samples: `{int(args.loss_samples)}`",
        f"- loss score: `{args.loss_score}`",
        "",
        "Original tau is diagnostic-only. The raw eig direction has arbitrary sign, so the loss-oriented tau is usually the more comparable distribution-method number.",
        "",
        "| source | raw tau | raw abs tau | loss-oriented tau | loss abs tau | selected reverse | score gap | asym | order original |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['source']} | {float(row['raw_original_tau']):+.6f} | "
            f"{float(row['raw_original_abs_tau']):.6f} | "
            f"{float(row['loss_oriented_original_tau']):+.6f} | "
            f"{float(row['loss_oriented_original_abs_tau']):.6f} | "
            f"{row['loss_oriented_selected_reverse']} | "
            f"{float(row['loss_oriented_score_gap']):.6f} | "
            f"{float(row['asym_score']):.6f} | "
            f"`[{', '.join(row['loss_oriented_order_original'].split())}]` |"
        )
    (args.out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
