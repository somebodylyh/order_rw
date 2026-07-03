#!/usr/bin/env python3
"""Evaluate a Fiedler-score Attn-MLP on historical head_information matrices."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attn_mlp_order_policy import load_frozen_attn_mlp_policy  # noqa: E402
from scripts.train.train_mlp_fiedler_score_distillation import (  # noqa: E402
    fiedler_vector_from_attention,
    minmax,
    pearson_np,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic-only evaluation of a score-distilled Attn-MLP on head_information artifacts."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, required=True)
    parser.add_argument(
        "--head_info_dir",
        type=Path,
        default=Path("Report/history/language/wikitext103/head_information"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_blocks", type=int, default=64)
    parser.add_argument(
        "--matrix_key",
        type=str,
        default="post_softmax_attention_without_none_raw_current_raw_average",
    )
    parser.add_argument("--include_l0_heads", action="store_true")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--example_limit", type=int, default=16)
    return parser.parse_args()


def all_pair_indices(num_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(int(num_blocks), k=1)


def pair_accuracy(left: np.ndarray, right: np.ndarray) -> float:
    pair_i, pair_j = all_pair_indices(int(left.shape[0]))
    left_pref = left[pair_i] > left[pair_j]
    right_pref = right[pair_i] > right[pair_j]
    return float(np.mean(left_pref == right_pref))


def format_order(order: np.ndarray, limit: int = 16) -> str:
    order = np.asarray(order, dtype=np.int64).reshape(-1)
    if int(limit) > 0:
        order = order[: int(limit)]
    return " ".join(str(int(value)) for value in order.tolist())


def fiedler_priorities(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    vector, meta = fiedler_vector_from_attention(matrix)
    raw_priority = 1.0 - minmax(vector)
    reverse_priority = minmax(vector)
    return raw_priority.astype(np.float32), reverse_priority.astype(np.float32), meta


def iter_l0_matrices(args: argparse.Namespace) -> Iterable[Dict]:
    aggregate_paths = sorted(
        (args.head_info_dir / "phenomenon_validation" / "checkpoints").glob("*/aggregate_matrices.npz")
    )
    if int(args.max_files) > 0:
        aggregate_paths = aggregate_paths[: int(args.max_files)]
    for path in aggregate_paths:
        data = np.load(path)
        if args.matrix_key not in data.files:
            continue
        matrices = np.asarray(data[args.matrix_key], dtype=np.float32)
        if matrices.ndim != 4 or matrices.shape[-2:] != (int(args.num_blocks), int(args.num_blocks)):
            continue
        checkpoint_label = path.parent.name
        yield {
            "artifact": str(path),
            "checkpoint_label": checkpoint_label,
            "matrix_key": str(args.matrix_key),
            "matrix_label": "L0_mean",
            "layer": 0,
            "head": "mean",
            "matrix": matrices[0].mean(axis=0),
        }
        if bool(args.include_l0_heads):
            for head in range(int(matrices.shape[1])):
                yield {
                    "artifact": str(path),
                    "checkpoint_label": checkpoint_label,
                    "matrix_key": str(args.matrix_key),
                    "matrix_label": f"L0H{head}",
                    "layer": 0,
                    "head": int(head),
                    "matrix": matrices[0, head],
                }


@torch.no_grad()
def evaluate_matrix(model: torch.nn.Module, matrix: np.ndarray, device: torch.device) -> Dict:
    raw_priority, reverse_priority, meta = fiedler_priorities(matrix)
    attn = torch.as_tensor(matrix, dtype=torch.float32, device=device).unsqueeze(0)
    pred_score = torch.sigmoid(model(attn).float()).squeeze(0).detach().cpu().numpy()
    pair_acc_raw = pair_accuracy(pred_score, raw_priority)
    pair_acc_reverse = pair_accuracy(pred_score, reverse_priority)
    tau_raw = 2.0 * pair_acc_raw - 1.0
    tau_reverse = 2.0 * pair_acc_reverse - 1.0
    pearson_raw = pearson_np(pred_score, raw_priority)
    pearson_reverse = pearson_np(pred_score, reverse_priority)
    pred_order = np.argsort(-pred_score)
    raw_order = np.argsort(-raw_priority)
    reverse_order = np.argsort(-reverse_priority)
    return {
        "pair_acc_raw": float(pair_acc_raw),
        "pair_acc_reverse": float(pair_acc_reverse),
        "tau_raw": float(tau_raw),
        "tau_reverse": float(tau_reverse),
        "axis_tau": float(max(tau_raw, tau_reverse)),
        "best_direction": "raw" if tau_raw >= tau_reverse else "reverse",
        "score_pearson_raw": float(pearson_raw),
        "score_pearson_reverse": float(pearson_reverse),
        "pred_score_mean": float(np.mean(pred_score)),
        "pred_score_std": float(np.std(pred_score)),
        "target_raw_score_std": float(np.std(raw_priority)),
        "fiedler_eigval": float(meta["laplacian_fiedler_eigval"]),
        "spectral_gap": float(meta["spectral_gap"]),
        "affinity_sum": float(meta["affinity_sum"]),
        "pred_order_first16": format_order(pred_order, 16),
        "pred_order_last16": format_order(pred_order[::-1], 16),
        "raw_order_first16": format_order(raw_order, 16),
        "reverse_order_first16": format_order(reverse_order, 16),
    }


def write_outputs(report_dir: Path, rows: List[Dict], example_limit: int) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "external_head_information_eval.csv"
    fields = [
        "artifact",
        "checkpoint_label",
        "matrix_key",
        "matrix_label",
        "layer",
        "head",
        "pair_acc_raw",
        "pair_acc_reverse",
        "tau_raw",
        "tau_reverse",
        "axis_tau",
        "best_direction",
        "score_pearson_raw",
        "score_pearson_reverse",
        "pred_score_mean",
        "pred_score_std",
        "target_raw_score_std",
        "fiedler_eigval",
        "spectral_gap",
        "affinity_sum",
        "diagnostic_only",
        "pred_order_first16",
        "pred_order_last16",
        "raw_order_first16",
        "reverse_order_first16",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    l0_mean_rows = [row for row in rows if row.get("matrix_label") == "L0_mean"]
    summary = {
        "diagnostic_only": True,
        "note": "Historical head_information artifacts do not contain loss-profile oriented teacher_order; raw/reverse/axis tau are diagnostic only.",
        "num_rows": int(len(rows)),
        "num_l0_mean_rows": int(len(l0_mean_rows)),
        "mean_axis_tau_all": float(np.mean([float(row["axis_tau"]) for row in rows])) if rows else float("nan"),
        "min_axis_tau_all": float(np.min([float(row["axis_tau"]) for row in rows])) if rows else float("nan"),
        "mean_axis_tau_l0_mean": float(np.mean([float(row["axis_tau"]) for row in l0_mean_rows]))
        if l0_mean_rows
        else float("nan"),
        "min_axis_tau_l0_mean": float(np.min([float(row["axis_tau"]) for row in l0_mean_rows]))
        if l0_mean_rows
        else float("nan"),
        "csv": str(csv_path),
        "order_examples": str(report_dir / "external_head_information_order_examples.md"),
    }
    (report_dir / "external_head_information_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        "# External Head Information Order Examples",
        "",
        "Diagnostic only: these artifacts do not provide the loss-profile orientation used for training labels.",
        "",
        "| artifact | matrix | axis tau | raw tau | reverse tau | best | pred first8 | raw first8 | reverse first8 |",
        "|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for row in rows[: max(0, int(example_limit))]:
        md_lines.append(
            f"| {Path(str(row['artifact'])).parent.name} | {row['matrix_label']} | "
            f"{float(row['axis_tau']):.4f} | {float(row['tau_raw']):.4f} | "
            f"{float(row['tau_reverse']):.4f} | {row['best_direction']} | "
            f"{' '.join(str(row['pred_order_first16']).split()[:8])} | "
            f"{' '.join(str(row['raw_order_first16']).split()[:8])} | "
            f"{' '.join(str(row['reverse_order_first16']).split()[:8])} |"
        )
    (report_dir / "external_head_information_order_examples.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, _ = load_frozen_attn_mlp_policy(args.checkpoint, int(args.num_blocks), device)
    model.eval()
    rows: List[Dict] = []
    for item in iter_l0_matrices(args):
        metrics = evaluate_matrix(model, item.pop("matrix"), device)
        row = {**item, **metrics, "diagnostic_only": True}
        rows.append(row)
    write_outputs(args.report_dir, rows, int(args.example_limit))
    print(
        json.dumps(
            {
                "rows": len(rows),
                "summary": str(args.report_dir / "external_head_information_summary.json"),
                "csv": str(args.report_dir / "external_head_information_eval.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
