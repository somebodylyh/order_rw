#!/usr/bin/env python3
"""Summarize protocol-matched g_beta loss comparison runs."""

from __future__ import annotations

import argparse
import json
import pathlib


REFERENCE = {
    "kendall_tau": 0.984,
    "pairwise_acc": 0.997,
    "prefix8": 0.973,
}


def _load_json(path: pathlib.Path) -> dict:
    with open(path) as f:
        return json.load(f)


def summarize_runs(root_dir: str) -> dict:
    root = pathlib.Path(root_dir)
    rows = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        config_path = run_dir / "config.json"
        training_path = run_dir / "training_summary.json"
        eval_path = run_dir / "evaluation" / "summary.json"
        if not (config_path.exists() and training_path.exists() and eval_path.exists()):
            continue
        config = _load_json(config_path)
        training = _load_json(training_path)
        evaluation = _load_json(eval_path)
        val = evaluation["results"]["val"]["normal"]
        test = evaluation["results"]["test"]["normal"]
        rows.append({
            "run": run_dir.name,
            "loss_type": config.get("loss_type", "pairwise_bce"),
            "temperature": float(config.get("rank_kl_temperature", 4.0)),
            "val_primary_loss": float(val["primary_loss"]),
            "test_tau": float(test["kendall_tau"]),
            "test_pairwise_acc": float(test["pairwise_acc"]),
            "test_prefix8": float(test["prefix8"]),
            "test_prefix16": float(test["prefix16"]),
            "train_seconds": float(training["train_seconds"]),
            "peak_cuda_memory_mb": float(training["peak_cuda_memory_mb"]),
        })

    summary = {
        "reference_pairwise_bce": REFERENCE,
        "note": (
            "Primary losses are on different scales; compare held-out order "
            "metrics, wall time, memory, and downstream NLL."
        ),
        "runs": rows,
    }
    with open(root / "loss_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# g_beta Loss Comparison",
        "",
        "| Run | Loss | Tau | Pairwise | Prefix@8 | Prefix@16 | "
        "Train sec | Peak MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['loss_type']} | "
            f"{row['test_tau']:.4f} | {row['test_pairwise_acc']:.4f} | "
            f"{row['test_prefix8']:.4f} | {row['test_prefix16']:.4f} | "
            f"{row['train_seconds']:.1f} | "
            f"{row['peak_cuda_memory_mb']:.1f} |"
        )
    lines.extend([
        "",
        "Reference Pairwise BCE: tau 0.984, pairwise accuracy 0.997, "
        "Prefix@8 0.973.",
        "",
        "Do not compare primary loss values across different objective types.",
    ])
    (root / "loss_comparison_summary.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    summary = summarize_runs(args.root)
    print(f"Summarized {len(summary['runs'])} runs under {args.root}")


if __name__ == "__main__":
    main()
