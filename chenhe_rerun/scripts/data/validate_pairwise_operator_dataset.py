#!/usr/bin/env python3
"""Validate and summarize an oriented pairwise operator dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate oriented pairwise dataset shards.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, required=True)
    parser.add_argument("--sample_shards_per_split", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260622)
    return parser.parse_args()


def shard_paths(dataset_dir: Path, split: str) -> List[Path]:
    return sorted((dataset_dir / split).glob("shard_*.pt"))


def validate_payload(payload: Dict) -> Dict:
    pair_i = payload["pair_i"].long()
    pair_j = payload["pair_j"].long()
    rank = payload["teacher_rank"].long()
    target = rank[:, pair_i] < rank[:, pair_j]
    pair_target_ok = bool(torch.equal(target, payload["pair_target"].bool()))
    q_ok = True
    q_diag_ok = True
    if "pairwise_Q" in payload:
        q = payload["pairwise_Q"].to(torch.int16)
        q_ok = bool(torch.equal(q, -q.transpose(1, 2)))
        diag = torch.diagonal(q, dim1=1, dim2=2)
        q_diag_ok = bool(torch.equal(diag, torch.zeros_like(diag)))
    return {
        "pair_target_ok": bool(pair_target_ok),
        "pairwise_q_skew_ok": bool(q_ok),
        "pairwise_q_diag_ok": bool(q_diag_ok),
    }


def summarize_split(dataset_dir: Path, split: str, sample_count: int, rng: random.Random) -> Dict:
    paths = shard_paths(dataset_dir, split)
    if not paths:
        return {
            "split": split,
            "num_shards": 0,
            "num_samples": 0,
            "record_min": None,
            "record_max": None,
            "iter_min": None,
            "iter_max": None,
            "head_counts": {},
            "selected_reverse_rate": None,
            "loss_score_gap_mean": None,
            "loss_score_gap_std": None,
            "validation": [],
        }
    num_samples = 0
    record_min = None
    record_max = None
    iter_min = None
    iter_max = None
    reverse_sum = 0.0
    gap_sum = 0.0
    gap_sq_sum = 0.0
    head_counts = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        count = int(payload["attention"].shape[0])
        num_samples += count
        records = payload["record_index"].long()
        iters = payload["iter"].long()
        record_min = int(records.min().item()) if record_min is None else min(record_min, int(records.min().item()))
        record_max = int(records.max().item()) if record_max is None else max(record_max, int(records.max().item()))
        iter_min = int(iters.min().item()) if iter_min is None else min(iter_min, int(iters.min().item()))
        iter_max = int(iters.max().item()) if iter_max is None else max(iter_max, int(iters.max().item()))
        reverse_sum += float(payload["selected_reverse"].float().sum().item())
        gap = payload["loss_score_gap"].float()
        gap_sum += float(gap.sum().item())
        gap_sq_sum += float((gap * gap).sum().item())
        for layer, head in zip(payload["layer"].long().tolist(), payload["head"].long().tolist()):
            key = f"L{int(layer)}H{int(head)}"
            head_counts[key] = int(head_counts.get(key, 0) + 1)
    validation = []
    sample_paths = paths if len(paths) <= int(sample_count) else rng.sample(paths, int(sample_count))
    for path in sorted(sample_paths):
        payload = torch.load(path, map_location="cpu")
        validation.append({"path": str(path), **validate_payload(payload)})
    mean_gap = gap_sum / max(1, num_samples)
    variance_gap = max(0.0, gap_sq_sum / max(1, num_samples) - mean_gap * mean_gap)
    return {
        "split": split,
        "num_shards": int(len(paths)),
        "num_samples": int(num_samples),
        "record_min": record_min,
        "record_max": record_max,
        "iter_min": iter_min,
        "iter_max": iter_max,
        "head_counts": dict(sorted(head_counts.items())),
        "selected_reverse_rate": float(reverse_sum / max(1, num_samples)),
        "loss_score_gap_mean": float(mean_gap),
        "loss_score_gap_std": float(variance_gap ** 0.5),
        "validation": validation,
    }


def write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))
    manifest = {}
    manifest_path = args.dataset_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_path = args.dataset_dir / "collection_summary.json"
    collection_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    train = summarize_split(args.dataset_dir, "train", int(args.sample_shards_per_split), rng)
    val = summarize_split(args.dataset_dir, "val", int(args.sample_shards_per_split), rng)
    payload = {
        "dataset_dir": str(args.dataset_dir),
        "manifest": manifest,
        "collection_summary": collection_summary,
        "splits": {"train": train, "val": val},
        "expected": {
            "records": int(collection_summary.get("records", 0)),
            "train_records": int(manifest.get("train_records", collection_summary.get("train_records", 0) or 0)),
            "num_layers": int(manifest.get("num_layers", collection_summary.get("num_layers", 0) or 0)),
            "num_heads": int(manifest.get("num_heads", collection_summary.get("num_heads", 0) or 0)),
        },
    }
    (args.report_dir / "dataset_validation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    head_rows = []
    for split_name, item in payload["splits"].items():
        for head, count in item["head_counts"].items():
            head_rows.append({"split": split_name, "head": head, "count": int(count)})
    write_csv(args.report_dir / "head_counts.csv", head_rows, ["split", "head", "count"])
    lines = [
        "# Pairwise Operator Dataset Validation",
        "",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Train samples: `{train['num_samples']}` across `{train['num_shards']}` shards",
        f"- Val samples: `{val['num_samples']}` across `{val['num_shards']}` shards",
        f"- Train record range: `{train['record_min']}` to `{train['record_max']}`",
        f"- Val record range: `{val['record_min']}` to `{val['record_max']}`",
        f"- Train selected_reverse rate: `{train['selected_reverse_rate']}`",
        f"- Val selected_reverse rate: `{val['selected_reverse_rate']}`",
        f"- Train gap mean/std: `{train['loss_score_gap_mean']}` / `{train['loss_score_gap_std']}`",
        f"- Val gap mean/std: `{val['loss_score_gap_mean']}` / `{val['loss_score_gap_std']}`",
        "",
        "## Validation Samples",
        "",
        "| split | shard | pair target | Q skew | Q diag |",
        "|---|---|---|---|---|",
    ]
    for split_name, item in payload["splits"].items():
        for row in item["validation"]:
            lines.append(
                f"| {split_name} | `{row['path']}` | `{row['pair_target_ok']}` | "
                f"`{row['pairwise_q_skew_ok']}` | `{row['pairwise_q_diag_ok']}` |"
            )
    (args.report_dir / "dataset_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"train_samples": train["num_samples"], "val_samples": val["num_samples"]}, indent=2))


if __name__ == "__main__":
    main()
