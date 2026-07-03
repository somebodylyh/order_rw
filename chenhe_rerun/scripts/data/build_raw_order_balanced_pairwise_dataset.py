#!/usr/bin/env python3
"""Build a raw-order, tau-balanced pairwise dataset from try8-style shards."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from order_utils import build_fixed_block_permutation, invert_permutation  # noqa: E402


BUCKETS = [
    ("neg_high", -1.0000001, -0.8),
    ("neg_mid", -0.8, -0.2),
    ("near_zero", -0.2, 0.2),
    ("pos_mid", 0.2, 0.8),
    ("pos_high", 0.8, 1.0000001),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--num_blocks", type=int, default=64)
    parser.add_argument("--permute_seed", type=int, default=42)
    parser.add_argument("--shard_size", type=int, default=512)
    parser.add_argument("--splits", type=str, default="train,val")
    return parser.parse_args()


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def original_l2r_target_rank(num_blocks: int, permute_seed: int) -> torch.Tensor:
    block_perm = build_fixed_block_permutation(int(num_blocks), int(permute_seed)).long()
    original_l2r = invert_permutation(block_perm).long()
    rank = torch.empty(int(num_blocks), dtype=torch.long)
    rank[original_l2r] = torch.arange(int(num_blocks), dtype=torch.long)
    return rank


def pair_indices(num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    pair_i: List[int] = []
    pair_j: List[int] = []
    for i in range(int(num_blocks)):
        for j in range(i + 1, int(num_blocks)):
            pair_i.append(i)
            pair_j.append(j)
    return torch.tensor(pair_i, dtype=torch.long), torch.tensor(pair_j, dtype=torch.long)


def tau_to_original_l2r(order: torch.Tensor, target_rank: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor) -> torch.Tensor:
    original_positions = target_rank[order.long()]
    concordant = (original_positions[:, pair_i] < original_positions[:, pair_j]).float().mean(dim=1)
    return 2.0 * concordant - 1.0


def assign_bucket(tau: float) -> str:
    for name, lo, hi in BUCKETS:
        if float(tau) >= lo and float(tau) < hi:
            return name
    raise ValueError(f"tau={tau} fell outside configured buckets")


def summarize_values(values: Iterable[float]) -> Dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p01": float(np.quantile(arr, 0.01)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
        "ge_0_9": int((arr >= 0.9).sum()),
        "ge_0_8": int((arr >= 0.8).sum()),
        "le_neg_0_9": int((arr <= -0.9).sum()),
        "le_neg_0_8": int((arr <= -0.8).sum()),
        "abs_le_0_1": int((np.abs(arr) <= 0.1).sum()),
        "abs_le_0_2": int((np.abs(arr) <= 0.2).sum()),
    }


def scan_split(source_dir: Path, split: str, target_rank: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor) -> Dict:
    rows = []
    bucket_to_indices = defaultdict(list)
    bucket_counts = defaultdict(int)
    head_counts = defaultdict(int)
    tau_values = []
    paths = sorted((source_dir / split).glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt found under {source_dir / split}")
    global_index = 0
    for shard_idx, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu")
        tau = tau_to_original_l2r(payload["raw_order"], target_rank, pair_i, pair_j).cpu().numpy()
        layer = payload["layer"].cpu().numpy()
        head = payload["head"].cpu().numpy()
        for local_idx, value in enumerate(tau.tolist()):
            bucket = assign_bucket(float(value))
            row = {
                "global_index": global_index,
                "shard_idx": shard_idx,
                "path": str(path),
                "local_idx": int(local_idx),
                "bucket": bucket,
                "tau": float(value),
                "layer": int(layer[local_idx]),
                "head": int(head[local_idx]),
                "label": f"L{int(layer[local_idx])}H{int(head[local_idx])}",
            }
            rows.append(row)
            bucket_to_indices[bucket].append(global_index)
            bucket_counts[bucket] += 1
            head_counts[row["label"]] += 1
            tau_values.append(float(value))
            global_index += 1
    return {
        "rows": rows,
        "bucket_to_indices": bucket_to_indices,
        "bucket_counts": dict(bucket_counts),
        "head_counts": dict(head_counts),
        "tau_summary": summarize_values(tau_values),
    }


def select_balanced(scan: Dict, seed: int) -> tuple[List[Dict], Dict]:
    rng = np.random.default_rng(int(seed))
    bucket_counts = {name: len(scan["bucket_to_indices"].get(name, [])) for name, _, _ in BUCKETS}
    n_per_bucket = min(bucket_counts.values())
    selected_global = []
    selected_counts = {}
    for bucket, _, _ in BUCKETS:
        candidates = np.asarray(scan["bucket_to_indices"].get(bucket, []), dtype=np.int64)
        if candidates.size == 0:
            raise ValueError(f"Bucket {bucket} is empty; cannot build balanced dataset.")
        chosen = rng.choice(candidates, size=n_per_bucket, replace=False)
        selected_global.extend(chosen.tolist())
        selected_counts[bucket] = int(chosen.size)
    rng.shuffle(selected_global)
    rows = [scan["rows"][idx] for idx in selected_global]
    summary = {
        "source_bucket_counts": bucket_counts,
        "selected_bucket_counts": selected_counts,
        "n_per_bucket": int(n_per_bucket),
        "num_selected": int(len(rows)),
    }
    return rows, summary


def make_raw_pairwise_fields(raw_rank: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rank = raw_rank.long()
    target = rank[:, pair_i] < rank[:, pair_j]
    sign = target.to(torch.int8) * 2 - 1
    q = torch.where(rank[:, :, None] < rank[:, None, :], 1, -1).to(torch.int8)
    diag = torch.arange(rank.size(1), dtype=torch.long)
    q[:, diag, diag] = 0
    return target.bool(), sign.to(torch.int8), q


def write_shard(path: Path, samples: List[Dict], pair_i: torch.Tensor, pair_j: torch.Tensor, metadata: Dict) -> None:
    if not samples:
        return
    keys = samples[0].keys()
    out = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values, dim=0)
    raw_pair_target, raw_pair_sign, raw_q = make_raw_pairwise_fields(out["raw_rank"], pair_i, pair_j)
    out["loss_oriented_teacher_order"] = out["teacher_order"].clone()
    out["loss_oriented_teacher_rank"] = out["teacher_rank"].clone()
    out["loss_oriented_selected_reverse"] = out["selected_reverse"].clone()
    out["teacher_order"] = out["raw_order"].clone()
    out["teacher_rank"] = out["raw_rank"].clone()
    out["pair_target"] = raw_pair_target
    out["pair_sign"] = raw_pair_sign
    out["pairwise_Q"] = raw_q
    out["selected_reverse"] = torch.zeros_like(out["selected_reverse"], dtype=torch.bool)
    out["pair_i"] = pair_i.to(torch.uint8)
    out["pair_j"] = pair_j.to(torch.uint8)
    out["metadata"] = dict(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)


def materialize_split(
    source_dir: Path,
    out_dir: Path,
    split: str,
    rows: List[Dict],
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    target_rank: torch.Tensor,
    shard_size: int,
    metadata: Dict,
) -> Dict:
    by_path = defaultdict(list)
    for row in rows:
        by_path[row["path"]].append(row)

    shard_samples: List[Dict] = []
    shard_idx = 0
    selected_tau = []
    selected_bucket_counts = defaultdict(int)
    selected_head_counts = defaultdict(int)
    for path_str in sorted(by_path.keys()):
        payload = torch.load(path_str, map_location="cpu")
        local_rows = sorted(by_path[path_str], key=lambda item: item["local_idx"])
        local_indices = torch.tensor([row["local_idx"] for row in local_rows], dtype=torch.long)
        tau = tau_to_original_l2r(payload["raw_order"].index_select(0, local_indices), target_rank, pair_i, pair_j)
        tensor_keys = [key for key, value in payload.items() if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == payload["attention"].size(0)]
        for offset, row in enumerate(local_rows):
            item = {key: payload[key][int(row["local_idx"])].clone() for key in tensor_keys}
            item["raw_original_l2r_tau"] = tau[offset].float()
            shard_samples.append(item)
            selected_tau.append(float(tau[offset].item()))
            selected_bucket_counts[row["bucket"]] += 1
            selected_head_counts[row["label"]] += 1
            if len(shard_samples) >= int(shard_size):
                write_shard(out_dir / split / f"shard_{shard_idx:03d}.pt", shard_samples, pair_i, pair_j, metadata)
                shard_samples = []
                shard_idx += 1
    if shard_samples:
        write_shard(out_dir / split / f"shard_{shard_idx:03d}.pt", shard_samples, pair_i, pair_j, metadata)
        shard_idx += 1
    return {
        "num_samples": int(len(rows)),
        "num_shards": int(shard_idx),
        "bucket_counts": dict(selected_bucket_counts),
        "head_counts": dict(selected_head_counts),
        "tau_summary": summarize_values(selected_tau),
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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    target_rank = original_l2r_target_rank(int(args.num_blocks), int(args.permute_seed))
    pair_i, pair_j = pair_indices(int(args.num_blocks))
    metadata = {
        "version": 1,
        "format": "raw_order_balanced_pairwise_dataset",
        "description": "Raw direct-asym-eig order labels, balanced by raw OriginalL2R tau buckets.",
        "source_dir": str(args.source_dir),
        "label_order": "raw_order",
        "label_rank": "raw_rank",
        "teacher_order_note": "teacher_order/teacher_rank are overwritten with raw_order/raw_rank for compatibility with existing MLP training code.",
        "balance_axis": "raw_order_original_l2r_tau",
        "buckets": [{"name": name, "lo": lo, "hi": hi} for name, lo, hi in BUCKETS],
        "num_blocks": int(args.num_blocks),
        "permute_seed": int(args.permute_seed),
        "seed": int(args.seed),
        "shard_size": int(args.shard_size),
    }

    collection = {"metadata": metadata, "splits": {}}
    for split_offset, split in enumerate(splits):
        scan = scan_split(args.source_dir, split, target_rank, pair_i, pair_j)
        selected_rows, select_summary = select_balanced(scan, int(args.seed) + split_offset)
        materialized = materialize_split(
            args.source_dir,
            args.out_dir,
            split,
            selected_rows,
            pair_i,
            pair_j,
            target_rank,
            int(args.shard_size),
            metadata,
        )
        collection["splits"][split] = {
            "source_bucket_counts": select_summary["source_bucket_counts"],
            "selected_bucket_counts": select_summary["selected_bucket_counts"],
            "n_per_bucket": select_summary["n_per_bucket"],
            "source_num_samples": int(len(scan["rows"])),
            "selected_num_samples": int(len(selected_rows)),
            "source_tau_summary": scan["tau_summary"],
            "selected": materialized,
        }
        write_csv(
            args.report_dir / f"{split}_bucket_counts.csv",
            [
                {
                    "bucket": name,
                    "lo": lo,
                    "hi": hi,
                    "source_count": select_summary["source_bucket_counts"][name],
                    "selected_count": select_summary["selected_bucket_counts"][name],
                }
                for name, lo, hi in BUCKETS
            ],
            ["bucket", "lo", "hi", "source_count", "selected_count"],
        )
        write_csv(
            args.report_dir / f"{split}_head_counts.csv",
            [
                {"head": head, "selected_count": count}
                for head, count in sorted(materialized["head_counts"].items())
            ],
            ["head", "selected_count"],
        )
    write_json(args.out_dir / "manifest.json", metadata)
    write_json(args.out_dir / "collection_summary.json", collection)
    write_json(args.report_dir / "collection_summary.json", collection)

    lines = [
        "# try13: Raw-Order Tau-Balanced Pairwise Dataset",
        "",
        "This try rewrites the existing try8 pairwise operator dataset so the compatible",
        "`teacher_order` / `teacher_rank` fields use `raw_order` / `raw_rank` directly.",
        "Samples are downsampled into five raw OriginalL2R tau buckets with equal counts",
        "inside each split.",
        "",
        f"- Source: `{args.source_dir}`",
        f"- Dataset: `{args.out_dir}`",
        f"- Seed: `{int(args.seed)}`",
        f"- Buckets: `{', '.join(name for name, _, _ in BUCKETS)}`",
        "",
        "## Split Summary",
        "",
        "| split | source samples | selected samples | per bucket | shards |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in splits:
        info = collection["splits"][split]
        lines.append(
            f"| {split} | {info['source_num_samples']} | {info['selected_num_samples']} | "
            f"{info['n_per_bucket']} | {info['selected']['num_shards']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `loss_oriented_teacher_order/rank` preserve the previous train-loss-oriented labels.",
            "- `selected_reverse` is reset to false because the active label is raw, not reverse-selected.",
            "- `raw_original_l2r_tau` is stored per sample for auditing only.",
        ]
    )
    (args.report_dir / "dataset_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
