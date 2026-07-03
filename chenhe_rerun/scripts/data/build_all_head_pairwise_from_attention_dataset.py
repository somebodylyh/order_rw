#!/usr/bin/env python3
"""Convert all-head attention records into direct-asym-eig pairwise targets."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.test_10k_direct_asym_eig_head_method import direct_asym_eig_order  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Turn try3/try4 all-head attention matrices into direct-asym-eig "
            "order/rank/pairwise-Q samples."
        )
    )
    parser.add_argument("--all_head_dir", type=Path, required=True)
    parser.add_argument("--split_spec", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, default=None)
    parser.add_argument("--direct_asym_eig_mode", type=str, default="raw_right_largest_real_real")
    parser.add_argument("--shard_size", type=int, default=512)
    parser.add_argument("--store_pairwise_q", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rank_from_order(order: Sequence[int], n: int) -> np.ndarray:
    rank = np.empty(int(n), dtype=np.uint8)
    for idx, value in enumerate(order):
        rank[int(value)] = int(idx)
    return rank


def pairwise_q_from_rank(rank: np.ndarray) -> np.ndarray:
    rank = np.asarray(rank)
    q = np.where(rank[:, None] < rank[None, :], 1, -1).astype(np.int8)
    np.fill_diagonal(q, 0)
    return q


def split_for_record(record_index: int, split_spec: Dict) -> str | None:
    for split in ("train", "val", "test"):
        item = split_spec.get(split)
        if not isinstance(item, dict):
            continue
        start = int(item["record_start"])
        stop = int(item["record_stop_exclusive"])
        if start <= int(record_index) < stop:
            return split
    return None


def existing_shards(out_dir: Path) -> List[Path]:
    return sorted(out_dir.glob("*/shard_*.pt"))


def save_shard(path: Path, samples: List[Dict], pair_i: torch.Tensor, pair_j: torch.Tensor, metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "attention": torch.stack([sample["attention"] for sample in samples], dim=0).to(torch.float16),
        "teacher_order": torch.stack([sample["teacher_order"] for sample in samples], dim=0).to(torch.uint8),
        "teacher_rank": torch.stack([sample["teacher_rank"] for sample in samples], dim=0).to(torch.uint8),
        "reverse_order": torch.stack([sample["reverse_order"] for sample in samples], dim=0).to(torch.uint8),
        "reverse_rank": torch.stack([sample["reverse_rank"] for sample in samples], dim=0).to(torch.uint8),
        "pair_i": pair_i.to(torch.uint8),
        "pair_j": pair_j.to(torch.uint8),
        "pair_target": torch.stack([sample["pair_target"] for sample in samples], dim=0).to(torch.bool),
        "pair_sign": torch.stack([sample["pair_sign"] for sample in samples], dim=0).to(torch.int8),
        "record_index": torch.tensor([sample["record_index"] for sample in samples], dtype=torch.int64),
        "iter": torch.tensor([sample["iter"] for sample in samples], dtype=torch.int64),
        "layer": torch.tensor([sample["layer"] for sample in samples], dtype=torch.uint8),
        "head": torch.tensor([sample["head"] for sample in samples], dtype=torch.uint8),
        "source_sample_count": torch.tensor([sample["source_sample_count"] for sample in samples], dtype=torch.int64),
        "source_probe_seed": torch.tensor([sample["source_probe_seed"] for sample in samples], dtype=torch.int64),
        "eigval_real": torch.tensor([sample["eigval_real"] for sample in samples], dtype=torch.float32),
        "eigval_imag": torch.tensor([sample["eigval_imag"] for sample in samples], dtype=torch.float32),
        "eigval_abs": torch.tensor([sample["eigval_abs"] for sample in samples], dtype=torch.float32),
        "vector_std": torch.tensor([sample["vector_std"] for sample in samples], dtype=torch.float32),
        "metadata": metadata,
    }
    if "pairwise_q" in samples[0]:
        payload["pairwise_Q"] = torch.stack([sample["pairwise_q"] for sample in samples], dim=0).to(torch.int8)
    torch.save(payload, path)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = args.report_dir if args.report_dir is not None else args.out_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    if existing_shards(args.out_dir) and not bool(args.overwrite):
        raise FileExistsError(f"Found existing shards under {args.out_dir}; pass --overwrite to replace.")
    if bool(args.overwrite):
        for path in existing_shards(args.out_dir):
            path.unlink()

    split_spec = json.loads(args.split_spec.read_text(encoding="utf-8"))
    manifest_in = json.loads((args.all_head_dir / "manifest.json").read_text(encoding="utf-8"))
    summary_in = json.loads((args.all_head_dir / "collection_summary.json").read_text(encoding="utf-8"))
    source_shards = sorted(args.all_head_dir.glob("all_head_attn_shard_*.pt"))
    if not source_shards:
        raise FileNotFoundError(f"No all_head_attn_shard_*.pt files found under {args.all_head_dir}")

    num_blocks = int(summary_in.get("num_blocks", 64))
    num_layers = int(summary_in.get("num_layers", 4))
    num_heads = int(summary_in.get("num_heads", 8))
    pair_i, pair_j = torch.triu_indices(num_blocks, num_blocks, offset=1)
    pairs_per_sample = int(pair_i.numel())

    command = " ".join(shlex.quote(part) for part in sys.argv)
    metadata = {
        "version": 1,
        "format": "all_head_direct_asym_eig_pairwise_dataset",
        "source_all_head_dir": str(args.all_head_dir),
        "split_spec": str(args.split_spec),
        "direct_asym_eig_mode": str(args.direct_asym_eig_mode),
        "orientation": "raw_unoriented",
        "orientation_note": (
            "teacher_order/teacher_rank are raw direct-asym-eig outputs. "
            "reverse_order/reverse_rank are also stored. No original-order or loss-based raw/reverse selection is used here."
        ),
        "sample_definition": "one sample = one layer/head 64x64 attention matrix",
        "pair_definition": "pair_target[k,p]=True iff pair_i[p] precedes pair_j[p] under teacher_rank[k]",
        "q_definition": "pairwise_Q[i,j]=+1 iff block i precedes block j, -1 iff j precedes i, 0 on diagonal",
        "num_blocks": int(num_blocks),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "pairs_per_sample": int(pairs_per_sample),
        "store_pairwise_q": bool(args.store_pairwise_q),
        "command": command,
        "source_manifest": manifest_in,
        "source_summary": summary_in,
    }
    write_json(args.out_dir / "manifest.json", metadata)
    write_json(report_dir / "pairwise_command.json", {"command": command})

    pending: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    shard_idx: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_records: Dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    errors: List[Dict] = []
    eig_abs_values: List[float] = []
    vector_std_values: List[float] = []
    record_offset = 0
    t0 = time.perf_counter()

    for source_idx, source_path in enumerate(source_shards):
        payload = torch.load(source_path, map_location="cpu")
        attention = payload["attention"]
        iters = payload["iter"].long()
        sample_counts = payload["sample_count"].long()
        probe_seeds = payload["probe_seed"].long()
        records_in_shard = int(attention.shape[0])
        for local_record in range(records_in_shard):
            record_index = int(record_offset + local_record)
            split = split_for_record(record_index, split_spec)
            if split is None:
                continue
            split_records[split].add(record_index)
            for layer_idx in range(int(attention.shape[1])):
                for head_idx in range(int(attention.shape[2])):
                    matrix_t = attention[local_record, layer_idx, head_idx].to(torch.float32)
                    matrix = matrix_t.numpy().astype(np.float64)
                    try:
                        order, eig_meta = direct_asym_eig_order(matrix, str(args.direct_asym_eig_mode))
                    except Exception as exc:
                        errors.append(
                            {
                                "source_shard": str(source_path),
                                "record_index": int(record_index),
                                "iter": int(iters[local_record].item()),
                                "layer": int(layer_idx),
                                "head": int(head_idx),
                                "error": str(exc),
                            }
                        )
                        continue
                    if len(order) != num_blocks or sorted(order) != list(range(num_blocks)):
                        errors.append(
                            {
                                "source_shard": str(source_path),
                                "record_index": int(record_index),
                                "iter": int(iters[local_record].item()),
                                "layer": int(layer_idx),
                                "head": int(head_idx),
                                "error": "direct_asym_eig produced an invalid permutation",
                            }
                        )
                        continue
                    reverse_order = list(reversed(order))
                    rank = rank_from_order(order, num_blocks)
                    reverse_rank = rank_from_order(reverse_order, num_blocks)
                    target_np = rank[pair_i.numpy()] < rank[pair_j.numpy()]
                    sign_np = np.where(target_np, 1, -1).astype(np.int8)
                    sample = {
                        "attention": matrix_t.to(torch.float16),
                        "teacher_order": torch.tensor(order, dtype=torch.uint8),
                        "teacher_rank": torch.from_numpy(rank),
                        "reverse_order": torch.tensor(reverse_order, dtype=torch.uint8),
                        "reverse_rank": torch.from_numpy(reverse_rank),
                        "pair_target": torch.from_numpy(target_np),
                        "pair_sign": torch.from_numpy(sign_np),
                        "record_index": int(record_index),
                        "iter": int(iters[local_record].item()),
                        "layer": int(layer_idx),
                        "head": int(head_idx),
                        "source_sample_count": int(sample_counts[local_record].item()),
                        "source_probe_seed": int(probe_seeds[local_record].item()),
                        "eigval_real": float(eig_meta.get("eigval_real", float("nan"))),
                        "eigval_imag": float(eig_meta.get("eigval_imag", float("nan"))),
                        "eigval_abs": float(eig_meta.get("eigval_abs", float("nan"))),
                        "vector_std": float(eig_meta.get("vector_std", float("nan"))),
                    }
                    if bool(args.store_pairwise_q):
                        sample["pairwise_q"] = torch.from_numpy(pairwise_q_from_rank(rank))
                    eig_abs_values.append(float(sample["eigval_abs"]))
                    vector_std_values.append(float(sample["vector_std"]))
                    pending[split].append(sample)
                    split_counts[split] += 1
                    if len(pending[split]) >= int(args.shard_size):
                        out_path = args.out_dir / split / f"shard_{shard_idx[split]:03d}.pt"
                        save_shard(out_path, pending[split], pair_i, pair_j, metadata)
                        shard_idx[split] += 1
                        pending[split] = []
        record_offset += records_in_shard
        print(
            f"[pairwise] source={source_idx + 1}/{len(source_shards)} "
            f"records_seen={record_offset} train={split_counts['train']} val={split_counts['val']} errors={len(errors)}",
            flush=True,
        )

    for split, samples in list(pending.items()):
        if not samples:
            continue
        out_path = args.out_dir / split / f"shard_{shard_idx[split]:03d}.pt"
        save_shard(out_path, samples, pair_i, pair_j, metadata)
        shard_idx[split] += 1
        pending[split] = []

    elapsed = time.perf_counter() - t0
    summary = {
        "dataset_dir": str(args.out_dir),
        "source_all_head_dir": str(args.all_head_dir),
        "elapsed_sec": float(elapsed),
        "records_seen": int(record_offset),
        "pairs_per_sample": int(pairs_per_sample),
        "splits": {
            split: {
                "records": int(len(split_records[split])),
                "head_matrix_samples": int(split_counts[split]),
                "pair_instances": int(split_counts[split] * pairs_per_sample),
                "shards": int(shard_idx[split]),
            }
            for split in ("train", "val", "test")
            if split_counts[split] > 0
        },
        "eigval_abs_mean": float(np.mean(eig_abs_values)) if eig_abs_values else None,
        "eigval_abs_median": float(np.median(eig_abs_values)) if eig_abs_values else None,
        "vector_std_mean": float(np.mean(vector_std_values)) if vector_std_values else None,
        "vector_std_median": float(np.median(vector_std_values)) if vector_std_values else None,
        "errors": errors,
    }
    write_json(args.out_dir / "collection_summary.json", summary)
    write_json(report_dir / "pairwise_collection_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
