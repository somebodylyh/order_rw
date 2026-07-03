#!/usr/bin/env python3
"""Merge compatible pairwise-operator distillation dataset shards.

The trainer consumes one dataset directory with train/ and val/ shard_*.pt
files. This utility creates such a directory from several compatible dataset
roots by hard-linking shards with fresh monotonically increasing names.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge pairwise operator dataset shards.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, action="append", required=True)
    parser.add_argument("--name", type=str, default="merged_pairwise_operator_dataset")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--copy", action="store_true", help="Copy shards instead of hard-linking.")
    return parser.parse_args()


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def link_or_copy(src: Path, dst: Path, copy: bool) -> str:
    if copy:
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy_fallback"


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        if not args.force:
            raise FileExistsError(f"{args.out_dir} exists; pass --force to replace it.")
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (args.out_dir / split).mkdir(parents=True, exist_ok=True)

    sources = []
    shard_counts = {"train": 0, "val": 0}
    sample_counts = {"train": 0, "val": 0}
    link_modes = {}

    for source_idx, dataset_dir in enumerate(args.dataset_dir):
        dataset_dir = dataset_dir.resolve()
        source = {
            "index": int(source_idx),
            "dataset_dir": str(dataset_dir),
            "manifest": load_json(dataset_dir / "manifest.json"),
            "collection_summary": load_json(dataset_dir / "collection_summary.json"),
            "splits": {},
        }
        for split in ("train", "val"):
            paths = sorted((dataset_dir / split).glob("shard_*.pt"))
            if not paths:
                raise FileNotFoundError(f"No {split}/shard_*.pt files under {dataset_dir}")
            split_rows = []
            for path in paths:
                dst = args.out_dir / split / f"shard_{shard_counts[split]:04d}.pt"
                mode = link_or_copy(path, dst, bool(args.copy))
                link_modes[mode] = int(link_modes.get(mode, 0)) + 1
                split_rows.append({"src": str(path), "dst": str(dst), "mode": mode})
                shard_counts[split] += 1
            summary = source["collection_summary"] or {}
            sample_counts[split] += int((summary.get("samples") or {}).get(split, 0))
            source["splits"][split] = {
                "num_shards": int(len(paths)),
                "summary_samples": int((summary.get("samples") or {}).get(split, 0)),
                "shards": split_rows,
            }
        sources.append(source)

    manifest = {
        "name": str(args.name),
        "format": "merged_pairwise_operator_dataset",
        "out_dir": str(args.out_dir.resolve()),
        "sources": sources,
        "shard_counts": shard_counts,
        "summary_sample_counts": sample_counts,
        "link_modes": link_modes,
    }
    (args.out_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "shard_counts": shard_counts,
        "summary_sample_counts": sample_counts,
        "link_modes": link_modes,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
