#!/usr/bin/env python3
"""Add hard weighted-consensus CDL orders to an existing g_beta dataset."""

from __future__ import annotations

import argparse
import os
import pathlib
import tempfile

import numpy as np

from build_l0_dynamic_gbeta_dataset import weighted_consensus_order


def _load_payload(path: str) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {key: z[key].copy() for key in z.files}


def _add_field(payload: dict) -> dict:
    if "teacher_ranks" not in payload:
        raise KeyError("teacher_ranks is required")
    if "teacher_weights" not in payload:
        raise KeyError("teacher_weights is required")

    ranks = payload["teacher_ranks"]
    weights = payload["teacher_weights"]
    if ranks.ndim != 3:
        raise ValueError(f"teacher_ranks must be [M, H, N], got {ranks.shape}")
    if weights.shape != ranks.shape[:2]:
        raise ValueError(
            "teacher_weights must match teacher_ranks [M, H], "
            f"got {weights.shape} vs {ranks.shape[:2]}"
        )

    payload["teacher_consensus_order"] = np.stack([
        weighted_consensus_order(ranks[m], weights[m])
        for m in range(ranks.shape[0])
    ])
    return payload


def add_consensus_order(src_path: str, dst_path: str) -> str:
    payload = _add_field(_load_payload(src_path))
    dst = pathlib.Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **payload)
    return str(dst)


def add_consensus_order_in_place(src_path: str) -> str:
    src = pathlib.Path(src_path)
    src.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{src.stem}.", suffix=".tmp.npz", dir=str(src.parent),
    )
    os.close(fd)
    try:
        add_consensus_order(str(src), tmp_name)
        os.replace(tmp_name, src)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return str(src)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dst")
    group.add_argument("--in-place", action="store_true")
    args = parser.parse_args()

    if args.in_place:
        output = add_consensus_order_in_place(args.src)
    else:
        output = add_consensus_order(args.src, args.dst)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
