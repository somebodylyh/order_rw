#!/usr/bin/env python3
"""Diagnose 65-node None-separated block graph discovery from A_with_none."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))

from none_separated_block_graph import (  # noqa: E402
    build_none_separated_B,
    content_label_permutation_control,
    discovery_metrics,
    entry_shuffled_control,
    rollout_from_none,
)


def _control_summary(B65: np.ndarray, kind: str, seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        if kind == "entry_shuffle":
            Bc = entry_shuffled_control(B65, seed=seed)
        elif kind == "content_label_permutation":
            Bc = content_label_permutation_control(B65, seed=seed)
        else:
            raise ValueError(kind)
        sigma = rollout_from_none(Bc)
        rows.append({"kind": kind, "seed": seed, **discovery_metrics(sigma)})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a-with-none", required=True, help="A[target_block, source_node] .npy")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-b65", default=None)
    p.add_argument("--control-seeds", type=int, nargs="*", default=list(range(20)))
    args = p.parse_args()

    A = np.load(args.a_with_none)
    B65 = build_none_separated_B(A)
    sigma = rollout_from_none(B65)
    result = {
        "input": args.a_with_none,
        "node_protocol": {
            "node0": "None / BOS start node",
            "content_node_i": "node = 1 + physical block i",
            "content_block_i": "x_{4i}..x_{4i+3}",
        },
        "shape": {"A_with_none": list(A.shape), "B65": list(B65.shape)},
        "main": discovery_metrics(sigma),
        "controls": (
            _control_summary(B65, "entry_shuffle", args.control_seeds)
            + _control_summary(B65, "content_label_permutation", args.control_seeds)
        ),
    }
    if args.out_b65:
        np.save(args.out_b65, B65.astype(np.float32))
        result["B65_npy"] = args.out_b65

    out_json = pathlib.Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
