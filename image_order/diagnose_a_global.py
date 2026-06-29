"""Compact A_global diagnostics — argmax-top-1 spatial locality + readiness signal.

Reads an A_global.npy (64x64 patch-patch attention, raster index) and reports:
  * mean_manh           = avg Manhattan distance from query q to argmax_k A[q,k]
  * P(d<=1), P(d<=2)    = fraction of queries whose top-1 lands within 1/2 patches
  * same_q              = fraction of queries whose top-1 stays in the same 2x2 quad
  * readiness_signal s  = std(in_degree) / mean(|A|)  where in_degree[k] = sum_q A[q,k]
  * hub_score           = max(in_degree) / mean(in_degree)

Also reports the same stats on two controls so it's clear what "random" looks like:
  * uniform random A    (sample uniform [0,1] then zero-diag, same shape)
  * shuffled A          (per-row column permutation of the real A — kills spatial info,
                          preserves row marginals; this is the cleanest "no spatial"
                          baseline)

Usage:
    python image_order/diagnose_a_global.py --a-global path/to/A_global.npy \\
        --label "ImageNet32 E1 baseline10k"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GRID = 8  # 8x8 patch grid for both toy CIFAR and ImageNet32


def grid_xy(idx: int):
    return idx // GRID, idx % GRID


def manhattan(a: int, b: int) -> int:
    ra, ca = grid_xy(a)
    rb, cb = grid_xy(b)
    return abs(ra - rb) + abs(ca - cb)


def same_quad(a: int, b: int) -> bool:
    ra, ca = grid_xy(a)
    rb, cb = grid_xy(b)
    return (ra // 2 == rb // 2) and (ca // 2 == cb // 2)


_MANH = np.array([[manhattan(i, j) for j in range(64)] for i in range(64)], dtype=np.int64)
_SAMEQ = np.array([[same_quad(i, j) for j in range(64)] for i in range(64)], dtype=np.bool_)


def _top1_keys(A: np.ndarray) -> np.ndarray:
    """For each query row, argmax key index, ignoring the diagonal."""
    A = A.copy()
    np.fill_diagonal(A, -np.inf)
    return A.argmax(axis=1)


def stats_for(A: np.ndarray, name: str) -> dict:
    N = A.shape[0]
    assert A.shape == (N, N)
    top1 = _top1_keys(A)
    qs = np.arange(N)

    manh = _MANH[qs, top1]
    sameq = _SAMEQ[qs, top1]

    in_deg = A.sum(axis=0)          # how much attention each key receives
    a_mean = np.abs(A).mean()
    readiness_s = float(in_deg.std() / max(a_mean, 1e-12))
    hub_score = float(in_deg.max() / max(in_deg.mean(), 1e-12))

    return {
        "label": name,
        "A_mean": float(A.mean()),
        "A_max": float(A.max()),
        "mean_manh": float(manh.mean()),
        "median_manh": float(np.median(manh)),
        "p_dist_le1": float((manh <= 1).mean()),
        "p_dist_le2": float((manh <= 2).mean()),
        "same_quad": float(sameq.mean()),
        "readiness_signal_s": readiness_s,
        "hub_score": hub_score,
    }


def shuffle_columns_per_row(A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute the columns within each row independently. Kills spatial structure,
    preserves row sums and value distribution."""
    out = np.empty_like(A)
    N = A.shape[1]
    for i in range(A.shape[0]):
        perm = rng.permutation(N)
        out[i] = A[i, perm]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a-global", type=Path, required=True,
                   help="Path to A_global.npy (NxN, raster index)")
    p.add_argument("--label", type=str, default="A_global",
                   help="Short label for the real A (e.g. 'E1 baseline10k')")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", type=Path, default=None,
                   help="Optional: write all rows as JSON to this path")
    args = p.parse_args()

    A = np.load(args.a_global)
    if A.shape != (64, 64):
        raise ValueError(f"Expected (64, 64), got {A.shape}")

    rng = np.random.default_rng(args.seed)

    rows = []
    rows.append(stats_for(A, args.label))

    A_shuf = shuffle_columns_per_row(A, rng)
    rows.append(stats_for(A_shuf, f"{args.label} [shuffled columns]"))

    A_rand = rng.random((64, 64)).astype(np.float32)
    np.fill_diagonal(A_rand, 0.0)
    rows.append(stats_for(A_rand, "uniform random"))

    print(f"\n{'label':40s} {'mean_manh':>10s} {'P(d<=1)':>8s} {'P(d<=2)':>8s} {'same_q':>7s} {'s_readiness':>12s} {'hub_score':>10s}")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['label']:40s} {r['mean_manh']:10.3f} "
            f"{r['p_dist_le1']:8.3f} {r['p_dist_le2']:8.3f} "
            f"{r['same_quad']:7.3f} {r['readiness_signal_s']:12.4f} "
            f"{r['hub_score']:10.3f}"
        )

    print("\nReference values from previous experiments:")
    print(f"  CIFAR baseline10k (toy E0)   : mean_manh ~ 2.4, s ~ 0.5-1.0")
    print(f"  ImageNet64 patch2x2 (E3)     : mean_manh ~ 11.7, s ~ 95")
    print(f"  uniform random on 8x8 grid   : mean_manh ~ 4.67 (theoretical)")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
