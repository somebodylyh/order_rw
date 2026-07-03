"""
Visualize directed pair scores exported by pair_margin_stability_probe.py.

Default usage for the seq256 signed_drop=0.2 sweep:

python scripts/analysis/visualize_pair_score_heatmaps.py

The default frame is original-frame, so cells (i, i+1) and (i, i-1) directly
show the adjacent original-order directions under permuted-data checkpoints.
"""

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BLOCK64_CSV = (
    "Report/analysis/signed_drop_sweep_seq256_fuller/block64/drop_p0p2/pair_stats.csv"
)
DEFAULT_TOKEN_CSV = (
    "Report/analysis/signed_drop_sweep_seq256_fuller/token_block1/drop_p0p2/pair_stats.csv"
)
DEFAULT_OUT_DIR = "Report/analysis/signed_drop_sweep_seq256_fuller/heatmaps_drop_p0p2"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build heatmaps from pair_stats.csv. Each heatmap cell is a directed "
            "pair score, with optional reverse-score filling from mean_reverse_score."
        )
    )
    parser.add_argument("--block64_csv", type=Path, default=Path(DEFAULT_BLOCK64_CSV))
    parser.add_argument("--token_csv", type=Path, default=Path(DEFAULT_TOKEN_CSV))
    parser.add_argument(
        "--csv",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Custom CSV input. Can be repeated. When provided, overrides the "
            "default block64/token inputs."
        ),
    )
    parser.add_argument("--out_dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--frame",
        type=str,
        default="original",
        choices=["original", "current"],
        help="Coordinate frame for the heatmap axes.",
    )
    parser.add_argument(
        "--score_col",
        type=str,
        default="mean_score",
        help="CSV column to place at cell (first, second). Usually mean_score.",
    )
    parser.add_argument(
        "--reverse_score_col",
        type=str,
        default="mean_reverse_score",
        help="CSV column used to fill cell (second, first).",
    )
    parser.add_argument(
        "--no_fill_reverse",
        action="store_true",
        help="Only fill (first, second) from score_col; do not fill the reverse cell.",
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=1.0,
        help="Symmetric percentile clipping per heatmap. 0 disables clipping.",
    )
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--mark_adjacent", action="store_true")
    parser.add_argument(
        "--save_matrix",
        action="store_true",
        help="Also save the numeric score matrix as .npy.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_custom_csv_inputs(raw_inputs):
    inputs = []
    for raw_input in raw_inputs:
        if "=" not in raw_input:
            raise ValueError(f"--csv must use LABEL=PATH format, got: {raw_input}")
        label, path = raw_input.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"--csv must use non-empty LABEL=PATH format, got: {raw_input}")
        safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
        inputs.append((safe_label, resolve_path(Path(path))))
    return inputs


def parse_singleton_list(raw_value):
    try:
        value = ast.literal_eval(str(raw_value))
    except (ValueError, SyntaxError):
        return None
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return int(value[0])
    return None


def finite_float(raw_value):
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(value):
        return value
    return None


def row_ids(row, frame: str):
    if frame == "current":
        return int(row["first_unit"]), int(row["second_unit"])

    first = parse_singleton_list(row.get("first_blocks_original", ""))
    second = parse_singleton_list(row.get("second_blocks_original", ""))
    if first is None or second is None:
        return None, None
    return first, second


def load_score_matrix(csv_path: Path, frame: str, score_col: str, reverse_score_col: str, fill_reverse: bool):
    entries = []
    skipped_non_singleton = 0
    skipped_nonfinite = 0

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            first, second = row_ids(row, frame)
            if first is None or second is None:
                skipped_non_singleton += 1
                continue

            score = finite_float(row.get(score_col))
            if score is None:
                skipped_nonfinite += 1
            else:
                entries.append((first, second, score))

            if fill_reverse:
                reverse_score = finite_float(row.get(reverse_score_col))
                if reverse_score is None:
                    skipped_nonfinite += 1
                else:
                    entries.append((second, first, reverse_score))

    if not entries:
        raise ValueError(f"No finite score entries found in {csv_path}")

    size = max(max(first, second) for first, second, _ in entries) + 1
    sums = np.zeros((size, size), dtype=np.float64)
    counts = np.zeros((size, size), dtype=np.int64)

    for first, second, score in entries:
        sums[first, second] += score
        counts[first, second] += 1

    matrix = np.full((size, size), np.nan, dtype=np.float64)
    np.divide(sums, counts, out=matrix, where=counts > 0)
    np.fill_diagonal(matrix, np.nan)

    diagnostics = {
        "csv_path": str(csv_path),
        "frame": frame,
        "score_col": score_col,
        "reverse_score_col": reverse_score_col if fill_reverse else None,
        "size": int(size),
        "num_finite_cells": int(np.isfinite(matrix).sum()),
        "coverage": float(np.isfinite(matrix).sum() / max(1, size * (size - 1))),
        "skipped_non_singleton_rows": int(skipped_non_singleton),
        "skipped_nonfinite_scores": int(skipped_nonfinite),
    }
    return matrix, diagnostics


def clipped_limits(matrix: np.ndarray, clip_percentile: float):
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return None, None
    if clip_percentile <= 0:
        return float(finite.min()), float(finite.max())
    lo = np.percentile(finite, clip_percentile)
    hi = np.percentile(finite, 100.0 - clip_percentile)
    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        return float(finite.min()), float(finite.max())
    return float(lo), float(hi)


def plot_heatmap(matrix: np.ndarray, label: str, out_path: Path, args, diagnostics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = matrix[np.isfinite(matrix)]
    vmin, vmax = clipped_limits(matrix, args.clip_percentile)

    cmap = plt.get_cmap(args.cmap).copy()
    cmap.set_bad(color="#eeeeee")

    size = matrix.shape[0]
    fig_size = 7.0 if size <= 80 else 9.5
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        interpolation="nearest",
        origin="upper",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(
        f"{label}: {args.score_col}, {args.frame}-frame\n"
        f"finite={diagnostics['num_finite_cells']} "
        f"coverage={diagnostics['coverage']:.3f}"
    )
    ax.set_xlabel("second id")
    ax.set_ylabel("first id")

    tick_step = 8 if size <= 80 else 32
    ticks = np.arange(0, size, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    if args.mark_adjacent and size > 1:
        marker_size = max(1.0, 1800.0 / size)
        upper = np.arange(size - 1)
        lower = np.arange(1, size)
        ax.scatter(
            upper + 1,
            upper,
            s=marker_size,
            marker="s",
            facecolors="none",
            edgecolors="#ff3b30",
            linewidths=0.35,
            label="i -> i+1",
        )
        ax.scatter(
            upper,
            lower,
            s=marker_size,
            marker="s",
            facecolors="none",
            edgecolors="#ff9500",
            linewidths=0.35,
            label="i -> i-1",
        )
        ax.legend(loc="upper right", fontsize=7, frameon=True)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(args.score_col)

    if finite.size:
        ax.text(
            0.01,
            -0.08,
            f"min={finite.min():.4g} mean={finite.mean():.4g} max={finite.max():.4g}",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="top",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        inputs = parse_custom_csv_inputs(args.csv)
    else:
        inputs = [
            ("block64", resolve_path(args.block64_csv)),
            ("token_block1", resolve_path(args.token_csv)),
        ]

    all_diagnostics = {}
    for label, csv_path in inputs:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        matrix, diagnostics = load_score_matrix(
            csv_path=csv_path,
            frame=args.frame,
            score_col=args.score_col,
            reverse_score_col=args.reverse_score_col,
            fill_reverse=not args.no_fill_reverse,
        )

        stem = f"{label}_{args.frame}_{args.score_col}"
        png_path = out_dir / f"{stem}.png"
        plot_heatmap(matrix, label, png_path, args, diagnostics)
        diagnostics["png_path"] = str(png_path)

        if args.save_matrix:
            npy_path = out_dir / f"{stem}.npy"
            np.save(npy_path, matrix)
            diagnostics["npy_path"] = str(npy_path)

        all_diagnostics[label] = diagnostics
        print(f"[heatmap] wrote {png_path}")

    meta_path = out_dir / f"heatmap_meta_{args.frame}_{args.score_col}.json"
    meta_path.write_text(json.dumps(all_diagnostics, indent=2), encoding="utf-8")
    print(f"[heatmap] wrote {meta_path}")


if __name__ == "__main__":
    main()
