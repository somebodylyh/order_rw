#!/usr/bin/env python3
"""Plot val_ori_l2r_block curves: random / CDL / g_beta / L2R comparison."""
from __future__ import annotations

import csv
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = ROOT / "block_lo_arm_order_network" / "probe_results"

# ── run definitions ──────────────────────────────────────────────────
RUNS = [
    # seed124 (clean 60k)
    ("random (seed124)",       PROBE / "random_baseline_b1_headscan_seed124",           "#333333", "-",  1.2),
    ("CDL from10k (s124)",     PROBE / "cdl_teacher_seed124_from10k_l0h0_b1",           "#d62728", "-",  1.0),
    ("CDL from20k (s124)",     PROBE / "cdl_teacher_from20k_seed124",                   "#d62728", "--", 1.0),
    ("CDL from40k (s124)",     PROBE / "cdl_teacher_from40k_seed124",                   "#d62728", ":",  1.0),
    ("g_beta from20k (s124)",  PROBE / "frozen_beta_from20k_seed124",                   "#1f77b4", "--", 1.2),
    # seed2
    ("random (seed2)",         PROBE / "random_baseline_continuous_jun08_seed2",        "#999999", "-",  1.0),
    ("CDL from10k (s2)",       PROBE / "cdl_teacher_seed2_from10k_l0h2",                "#ff7f0e", "-",  1.0),
    ("CDL from20k (s2)",       PROBE / "cdl_teacher_seed2_from20k_l0h2",                "#ff7f0e", "--", 1.0),
    ("CDL from40k (s2)",       PROBE / "cdl_teacher_seed2_from40k_l0h2",                "#ff7f0e", ":",  1.0),
    ("g_beta from10k (s2)",    PROBE / "frozen_beta_b1_seed2_from10000_l0h2_headmaps_20260623_0110", "#2ca02c", "-",  1.0),
    ("g_beta from20k (s2)",    PROBE / "frozen_beta_b1_seed2_from20000",                "#2ca02c", "--", 1.2),
    ("g_beta from40k (s2)",    PROBE / "frozen_beta_b1_seed2_from40000",                "#2ca02c", ":",  1.0),
    # L2R reference (clean 0→60k, no resume)
    ("L2R (seed2)",            PROBE / "l2r_continuous_seed2",                          "#9467bd", "-",  1.2),
    ("L2R (seed42)",           PROBE / "l2r_continuous_seed42",                         "#8e44ad", "-",  1.0),
]

OUT = ROOT / "reports" / "old_gbeta_comparison.png"


def load_curve(path: pathlib.Path):
    tsv = path / "eval_curve.tsv"
    if not tsv.exists():
        return None, None
    with tsv.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    steps = [int(r["step"]) for r in rows]
    vals = []
    for r in rows:
        v = r.get("val_ori_l2r_block", "")
        vals.append(float(v) if v and v != "nan" else np.nan)
    return np.array(steps), np.array(vals)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # ── Left: full view ──
    for label, d, color, ls, lw in RUNS:
        steps, vals = load_curve(d)
        if steps is None:
            print(f"  SKIP {label}: no eval_curve.tsv")
            continue
        ax1.plot(steps, vals, color=color, linestyle=ls, linewidth=lw,
                 alpha=0.85, label=label)

    ax1.set_xlabel("training step", fontsize=11)
    ax1.set_ylabel("val_ori_l2r_block", fontsize=11)
    ax1.set_title("Full training curves (all runs)", fontsize=12)
    ax1.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.8)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(0, 62000)
    ax1.set_ylim(top=4.0)

    # ── Right: zoom 40k-60k (the comparison region) ──
    for label, d, color, ls, lw in RUNS:
        steps, vals = load_curve(d)
        if steps is None:
            continue
        mask = steps >= 38000
        if mask.sum() < 2:
            continue
        ax2.plot(steps[mask], vals[mask], color=color, linestyle=ls,
                 linewidth=lw, alpha=0.85)

    ax2.set_xlabel("training step", fontsize=11)
    ax2.set_ylabel("val_ori_l2r_block", fontsize=11)
    ax2.set_title("Zoom: 40k – 60k", fontsize=12)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(top=4.0)

    # ── Annotation: reference levels ──
    ax2.axhline(3.477, color="#333333", linestyle="--", linewidth=0.7, alpha=0.5)
    ax2.text(60500, 3.477, "rand s124 3.477", fontsize=7, color="#333333", va="center")
    ax2.axhline(3.445, color="#999999", linestyle="--", linewidth=0.7, alpha=0.5)
    ax2.text(60500, 3.445, "rand s2 3.445", fontsize=7, color="#999999", va="center")
    ax2.axhline(3.364, color="#9467bd", linestyle="--", linewidth=0.5, alpha=0.35)
    ax2.text(60500, 3.364, "L2R s2 3.364", fontsize=6, color="#9467bd", va="center")
    ax2.axhline(3.338, color="#8e44ad", linestyle="--", linewidth=0.5, alpha=0.35)
    ax2.text(60500, 3.338, "L2R s42 3.338", fontsize=6, color="#8e44ad", va="center")

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
