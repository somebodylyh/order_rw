#!/usr/bin/env python3
"""Per-seed val_ori_l2r_block curves: random / CDL / g_beta / L2R."""
from __future__ import annotations

import csv, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = ROOT / "block_lo_arm_order_network" / "probe_results"

# ── per-seed run groups ────────────────────────────────────────────────
SEEDS = {
    "seed124": {
        "random":     PROBE / "random_baseline_b1_headscan_seed124",
        "CDL from10k": PROBE / "cdl_teacher_seed124_from10k_l0h0_b1",
        "CDL from20k": PROBE / "cdl_teacher_from20k_seed124",
        "CDL from40k": PROBE / "cdl_teacher_from40k_seed124",
        "g_beta from20k": PROBE / "frozen_beta_from20k_seed124",
    },
    "seed2": {
        "random":     PROBE / "random_baseline_continuous_jun08_seed2",
        "CDL from10k": PROBE / "cdl_teacher_seed2_from10k_l0h2",
        "CDL from20k": PROBE / "cdl_teacher_seed2_from20k_l0h2",
        "CDL from40k": PROBE / "cdl_teacher_seed2_from40k_l0h2",
        "g_beta from10k": PROBE / "frozen_beta_b1_seed2_from10000_l0h2_headmaps_20260623_0110",
        "g_beta from20k": PROBE / "frozen_beta_b1_seed2_from20000",
        "g_beta from40k": PROBE / "frozen_beta_b1_seed2_from40000",
    },
}

# colour palette per policy family
COLORS = {
    "random":        "#333333",
    "CDL from10k":   "#d62728",
    "CDL from20k":   "#e57373",
    "CDL from40k":   "#ef9a9a",
    "g_beta from10k":"#1f77b4",
    "g_beta from20k":"#4da6e8",
    "g_beta from40k":"#90caf9",
}

LINESTYLES = {
    "random": "-",
    "CDL from10k": "-",
    "CDL from20k": "--",
    "CDL from40k": ":",
    "g_beta from10k": "-",
    "g_beta from20k": "--",
    "g_beta from40k": ":",
}

OUT_DIR = ROOT / "reports"


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


def plot_seed(seed_name: str, runs: dict, l2r_path: pathlib.Path | None,
              ax_full, ax_zoom, ymax=4.0):
    """Plot one seed's runs onto the given axes."""
    for label, d in runs.items():
        steps, vals = load_curve(d)
        if steps is None:
            print(f"  SKIP {seed_name}/{label}: no eval_curve.tsv")
            continue
        color = COLORS.get(label, "#666666")
        ls = LINESTYLES.get(label, "-")
        lw = 1.4 if "CDL" in label or "g_beta" in label else 1.0
        ax_full.plot(steps, vals, color=color, linestyle=ls, linewidth=lw,
                     alpha=0.85, label=f"{seed_name} {label}")
        mask = steps >= 38000
        if mask.sum() >= 2:
            ax_zoom.plot(steps[mask], vals[mask], color=color, linestyle=ls,
                         linewidth=lw, alpha=0.85)

    # L2R reference
    if l2r_path is not None:
        steps, vals = load_curve(l2r_path)
        if steps is not None:
            ax_full.plot(steps, vals, color="#2ca02c", linestyle="-", linewidth=1.2,
                         alpha=0.75, label=f"L2R ({seed_name})")
            mask = steps >= 38000
            if mask.sum() >= 2:
                ax_zoom.plot(steps[mask], vals[mask], color="#2ca02c",
                             linestyle="-", linewidth=1.2, alpha=0.75)

    ax_full.set_ylim(top=ymax)
    ax_zoom.set_ylim(top=ymax)


def main():
    for seed_name, runs in SEEDS.items():
        fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 6.2))

        # match L2R seed
        if seed_name == "seed2":
            l2r = PROBE / "l2r_continuous_seed2"
        elif seed_name == "seed124":
            l2r = None  # no seed124 L2R
        else:
            l2r = None

        plot_seed(seed_name, runs, l2r, ax_full, ax_zoom)

        ax_full.set_xlabel("step", fontsize=11)
        ax_full.set_ylabel("val_ori_l2r_block", fontsize=11)
        ax_full.set_title(f"{seed_name} — full curves", fontsize=13)
        ax_full.legend(fontsize=7.5, ncol=1, loc="upper right", framealpha=0.85)
        ax_full.grid(True, alpha=0.2)
        ax_full.set_xlim(0, 62000)

        ax_zoom.set_xlabel("step", fontsize=11)
        ax_zoom.set_ylabel("val_ori_l2r_block", fontsize=11)
        ax_zoom.set_title(f"{seed_name} — zoom 40k–60k", fontsize=13)
        ax_zoom.grid(True, alpha=0.2)

        out = OUT_DIR / f"old_gbeta_{seed_name}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
