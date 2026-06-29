"""
Multi-start frozen_beta comparison — seed2 vs seed42, separate panels.
Plots ori_l2r vs absolute step for each seed's baseline + from10k/20k/40k.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

BASE = Path("/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/probe_results")

# ── Data sources ──────────────────────────────────────────────────
SEEDS = {
    "seed2 (L0H2)": {
        "baseline": BASE / "random_baseline_continuous_jun08_seed2/eval_curve.tsv",
        "from10k":  BASE / "frozen_beta_seed2_from10k_l0h2/eval_curve.tsv",
        "from20k":  BASE / "frozen_beta_seed2_from20k_l0h2/eval_curve.tsv",
        "from40k":  BASE / "frozen_beta_seed2_from40k_l0h2/eval_curve.tsv",
    },
    "seed42 (L0H4)": {
        "baseline": BASE / "random_baseline_continuous_jun05/eval_curve.tsv",
        "from10k":  BASE / "frozen_beta_random_jun05_from10k/eval_curve.tsv",
        "from20k":  BASE / "frozen_beta_random_jun05_from20k/eval_curve.tsv",
        "from40k":  BASE / "frozen_beta_random_jun05_from40k/eval_curve.tsv",
    },
}

COLORS = {
    "baseline": "#333333",
    "from10k":  "#e41a1c",
    "from20k":  "#377eb8",
    "from40k":  "#4daf4a",
}
LABELS = {
    "baseline": "ori-L2R baseline",
    "from10k":  "hook from 10k",
    "from20k":  "hook from 20k",
    "from40k":  "hook from 40k",
}
MARKERS = {"from10k": 15000, "from20k": 25000, "from40k": 45000}  # α=1 step for each
MARKER_COLORS = {"from10k": "#e41a1c", "from20k": "#377eb8", "from40k": "#4daf4a"}


def load_curve(path):
    df = pd.read_csv(path, sep="\t")
    df = df[["step", "val_ori_l2r_block"]].dropna()
    return df


# ── Plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=False)

for ax, (seed_name, paths) in zip(axes, SEEDS.items()):
    # ── baseline ──
    bl = load_curve(paths["baseline"])
    ax.plot(bl["step"], bl["val_ori_l2r_block"],
            color=COLORS["baseline"], linewidth=1.2, alpha=0.7,
            label=LABELS["baseline"])

    # ── frozen_beta curves ──
    for run_key in ["from10k", "from20k", "from40k"]:
        df = load_curve(paths[run_key])
        ax.plot(df["step"], df["val_ori_l2r_block"],
                color=COLORS[run_key], linewidth=1.8,
                label=LABELS[run_key])

        # Mark α=1 point (end of warmup)
        alpha1_step = MARKERS[run_key]
        row = df[df["step"] == alpha1_step]
        if not row.empty:
            ori = row["val_ori_l2r_block"].values[0]
            ax.scatter(alpha1_step, ori, color=COLORS[run_key],
                       s=60, zorder=5, edgecolors="white", linewidth=0.8)
            ax.annotate(f"α=1\n{ori:.3f}",
                        (alpha1_step, ori),
                        textcoords="offset points", xytext=(0, -22),
                        fontsize=7.5, color=COLORS[run_key],
                        ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor=COLORS[run_key], alpha=0.85))

        # Mark final point
        final = df.iloc[-1]
        ax.scatter(final["step"], final["val_ori_l2r_block"],
                   color=COLORS[run_key], s=60, zorder=5,
                   edgecolors="white", linewidth=0.8, marker="D")
        offset = (8, -18) if run_key != "from40k" else (8, 8)
        ax.annotate(f"{final['val_ori_l2r_block']:.3f}",
                    (final["step"], final["val_ori_l2r_block"]),
                    textcoords="offset points", xytext=offset,
                    fontsize=8, color=COLORS[run_key], fontweight="bold")

    # ── Annotate: step-to-baseline-50k ──
    bl50k = bl[bl["step"] == 50000]
    if not bl50k.empty:
        baseline_50k_ori = bl50k["val_ori_l2r_block"].values[0]
        ax.axhline(y=baseline_50k_ori, color="#888888", linestyle="--",
                   linewidth=0.8, alpha=0.5)
        ax.text(52000, baseline_50k_ori, f"baseline@50k={baseline_50k_ori:.3f}",
                fontsize=7.5, color="#666666", va="bottom", ha="left")

    # ── Styling ──
    ax.set_title(seed_name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("ori_l2r NLL  ↓", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(5000, 62000)

    # ── Inset: Δ vs baseline@50k ──
    inset = ax.inset_axes([0.18, 0.22, 0.35, 0.32])
    for run_key in ["from10k", "from20k", "from40k"]:
        df = load_curve(paths[run_key])
        final_ori = df["val_ori_l2r_block"].values[-1]
        delta = baseline_50k_ori - final_ori
        inset.bar(run_key, delta, color=COLORS[run_key], edgecolor="white", linewidth=0.8)
        inset.text(run_key, delta + 0.002, f"{delta:.3f}",
                   ha="center", fontsize=9, fontweight="bold", color=COLORS[run_key])
    inset.axhline(y=0, color="#888888", linewidth=0.7)
    inset.set_title("Δ vs baseline@50k", fontsize=9, fontweight="bold")
    inset.set_ylabel("ori_l2r improvement", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(axis="y", alpha=0.2)

plt.tight_layout(pad=2)
outpath = Path("/home/admin/lyuyuhuan/order_lyu/analyses/figures/multistart_seed2_vs_seed42.png")
outpath.parent.mkdir(exist_ok=True)
fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {outpath}")
plt.close()

# ── Summary table ─────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY TABLE")
print("=" * 72)
for seed_name, paths in SEEDS.items():
    print(f"\n── {seed_name} ──")
    bl = load_curve(paths["baseline"])
    bl50k = bl[bl["step"] == 50000]
    bl50k_ori = bl50k["val_ori_l2r_block"].values[0] if not bl50k.empty else float("nan")
    print(f"  Baseline @50k:  {bl50k_ori:.4f}")
    print(f"  {'Run':<12s} {'Final ori_l2r':>14s} {'Δ vs bl@50k':>12s} {'Steps':>8s}")
    print(f"  {'─'*50}")
    for run_key in ["from10k", "from20k", "from40k"]:
        df = load_curve(paths[run_key])
        final = df.iloc[-1]
        delta = bl50k_ori - final["val_ori_l2r_block"]
        print(f"  {run_key:<12s} {final['val_ori_l2r_block']:14.4f} {delta:+12.4f} {int(final['step']):>8d}")
