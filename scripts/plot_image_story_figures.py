#!/usr/bin/env python3
"""Draft 'image story' figures for advisor review.

Does NOT train, does NOT read ckpt/npy. Figures 1-4 use draft summary numbers
hard-coded below (each annotated with its source file). Figure 5 reads the real
Long-1 eval_curve.tsv files to plot the cross-avg gap over steps.

Run:  python scripts/plot_image_story_figures.py
Out:  probe_results_image_large/structure_adaptive/image_story_figures/

----------------------------------------------------------------------------
DRAFT SUMMARY NUMBERS (verified against committed report files):
  E3-ctrl-small B top-1 locality: mean_manh=1.156, P(d<=1)=0.984
      <- probe_results_image_large/grw_e3ctrlsmall/SUMMARY.md
  readout-diagnostic mean_manh: random=5.349, raster=1.778, v1_graph_rw=4.918
      <- probe_results_image_large/grw_e3ctrlsmall/readout_diagnostic/metrics.tsv
  structure probes (mean_manh, top4-follow on real B):
      random=(5.349,0.059), Hilbert=(1.0,0.397),
      distance_only=(1.13,0.531), Bcov_balanced=(2.407,0.846)
      <- structure_adaptive/structural_probes/preview_on_saved_orders.json
  shuffled-B control evaluated on real B: top4-follow=0.093 (mean_manh 3.31)
      <- structure_adaptive/structural_probes/shuffledB_reverse_control.json
  Round-2 multi-seed (5 seeds) task gain (mean Δ, 95% CI, wins):
      Bcov-distance  cross=(-0.0021,0.0002,5/5)  structured=(-0.0012,0.0002,5/5)
      Bcov-shuffled  cross=(-0.0049,0.0007,5/5)  structured=(-0.0075,0.0007,5/5)
      <- structure_adaptive/round2_bcov_ci/REPORT.md
----------------------------------------------------------------------------
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path("/home/admin/lyuyuhuan/order_lyu")
OUTDIR = ROOT / "probe_results_image_large/structure_adaptive/image_story_figures"
LONG_ROOT = ROOT / "probe_results_image_large/grw_e3ctrlsmall_round2_long10000_seed42"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Locked cross schema (7 shared probes; no dist) — matches summarize_round2_e3.py
CROSS7 = ["val_random", "val_raster", "val_hilbert", "val_Bcov_balanced",
          "val_rw_top4_eps0", "val_rw_eps015", "val_rw_topk8"]

C_BLUE, C_ORANGE, C_GREEN, C_RED, C_GRAY = "#3b6ea5", "#e08214", "#4d9221", "#c0392b", "#888888"


def save(fig, name):
    path = OUTDIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")
    return path


# ---------------------------------------------------------------- Figure 1
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_xlim(0, 13.4); ax.set_ylim(0, 3); ax.axis("off")
    boxes = [
        ("AO-GPT image\nattention  A", C_BLUE),
        (r"directed graph" + "\n" + r"B = A$^\mathsf{T}$", C_BLUE),
        ("structure\nprobes", C_GREEN),
        (r"readout order  $\sigma$" + "\n(Bcov_balanced)", C_ORANGE),
        ("continuation\nvalidation gain", C_RED),
    ]
    w, h, gap = 2.1, 1.4, 0.55
    x = 0.4
    centers = []
    for label, color in boxes:
        box = FancyBboxPatch((x, 0.8), w, h, boxstyle="round,pad=0.06",
                             linewidth=1.6, edgecolor=color, facecolor=color + "22")
        ax.add_patch(box)
        ax.text(x + w / 2, 0.8 + h / 2, label, ha="center", va="center", fontsize=11)
        centers.append(x + w)
        x += w + gap
    for i in range(len(boxes) - 1):
        arr = FancyArrowPatch((centers[i], 1.5), (centers[i] + gap, 1.5),
                              arrowstyle="-|>", mutation_scale=16, linewidth=1.4, color="#444")
        ax.add_patch(arr)
    ax.text(6.85, 2.7, "Image-side framework: learned attention -> directed graph -> structure-guided readout -> task gain",
            ha="center", va="center", fontsize=12, fontweight="bold")
    return save(fig, "fig1_image_pipeline.png")


# ---------------------------------------------------------------- Figure 2
def fig2_representation_and_mismatch():
    # mean_manh: lower = more local. B top-1 is local; v1 readout fails to read it out.
    labels = ["B top-1\n(real graph)", "raster\n(geometry)", "v1 Graph-RW\n(readout)", "random"]
    vals = [1.156, 1.778, 4.918, 5.349]
    colors = [C_GREEN, C_BLUE, C_RED, C_GRAY]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    ax.axhline(5.349, ls="--", lw=1.0, color=C_GRAY)
    ax.text(3.0, 5.45, "random baseline", color=C_GRAY, fontsize=9, ha="center")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("mean Manhattan distance (lower = more local)")
    ax.set_title("Representation has local structure, but v1 readout does not read it out\n"
                 "(E3-control-small, 8x8 block graph, N=64)", fontsize=11)
    ax.set_ylim(0, 6.2)
    return save(fig, "fig2_representation_and_mismatch.png")


# ---------------------------------------------------------------- Figure 3
def fig3_beyond_locality_bedge():
    orders = ["random", "Hilbert", "distance-only", "Bcov_balanced"]
    manh = [5.349, 1.0, 1.13, 2.407]
    follow = [0.059, 0.397, 0.531, 0.846]
    shuffled_follow = 0.093
    x = range(len(orders))
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    width = 0.38
    b1 = ax1.bar([i - width / 2 for i in x], manh, width, color=C_BLUE, label="mean_manh (left)")
    ax1.set_ylabel("mean Manhattan distance", color=C_BLUE)
    ax1.tick_params(axis="y", labelcolor=C_BLUE)
    ax1.set_ylim(0, 6.0)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(orders)

    ax2 = ax1.twinx()
    b2 = ax2.bar([i + width / 2 for i in x], follow, width, color=C_ORANGE, label="top4-follow real B (right)")
    ax2.set_ylabel("top-4 follow rate on real B", color=C_ORANGE)
    ax2.tick_params(axis="y", labelcolor=C_ORANGE)
    ax2.set_ylim(0, 1.0)
    ax2.axhline(shuffled_follow, ls="--", lw=1.4, color=C_RED)
    ax2.text(len(orders) - 1.0, shuffled_follow + 0.02,
             f"shuffled-B control on real B = {shuffled_follow:.2f}", color=C_RED, fontsize=9, ha="center")
    for i, v in zip(x, follow):
        ax2.text(i + width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color=C_ORANGE)
    for i, v in zip(x, manh):
        ax1.text(i - width / 2, v + 0.08, f"{v:.2f}", ha="center", fontsize=9, color=C_BLUE)

    ax1.set_title("Beyond locality: Bcov is less local but follows real B edges most\n"
                  "shuffled-B control collapses to ~random (not geometry coincidence)", fontsize=11)
    lines = [b1, b2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper center", fontsize=9)
    return save(fig, "fig3_beyond_locality_bedge.png")


# ---------------------------------------------------------------- Figure 4
def fig4_round2_task_gain_ci():
    labels = ["Bcov - distance\ncross", "Bcov - distance\nstructured",
              "Bcov - shuffled\ncross", "Bcov - shuffled\nstructured"]
    means = [-0.0021, -0.0012, -0.0049, -0.0075]
    errs = [0.0002, 0.0002, 0.0007, 0.0007]
    colors = [C_GREEN, C_GREEN, C_BLUE, C_BLUE]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, means, yerr=errs, capsize=5, color=colors, width=0.6,
                  error_kw=dict(ecolor="#333", lw=1.4))
    ax.axhline(0, color="black", lw=1.0)
    for b, m, e in zip(bars, means, errs):
        ax.text(b.get_x() + b.get_width() / 2, m - e - 0.0004, f"{m:+.4f}", ha="center", va="top", fontsize=9)
    ax.set_ylabel("mean Δ val loss  (negative = Bcov better)")
    ax.set_title("Round-2 task gain: 5 seeds, all negative, CIs do not cross 0\n"
                 "(small but consistent; wins 5/5 each)", fontsize=11)
    ax.set_ylim(-0.0095, 0.0015)
    return save(fig, "fig4_round2_task_gain_ci.png")


# ---------------------------------------------------------------- Figure 5
def _read_cross_curve(arm):
    """Return (steps, cross_avg) from a Long-1 arm eval_curve.tsv. Real data."""
    path = LONG_ROOT / arm / "eval_curve.tsv"
    if not path.exists():
        return None, None
    steps, cross = [], []
    for r in csv.DictReader(path.open(), delimiter="\t"):
        s = int(r["step"])
        if s == 0:
            continue
        steps.append(s)
        cross.append(sum(float(r[c]) for c in CROSS7) / len(CROSS7))
    return steps, cross


def _read_train_curve(arm):
    """Return (steps, train_loss) from a Long-1 arm eval_curve.tsv. Real data.
    Skips step=0 (train_loss is nan there: alpha=0, no backward yet)."""
    path = LONG_ROOT / arm / "eval_curve.tsv"
    if not path.exists():
        return None, None
    steps, tr = [], []
    for r in csv.DictReader(path.open(), delimiter="\t"):
        s = int(r["step"])
        if s == 0:
            continue
        try:
            v = float(r["train_loss"])
        except ValueError:
            continue
        steps.append(s)
        tr.append(v)
    return steps, tr


def _smooth(y, k=2):
    """Centered moving average, window 2k+1."""
    out = []
    for i in range(len(y)):
        lo, hi = max(0, i - k), min(len(y), i + k + 1)
        out.append(sum(y[lo:hi]) / (hi - lo))
    return out


def fig6_train_loss_curves():
    arms = [
        ("cont_random", "random", C_GRAY),
        ("cont_Bcov_balanced", "Bcov_balanced", C_ORANGE),
        ("cont_distance_only_coverage", "distance_only", C_GREEN),
        ("cont_shuffled_Bcov_balanced", "shuffled_Bcov", C_BLUE),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    any_real = False
    for arm, label, color in arms:
        steps, tr = _read_train_curve(arm)
        if steps is None:
            continue
        any_real = True
        ax.plot(steps, tr, color=color, alpha=0.22, lw=1.0)             # raw (noisy)
        ax.plot(steps, _smooth(tr), color=color, lw=1.9, label=label)   # smoothed trend
    ax.axvline(5000, ls=":", color=C_GRAY, lw=1.0)
    ax.text(5080, ax.get_ylim()[1] - 0.002, "5k", color=C_GRAY, fontsize=9)
    ax.set_xlabel("continuation step")
    ax.set_ylabel("train loss (CE, lower = better)")
    ax.set_title("Long-1 training loss curves (4 arms, seed=42)\n"
                 "REAL data from eval_curve.tsv; faint = raw per-eval, solid = smoothed (±2)", fontsize=11)
    ax.legend(fontsize=9)
    return save(fig, "fig6_train_loss_curves.png"), any_real


def fig5_long1_trend():
    bcov = _read_cross_curve("cont_Bcov_balanced")
    dist = _read_cross_curve("cont_distance_only_coverage")
    shuf = _read_cross_curve("cont_shuffled_Bcov_balanced")
    used_real = all(v[0] is not None for v in (bcov, dist, shuf))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    if used_real:
        steps = bcov[0]
        gap_d = [b - d for b, d in zip(bcov[1], dist[1])]
        gap_s = [b - s for b, s in zip(bcov[1], shuf[1])]
        ax.plot(steps, gap_d, "-o", ms=3, color=C_GREEN, label="Bcov - distance (cross)")
        ax.plot(steps, gap_s, "-o", ms=3, color=C_BLUE, label="Bcov - shuffled (cross)")
        ax.axhline(0, color="black", lw=1.0)
        ax.axvline(5000, ls=":", color=C_GRAY, lw=1.0)
        ax.text(5050, ax.get_ylim()[1] * 0.9, "5k", color=C_GRAY, fontsize=9)
        ax.set_xlabel("continuation step")
        src = "REAL data from Long-1 eval_curve.tsv"
    else:
        # placeholder schema
        steps = [5000, 10000]
        ax.plot(steps, [-0.0027, -0.0038], "-o", color=C_GREEN, label="Bcov - distance (cross)")
        ax.plot(steps, [-0.0052, -0.0057], "-o", color=C_BLUE, label="Bcov - shuffled (cross)")
        ax.axhline(0, color="black", lw=1.0)
        src = "PLACEHOLDER (eval_curve not found)"
    ax.set_ylabel("cross-avg gap  (negative = Bcov better)")
    ax.set_title(f"Long-1: Bcov gap persists 5k -> 10k (single-seed trend, seed=42)\n{src}", fontsize=11)
    ax.legend(fontsize=9)
    return save(fig, "fig5_long1_trend_placeholder.png"), used_real


def main():
    print(f"[outdir] {OUTDIR}")
    fig1_pipeline()
    fig2_representation_and_mismatch()
    fig3_beyond_locality_bedge()
    fig4_round2_task_gain_ci()
    _, used_real = fig5_long1_trend()
    print(f"[fig5] used_real_long1_data = {used_real}")
    _, used_real6 = fig6_train_loss_curves()
    print(f"[fig6] used_real_long1_data = {used_real6}")


if __name__ == "__main__":
    main()
