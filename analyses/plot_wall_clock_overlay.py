"""Wall-clock overlay plot: random vs V3 matched vs ori-L2R."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "attention_diagnostic_20260609")

data = {"random": [], "V3_matched": [], "ori_L2R": []}
with open(os.path.join(OUT_DIR, "wall_clock_plot_data.tsv")) as f:
    header = f.readline()
    for line in f:
        run, wall_s, ori = line.strip().split("\t")
        data[run].append((int(wall_s), float(ori)))

fig, ax = plt.subplots(figsize=(8, 5.5))

colors = {"random": "#d62728", "V3_matched": "#2ca02c", "ori_L2R": "#1f77b4"}
labels = {"random": "Random baseline", "V3_matched": "gβ hook (V3 matched @10k)",
          "ori_L2R": "ori-L2R (upper bound)"}
markers = {"random": "s", "V3_matched": "o", "ori_L2R": "D"}
zorder = {"V3_matched": 3, "random": 2, "ori_L2R": 1}

# Filter ori-L2R to show only every 5th point (it's very dense)
for run in data:
    pts = data[run]
    xs = [p[0]/3600 for p in pts]  # hours
    ys = [p[1] for p in pts]
    if run == "ori_L2R":
        # sparse sampling to avoid clutter
        xs = xs[::5]
        ys = ys[::5]
    ax.plot(xs, ys, color=colors[run], marker=markers[run], markersize=4 if run != "ori_L2R" else 3,
            linewidth=1.5, label=labels[run], zorder=zorder[run],
            markevery=max(1, len(xs)//12))

# Horizontal threshold lines
thresholds = [3.60, 3.55, 3.50, 3.47, 3.45]
for t in thresholds:
    ax.axhline(y=t, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.text(0.02, t, f"ori_l2r={t}", fontsize=7, va="bottom", ha="left",
            color="gray", alpha=0.8)

# Mark "step to 3.47" for V3 and random
ax.annotate("", xy=(0.78, 3.47), xytext=(3.23, 3.47),
            arrowprops=dict(arrowstyle="<->", color="green", lw=2, shrinkA=0, shrinkB=0))
ax.text(2.0, 3.44, "76% wall saving\n(58% incl. gβ pretrain)",
        fontsize=8, fontweight="bold", color="green", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

# gβ pretrain cost bar at the start of V3 curve
v3_start_x = data["V3_matched"][0][0] / 3600
ax.axvspan(0, 2100/3600, color="orange", alpha=0.15, label="gβ pretrain cost (0.58h)")
ax.text(2100/3600/2, 4.5, "gβ\npretrain", fontsize=7, ha="center",
        color="orange", fontweight="bold")

ax.set_xlabel("Wall-Clock Time (hours)", fontsize=11)
ax.set_ylabel("ori_l2r NLL", fontsize=11)
ax.set_title("Wall-Clock to Target: Random vs. gβ Hook (V3, seed2)",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
ax.set_xlim(0, None)
ax.grid(True, alpha=0.3)

# Inset: zoom on final region
ax_inset = fig.add_axes([0.48, 0.25, 0.38, 0.35])
for run in ["random", "V3_matched", "ori_L2R"]:
    pts = data[run]
    xs = [p[0]/3600 for p in pts if p[1] < 3.55]
    ys = [p[1] for p in pts if p[1] < 3.55]
    ax_inset.plot(xs, ys, color=colors[run], marker=markers[run], markersize=2,
                  linewidth=1, alpha=0.8)
ax_inset.axhline(y=3.47, color="gray", linestyle=":", alpha=0.5)
ax_inset.axhline(y=3.45, color="gray", linestyle=":", alpha=0.5)
ax_inset.text(0.02, 3.47, "3.47", fontsize=6, va="bottom", ha="left", color="gray")
ax_inset.text(0.02, 3.45, "3.45", fontsize=6, va="bottom", ha="left", color="gray")
ax_inset.set_title("Zoom: NLL < 3.55", fontsize=8)
ax_inset.tick_params(labelsize=7)

fig.tight_layout()
png_path = os.path.join(OUT_DIR, "wall_clock_overlay.png")
fig.savefig(png_path, dpi=200, bbox_inches="tight")
print(f"Saved: {png_path}")

# Also save a clean pair plot (just random vs V3) for slides
fig2, ax2 = plt.subplots(figsize=(7, 5))
for run in ["random", "V3_matched"]:
    pts = data[run]
    xs = [p[0]/3600 for p in pts]
    ys = [p[1] for p in pts]
    ax2.plot(xs, ys, color=colors[run], marker=markers[run], markersize=4,
             linewidth=2, label=labels[run])
ax2.axhline(y=3.47, color="gray", linestyle=":", alpha=0.5)
ax2.text(0.02, 3.47, "ori_l2r=3.47", fontsize=8, va="bottom", ha="left", color="gray")
ax2.set_xlabel("Wall-Clock Time (hours)", fontsize=11)
ax2.set_ylabel("ori_l2r NLL", fontsize=11)
ax2.set_title("Random vs. gβ Hook — Wall-Clock Efficiency", fontsize=12, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
png2 = os.path.join(OUT_DIR, "wall_clock_pair.png")
fig2.savefig(png2, dpi=200, bbox_inches="tight")
print(f"Saved: {png2}")

# Print summary stats
print("\n=== Summary Stats ===")
for run in ["random", "V3_matched", "ori_L2R"]:
    pts = data[run]
    print(f"{labels[run]}: {len(pts)} points, "
          f"wall range={pts[0][0]/3600:.1f}h–{pts[-1][0]/3600:.1f}h, "
          f"ori_l2r range={pts[0][1]:.4f}–{pts[-1][1]:.4f}")
