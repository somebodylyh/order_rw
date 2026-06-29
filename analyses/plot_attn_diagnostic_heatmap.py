"""Generate paper-ready heatmap figure from CDL τ diagnostic data."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "attention_diagnostic_20260609")

with open(os.path.join(OUT_DIR, "cdl_tau_diagnostic.json")) as f:
    data = json.load(f)

models = ["shuffled-L2R", "random-order", "ori-L2R"]
titles = [
    "Shuffled-L2R (fixed wrong order)",
    "Random-order (random permutation)",
    "Ori-L2R (natural L2R)",
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
vmin, vmax = -0.8, 0.6
cmap = plt.cm.RdBu_r

for ax, name, title in zip(axes, models, titles):
    d = data[name]
    mat = np.array(d["tau_matrix"])  # L×H
    L, H = mat.shape
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Annotate cells
    for l in range(L):
        for h in range(H):
            val = mat[l, h]
            color = "white" if abs(val) > 0.35 else "black"
            ax.text(h, l, f"{val:+.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if abs(val) > 0.3 else "normal")

    ax.set_xticks(range(H))
    ax.set_yticks(range(L))
    ax.set_xticklabels([f"H{h}" for h in range(H)], fontsize=7)
    ax.set_yticklabels([f"L{l}" for l in range(L)], fontsize=7)
    ax.set_xlabel("Head", fontsize=9)
    ax.set_ylabel("Layer", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")

    # Mark best head
    bl, bh = d["best_head"]
    ax.plot(bh, bl, marker="*", color="lime", markersize=16, markeredgecolor="black",
            markeredgewidth=0.5)

    # Stats text box
    s = d["stats"]
    stats_text = (f"Best: τ={d['best_tau']:+.3f} (L{bl}H{bh})\n"
                  f"Top-3 mean: {s['3']:+.3f}\n"
                  f"Top-5 mean: {s['5']:+.3f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=7.5,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

# Colorbar
cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
cbar.set_label("Kendall τ (CDL vs L2R)", fontsize=9)

fig.suptitle("CDL Teacher τ_vs_L2R per Head — Three Training Regimes",
             fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()

png_path = os.path.join(OUT_DIR, "cdl_tau_heatmap_3panel.png")
fig.savefig(png_path, dpi=200, bbox_inches="tight")
print(f"Saved: {png_path}")

# Also individual heatmaps for slides
for name, title in zip(models, titles):
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    d = data[name]
    mat = np.array(d["tau_matrix"])
    L, H = mat.shape
    im2 = ax2.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for l in range(L):
        for h in range(H):
            val = mat[l, h]
            color = "white" if abs(val) > 0.35 else "black"
            ax2.text(h, l, f"{val:+.2f}", ha="center", va="center",
                     fontsize=10, color=color, fontweight="bold" if abs(val) > 0.3 else "normal")
    ax2.set_xticks(range(H)); ax2.set_yticks(range(L))
    ax2.set_xticklabels([f"H{h}" for h in range(H)])
    ax2.set_yticklabels([f"L{l}" for l in range(L)])
    ax2.set_xlabel("Head"); ax2.set_ylabel("Layer")
    bl, bh = d["best_head"]
    ax2.plot(bh, bl, marker="*", color="lime", markersize=20, markeredgecolor="black", markeredgewidth=0.5)
    ax2.set_title(f"{title}\nBest: L{bl}H{bh} τ={d['best_tau']:+.3f}", fontweight="bold")
    fig2.colorbar(im2, ax=ax2, label="Kendall τ (CDL vs L2R)")
    slug = name.lower().replace("-", "_")
    fig2.savefig(os.path.join(OUT_DIR, f"cdl_tau_heatmap_{slug}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig2)

print("All heatmaps saved.")
