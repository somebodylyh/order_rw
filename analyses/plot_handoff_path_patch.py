"""Plot Pillar-3 path-patch summaries: a src-ablation x downstream-layer Delta-tau
heatmap per seed.

Usage:
    python analyses/plot_handoff_path_patch.py runs/handoff_pathpatch/seed2/summary.json [...]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_src_downstream_heatmap(summary_json, out_png):
    s = json.load(open(summary_json))
    # rows = interventions (Stage1a variants + Stage1b source + nulls);
    # cols = the readout layer's mean_tau_after - mean_tau_before delta.
    rows = s["stage1"]
    labels = [f"{r['stage']}:{r['intervention']}@L{r['target_layer']}" for r in rows]
    deltas = np.array([[r["mean_tau_after"] - r["mean_tau_before"]] for r in rows])
    fig, ax = plt.subplots(figsize=(4, 0.5 * len(rows) + 1))
    im = ax.imshow(deltas, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels(["readout-layer mean dtau"], fontsize=8)
    ax.set_title(f"seed{s['seed']} Stage-1 mean dtau (tier={s['dst_tier']})", fontsize=9)
    for i in range(len(rows)):
        ax.text(0, i, f"{deltas[i,0]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.05)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def main(argv):
    for summary_json in argv:
        out_png = summary_json.replace("summary.json", "stage1_heatmap.png")
        path = plot_src_downstream_heatmap(summary_json, out_png)
        print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
