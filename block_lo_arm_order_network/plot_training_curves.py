"""Plot training curves for all Graph-RW experiments, including baselines."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_results")

# (label, dir, linestyle, linewidth, color)
EXPS = [
    ("L2R (upper bound)",  "stageA_l2r_from30k",                    "-",  2.5, "#2ecc40"),
    ("random (lower bound)","clean_base_random_perm",                "--", 2.5, "#e74c3c"),
    ("v2  α=0.9",          "clean_method_graph_rw_a09_from20k_v2",  "-",  2.5, "#3498db"),
    ("α=1.0→0.9 @30k",     "clean_method_graph_rw_a10_to_a09_30k40k","--",2.0, "#e67e22"),
    ("α=1.0→0.95 @30k",    "clean_method_graph_rw_a10_to_a095_30k60k","-.",2.0,"#9b59b6"),
    ("α=1.0 (pure RW)",    "clean_method_graph_rw_a10_from20k",     ":",  2.0, "#95a5a6"),
    ("ε=0.15",             "clean_method_graph_rw_a09_from20k_eps015","-.",1.5,"#1abc9c"),
    ("ε=0.20",             "clean_method_graph_rw_a09_from20k_eps020","-.",1.5,"#16a085"),
    ("top_k=8 τ=0.5",     "clean_method_graph_rw_a09_from20k_topk8_tau05",(0,(3,1,1,1)),1.5,"#f39c12"),
    ("no-refresh",         "clean_method_graph_rw_a09_v2_no_refresh_20k40k",":",2.0,"#c0392b"),
    ("random-refresh",     "ablation_rand_refresh_a09_from20k",      "--", 1.5, "#7f8c8d"),
]


def load_tsv(path):
    data = np.genfromtxt(path, delimiter="\t", names=True, dtype=None, encoding="utf-8")
    steps = data["step"].astype(int)
    ori_l2r = data["val_ori_l2r_block"].astype(float)
    train_obj = data["val_train_objective"].astype(float)
    return steps, ori_l2r, train_obj


def main():
    plt.rcParams.update({"font.size": 11})

    fig, axes = plt.subplots(1, 2, figsize=(22, 8), constrained_layout=True)

    for ax, title, ycol in [
        (axes[0], "val_ori_l2r_block (L2R-generalization CE)", "ori_l2r"),
        (axes[1], "val_train_objective (training CE)", "train_obj"),
    ]:
        for label, d, ls, lw, color in EXPS:
            tsv = os.path.join(BASE, d, "eval_curve.tsv")
            if not os.path.exists(tsv):
                continue
            steps, ori_l2r, train_obj = load_tsv(tsv)
            y = ori_l2r if ycol == "ori_l2r" else train_obj
            ax.plot(steps, y, linestyle=ls, linewidth=lw, color=color,
                    label=label, alpha=0.9)

        ax.axvline(x=30000, color="gray", linestyle=":", alpha=0.4, linewidth=1)
        ax.text(31000, 3.98, "α-switch", fontsize=8, color="gray")

        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("CE", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(left=10000)
        ax.set_ylim(3.35, 4.0)
        ax.legend(fontsize=8, loc="lower left", ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.2)

    # Summary
    print(f"{'Experiment':<28s} {'Max Step':>8s}  {'ori_l2r':>10s}  {'train_obj':>10s}")
    print("-" * 62)
    for label, d, *_ in EXPS:
        tsv = os.path.join(BASE, d, "eval_curve.tsv")
        if not os.path.exists(tsv):
            continue
        steps, ori_l2r, train_obj = load_tsv(tsv)
        print(f"{label:<28s} {steps[-1]:>8d}  {ori_l2r[-1]:>10.4f}  {train_obj[-1]:>10.4f}")
    print("-" * 62)

    for p in [os.path.join(BASE, "training_curves.png"),
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training_curves.png")]:
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
