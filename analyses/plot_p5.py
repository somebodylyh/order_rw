"""Plot P5 Phase-0 utility metrics."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_p5(json_path, out_dir):
    r = json.load(open(json_path)); out = pathlib.Path(out_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    hd = r["headroom"]; bd = hd["best_dist"]
    ax1.bar(range(len(bd)), list(bd.values()))
    ax1.set_xticks(range(len(bd))); ax1.set_xticklabels(list(bd), rotation=60, fontsize=6)
    ax1.set_title(f"best-candidate dist | headroom_abs={hd['abs_mean']:.3f} "
                  f"gate={hd['gate_pass']}")
    if "metrics" in r:
        m = r["metrics"]
        bars = {"B-only": m["nll_b_only"], "B+H": m["nll_bh"]}
        ax2.bar(range(len(bars)), list(bars.values()))
        ax2.set_xticks(range(len(bars))); ax2.set_xticklabels(list(bars))
        ax2.set_title(f"ΔNLL={m['delta_nll']:.4f}  shuf_drop={m['h_shuffle_drop']:.4f}  "
                      f"verdict={m['verdict']}")
    else:
        ax2.text(0.5, 0.5, "headroom gate FAILED\n(no training)", ha="center")
    fig.tight_layout(); fig.savefig(out / "phase0_metrics.png", dpi=120); plt.close(fig)
