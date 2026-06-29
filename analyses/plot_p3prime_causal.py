"""Plot P3' causal-verification metrics (B joint tau+R2; C residual vs floor)."""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_p3prime(json_path, out_dir):
    s = json.load(open(json_path))
    out = pathlib.Path(out_dir)
    bh = s["B_position"]["per_head"]
    ch = {row["head"]: row for row in s["C_content"]["per_head"]}
    heads = [row["head"] for row in bh]
    x = np.arange(len(heads))
    w = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(x - 1.5 * w, [row["tau_none"] for row in bh], w, label="tau none", color="C0")
    ax1.bar(x - 0.5 * w, [row["tau_abl"] for row in bh], w, label="tau pe-abl", color="C0", alpha=0.5)
    ax1.bar(x + 0.5 * w, [row["r2_none"] for row in bh], w, label="R2 none", color="C3")
    ax1.bar(x + 1.5 * w, [row["r2_abl"] for row in bh], w, label="R2 pe-abl", color="C3", alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"H{head}" for head in heads])
    ax1.set_title(f"seed{s['seed']} P3'-B base-map (joint tau+R2)")
    ax1.legend(fontsize=7)

    resid = [ch[head]["resid_change"] for head in heads]
    floor = [ch[head]["noise_floor"] for head in heads]
    ax2.bar(x - w / 2, resid, w, label="resid change", color="C2")
    ax2.bar(x + w / 2, floor, w, label="noise floor", color="C7", alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"H{head}" for head in heads])
    ax2.set_title(f"seed{s['seed']} P3'-C content residual | verdict={s['verdict']}")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out / "p3prime_metrics.png", dpi=120)
    plt.close(fig)
