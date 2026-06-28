"""Plot P2 physical-order signal-source disambiguation metrics."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_source(source_json, out_dir):
    s = json.load(open(source_json))
    out = pathlib.Path(out_dir)
    rows = s["per_head"]
    heads = [f"L{r['layer']}H{r['head']}" for r in rows]
    x = np.arange(len(rows))
    cvar = [r["content_variance"] for r in rows]
    floor = [r["noise_floor"] for r in rows]
    r2 = [r.get("r2_raw", r.get("r2_slot_only", 0.0)) for r in rows]
    verdicts = [r["verdict"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    # left: content variance vs within-text noise floor per head
    w = 0.38
    ax1.bar(x - w / 2, cvar, w, label="content variance (cross-text − floor)", color="C3")
    ax1.bar(x + w / 2, floor, w, label="within-text noise floor", color="C7", alpha=0.7)
    ax1.set_xticks(x); ax1.set_xticklabels(heads)
    ax1.set_ylabel("variance (row-normalized B)")
    ax1.set_title(f"seed{s['seed']} content vs sampling-noise")
    ax1.legend(fontsize=8)
    for i, v in enumerate(verdicts):
        ax1.text(i, max(cvar[i], floor[i]) + 1e-4, v, ha="center", fontsize=8, fontweight="bold")
    # right: slot-only predictor R2 vs randomized null
    null = s["anchors"]["r2_raw_null"]
    ax2.bar(x, r2, 0.5, color="C0", label="slot-only R² (raw)")
    ax2.axhline(null, color="gray", ls=":", label=f"randomized null ({null:.2f})")
    ax2.set_xticks(x); ax2.set_xticklabels(heads)
    ax2.set_ylim(min(-0.1, null - 0.1), 1.05)
    ax2.set_ylabel("held-out R² (content-free table)")
    ax2.set_title(f"seed{s['seed']} slot-only predictability  | verdict={s['seed_verdict']}")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "source_metrics.png", dpi=120)
    plt.close(fig)
