"""Plot Spec A emergence characterization: A1 concentration + event window,
A2 predictability, A3 schedule/loss overlay; and A4 carrier-B structure.

Usage (per seed, after run_seed_emergence):
    plot_emergence(summary.json, concentration.csv, eval_curve.tsv, out_dir)
"""
import csv
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _read_concentration(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    out = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return out


def _read_eval(tsv_path):
    rows = list(csv.DictReader(open(tsv_path), delimiter="\t"))

    def col(n):
        return np.array([float(r[n]) if r[n] not in ("", "nan") else np.nan for r in rows])

    return {"step": col("step"), "lr": col("lr"),
            "val_train_objective": col("val_train_objective")}


def plot_emergence(summary_json, concentration_csv, eval_tsv, out_dir):
    s = json.load(open(summary_json))
    c = _read_concentration(concentration_csv)
    ec = _read_eval(eval_tsv)
    ev = s["event"]
    out = pathlib.Path(out_dir)
    seed = s["seed"]

    # ── A1: concentration (diffuse_count + per-layer strong) + event window ──
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(c["step"], c["diffuse_count"], "-o", ms=3, label="diffuse_count (|τ|≥0.7)")
    for L in range(4):
        ax.plot(c["step"], c[f"strong_L{L}"], "--", alpha=0.6, label=f"strong L{L} (|τ|≥0.95)")
    ax.axvspan(ev["onset"], ev["completion"], color="orange", alpha=0.15)
    ax.axvline(ev["midpoint"], color="red", ls=":", label=f"midpoint {ev['midpoint']}")
    ax.set_xlabel("step"); ax.set_ylabel("count")
    ax.set_title(f"seed{seed} A1 pruning event  "
                 f"(onset {ev['onset']} / mid {ev['midpoint']} / comp {ev['completion']})")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / "concentration.png", dpi=120); plt.close(fig)

    # ── A2: predictability (AUC + Spearman vs early step) ──
    pr = s["predictability"]
    steps = sorted(int(k) for k in pr["auc"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(steps, [pr["auc"][str(k)] if str(k) in pr["auc"] else pr["auc"][k] for k in steps],
            "-o", label="early→final AUC")
    ax.plot(steps, [pr["spearman"][str(k)] if str(k) in pr["spearman"] else pr["spearman"][k]
                    for k in steps], "-s", label="early→final Spearman")
    ax.axhline(0.5, color="gray", ls=":"); ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("early step"); ax.set_ylabel("predictability")
    ax.set_title(f"seed{seed} A2 winner predictability — {pr['verdict']}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "predictability.png", dpi=120); plt.close(fig)

    # ── A3: schedule/loss overlay with event band ──
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ec["step"], ec["val_train_objective"], "b-", label="val_train_objective")
    ax1.set_xlabel("step"); ax1.set_ylabel("loss", color="b")
    ax2 = ax1.twinx()
    ax2.plot(ec["step"], ec["lr"], "g-", alpha=0.6, label="lr")
    ax2.set_ylabel("lr", color="g")
    ax1.axvspan(ev["onset"], ev["completion"], color="orange", alpha=0.15)
    ax1.set_title(f"seed{seed} A3 schedule/loss — {s['schedule']['classification']}")
    fig.tight_layout(); fig.savefig(out / "schedule_overlay.png", dpi=120); plt.close(fig)
