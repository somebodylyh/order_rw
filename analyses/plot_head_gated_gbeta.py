#!/usr/bin/env python3
"""Plot head-gated g_beta training curves and gate analysis.

Usage:
  python3 analyses/plot_head_gated_gbeta.py \
    --runs /tmp/head_gated_smoke/*/ \
    --out-dir reports/head_gated_gbeta_eval
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not available; skipping plots")
    sys.exit(0)


def _load_history(run_dir: pathlib.Path) -> dict | None:
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        return None
    h = json.loads(hist_path.read_text())
    h["_run_name"] = run_dir.name
    return h


def _plot_tau_curves(histories: list[dict], out_path: str):
    """Plot tau vs epoch for all runs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    tau_keys = [
        ("tau_model_vs_teacher", "τ vs Teacher"),
        ("tau_model_vs_semantic_path", "τ vs Semantic Path"),
        ("tau_model_vs_identity", "τ vs Identity"),
    ]

    for ax, (key, title) in zip(axes, tau_keys):
        for h in histories:
            epochs = [e["epoch"] for e in h["history"]]
            values = [e.get(key, float("nan")) for e in h["history"]]
            ax.plot(epochs, values, marker=".", label=h["_run_name"])
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Kendall τ")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Head-Gated gβ Training Curves", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved tau curves to {out_path}")


def _plot_loss_curves(histories: list[dict], out_path: str):
    """Plot train/val loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for h in histories:
        epochs = [e["epoch"] for e in h["history"]]
        train = [e.get("train_loss", float("nan")) for e in h["history"]]
        val = [e.get("val_loss", float("nan")) for e in h["history"]]
        ax1.plot(epochs, train, marker=".", label=h["_run_name"])
        ax2.plot(epochs, val, marker=".", label=h["_run_name"])

    ax1.set_title("Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Val Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Head-Gated gβ Loss Curves", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved loss curves to {out_path}")


def _plot_gate_analysis(histories: list[dict], out_path: str):
    """Plot gate entropy and head weights for gated variants."""
    gated = [h for h in histories
             if "gated" in h["_run_name"].lower()
             or "head_gated" in h["_run_name"].lower()]

    if not gated:
        print("  (no gated variants — skipping gate analysis)")
        return

    fig, axes = plt.subplots(1, len(gated), figsize=(5 * len(gated), 4))
    if len(gated) == 1:
        axes = [axes]

    for ax, h in zip(axes, gated):
        epochs = [e["epoch"] for e in h["history"]]
        entropy = [e.get("gate_entropy", float("nan")) for e in h["history"]]
        ax.plot(epochs, entropy, "b-o", markersize=4, label="gate entropy")

        # Reference: max entropy = log(H)
        H = 8  # default; could parse from config
        ax.axhline(y=np.log(H), color="gray", linestyle="--", alpha=0.5,
                   label=f"max ent (log({H})={np.log(H):.2f})")
        ax.set_title(f"Gate Entropy — {h['_run_name']}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Entropy (nats)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Gate Entropy Evolution", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved gate analysis to {out_path}")


def _plot_bar_comparison(histories: list[dict], out_path: str):
    """Bar chart comparing best tau across runs."""
    names = [h["_run_name"] for h in histories]
    taus_teacher = []
    taus_semantic = []

    for h in histories:
        best = h.get("best", {}).get("metrics", {})
        taus_teacher.append(best.get("tau_model_vs_teacher", 0))
        taus_semantic.append(best.get("tau_model_vs_semantic_path", 0))

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    bars1 = ax.bar(x - width/2, taus_teacher, width, label="τ vs Teacher", color="#2196F3")
    bars2 = ax.bar(x + width/2, taus_semantic, width, label="τ vs Semantic Path", color="#4CAF50")

    ax.set_ylabel("Kendall τ")
    ax.set_title("Best τ Across Variants")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved bar comparison to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Plot head-gated g_beta results")
    p.add_argument("--runs", nargs="+", required=True,
                   help="Run output directories")
    p.add_argument("--out-dir", default="reports/head_gated_gbeta_eval",
                   help="Output directory for plots")
    args = p.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    histories = []
    for d in args.runs:
        d = pathlib.Path(d)
        h = _load_history(d)
        if h is not None:
            histories.append(h)

    if not histories:
        print("No valid runs found.")
        sys.exit(1)

    print(f"Plotting {len(histories)} runs...")

    _plot_tau_curves(histories, str(out / "tau_curves.png"))
    _plot_loss_curves(histories, str(out / "loss_curves.png"))
    _plot_gate_analysis(histories, str(out / "gate_analysis.png"))
    _plot_bar_comparison(histories, str(out / "bar_comparison.png"))

    print(f"\nAll plots saved to {out}/")


if __name__ == "__main__":
    main()
