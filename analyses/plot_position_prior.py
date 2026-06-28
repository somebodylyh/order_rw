"""Plots for Pillar-5-core position-prior decomposition + content binding."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_part1(part1_json, out_dir):
    s = json.load(open(part1_json))
    out = pathlib.Path(out_dir)
    arms = s["arms_carrier_max"]
    labels = ["full", "zero_wpe", "zero_wtpe", "zero_both",
              "uniform_causal", "random_B"]
    vals = [arms["full"], arms["zero_wpe"], arms["zero_wtpe"], arms["zero_both"],
            s["floor"]["uniform_causal"], s["floor"]["random_B"]]
    colors = ["C0", "C1", "C1", "C1", "C2", "C7"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals, color=colors)
    ax.axhline(s["floor"]["uniform_causal"], color="C2", ls=":", alpha=0.7)
    ax.set_ylabel("carrier max τ (C-D+L)")
    ax.set_title(f"seed{s['seed']} Part-1 step-0 prior decomposition "
                 f"(L{s['winning_layer']} carrier, tier={s['tier']})")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(out / "part1_decomposition.png", dpi=120)
    plt.close(fig)


def plot_part2(part2_json, out_dir):
    s = json.load(open(part2_json))
    out = pathlib.Path(out_dir)
    steps = sorted(int(k) for k in s["by_step"])
    pos = [s["by_step"][str(k)]["relayout_mean_pos"] for k in steps]
    cont = [s["by_step"][str(k)]["relayout_mean_content"] for k in steps]
    anch = [s["by_step"][str(k)]["anchor_tau_pos"] for k in steps]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, pos, "-o", label="τ_pos (relayout mean)")
    ax.plot(steps, cont, "-s", label="τ_content (relayout mean)")
    ax.plot(steps, anch, "--^", color="gray", alpha=0.7, label="anchor τ_pos")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("step"); ax.set_ylabel("τ")
    ax.set_title(f"seed{s['seed']} Part-2 content vs position binding (OOD relayout)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "binding.png", dpi=120)
    plt.close(fig)
