"""Aggregate per-(step,seed) L0H5 evolution scans into curves + figure + verdict.

Reads block_lo_arm_order_network/batch_readout/logs/l0h5_evo/scan_step{S}_seed{D}.json
(produced by scripts/diag_l0h5_evolution_scan.py), aggregates over sampling
seeds, and answers:
  Q1 (strength): tau_vs_l2r(L0H5) mean+/-std vs training step.
  Q2 (localization): is the |tau_vs_l2r| winner head always L0H5, or does it drift?

Outputs:
  - report.json  (per-step aggregates + verdict)
  - l0h5_evolution.png  (multi-panel figure)

Run:
  PYTHONPATH=block_lo_arm_order_network \
    python scripts/aggregate_l0h5_evolution.py \
      --logdir block_lo_arm_order_network/batch_readout/logs/l0h5_evo
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_FNAME = re.compile(r"scan_step(\d+)_seed(\d+)\.json$")


def _row(table, layer, head):
    for r in table:
        if r["layer"] == layer and r["head"] == head:
            return r
    return None


def load_scans(logdir):
    """-> {step: {seed: report}}"""
    out = defaultdict(dict)
    for path in glob.glob(str(pathlib.Path(logdir) / "scan_step*_seed*.json")):
        m = _FNAME.search(path)
        if not m:
            continue
        step, seed = int(m.group(1)), int(m.group(2))
        with open(path) as f:
            out[step][seed] = json.load(f)
    return out


def aggregate(scans):
    steps = sorted(scans.keys())
    per_step = []
    for s in steps:
        seeds = sorted(scans[s].keys())
        l0h5_tau, l0h5_pw, l0h5_fh, l0h5_rank = [], [], [], []
        l0h1_tau = []
        winners = []          # winner (layer,head) per seed
        layer0_frac = []      # max|tau| in layer0 / max|tau| overall
        heavy_tau = []
        for d in seeds:
            rep = scans[s][d]
            table = rep["per_head_layer_sorted_by_abs_tau_vs_l2r"]
            r5 = _row(table, 0, 5)
            r1 = _row(table, 0, 1)
            if r5 is not None:
                l0h5_tau.append(r5["tau_vs_l2r"])
                l0h5_pw.append(r5["mean_pairwise_tau"])
                l0h5_fh.append(r5["first_step_entropy"])
                l0h5_rank.append(table.index(r5) + 1)
            if r1 is not None:
                l0h1_tau.append(r1["tau_vs_l2r"])
            top = table[0]
            winners.append((top["layer"], top["head"]))
            by_layer = defaultdict(float)
            for r in table:
                by_layer[r["layer"]] = max(by_layer[r["layer"]], abs(r["tau_vs_l2r"]))
            overall = max(by_layer.values()) if by_layer else 1.0
            layer0_frac.append(by_layer.get(0, 0.0) / overall if overall else 0.0)
            heavy_tau.append(rep["heavy_baseline"]["tau_vs_l2r"])

        winner_counts = defaultdict(int)
        for w in winners:
            winner_counts[w] += 1
        modal_winner = max(winner_counts, key=winner_counts.get)
        per_step.append({
            "step": s, "n_seeds": len(seeds),
            "l0h5_tau_mean": float(np.mean(l0h5_tau)), "l0h5_tau_std": float(np.std(l0h5_tau)),
            "l0h5_pairwise_mean": float(np.mean(l0h5_pw)),
            "l0h5_first_H_mean": float(np.mean(l0h5_fh)),
            "l0h5_rank_mean": float(np.mean(l0h5_rank)),
            "l0h5_rank_per_seed": l0h5_rank,
            "l0h1_tau_mean": float(np.mean(l0h1_tau)) if l0h1_tau else None,
            "winner_per_seed": [f"L{l}H{h}" for (l, h) in winners],
            "modal_winner": f"L{modal_winner[0]}H{modal_winner[1]}",
            "winner_is_l0h5_all_seeds": all(w == (0, 5) for w in winners),
            "layer0_frac_mean": float(np.mean(layer0_frac)),
            "heavy_tau_mean": float(np.mean(heavy_tau)),
        })
    return steps, per_step


def verdict(per_step):
    """稳定 / 涌现 / 漂移 classification."""
    if not per_step:
        return {"label": "NO_DATA", "detail": "no scans found"}
    trained = [p for p in per_step if p["step"] > 0]
    if not trained:
        return {"label": "NO_DATA", "detail": "only init checkpoint scanned"}

    winner_always_l0h5 = all(p["winner_is_l0h5_all_seeds"] for p in trained)
    taus = [p["l0h5_tau_mean"] for p in trained]
    signs_const = all((t > 0) for t in taus) or all((t < 0) for t in taus)
    abs_taus = [abs(t) for t in taus]
    # emergence: low at first trained step, rises notably
    first, last = abs_taus[0], abs_taus[-1]
    rising = last - first > 0.10

    if winner_always_l0h5 and signs_const and min(abs_taus) >= 0.40:
        label = "STABLE"
        detail = ("L0H5 is top-1 |tau_vs_l2r| head at every trained step across all seeds, "
                  f"sign constant, |tau| in [{min(abs_taus):.2f},{max(abs_taus):.2f}].")
    elif not winner_always_l0h5:
        drift = [(p["step"], p["modal_winner"]) for p in trained if p["modal_winner"] != "L0H5"]
        label = "DRIFT"
        detail = f"winner head leaves L0H5 at steps {drift}; L0H5 not a training-invariant."
    elif rising:
        label = "EMERGENT"
        detail = (f"L0H5 |tau| rises {first:.2f}->{last:.2f} over training "
                  "(signal develops rather than present at init).")
    else:
        label = "WEAK_OR_NOISY"
        detail = (f"L0H5 stays winner but |tau| modest/flat ({first:.2f}->{last:.2f}); "
                  "inspect curve.")
    return {"label": label, "detail": detail}


def make_figure(per_step, out_png):
    steps = [p["step"] for p in per_step]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    mean = [p["l0h5_tau_mean"] for p in per_step]
    std = [p["l0h5_tau_std"] for p in per_step]
    ax.errorbar(steps, mean, yerr=std, marker="o", capsize=4, label="L0H5", color="C0")
    l0h1 = [p["l0h1_tau_mean"] for p in per_step]
    if all(v is not None for v in l0h1):
        ax.plot(steps, l0h1, marker="s", ls="--", color="C3", label="L0H1 (anti-L2R)")
    ax.plot(steps, [p["heavy_tau_mean"] for p in per_step], marker="^", ls=":",
            color="C7", label="heavy")
    ax.axhspan(0.56, 0.64, alpha=0.12, color="C0",
               label="L0H5 prior @5k (3-seed)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("training step"); ax.set_ylabel("tau_vs_l2r")
    ax.set_title("Q1: L0H5 order-signal strength vs training"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(steps, [p["l0h5_rank_mean"] for p in per_step], marker="o", color="C0")
    ax.axhline(1, color="g", ls="--", lw=0.8, label="rank 1 (winner)")
    ax.set_xlabel("training step"); ax.set_ylabel("L0H5 rank (|tau| order)")
    ax.set_title("Q2: is L0H5 the winner head?"); ax.invert_yaxis(); ax.legend(fontsize=8)
    for p in per_step:
        ax.annotate(p["modal_winner"], (p["step"], p["l0h5_rank_mean"]),
                    fontsize=7, xytext=(0, 6), textcoords="offset points", ha="center")

    ax = axes[1, 0]
    ax.plot(steps, [p["layer0_frac_mean"] for p in per_step], marker="o", color="C2")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("training step"); ax.set_ylabel("layer-0 |tau| / overall max")
    ax.set_title("order-signal concentration in Layer 0")

    ax = axes[1, 1]
    ax.plot(steps, [p["l0h5_first_H_mean"] for p in per_step], marker="o", color="C4",
            label="first_step_entropy")
    ax.plot(steps, [p["l0h5_pairwise_mean"] for p in per_step], marker="s", color="C5",
            label="mean_pairwise_tau")
    ax.set_xlabel("training step"); ax.set_ylabel("diversity health")
    ax.set_title("L0H5 teacher diversity health"); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="block_lo_arm_order_network/batch_readout/logs/l0h5_evo")
    args = ap.parse_args()

    scans = load_scans(args.logdir)
    if not scans:
        print(f"no scans found in {args.logdir}")
        return
    steps, per_step = aggregate(scans)
    vd = verdict(per_step)

    report = {"steps": steps, "per_step": per_step, "verdict": vd}
    out_json = pathlib.Path(args.logdir) / "report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    make_figure(per_step, str(pathlib.Path(args.logdir) / "l0h5_evolution.png"))

    print(f"\n=== VERDICT: {vd['label']} ===\n{vd['detail']}\n")
    print(f"{'step':>7} {'L0H5 tau':>16} {'rank':>5} {'winner':>7} {'L0frac':>7} {'firstH':>7}")
    for p in per_step:
        print(f"{p['step']:>7} {p['l0h5_tau_mean']:>+9.3f}+-{p['l0h5_tau_std']:.3f} "
              f"{p['l0h5_rank_mean']:>5.1f} {p['modal_winner']:>7} "
              f"{p['layer0_frac_mean']:>7.2f} {p['l0h5_first_H_mean']:>7.2f}")
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
