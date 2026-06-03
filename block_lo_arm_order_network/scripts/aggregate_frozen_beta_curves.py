"""Aggregate the frozen-β hook ladder eval curves vs the clean_base baseline.

Fair comparison (see spec §5 note): raw train loss is NOT comparable across runs
because the hook trains under g_β-order (≈L2R, easier) while baseline trains under
random order. We therefore compare the FIXED-protocol eval NLLs that every run logs
identically: `unstructured` (order-agnostic, the key metric) and `ori_l2r` (L2R order).

Sources:
  - hook runs: parse `[Eval @ N] ... unstructured=.. | rw=..` lines from each
    probe_results/frozen_beta_from{START}.log
  - baseline: read last_eval from probe_results/clean_base_random_perm/ckpt_step{N}.pt
"""
import glob
import json
import pathlib
import re
import sys

import numpy as np
import torch

PKG = pathlib.Path(__file__).resolve().parent.parent
CLEAN = PKG / "probe_results" / "clean_base_random_perm"
EVAL_RE = re.compile(
    r"\[Eval @ (\d+)\].*?train_obj=([\d.]+).*?ori_l2r=([\d.]+).*?"
    r"model_order=([\d.]+).*?unstructured=([\d.]+).*?rw=([\d.]+)"
)
KEYS = ["train_obj", "ori_l2r", "model_order", "unstructured", "rw"]


def parse_hook_log(path):
    out = {}
    for line in pathlib.Path(path).read_text(errors="ignore").splitlines():
        m = EVAL_RE.search(line)
        if m:
            step = int(m.group(1))
            out[step] = {k: float(m.group(i + 2)) for i, k in enumerate(KEYS)}
    return out


def baseline_curve():
    curve = {}
    for f in sorted(glob.glob(str(CLEAN / "ckpt_step*.pt"))):
        step = int(re.search(r"ckpt_step(\d+)", f).group(1))
        c = torch.load(f, map_location="cpu", weights_only=False)
        le = c.get("last_eval")
        if not le:
            continue
        curve[step] = {
            "train_obj": le["val_train_objective"]["loss_token_avg"],
            "ori_l2r": le["val_ori_l2r_block"]["loss_token_avg"],
            "model_order": le["val_model_order"]["loss_token_avg"],
            "unstructured": le["val_unstructured_order"]["loss_token_avg"],
            "rw": le["val_rw_order"]["loss_token_avg"],
        }
    return curve


def fmt_curve(name, curve, metric):
    pts = sorted(curve.items())
    s = " ".join(f"{st//1000}k:{v[metric]:.3f}" for st, v in pts)
    return f"  {name:18s} {s}"


def main():
    runs = {}
    for f in sorted(glob.glob(str(PKG / "probe_results" / "frozen_beta_from*.log"))):
        start = re.search(r"from(\d+)\.log", f)
        if not start:
            continue
        runs[f"hook_from{int(start.group(1))//1000}k"] = parse_hook_log(f)
    base = baseline_curve()

    report = {"baseline": base, "hook_runs": runs}
    (PKG / "probe_results" / "frozen_beta_curves.json").write_text(json.dumps(report, indent=2, default=float))

    for metric in ("unstructured", "ori_l2r", "rw"):
        print(f"\n=== {metric} (NLL/token, lower=better) ===")
        print(fmt_curve("baseline(random)", base, metric))
        for name in sorted(runs):
            if runs[name]:
                print(fmt_curve(name, runs[name], metric))

    # final-step comparison at 60k
    print("\n=== @60k final ===")
    b60 = base.get(60000, {})
    print(f"  baseline    unstructured={b60.get('unstructured', float('nan')):.4f} "
          f"ori_l2r={b60.get('ori_l2r', float('nan')):.4f}")
    for name in sorted(runs):
        r = runs[name]
        steps = sorted(r)
        if not steps:
            continue
        last = steps[-1]
        print(f"  {name:14s} @{last}: unstructured={r[last]['unstructured']:.4f} "
              f"ori_l2r={r[last]['ori_l2r']:.4f}")

    print(f"\nsaved -> {PKG / 'probe_results' / 'frozen_beta_curves.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, metric in zip(axes, ("unstructured", "ori_l2r")):
            if base:
                xs = sorted(base)
                ax.plot([x / 1000 for x in xs], [base[x][metric] for x in xs],
                        "k-o", lw=2, label="baseline (random order)")
            for name in sorted(runs):
                r = runs[name]
                if not r:
                    continue
                xs = sorted(r)
                ax.plot([x / 1000 for x in xs], [r[x][metric] for x in xs], "-", label=name)
            ax.set_xlabel("step (k)"); ax.set_ylabel(f"{metric} NLL/token")
            ax.set_title(f"frozen-β hook vs baseline: {metric}")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        out = PKG / "probe_results" / "frozen_beta_curves.png"
        fig.tight_layout(); fig.savefig(out, dpi=120)
        print(f"plot  -> {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
