#!/usr/bin/env python3
r"""Live summary for the Phase-2 text arms vs random/v3 baselines (no GPU).

Reads the (possibly partial) eval_curves and prints / writes a comparison of
val_ori_l2r_block at every step the arm has reached, plus deltas vs random and v3,
highlighting step 25000 / 30000. Safe to run any time during training.

Run:
    python block_lo_arm_order_network/scripts/summarize_phase2_text.py
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
_REPO = _HERE.parent
ARM_DIR = _REPO / "probe_results/attention_order_mlp"   # arms write to the REPO-level tree
RANDOM = _HERE / "probe_results/clean_base_random_perm/eval_curve.tsv"
V3 = _HERE / "probe_results/clean_method_graph_rw_v3_from20k/eval_curve.tsv"
ARMS = ["cont_MLP_CDL_source_start", "cont_MLP_CDL_reversed", "cont_MLP_CDL_original"]
KEY = "val_ori_l2r_block"


def load(path):
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    lines = p.read_text().strip().splitlines()
    if len(lines) < 2:
        return out
    hdr = lines[0].split("\t")
    si, vi = hdr.index("step"), hdr.index(KEY)
    for ln in lines[1:]:
        c = ln.split("\t")
        try:
            out[int(c[si])] = float(c[vi])
        except (ValueError, IndexError):
            pass
    return out


def fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else "—"


def main():
    rnd, v3 = load(RANDOM), load(V3)
    arms = {a: load(ARM_DIR / a / "eval_curve.tsv") for a in ARMS}
    arms = {a: c for a, c in arms.items() if c}            # only arms with data

    lines = ["# Phase-2 text — live summary (val_ori_l2r_block, lower=better)\n",
             f"baselines @25000: random {fmt(rnd.get(25000))} | v3 {fmt(v3.get(25000))}",
             f"baselines @30000: random {fmt(rnd.get(30000))} | v3 {fmt(v3.get(30000))}\n"]
    if not arms:
        lines.append("(no arm eval rows yet — training still warming up)")
        out = "\n".join(lines)
        print(out)
        (ARM_DIR / "SUMMARY.md").write_text(out + "\n")
        return

    for arm, curve in arms.items():
        steps = sorted(curve)
        lines.append(f"## {arm}  (latest step {steps[-1]}, {len(steps)} evals)")
        lines.append("| step | arm | random | v3 | arm−random | arm−v3 |")
        lines.append("|---|---|---|---|---|---|")
        show = [s for s in steps if s in (25000, 30000) or s == steps[-1]]
        for s in sorted(set(show)):
            a = curve[s]; r = rnd.get(s); v = v3.get(s)
            dr = f"{a-r:+.4f}" if r is not None else "—"
            dv = f"{a-v:+.4f}" if v is not None else "—"
            tag = " ⬅ latest" if s == steps[-1] and s not in (25000, 30000) else ""
            lines.append(f"| {s}{tag} | {fmt(a)} | {fmt(r)} | {fmt(v)} | {dr} | {dv} |")
        # verdict line at 30000 if present
        if 30000 in curve:
            a = curve[30000]; r = rnd.get(30000); v = v3.get(30000)
            beat_r = (r is not None and a < r)
            near_v = (v is not None and abs(a - v) < 0.01)
            lines.append(f"\n@30000: beat_random={beat_r}"
                         + (f" (Δ={a-r:+.4f})" if r is not None else "")
                         + f"; close_to_v3(<0.01)={near_v}"
                         + (f" (Δ={a-v:+.4f})" if v is not None else "") + "\n")
        else:
            lines.append("")

    out = "\n".join(lines)
    print(out)
    (ARM_DIR / "SUMMARY.md").write_text(out + "\n")


if __name__ == "__main__":
    main()
