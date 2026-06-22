#!/usr/bin/env python3
"""Evaluate small-train audition results for head-gated g_beta candidates.

Bridges offline τ ranking (Task 5) and full frozen feedback (Task 7).
Reads eval_curve.tsv from each audition run, computes delta vs random
continuation, and applies the decision rule:

  offline τ high + small-train win        → strongest candidate → full 60k
  offline τ high + small-train no-improve → readable but not useful
  offline τ medium + small-train win      → usable despite modest τ
  offline τ low + small-train lose        → drop

Usage:
  python3 analyses/eval_head_gated_audition.py \
    --runs audition_results/*/ \
    --random-baseline audition_results/random_continuation/ \
    --offline-eval reports/head_gated_gbeta_eval/runs.json \
    --out-dir reports/head_gated_audition/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional

import numpy as np


def _load_eval_curve(run_dir: pathlib.Path) -> Optional[dict]:
    """Load eval_curve.tsv from a run directory. Returns summary dict."""
    tsv_path = run_dir / "eval_curve.tsv"
    if not tsv_path.exists():
        # Try train_log.txt fallback
        log_path = run_dir / "train_log.txt"
        if log_path.exists():
            return {"run": run_dir.name, "status": "log_only",
                    "note": "eval_curve.tsv not found; manual inspection needed"}
        return None

    # Read TSV
    lines = tsv_path.read_text().strip().split("\n")
    if len(lines) < 2:
        return None

    header = lines[0].split("\t")
    data = []
    for line in lines[1:]:
        vals = line.strip().split("\t")
        row = {}
        for i, h in enumerate(header):
            if i < len(vals):
                try:
                    row[h] = float(vals[i])
                except ValueError:
                    row[h] = vals[i]
        data.append(row)

    # Summary metrics
    final = data[-1] if data else {}
    first = data[0] if data else {}

    return {
        "run": run_dir.name,
        "path": str(run_dir),
        "steps": len(data),
        "final_step": final.get("step", final.get("global_step", None)),
        "final_val_ori_l2r": final.get("val_ori_l2r_block", final.get("val_ori_l2r", None)),
        "final_train_obj": final.get("train_obj", final.get("train_loss", None)),
        "first_val_ori_l2r": first.get("val_ori_l2r_block", first.get("val_ori_l2r", None)),
        "delta_val": (final.get("val_ori_l2r_block", 0) - first.get("val_ori_l2r_block", 0)
                      if final and first else None),
        "raw_data": data,
    }


def _compute_decision(offline_tau: Optional[float], delta_vs_random: Optional[float],
                      run_name: str) -> str:
    """Apply the 4-way decision rule."""
    if offline_tau is None or delta_vs_random is None:
        return "insufficient_data"

    if offline_tau >= 0.10 and delta_vs_random < 0:
        return "strongest → full 60k"
    elif offline_tau >= 0.10 and delta_vs_random >= 0:
        return "readable but not useful as controller"
    elif offline_tau < 0.10 and delta_vs_random < 0:
        return "usable controller despite modest τ"
    else:
        return "drop"


def main():
    p = argparse.ArgumentParser(description="Evaluate small-train audition results")
    p.add_argument("--runs", nargs="+", required=True,
                   help="Audition run directories")
    p.add_argument("--random-baseline", required=True,
                   help="Random continuation audition run (baseline)")
    p.add_argument("--offline-eval", default=None,
                   help="Offline eval runs.json (for τ cross-reference)")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for report")
    args = p.parse_args()

    # Load audition runs
    runs = []
    for d in args.runs:
        d = pathlib.Path(d)
        summary = _load_eval_curve(d)
        if summary:
            runs.append(summary)
        else:
            print(f"Warning: no eval data in {d}")

    # Load random baseline
    random_dir = pathlib.Path(args.random_baseline)
    random_summary = _load_eval_curve(random_dir)
    if not random_summary:
        print("Error: random baseline not found or invalid")
        sys.exit(1)

    random_final = random_summary.get("final_val_ori_l2r")
    if random_final is None:
        print("Error: random baseline has no final val_ori_l2r")
        sys.exit(1)

    print(f"Random baseline final val_ori_l2r: {random_final:.4f}")

    # Load offline τ (optional)
    offline_taus = {}
    if args.offline_eval:
        runs_json = pathlib.Path(args.offline_eval)
        if runs_json.exists():
            offline_data = json.loads(runs_json.read_text())
            for r in offline_data:
                offline_taus[r.get("run", "")] = r.get("tau_model_vs_teacher")

    # Compare each run vs random
    print("\n=== Small-Train Audition Results ===\n")
    lines = []
    lines.append("run\tfinal_val_ori_l2r\tdelta_vs_random\toffline_τ\tdecision")
    lines.append("-" * 80)

    decisions = []
    for r in runs:
        name = r["run"]
        final_val = r.get("final_val_ori_l2r")
        if final_val is None:
            lines.append(f"{name}\tN/A\tN/A\tN/A\tno_val_data")
            continue

        delta = final_val - random_final
        offline_tau = offline_taus.get(name)
        decision = _compute_decision(offline_tau, delta, name)

        lines.append(f"{name}\t{final_val:.4f}\t{delta:+.4f}\t"
                     f"{offline_tau:.4f}" if offline_tau is not None else f"{name}\t{final_val:.4f}\t{delta:+.4f}\tN/A"
                     f"\t{decision}")

        decisions.append({
            "run": name,
            "final_val_ori_l2r": final_val,
            "delta_vs_random": delta,
            "offline_tau": offline_tau,
            "decision": decision,
        })

    report = "\n".join(lines)
    print(report)

    # Summary
    strongest = [d for d in decisions if "strongest" in d["decision"]]
    usable = [d for d in decisions if "usable" in d["decision"]]
    print(f"\nStrongest candidates (→ full 60k): {len(strongest)}")
    for d in strongest:
        print(f"  {d['run']}: Δ={d['delta_vs_random']:+.4f}, τ={d['offline_tau']}")
    print(f"Usable candidates: {len(usable)}")
    print(f"Drop/not-useful: {len(decisions) - len(strongest) - len(usable)}")

    # Save
    if args.out_dir:
        out = pathlib.Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "audition_report.txt").write_text(report)
        (out / "audition_decisions.json").write_text(
            json.dumps({"decisions": decisions, "random_baseline": random_summary},
                       indent=2, default=str)
        )
        print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
