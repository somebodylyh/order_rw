#!/usr/bin/env python3
"""Evaluate and compare head-gated g_beta training runs.

Reads history.json from each run directory, produces a comparison table
(TSV) with per-run metrics, and performs the selected-head sanity check:
the head selected by sparse gate should perform similarly as a single-head
g_beta. If sparse gate greatly exceeds every selected single head, the gate
is doing nontrivial head composition, not simple head selection.

Usage:
  python3 analyses/eval_head_gated_gbeta.py \
    --runs /tmp/head_gated_smoke/single_head \
           /tmp/head_gated_smoke/mean_head \
           /tmp/head_gated_smoke/head_gated_soft_all \
           /tmp/head_gated_smoke/head_gated_topk2 \
    --out-dir reports/head_gated_gbeta_eval
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional

import numpy as np


def _load_history(run_dir: pathlib.Path) -> dict:
    """Load history.json from a run directory. Returns None if not found."""
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        return None
    return json.loads(hist_path.read_text())


def _extract_config(run_dir: pathlib.Path, history: dict) -> dict:
    """Infer variant/config from run directory name and best checkpoint."""
    config = {"run": run_dir.name, "path": str(run_dir)}
    best = history.get("best", {}).get("metrics", {})

    # Try to infer variant from dir name
    name = run_dir.name.lower()
    if "single" in name:
        config["variant"] = "single_head"
    elif "mean" in name:
        config["variant"] = "mean_head"
    elif "gated" in name or "head_gated" in name:
        config["variant"] = "head_gated"
    else:
        config["variant"] = "unknown"

    # Try to load config from checkpoint
    ckpt_path = run_dir / "g_beta_best.pt"
    if ckpt_path.exists():
        import torch
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt_cfg = ckpt.get("config", {})
            config["variant"] = ckpt_cfg.get("variant", config["variant"])
            config["H"] = ckpt_cfg.get("H", None)
            config["N"] = ckpt_cfg.get("N", None)
            # Only gate-relevant fields for head_gated variant
            if config["variant"] == "head_gated":
                config["gate_mode"] = ckpt_cfg.get("gate_mode", "unknown")
                config["topk"] = ckpt_cfg.get("topk", None)
            elif config["variant"] == "single_head":
                config["gate_mode"] = "N/A"
                config["topk"] = "N/A"
                config["head_idx"] = ckpt_cfg.get("head_idx", None)
            elif config["variant"] == "mean_head":
                config["gate_mode"] = "N/A"
                config["topk"] = "N/A"
            else:
                config["gate_mode"] = ckpt_cfg.get("gate_mode", "unknown")
                config["topk"] = ckpt_cfg.get("topk", None)
        except Exception:
            pass

    # Extract metrics from best epoch
    for key in ["tau_model_vs_teacher", "tau_model_vs_semantic_path",
                "tau_model_vs_identity", "gate_entropy", "top_heads",
                "train_loss", "val_loss", "epoch"]:
        config[key] = best.get(key, None)

    return config


def _sanity_check(runs: list[dict]) -> str:
    """Selected-head sanity check: sparse vs its top heads as single-head.

    For each head_gated run, identifies the top-2 selected heads, then looks
    for single_head runs (if available) targeting those heads and compares taus.
    """
    lines = []
    lines.append("# Selected-Head Sanity Check")
    lines.append("")

    gated_runs = [r for r in runs if r.get("variant") == "head_gated"]
    single_runs = {r.get("head_idx"): r for r in runs
                   if r.get("variant") == "single_head" and r.get("head_idx") is not None}

    if not gated_runs:
        lines.append("(no head_gated runs found — skipping)")
        return "\n".join(lines)

    for gr in gated_runs:
        lines.append(f"## {gr['run']}")
        lines.append(f"  gate_mode={gr.get('gate_mode')}, topk={gr.get('topk')}")
        tau_gated = gr.get("tau_model_vs_teacher")
        lines.append(f"  tau_vs_teacher (gated): {tau_gated:.4f}" if tau_gated is not None
                     else "  tau_vs_teacher (gated): N/A")

        top_heads = gr.get("top_heads", [])
        if not top_heads:
            lines.append("  (no top_heads info)")
            continue

        for i, h in enumerate(top_heads[:2]):
            h = int(h)
            lines.append(f"  Top-{i+1} head: H{h}")
            if h in single_runs:
                tau_single = single_runs[h].get("tau_model_vs_teacher")
                if tau_single is not None and tau_gated is not None:
                    delta = tau_gated - tau_single
                    verdict = ("gate ≈ single head (clean selection)"
                               if abs(delta) < 0.05 else
                               ("gate >> single head (nontrivial composition)"
                                if delta > 0.05 else
                                "gate < single head (gate hurts signal)"))
                    lines.append(f"    single_H{h} tau={tau_single:.4f}, delta={delta:+.4f} → {verdict}")
                else:
                    lines.append(f"    single_H{h} tau={tau_single}")
            else:
                lines.append(f"    (no single_head H{h} run available for comparison)")

        # Gate entropy
        ent = gr.get("gate_entropy")
        if ent is not None:
            H = gr.get("H", 8)
            max_ent = np.log(H)
            lines.append(f"  gate_entropy={ent:.3f} / max={max_ent:.3f} "
                         f"({ent/max_ent*100:.0f}% of max)")
        lines.append("")

    return "\n".join(lines)


def _comparison_table(runs: list[dict]) -> str:
    """TSV comparison table of all runs."""
    columns = [
        "run", "variant", "gate_mode", "topk", "head_idx",
        "tau_vs_teacher", "tau_vs_semantic", "tau_vs_identity",
        "gate_entropy", "top1_head", "top2_head",
        "val_loss", "best_epoch",
    ]
    header = "\t".join(columns)

    rows = []
    for r in runs:
        top_heads = r.get("top_heads", []) or []
        row = [
            r.get("run", ""),
            r.get("variant", ""),
            r.get("gate_mode", ""),
            str(r.get("topk", "")),
            str(r.get("head_idx", "")),
            f"{r.get('tau_model_vs_teacher', 0):.4f}" if r.get("tau_model_vs_teacher") is not None else "N/A",
            f"{r.get('tau_model_vs_semantic_path', 0):.4f}" if r.get("tau_model_vs_semantic_path") is not None else "N/A",
            f"{r.get('tau_model_vs_identity', 0):.4f}" if r.get("tau_model_vs_identity") is not None else "N/A",
            f"{r.get('gate_entropy', 0):.3f}" if r.get("gate_entropy") is not None else "N/A",
            str(int(top_heads[0])) if len(top_heads) > 0 else "",
            str(int(top_heads[1])) if len(top_heads) > 1 else "",
            f"{r.get('val_loss', 0):.4f}" if r.get("val_loss") is not None else "N/A",
            str(r.get("epoch", "")),
        ]
        rows.append("\t".join(row))

    return "\n".join([header] + rows)


def main():
    p = argparse.ArgumentParser(description="Evaluate head-gated g_beta runs")
    p.add_argument("--runs", nargs="+", required=True,
                   help="Run output directories (containing history.json)")
    p.add_argument("--out-dir", default=None,
                   help="Directory for output reports")
    args = p.parse_args()

    # Load all runs
    runs = []
    for d in args.runs:
        d = pathlib.Path(d)
        hist = _load_history(d)
        if hist is None:
            print(f"Warning: no history.json in {d}, skipping")
            continue
        config = _extract_config(d, hist)
        runs.append(config)

    if not runs:
        print("No valid runs found.")
        sys.exit(1)

    print(f"Loaded {len(runs)} runs")

    # Comparison table
    table = _comparison_table(runs)
    print("\n=== Comparison Table ===")
    print(table)

    # Sanity check
    sanity = _sanity_check(runs)
    print(f"\n{sanity}")

    # Save outputs
    if args.out_dir:
        out = pathlib.Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "comparison.tsv").write_text(table)
        (out / "sanity_check.md").write_text(sanity)
        # Full JSON
        (out / "runs.json").write_text(json.dumps(runs, indent=2, default=str))
        print(f"\nSaved reports to {out}/")

    # Selection recommendation
    best = max(runs, key=lambda r: r.get("tau_model_vs_teacher") or -1.0)
    print(f"\nTop run by tau_vs_teacher: {best['run']} "
          f"({best.get('tau_model_vs_teacher', 'N/A')})")


if __name__ == "__main__":
    main()
