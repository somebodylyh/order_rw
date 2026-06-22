#!/usr/bin/env python3
"""Head-drift diagnostic for head-gated g_beta feedback training.

Answers: did MLP/controller participation alter the input head's readable
order signal beyond normal training drift?

Method: difference-in-differences across checkpoint ladders (20k→60k).

  delta_random   = metric_60k_random - metric_20k_start
  delta_feedback = metric_60k_feedback - metric_20k_start
  feedback_effect = delta_feedback - delta_random

If |feedback_effect| is small, the head's order signal is stable under
feedback. If large, the controller may be reward-hacking the input head.

Usage:
  python3 analyses/head_drift_diagnostic.py \
    --arms random=path/to/random_ckpts/ \
           feedback=path/to/head_gated_ckpts/ \
    --start-step 20000 \
    --steps 30000 40000 50000 60000 \
    --gbeta-ckpt path/to/g_beta_best.pt \
    --out-dir reports/head_drift_head_gated_gbeta/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict
from typing import Optional

import numpy as np
from scipy.stats import kendalltau

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))

from none_separated_block_graph import build_none_separated_B, rollout_by_method  # noqa: E402


def _tau(a: np.ndarray, b: np.ndarray) -> float:
    val, _ = kendalltau(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
    return float(val) if not np.isnan(val) else float("nan")


def _block_perm_from_inv(inv_perm: np.ndarray) -> np.ndarray:
    inv = np.asarray(inv_perm, dtype=np.int64)
    block = np.empty_like(inv)
    block[inv] = np.arange(inv.shape[0], dtype=np.int64)
    return block


def _find_checkpoint(run_dir: pathlib.Path, step: int) -> Optional[pathlib.Path]:
    """Find a checkpoint at a specific step in a run directory.

    Searches for files matching ckpt_step{step}.pt.
    """
    pattern = f"ckpt_step{step}.pt"
    candidates = list(run_dir.glob(pattern))
    if candidates:
        return candidates[0]
    # Try recursive
    candidates = list(run_dir.glob(f"**/{pattern}"))
    if candidates:
        return candidates[0]
    # Try any file containing the step number
    for f in sorted(run_dir.glob("*.pt")):
        if f"step{step}" in f.name:
            return f
    return None


def _extract_per_head_metrics(ckpt_path: pathlib.Path, device: str = "cuda:0",
                              M: int = 12, batch_size: int = 4, fwd_batch: int = 4,
                              seed: int = 0) -> dict:
    """Extract per-head model-frame metrics from one checkpoint.

    Uses the same extraction path as diag_model_frame_feedback.py.
    Returns dict with per-(layer,head) metrics and metadata.
    """
    from search_strict_label_free_65 import _extract_A_with_none_lh_model_frame

    run_args = argparse.Namespace(
        ckpt=str(ckpt_path), device=device, M=M, batch_size=batch_size,
        fwd_batch=fwd_batch, seed=seed,
        perm_orientation="model_to_phys",
    )

    A_lh, inv_perm, meta = _extract_A_with_none_lh_model_frame(run_args)
    inv_perm = np.asarray(inv_perm, dtype=np.int64)
    block_perm = _block_perm_from_inv(inv_perm)
    identity = np.arange(inv_perm.shape[0], dtype=np.int64)

    L, H = A_lh.shape[:2]
    methods = ["C-D+L", "L", "C-D", "C+L", "C", "-D"]

    rows = []
    for layer in range(L):
        for head in range(H):
            B65_model = build_none_separated_B(A_lh[layer, head])
            for method in methods:
                sigma_model = rollout_by_method(B65_model, method)
                rows.append({
                    "layer": layer,
                    "head": head,
                    "method": method,
                    "tau_model_vs_semantic_path": _tau(sigma_model, block_perm),
                    "tau_model_vs_identity": _tau(sigma_model, identity),
                    "sigma_model_first16": " ".join(str(int(x)) for x in sigma_model[:16]),
                })

    return {
        "rows": rows,
        "L": L, "H": H,
        "block_perm": block_perm.tolist(),
        "inv_perm": inv_perm.tolist(),
        "meta": dict(meta),
    }


def _extract_gate_analysis(ckpt_path: pathlib.Path,
                           g_beta_ckpt: Optional[str] = None,
                           device: str = "cuda:0",
                           M: int = 12, batch_size: int = 4, fwd_batch: int = 4,
                           seed: int = 0) -> Optional[dict]:
    """Extract gate analysis using a head-gated g_beta model.

    If g_beta_ckpt is None, returns None.
    """
    if g_beta_ckpt is None:
        return None

    import torch
    from batch_readout.head_gated_gbeta import HeadGatedGBeta
    from batch_readout.hook_order_provider import extract_all_heads_A_for_batch
    from clean_training_protocol import CleanPermutation
    from training_utils import SEQ_LEN
    from neural_readout.extract_b import _load_model_and_chunks

    # Load g_beta
    state = torch.load(g_beta_ckpt, map_location="cpu", weights_only=False)
    cfg = state["config"]
    gb_N = cfg.get("N", 64)
    gb_H = cfg.get("H", 16)
    gb_gate_mode = cfg.get("gate_mode", "topk")
    gb_topk = cfg.get("topk", 2)

    gbeta = HeadGatedGBeta(N=gb_N, H=gb_H, gate_mode=gb_gate_mode, topk=gb_topk)
    gbeta.load_state_dict(state["model"])
    gbeta.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        gbeta.to(device)

    # Load model + chunks
    try:
        model, chunks, clean_perm, dev, chunk_idx = _load_model_and_chunks(
            str(ckpt_path), M * batch_size, seed, device, "train"
        )
    except Exception as e:
        print(f"  [gate] failed to load model: {e}")
        return None

    # Extract all heads from layer 0
    from batch_readout.hook_order_provider import random_probe_token_orders

    probe = random_probe_token_orders(M * batch_size, seed, 0, str(dev))
    B_heads_list = extract_all_heads_A_for_batch(
        model, chunks.to(dev), 0, clean_perm, dev, probe, none_mode="model"
    )  # (M*B, H, N, N) in model frame

    # Group into batch-mean
    total = M * batch_size
    B_heads_mean = B_heads_list.reshape(M, batch_size, gb_H, gb_N, gb_N).mean(dim=1)

    # Run gate
    gate_weights_all = []
    for m in range(M):
        Bh = B_heads_mean[m:m+1].to(gbeta.readout[0].weight.device)  # (1, H, N, N)
        with torch.no_grad():
            _, aux = gbeta(Bh)
        gate_weights_all.append(aux["gate_weights"].cpu().numpy())

    gate_weights_all = np.concatenate(gate_weights_all, axis=0)  # (M, H)
    mean_gw = gate_weights_all.mean(axis=0)  # (H,)
    top_order = np.argsort(-mean_gw)

    # Entropy
    w = gate_weights_all
    eps = 1e-9
    entropy = float(-(w * np.log(w + eps)).sum(axis=1).mean())

    return {
        "gate_entropy_mean": entropy,
        "gate_weights_mean": mean_gw.tolist(),
        "top_heads": top_order[:4].tolist(),
        "top1_head": int(top_order[0]),
        "top2_heads": top_order[:2].tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="+", required=True,
                   help="Arms as name=path, e.g. random=path/to/random/ feedback=path/to/fb/")
    p.add_argument("--start-step", type=int, default=20000,
                   help="Baseline step (e.g., 20000)")
    p.add_argument("--steps", nargs="+", type=int, default=[30000, 40000, 50000, 60000],
                   help="Ladder steps to scan")
    p.add_argument("--end-step", type=int, default=60000,
                   help="Final step for delta computation")
    p.add_argument("--gbeta-ckpt", default=None,
                   help="Path to head-gated g_beta checkpoint for gate analysis")
    p.add_argument("--tracked-head", nargs=2, type=int, default=[0, 2],
                   metavar=("LAYER", "HEAD"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--M", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--fwd-batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="reports/head_drift_head_gated_gbeta")
    p.add_argument("--no-gate", action="store_true",
                   help="Skip gate analysis (faster)")
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse arms
    arms = {}
    for spec in args.arms:
        name, path = spec.split("=", 1)
        arms[name.strip()] = pathlib.Path(path.strip())

    all_steps = [args.start_step] + args.steps
    tracked_l, tracked_h = args.tracked_head

    # Extract per-step per-arm metrics
    all_data = {}  # arm_name -> step -> {rows, gate}

    for arm_name, arm_dir in arms.items():
        print(f"\n=== Arm: {arm_name} ({arm_dir}) ===")
        arm_data = {}
        for step in all_steps:
            ckpt = _find_checkpoint(arm_dir, step)
            if ckpt is None:
                print(f"  step {step}: checkpoint NOT FOUND (skipping)")
                continue
            print(f"  step {step}: {ckpt.name}")
            metrics = _extract_per_head_metrics(
                ckpt, device=args.device,
                M=args.M, batch_size=args.batch_size,
                fwd_batch=args.fwd_batch, seed=args.seed,
            )
            arm_data[step] = metrics

            # Gate analysis (only at start and end to save time)
            if not args.no_gate and args.gbeta_ckpt and step in (args.start_step, args.end_step):
                gate = _extract_gate_analysis(
                    ckpt, g_beta_ckpt=args.gbeta_ckpt,
                    device=args.device, M=args.M, batch_size=args.batch_size,
                    fwd_batch=args.fwd_batch, seed=args.seed,
                )
                if gate:
                    arm_data[step]["_gate"] = gate
                    print(f"    gate: top1=H{gate['top1_head']}, ent={gate['gate_entropy_mean']:.3f}")

        all_data[arm_name] = arm_data

    # --- Difference-in-differences ---
    print(f"\n=== Difference-in-Differences (Δ{args.start_step}→{args.end_step}) ===\n")

    start = args.start_step
    end = args.end_step
    report_lines = []

    # Identify "random" arm (baseline)
    random_arm = None
    for name in arms:
        if "random" in name.lower():
            random_arm = name
            break

    # For tracked head: compute DiD
    tracked_deltas = {}
    for arm_name, arm_data in all_data.items():
        if start not in arm_data or end not in arm_data:
            print(f"  {arm_name}: missing start or end data, skipping")
            continue

        start_rows = arm_data[start]["rows"]
        end_rows = arm_data[end]["rows"]

        # Find tracked head rows for C-D+L method
        def _find_row(rows, layer, head, method="C-D+L"):
            for r in rows:
                if r["layer"] == layer and r["head"] == head and r["method"] == method:
                    return r
            return None

        r_start = _find_row(start_rows, tracked_l, tracked_h)
        r_end = _find_row(end_rows, tracked_l, tracked_h)

        if r_start is None or r_end is None:
            print(f"  {arm_name}: tracked head L{tracked_l}H{tracked_h} not found")
            continue

        delta_sem = r_end["tau_model_vs_semantic_path"] - r_start["tau_model_vs_semantic_path"]
        delta_id = r_end["tau_model_vs_identity"] - r_start["tau_model_vs_identity"]

        tracked_deltas[arm_name] = {
            "tau_start_semantic": r_start["tau_model_vs_semantic_path"],
            "tau_end_semantic": r_end["tau_model_vs_semantic_path"],
            "delta_semantic": delta_sem,
            "tau_start_identity": r_start["tau_model_vs_identity"],
            "tau_end_identity": r_end["tau_model_vs_identity"],
            "delta_identity": delta_id,
            "sigma_start_first16": r_start["sigma_model_first16"],
            "sigma_end_first16": r_end["sigma_model_first16"],
        }

    # Compute feedback effect vs random
    if random_arm and random_arm in tracked_deltas:
        rand_deltas = tracked_deltas[random_arm]
        print(f"Random baseline ({random_arm}):")
        print(f"  Δsemantic = {rand_deltas['delta_semantic']:+.4f}")
        print(f"  Δidentity = {rand_deltas['delta_identity']:+.4f}")
        print()

        for arm_name, deltas in tracked_deltas.items():
            if arm_name == random_arm:
                continue
            fb_effect_sem = deltas["delta_semantic"] - rand_deltas["delta_semantic"]
            fb_effect_id = deltas["delta_identity"] - rand_deltas["delta_identity"]

            verdict_sem = ("⚠ reward hacking?" if abs(fb_effect_sem) > 0.15
                           else "✓ stable" if abs(fb_effect_sem) < 0.05
                           else "~ mild drift")
            verdict_id = ("⚠ reward hacking?" if abs(fb_effect_id) > 0.15
                          else "✓ stable" if abs(fb_effect_id) < 0.05
                          else "~ mild drift")

            print(f"{arm_name}:")
            print(f"  Δsemantic = {deltas['delta_semantic']:+.4f}  "
                  f"feedback_effect = {fb_effect_sem:+.4f}  {verdict_sem}")
            print(f"  Δidentity = {deltas['delta_identity']:+.4f}  "
                  f"feedback_effect = {fb_effect_id:+.4f}  {verdict_id}")

            report_lines.append({
                "arm": arm_name,
                "delta_semantic": deltas["delta_semantic"],
                "feedback_effect_semantic": fb_effect_sem,
                "verdict_semantic": verdict_sem,
                "delta_identity": deltas["delta_identity"],
                "feedback_effect_identity": fb_effect_id,
                "verdict_identity": verdict_id,
            })
    else:
        print("(no random arm found — skipping DiD computation)")

    # Gate drift
    print(f"\n=== Gate Drift ({start}→{end}) ===")
    for arm_name, arm_data in all_data.items():
        gate_start = arm_data.get(start, {}).get("_gate")
        gate_end = arm_data.get(end, {}).get("_gate")
        if gate_start and gate_end:
            print(f"  {arm_name}:")
            print(f"    top1: H{gate_start['top1_head']} → H{gate_end['top1_head']}")
            print(f"    ent:  {gate_start['gate_entropy_mean']:.3f} → {gate_end['gate_entropy_mean']:.3f}")

    # Save report
    report = {
        "convention": {
            "feedback_effect": "delta_arm - delta_random",
            "verdict": "|effect| < 0.05 = stable, > 0.15 = reward hacking",
        },
        "tracked_head": {"layer": tracked_l, "head": tracked_h},
        "arms": list(arms.keys()),
        "tracked_deltas": tracked_deltas,
        "did_report": report_lines,
    }
    (out_dir / "head_drift_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved to {out_dir}/head_drift_report.json")


if __name__ == "__main__":
    main()
