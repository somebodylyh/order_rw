#!/usr/bin/env python3
"""Matched old/new checkpoint audit for strict65 CDL attention signals.

The script fixes eval samples, probe orders, strict65 aggregation, B convention,
and physical remapping across checkpoint groups.  It reports per-head and
consensus CDL tau in both model and physical frames, and repeats the rollout on
B and B.T to expose direction-convention flips.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr

REPO = pathlib.Path(__file__).resolve().parents[1]
PKG = REPO / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))

from batch_readout.label_free_cdl_teacher import (  # noqa: E402
    compute_label_free_quality,
    cdl_rollout_with_standardized_margin,
    headwise_zscore,
    rank_agreement,
    soft_pairwise_teacher,
)
from batch_readout.l0_strict65 import build_model_frame_strict65  # noqa: E402
from clean_training_protocol import expand_model_blocks_to_token_order  # noqa: E402
from neural_readout.extract_b import _load_model_and_chunks  # noqa: E402
from none_separated_block_graph import build_none_separated_B  # noqa: E402
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_vec  # noqa: E402
from training_utils import BLOCK_LEN, N, SEQ_LEN  # noqa: E402


@dataclass(frozen=True)
class CkptJob:
    group: str
    step: int
    path: pathlib.Path


def _tau(order: np.ndarray, ref: np.ndarray) -> float:
    val, _ = kendalltau(np.asarray(order, dtype=np.int64), np.asarray(ref, dtype=np.int64))
    return float(val) if not np.isnan(val) else float("nan")


def _mean(xs: list[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _std(xs: list[float]) -> float:
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    return float(arr.std()) if arr.size else float("nan")


def _safe_corr(x: list[float], y: list[float]) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = ~(np.isnan(xa) | np.isnan(ya))
    xa = xa[mask]
    ya = ya[mask]
    if xa.size < 2 or float(np.std(xa)) < 1e-12 or float(np.std(ya)) < 1e-12:
        return float("nan")
    val, _ = pearsonr(xa, ya)
    return float(val)


def _row_entropy_top1(B: np.ndarray) -> tuple[float, float]:
    rows = np.asarray(B[:, 1:], dtype=np.float64)
    row_sums = rows.sum(axis=1, keepdims=True)
    mask = row_sums[:, 0] > 1e-12
    if not np.any(mask):
        return float("nan"), float("nan")
    p = rows[mask] / row_sums[mask]
    entropy = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)
    top1 = p.max(axis=1)
    return float(entropy.mean()), float(top1.mean())


def _near_diag_mass(B: np.ndarray, width: int = 3) -> float:
    content = np.asarray(B[1:, 1:], dtype=np.float64)
    total = float(content.sum())
    if total <= 1e-12:
        return float("nan")
    mass = 0.0
    for d in range(1, width + 1):
        mass += float(np.diag(content, k=d).sum())
        mass += float(np.diag(content, k=-d).sum())
    return float(mass / total)


def _consensus_order(ranks: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    H, n_blocks = ranks.shape
    if weights is None:
        weights = np.full(H, 1.0 / H, dtype=np.float64)
    pairwise = soft_pairwise_teacher(ranks, weights)
    scores = pairwise.sum(axis=1)
    return np.argsort(-scores, kind="stable").astype(np.int64)[:n_blocks]


def _quality_for_heads(B_heads: np.ndarray, destroy_seed: int, n_destroy_replicas: int) -> dict:
    if n_destroy_replicas > 0:
        return compute_label_free_quality(
            B_heads,
            destroy_seed=destroy_seed,
            n_destroy_replicas=n_destroy_replicas,
        )

    H = B_heads.shape[0]
    orders = np.empty((H, N), dtype=np.int64)
    ranks = np.empty((H, N), dtype=np.int64)
    margins = np.empty(H, dtype=np.float64)
    for h in range(H):
        res = cdl_rollout_with_standardized_margin(B_heads[h])
        order = res["content_order"]
        rank = np.empty(N, dtype=np.int64)
        rank[order] = np.arange(N, dtype=np.int64)
        orders[h] = order
        ranks[h] = rank
        margins[h] = float(res["avg_margin"])
    agreement = rank_agreement(ranks)
    gaps = np.zeros(H, dtype=np.float64)
    quality = headwise_zscore(margins) + headwise_zscore(agreement)
    return {
        "orders": orders,
        "ranks": ranks,
        "margin": margins,
        "destroyed_gap": gaps,
        "agreement": agreement,
        "quality": quality,
    }


def _transpose_strict65(B: np.ndarray) -> np.ndarray:
    out = np.asarray(B, dtype=np.float64).T.copy()
    out[:, 0] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


def _physical_strict65_from_l0(
    attn_l0: np.ndarray,
    probe_orders: np.ndarray,
    inv_perm: np.ndarray,
) -> np.ndarray:
    total, n_heads = attn_l0.shape[:2]
    out = np.empty((total, n_heads, N + 1, N + 1), dtype=np.float32)
    for i in range(total):
        A = _attn_to_A_block_loss_aligned_with_none_vec(
            attn_l0[i],
            probe_orders[i],
            inv_perm,
        )
        for h in range(n_heads):
            out[i, h] = build_none_separated_B(A[h])
    return out


@torch.no_grad()
def extract_l0_strict65_graphs(
    model,
    chunks: torch.Tensor,
    clean_perm,
    device: torch.device,
    seed: int,
    fwd_batch: int,
    layer: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = len(chunks)
    probe_orders = torch.empty((total, SEQ_LEN), dtype=torch.long)
    for i in range(total):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + int(i))
        rand_blocks = torch.randperm(N, generator=gen, device="cpu")
        probe_orders[i] = expand_model_blocks_to_token_order(
            rand_blocks.unsqueeze(0), BLOCK_LEN
        )[0]

    attn_l0_batches = []
    model.eval()
    for start in range(0, total, max(1, int(fwd_batch))):
        stop = min(start + max(1, int(fwd_batch)), total)
        tokens = chunks[start:stop].to(device)
        orders = probe_orders[start:stop].to(device)
        _, _, attn_list = model.forward_fn(tokens, orders, return_attentions=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        attn_l0_batches.append(attn_list[int(layer)].detach().cpu().numpy())
        print(f"    extracted L{int(layer)} attention {stop}/{total}", flush=True)

    attn_l0 = np.concatenate(attn_l0_batches, axis=0)
    probe_np = probe_orders.numpy()
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy().astype(np.int64)
    B_model = build_model_frame_strict65(attn_l0, probe_np)
    B_phys = _physical_strict65_from_l0(attn_l0, probe_np, inv_perm)
    return B_model, B_phys, inv_perm


def _batch_mean(B: np.ndarray, M: int, bs_mean: int) -> np.ndarray:
    return B.reshape(M, bs_mean, *B.shape[1:]).mean(axis=1).astype(np.float32)


def analyze_frame(
    B_mean: np.ndarray,
    frame: str,
    inv_perm: np.ndarray,
    group: str,
    step: int,
    b_label: str,
    n_destroy_replicas: int,
) -> tuple[list[dict], dict]:
    M, H = B_mean.shape[:2]
    l2r = np.arange(N, dtype=np.int64)
    rev = l2r[::-1]

    per_head_orders = np.empty((M, H, N), dtype=np.int64)
    per_head_ranks = np.empty((M, H, N), dtype=np.int64)
    qualities = np.empty((M, H), dtype=np.float64)
    margins = np.empty((M, H), dtype=np.float64)
    near_diag = np.empty((M, H), dtype=np.float64)
    entropy = np.empty((M, H), dtype=np.float64)
    top1 = np.empty((M, H), dtype=np.float64)

    for m in range(M):
        qr = _quality_for_heads(
            B_mean[m],
            destroy_seed=step * 1000 + m,
            n_destroy_replicas=n_destroy_replicas,
        )
        per_head_orders[m] = qr["orders"]
        per_head_ranks[m] = qr["ranks"]
        qualities[m] = qr["quality"]
        margins[m] = qr["margin"]
        for h in range(H):
            near_diag[m, h] = _near_diag_mass(B_mean[m, h])
            entropy[m, h], top1[m, h] = _row_entropy_top1(B_mean[m, h])

    per_head_rows: list[dict] = []
    all_abs_tau: list[float] = []
    all_quality: list[float] = []
    for h in range(H):
        tau_l2r_samples = []
        tau_rev_samples = []
        for m in range(M):
            order = per_head_orders[m, h]
            if frame == "model":
                score_order = inv_perm[order]
            else:
                score_order = order
            tau_l2r_samples.append(_tau(score_order, l2r))
            tau_rev_samples.append(_tau(score_order, rev))
            all_abs_tau.append(abs(tau_l2r_samples[-1]))
            all_quality.append(float(qualities[m, h]))
        per_head_rows.append({
            "ckpt_group": group,
            "step": step,
            "head": h,
            "tau_l2r": _mean(tau_l2r_samples),
            "tau_rev_l2r": _mean(tau_rev_samples),
            "abs_tau_gt_0.5": float(np.mean(np.abs(tau_l2r_samples) > 0.5)),
            "quality": float(np.mean(qualities[:, h])),
            "quality_std": float(np.std(qualities[:, h])),
            "frame": frame,
            "B_or_BT": b_label,
            "near_diag_mass": float(np.mean(near_diag[:, h])),
            "row_entropy": float(np.mean(entropy[:, h])),
            "top1_mass": float(np.mean(top1[:, h])),
            "margin": float(np.mean(margins[:, h])),
        })

    consensus_tau_l2r = []
    consensus_tau_rev = []
    consensus_uniform_tau_l2r = []
    consensus_uniform_tau_rev = []
    for m in range(M):
        # Quality-weighted consensus mirrors the label-free teacher.  The
        # uniform consensus is a useful control for whether quality weighting is
        # doing the work or heads already agree.
        q = qualities[m]
        q = q - np.max(q)
        weights = np.exp(q)
        weights = weights / weights.sum()
        for order, out_l2r, out_rev in [
            (_consensus_order(per_head_ranks[m], weights), consensus_tau_l2r, consensus_tau_rev),
            (_consensus_order(per_head_ranks[m], None), consensus_uniform_tau_l2r, consensus_uniform_tau_rev),
        ]:
            score_order = inv_perm[order] if frame == "model" else order
            out_l2r.append(_tau(score_order, l2r))
            out_rev.append(_tau(score_order, rev))

    consensus = {
        "ckpt_group": group,
        "step": step,
        "frame": frame,
        "B_or_BT": b_label,
        "consensus_tau_l2r": _mean(consensus_tau_l2r),
        "consensus_tau_rev_l2r": _mean(consensus_tau_rev),
        "uniform_consensus_tau_l2r": _mean(consensus_uniform_tau_l2r),
        "uniform_consensus_tau_rev_l2r": _mean(consensus_uniform_tau_rev),
        "mean_quality": float(np.mean(qualities)),
        "std_quality": float(np.std(qualities)),
        "r_quality_abs_tau": _safe_corr(all_quality, all_abs_tau),
        "mean_near_diag_mass": float(np.mean(near_diag)),
        "mean_row_entropy": float(np.mean(entropy)),
        "mean_top1_mass": float(np.mean(top1)),
    }
    return per_head_rows, consensus


def parse_ckpt_specs(specs: list[str], steps: list[int]) -> list[CkptJob]:
    jobs = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--ckpt-dir entries must be name=path, got {spec!r}")
        group, raw_path = spec.split("=", 1)
        ckpt_dir = pathlib.Path(raw_path)
        for step in steps:
            ckpt = ckpt_dir / f"ckpt_step{step}.pt"
            if ckpt.exists():
                jobs.append(CkptJob(group=group, step=int(step), path=ckpt))
            else:
                print(f"[warn] missing {group} step {step}: {ckpt}", flush=True)
    return jobs


def write_csv(path: pathlib.Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", action="append", required=True,
                   help="checkpoint group as name=/path/to/run_dir; repeatable")
    p.add_argument("--steps", type=int, nargs="+", default=[5000, 10000, 20000])
    p.add_argument("--M", type=int, default=500)
    p.add_argument("--bs-mean", type=int, default=4)
    p.add_argument("--layer", type=int, default=0,
                   help="transformer layer index to read attention from (default 0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", default="train")
    p.add_argument("--fwd-batch", type=int, default=32)
    p.add_argument("--n-destroy-replicas", type=int, default=1)
    p.add_argument("--out-dir", default=str(REPO / "reports/old_new_ckpt_matched_audit_20260626"))
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = parse_ckpt_specs(args.ckpt_dir, args.steps)
    if not jobs:
        raise SystemExit("no checkpoint jobs found")

    total = int(args.M) * int(args.bs_mean)
    all_head_rows: list[dict] = []
    all_consensus_rows: list[dict] = []
    manifest = {
        "M": args.M,
        "bs_mean": args.bs_mean,
        "total_eval_samples": total,
        "seed": args.seed,
        "split": args.split,
        "fwd_batch": args.fwd_batch,
        "n_destroy_replicas": args.n_destroy_replicas,
        "jobs": [{"group": j.group, "step": j.step, "path": str(j.path)} for j in jobs],
        "conventions": {
            "strict65": "node 0 None/BOS; content nodes 1..64",
            "B": "B[source,target] from build_none_separated_B(A[target,source])",
            "model_to_physical": "sigma_phys = inv_perm_model_to_phys[sigma_model]",
            "tau_refs": "physical L2R [0..63] and reversed physical L2R [63..0]",
            "probe_order": "torch.randperm(N) with manual_seed(seed + sample_index)",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for idx, job in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] {job.group} step={job.step} {job.path}", flush=True)
        model, chunks, clean_perm, dev, chunk_index = _load_model_and_chunks(
            str(job.path), total, int(args.seed), args.device, args.split
        )
        B_model, B_phys, inv_perm = extract_l0_strict65_graphs(
            model, chunks, clean_perm, dev, int(args.seed), int(args.fwd_batch),
            layer=int(args.layer),
        )
        Bm_model = _batch_mean(B_model, int(args.M), int(args.bs_mean))
        Bm_phys = _batch_mean(B_phys, int(args.M), int(args.bs_mean))

        ckpt_rows = []
        ckpt_consensus = []
        for frame, Bm in [("model", Bm_model), ("physical", Bm_phys)]:
            for b_label, B_use in [("B", Bm), ("BT", np.stack([
                np.stack([_transpose_strict65(Bm[m, h]) for h in range(Bm.shape[1])])
                for m in range(Bm.shape[0])
            ]))]:
                rows, cons = analyze_frame(
                    B_use, frame, inv_perm, job.group, job.step, b_label,
                    int(args.n_destroy_replicas),
                )
                ckpt_rows.extend(rows)
                ckpt_consensus.append(cons)

        all_head_rows.extend(ckpt_rows)
        all_consensus_rows.extend(ckpt_consensus)
        detail = {
            "group": job.group,
            "step": job.step,
            "ckpt": str(job.path),
            "chunk_index": np.asarray(chunk_index).astype(int).tolist(),
            "inv_perm_model_to_phys": inv_perm.astype(int).tolist(),
            "per_head": ckpt_rows,
            "consensus": ckpt_consensus,
        }
        detail_path = out_dir / f"{job.group}_step{job.step}_detail.json"
        detail_path.write_text(json.dumps(detail, indent=2))
        print(f"  wrote {detail_path}", flush=True)

        del model, chunks, B_model, B_phys, Bm_model, Bm_phys
        gc.collect()
        if torch.cuda.is_available() and str(dev).startswith("cuda"):
            torch.cuda.empty_cache()

    head_fields = [
        "ckpt_group", "step", "head", "tau_l2r", "tau_rev_l2r",
        "abs_tau_gt_0.5", "quality", "quality_std", "frame", "B_or_BT",
        "near_diag_mass", "row_entropy", "top1_mass", "margin",
    ]
    consensus_fields = [
        "ckpt_group", "step", "frame", "B_or_BT",
        "consensus_tau_l2r", "consensus_tau_rev_l2r",
        "uniform_consensus_tau_l2r", "uniform_consensus_tau_rev_l2r",
        "mean_quality", "std_quality", "r_quality_abs_tau",
        "mean_near_diag_mass", "mean_row_entropy", "mean_top1_mass",
    ]
    write_csv(out_dir / "per_head_audit.csv", all_head_rows, head_fields)
    write_csv(out_dir / "consensus_audit.csv", all_consensus_rows, consensus_fields)
    print(f"[done] wrote {out_dir / 'per_head_audit.csv'}", flush=True)
    print(f"[done] wrote {out_dir / 'consensus_audit.csv'}", flush=True)


if __name__ == "__main__":
    main()
