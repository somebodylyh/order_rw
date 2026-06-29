#!/usr/bin/env python3
"""Post-hoc characterization of a trained L0DynamicGBeta checkpoint.

THIS SCRIPT IS EVALUATION-ONLY.  It reads physical-coordinate metadata
(``inv_perm_model_to_phys``, ``block_perm_phys_to_model``) from the
source AOGPT checkpoint ONLY for post-hoc interpretation.  These fields
are NEVER used for training, checkpoint selection, or dataset building.

Output: characterization.json with Kendall-tau comparisons, prefix
overlap, and destroyed-input diagnostics.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional

import numpy as np
import torch
from scipy.stats import kendalltau

_ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = _ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))

from batch_readout.l0_dynamic_gbeta import L0DynamicGBeta  # noqa: E402
from batch_readout.train_l0_dynamic_gbeta import (  # noqa: E402
    load_pretrain_dataset,
)
from batch_readout.label_free_cdl_teacher import (  # noqa: E402
    destroy_strict65,
    order_to_rank,
)


# ---------------------------------------------------------------------------
# Kendall tau helper (rank-based)
# ---------------------------------------------------------------------------

def _tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau between two order vectors (rank-based, safe)."""
    ra = order_to_rank(np.asarray(a, dtype=np.int64))
    rb = order_to_rank(np.asarray(b, dtype=np.int64))
    val, _ = kendalltau(ra, rb)
    return float(val) if not np.isnan(val) else float("nan")


# ---------------------------------------------------------------------------
# Teacher consensus hard order
# ---------------------------------------------------------------------------

def _teacher_consensus_order(
    teacher_ranks: np.ndarray,
    teacher_weights: np.ndarray,
) -> np.ndarray:
    """Derive a hard consensus order from per-head ranks and weights.

    mean_rank[block] = sum_h w_h * rank_h[block]
    consensus_order = argsort(mean_rank)
    """
    mean_rank = (teacher_ranks.astype(np.float64) *
                 teacher_weights[:, None]).sum(axis=0)
    return np.argsort(mean_rank).astype(np.int64)


# ---------------------------------------------------------------------------
# Prefix overlap
# ---------------------------------------------------------------------------

def _prefix_overlap(order: np.ndarray, ref: np.ndarray, k: int) -> int:
    return int(np.intersect1d(order[:k], ref[:k]).size)


# ---------------------------------------------------------------------------
# Main characterization
# ---------------------------------------------------------------------------

@torch.no_grad()
def characterize(
    dataset_path: str,
    ckpt_path: str,
    source_ckpt_path: str,
    device: str = "cuda:0",
    destroy_seed: int = 0,
    out_dir: Optional[str] = None,
) -> dict:
    """Run post-hoc physical-space characterization.

    Args:
        dataset_path: path to Task-3 .npz.
        ckpt_path: path to g_beta_best.pt.
        source_ckpt_path: path to the ORIGINAL AOGPT checkpoint (for
            clean_protocol / physical-coordinate metadata).
        device: torch device.
        destroy_seed: seed for destroy RNG.
        out_dir: output directory (default: alongside ckpt_path).

    Returns:
        dict of all characterization results.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")

    # ── Load dataset ──
    data = load_pretrain_dataset(
        dataset_path,
        device=str(dev),
        require_consensus_order=True,
    )
    B_raw = data["B_raw"]
    teacher_consensus_order = data["teacher_consensus_order"].cpu().numpy()
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]

    H = B_raw.shape[1]

    # ── Load model ──
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = L0DynamicGBeta(
        heads=H, nodes=65,
        scorer_hidden=tuple(cfg.get("scorer_hidden", (256, 64))),
        gate_hidden=cfg.get("gate_hidden", 32),
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Load physical metadata from SOURCE AOGPT checkpoint ──
    # ══════════════════════════════════════════════════════════════════════
    # POST-HOC ONLY: these fields are read EXCLUSIVELY in this script for
    # interpretation.  They are never used for training, checkpoint
    # selection, or dataset building.
    # ══════════════════════════════════════════════════════════════════════
    src = torch.load(source_ckpt_path, map_location="cpu", weights_only=False)
    protocol = src.get("clean_protocol", None)
    if protocol is None:
        raise RuntimeError(
            "Source checkpoint has no 'clean_protocol' — cannot extract "
            "physical-coordinate metadata for post-hoc characterization."
        )
    inv_perm = np.asarray(protocol["inv_perm_model_to_phys"], dtype=np.int64)
    block_perm = np.asarray(protocol["block_perm_phys_to_model"], dtype=np.int64)
    # block_perm[p] = model block m that corresponds to physical block p.
    # The model-frame path that yields physical L2R is block_perm itself.
    layout_path = block_perm.copy()                 # model blocks in L2R order
    identity_model = np.arange(64, dtype=np.int64)   # [0, 1, ..., 63]

    # ── Output directory ──
    if out_dir is None:
        out_dir = str(pathlib.Path(ckpt_path).parent)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Per-split characterization ──
    results = {}

    for split_name, split_idx in (("val", val_idx), ("test", test_idx)):
        Bs = B_raw[split_idx].to(dev)
        Ns = len(split_idx)

        # --- Normal forward ---
        scores, _ = model(Bs, apply_head_dropout=False)
        scores_np = scores.cpu().numpy()  # (N, 64)

        # --- Destroyed forward ---
        B_np = Bs.cpu().numpy()
        B_destroyed_np = np.zeros_like(B_np)
        for b in range(Ns):
            for h in range(H):
                rng = np.random.default_rng(
                    destroy_seed * 1000 + int(split_idx[b]) * 100 + h
                )
                B_destroyed_np[b, h] = destroy_strict65(B_np[b, h], rng)
        B_d = torch.from_numpy(B_destroyed_np).float().to(dev)
        scores_destroyed, _ = model(B_d, apply_head_dropout=False)
        scores_d_np = scores_destroyed.cpu().numpy()

        # Per-sample
        taus_normal = []
        taus_destroyed = []
        first_blocks = []
        prefix4 = []
        prefix8 = []
        prefix16 = []

        for i in range(Ns):
            pred_model = np.argsort(-scores_np[i]).astype(np.int64)
            pred_destr = np.argsort(-scores_d_np[i]).astype(np.int64)

            teacher_order = teacher_consensus_order[split_idx[i]]

            # Translate to physical space
            pred_phys = inv_perm[pred_model]
            pred_phys_d = inv_perm[pred_destr]

            taus_normal.append({
                "sample": int(split_idx[i]),
                "tau_vs_teacher": _tau(pred_model, teacher_order),
                "tau_vs_layout_path": _tau(pred_model, layout_path),
                "tau_vs_identity": _tau(pred_model, identity_model),
                "tau_vs_physical_l2r": _tau(pred_phys, np.arange(64)),
            })
            taus_destroyed.append({
                "sample": int(split_idx[i]),
                "tau_vs_teacher": _tau(pred_destr, teacher_order),
                "tau_vs_layout_path": _tau(pred_destr, layout_path),
                "tau_vs_identity": _tau(pred_destr, identity_model),
                "tau_vs_physical_l2r": _tau(pred_phys_d, np.arange(64)),
            })

            first_blocks.append({
                "sample": int(split_idx[i]),
                "pred_first_model": int(pred_model[0]),
                "pred_first_physical": int(pred_phys[0]),
                "teacher_first_model": int(teacher_order[0]),
                "teacher_first_physical": int(inv_perm[teacher_order[0]]),
            })
            prefix4.append({
                "sample": int(split_idx[i]),
                "overlap_vs_teacher": _prefix_overlap(pred_model, teacher_order, 4),
                "overlap_vs_layout_path": _prefix_overlap(pred_model, layout_path, 4),
            })
            prefix8.append({
                "sample": int(split_idx[i]),
                "overlap_vs_teacher": _prefix_overlap(pred_model, teacher_order, 8),
                "overlap_vs_layout_path": _prefix_overlap(pred_model, layout_path, 8),
            })
            prefix16.append({
                "sample": int(split_idx[i]),
                "overlap_vs_teacher": _prefix_overlap(
                    pred_model, teacher_order, 16,
                ),
                "overlap_vs_layout_path": _prefix_overlap(
                    pred_model, layout_path, 16,
                ),
            })

        # Aggregate
        def _mean_tau(records, key):
            vals = [r[key] for r in records if not np.isnan(r[key])]
            return float(np.mean(vals)) if vals else float("nan")

        results[split_name] = {
            "n_samples": Ns,
            "normal": {
                "tau_vs_teacher": _mean_tau(taus_normal, "tau_vs_teacher"),
                "tau_vs_layout_path": _mean_tau(taus_normal, "tau_vs_layout_path"),
                "tau_vs_identity": _mean_tau(taus_normal, "tau_vs_identity"),
                "tau_vs_physical_l2r": _mean_tau(taus_normal, "tau_vs_physical_l2r"),
            },
            "destroyed": {
                "tau_vs_teacher": _mean_tau(taus_destroyed, "tau_vs_teacher"),
                "tau_vs_layout_path": _mean_tau(taus_destroyed, "tau_vs_layout_path"),
                "tau_vs_identity": _mean_tau(taus_destroyed, "tau_vs_identity"),
                "tau_vs_physical_l2r": _mean_tau(taus_destroyed, "tau_vs_physical_l2r"),
            },
            "per_sample_taus_normal": taus_normal,
            "per_sample_taus_destroyed": taus_destroyed,
            "first_blocks": first_blocks,
            "prefix4": prefix4,
            "prefix8": prefix8,
            "prefix16": prefix16,
        }

    # ── Summary ──
    physical_meta = {
        "inv_perm_first16": inv_perm[:16].tolist(),
        "block_perm_first16": block_perm[:16].tolist(),
        "NOTE": ("POST-HOC ONLY: physical-coordinate metadata read from "
                 "source AOGPT checkpoint for interpretation.  These values "
                 "are NEVER used for training, checkpoint selection, or "
                 "dataset building."),
    }

    summary = {
        "dataset_path": str(dataset_path),
        "ckpt_path": str(ckpt_path),
        "source_ckpt_path": str(source_ckpt_path),
        "ckpt_epoch": ckpt.get("epoch"),
        "post_hoc_only": True,
        "physical_metadata": physical_meta,
        "results": results,
    }
    with open(out / "characterization.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print key numbers ──
    for split_name in ("val", "test"):
        r = results[split_name]
        print(f"\n{split_name} (N={r['n_samples']}):")
        print(f"  NORMAL:")
        print(f"    τ_vs_teacher:       {r['normal']['tau_vs_teacher']:+.4f}")
        print(f"    τ_vs_layout_path:   {r['normal']['tau_vs_layout_path']:+.4f}")
        print(f"    τ_vs_identity:      {r['normal']['tau_vs_identity']:+.4f}")
        print(f"    τ_vs_physical_l2r:  {r['normal']['tau_vs_physical_l2r']:+.4f}")
        print(f"  DESTROYED:")
        print(f"    τ_vs_teacher:       {r['destroyed']['tau_vs_teacher']:+.4f}")
        print(f"    τ_vs_layout_path:   {r['destroyed']['tau_vs_layout_path']:+.4f}")
        print(f"    τ_vs_identity:      {r['destroyed']['tau_vs_identity']:+.4f}")
        print(f"    τ_vs_physical_l2r:  {r['destroyed']['tau_vs_physical_l2r']:+.4f}")

    print(f"\nSaved: {out / 'characterization.json'}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Post-hoc characterization of L0DynamicGBeta "
                    "(evaluation-only, physical coordinates for interpretation)"
    )
    p.add_argument("--dataset", required=True,
                   help="Path to Task-3 .npz")
    p.add_argument("--ckpt", required=True,
                   help="Path to g_beta_best.pt")
    p.add_argument("--source-ckpt", required=True,
                   help="Path to ORIGINAL AOGPT checkpoint (for clean_protocol)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--destroy-seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    characterize(
        dataset_path=args.dataset,
        ckpt_path=args.ckpt,
        source_ckpt_path=args.source_ckpt,
        device=args.device,
        destroy_seed=args.destroy_seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
