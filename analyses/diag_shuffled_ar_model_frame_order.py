#!/usr/bin/env python3
"""Model-frame order diagnostic for the shuffled-AR control.

This complements the older shuffled-L2R diagnostic, which only checked whether
the recovered order matched physical L2R. Here we test the stronger mechanism
contrast:

  - identity model-frame order: 0,1,2,...,63
  - semantic-neighbor model path: physical L2R mapped to model frame
    using local convention block_perm[physical] = model
  - posthoc physical L2R after inv_perm[model_order]
  - posthoc imposed shuffled physical order inv_perm[identity]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
from scipy.stats import kendalltau

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(ROOT / "scripts"))

from none_separated_block_graph import build_none_separated_B, rollout_by_method  # noqa: E402
from search_strict_label_free_65 import METHODS, _extract_A_with_none_lh_model_frame  # noqa: E402


def _tau(a: np.ndarray, b: np.ndarray) -> float:
    val, _ = kendalltau(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
    return float(val) if not np.isnan(val) else float("nan")


def _block_perm_from_inv(inv_perm: np.ndarray) -> np.ndarray:
    inv = np.asarray(inv_perm, dtype=np.int64)
    block = np.empty_like(inv)
    block[inv] = np.arange(inv.shape[0], dtype=np.int64)
    return block


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=str(ROOT / "block_lo_arm_order_network/probe_results/shuffled_l2r_continuous_jun05/ckpt_step50000.pt"))
    p.add_argument("--out-dir", default=str(ROOT / "reports/shuffled_ar_model_frame_diag"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--M", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--fwd-batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perm-orientation", choices=["phys_to_model", "model_to_phys"], default="model_to_phys")
    p.add_argument("--methods", nargs="*", default=["C-D+L", "L", "C-D", "C+L", "C"])
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    A_lh, inv_perm, meta = _extract_A_with_none_lh_model_frame(args)
    inv_perm = np.asarray(inv_perm, dtype=np.int64)  # local convention: model -> physical
    block_perm = _block_perm_from_inv(inv_perm)      # local convention: physical -> model
    identity = np.arange(inv_perm.shape[0], dtype=np.int64)

    np.save(out_dir / "A_with_none_lh_mean_MODEL_FRAME.npy", A_lh)
    np.save(out_dir / "inv_perm_model_to_phys.npy", inv_perm)
    np.save(out_dir / "block_perm_phys_to_model.npy", block_perm)

    rows = []
    L, H = A_lh.shape[:2]
    for layer in range(L):
        for head in range(H):
            B65_model = build_none_separated_B(A_lh[layer, head])
            for method in args.methods:
                sigma_model = rollout_by_method(B65_model, method)
                sigma_phys = inv_perm[sigma_model]
                rows.append({
                    "layer": layer,
                    "head": head,
                    "method": method,
                    "tau_model_vs_identity": _tau(sigma_model, identity),
                    "tau_model_vs_semantic_model_path": _tau(sigma_model, block_perm),
                    "tau_phys_vs_l2r": _tau(sigma_phys, identity),
                    "tau_phys_vs_imposed_shuffled_order": _tau(sigma_phys, inv_perm),
                    "sigma_model_first16": " ".join(str(int(x)) for x in sigma_model[:16]),
                    "sigma_phys_first16": " ".join(str(int(x)) for x in sigma_phys[:16]),
                })

    tsv_path = out_dir / "shuffled_ar_model_frame_order.tsv"
    with tsv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "meta": meta,
        "local_convention": {
            "block_perm": "block_perm[physical_block] = model_block",
            "inv_perm": "inv_perm[model_block] = physical_block",
            "identity_model_order": "0,1,2,...,63",
            "semantic_model_path": "block_perm[np.arange(N)]",
            "imposed_shuffled_physical_order": "inv_perm[np.arange(N)]",
        },
        "best_by_identity": max(rows, key=lambda r: r["tau_model_vs_identity"]),
        "best_by_semantic_model_path": max(rows, key=lambda r: r["tau_model_vs_semantic_model_path"]),
        "best_by_physical_l2r": max(rows, key=lambda r: r["tau_phys_vs_l2r"]),
        "best_by_imposed_shuffled_order": max(rows, key=lambda r: r["tau_phys_vs_imposed_shuffled_order"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("=== shuffled-AR model-frame diagnostic ===")
    for key in [
        "best_by_identity",
        "best_by_semantic_model_path",
        "best_by_physical_l2r",
        "best_by_imposed_shuffled_order",
    ]:
        row = summary[key]
        print(
            f"{key}: L{row['layer']}H{row['head']} {row['method']} "
            f"tau_id={row['tau_model_vs_identity']:.4f} "
            f"tau_sem_model={row['tau_model_vs_semantic_model_path']:.4f} "
            f"tau_phys_l2r={row['tau_phys_vs_l2r']:.4f} "
            f"tau_phys_shuf={row['tau_phys_vs_imposed_shuffled_order']:.4f}"
        )
    print(f"Saved: {tsv_path}")
    print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
