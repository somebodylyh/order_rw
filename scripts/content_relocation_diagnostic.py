#!/usr/bin/env python3
"""Content-relocation diagnostic: test whether attention order signal is
content-bound or position-bound.

Trained model uses fixed input arrangement (clean_perm maps phys→model).
We test: if we relocate the SAME physical content blocks to DIFFERENT model
positions, does the CDL-recovered order follow the content (physical L2R
still recovered) or stay at old model positions (signal breaks)?

Protocol:
  1. Extract A_with_none using ORIGINAL clean_perm → baseline τ
  2. Generate NEW random clean_perm (different phys→model mapping)
  3. Extract A_with_none using NEW clean_perm → relocated τ
  4. Compare: if τ survives relocation → content-bound
             if τ collapses → position-bound
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(ROOT / "scripts"))

from clean_training_protocol import (
    CleanPermutation,
    build_clean_block_permutation,
    expand_model_blocks_to_token_order,
)
from none_separated_block_graph import (
    build_none_separated_B,
    classify_gate_status,
    content_label_permutation_control,
    discovery_metrics,
    entry_shuffled_control,
    rollout_by_method,
)
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_model_vec
from scan_collaborator_ckpt_b1 import (
    _clean_perm_from_ckpt,
    _load_model,
)
from training_utils import BLOCK_LEN, N, SEQ_LEN, load_train_chunks

METHODS = ("L", "C-D+L")


def _phys_chunks_to_model_positions(chunks_phys, clean_perm):
    """Place physical tokens at model positions according to clean_perm."""
    # chunks_phys: (n, seq_len) physical-order tokens
    # Need to rearrange to model-order
    n, seq_len = chunks_phys.shape
    chunks_model = torch.empty_like(chunks_phys)
    bp = clean_perm.block_perm_phys_to_model
    for pos in range(seq_len):
        phys_block = pos // BLOCK_LEN
        offset = pos % BLOCK_LEN
        model_block = int(bp[phys_block].item())
        chunks_model[:, model_block * BLOCK_LEN + offset] = chunks_phys[:, pos]
    return chunks_model


def extract_A_lh_for_perm(
    args, clean_perm, tag: str
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract A_with_none in model-frame using the given clean_perm."""
    ckpt, model, dev = _load_model(args.ckpt, args.device)
    inv_perm_np = clean_perm.inv_perm_model_to_phys.cpu().numpy()

    total = int(args.M) * int(args.batch_size)
    chunks_phys = load_train_chunks(n_chunks=total)
    chunks_model = _phys_chunks_to_model_positions(chunks_phys, clean_perm)

    token_orders = torch.empty((total, SEQ_LEN), dtype=torch.long)
    for i in range(total):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(args.seed) + int(i))
        rand_blocks = torch.randperm(N, generator=gen, device="cpu")
        token_orders[i] = expand_model_blocks_to_token_order(
            rand_blocks.unsqueeze(0), BLOCK_LEN
        )[0]

    sums = None
    count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, total, max(1, int(args.fwd_batch))):
            stop = min(start + max(1, int(args.fwd_batch)), total)
            tokens = chunks_model[start:stop].to(dev)
            orders = token_orders[start:stop].to(dev)
            _, _, attn_list = model.forward_fn(tokens, orders, return_attentions=True)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            attn_batch = torch.stack(attn_list).cpu().numpy()
            for bi in range(stop - start):
                sample = attn_batch[:, bi]
                A_lh_sample = _attn_to_A_block_loss_aligned_with_none_model_vec(
                    sample, token_orders[start + bi].numpy(),
                )
                if sums is None:
                    sums = np.zeros_like(A_lh_sample, dtype=np.float64)
                sums += A_lh_sample
                count += 1
            print(f"  [{tag}] {stop}/{total}", flush=True)

    A_mean = (sums / count).astype(np.float32)
    meta = {
        "ckpt": args.ckpt,
        "tag": tag,
        "M": args.M, "batch_size": args.batch_size, "seed": args.seed,
        "block_perm_first16": clean_perm.block_perm_phys_to_model[:16].tolist(),
    }
    return A_mean, inv_perm_np, meta


def scan_heads(A_lh, inv_perm, methods, control_seeds):
    """Run CDL rollout on all heads×methods, return rows."""
    rows = []
    L, H = A_lh.shape[:2]
    for layer in range(L):
        for head in range(H):
            B65 = build_none_separated_B(A_lh[layer, head])
            for method in methods:
                sigma_model = rollout_by_method(B65, method)
                sigma_phys = np.asarray(
                    [inv_perm[s] for s in sigma_model], dtype=np.int64
                )
                metrics = discovery_metrics(sigma_phys)
                # destroyed controls
                destroyed_vals = []
                for seed in control_seeds:
                    for Bc in (
                        entry_shuffled_control(B65, seed=seed),
                        content_label_permutation_control(B65, seed=seed),
                    ):
                        sm = rollout_by_method(Bc, method)
                        sp = np.asarray(
                            [inv_perm[s] for s in sm], dtype=np.int64
                        )
                        destroyed_vals.append(
                            abs(discovery_metrics(sp)["tau_vs_l2r"])
                        )
                destroyed = float(np.mean(destroyed_vals))
                gate = classify_gate_status(metrics, destroyed)
                rows.append({
                    "layer": layer, "head": head, "method": method,
                    "tau": float(metrics["tau_vs_l2r"]),
                    "first": int(metrics["first_block"]),
                    "phys0_rank": int(metrics["phys0_rank"]),
                    "p4": int(metrics["prefix4_overlap"]),
                    "p8": int(metrics["prefix8_overlap"]),
                    "destroyed_tau": destroyed,
                    "gate": gate,
                    "order_first16": " ".join(
                        str(x) for x in metrics["order_first16"]
                    ),
                })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", default="reports/content_relocation_diagnostic")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--M", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--fwd-batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--control-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load original clean_perm from ckpt ──
    ckpt, _, _ = _load_model(args.ckpt, "cpu")
    clean_perm_orig = _clean_perm_from_ckpt(ckpt, "model_to_phys")

    # ── Generate new relocated clean_perm ──
    N_blocks = 64
    new_seed = 99999  # different from any training seed
    clean_perm_new = build_clean_block_permutation(N_blocks, new_seed)

    print(f"Original block_perm[:8]: {clean_perm_orig.block_perm_phys_to_model[:8].tolist()}")
    print(f"New      block_perm[:8]: {clean_perm_new.block_perm_phys_to_model[:8].tolist()}")
    print(f"Overlap in mapping: {sum(1 for i in range(N_blocks) if clean_perm_orig.block_perm_phys_to_model[i] == clean_perm_new.block_perm_phys_to_model[i])}/64")

    # ── Extract & scan with ORIGINAL perm ──
    print("\n=== ORIGINAL clean_perm ===")
    A_orig, inv_orig, meta_orig = extract_A_lh_for_perm(args, clean_perm_orig, "original")
    rows_orig = scan_heads(A_orig, inv_orig, METHODS, args.control_seeds)

    # ── Extract & scan with RELOCATED perm ──
    print("\n=== RELOCATED clean_perm ===")
    A_new, inv_new, meta_new = extract_A_lh_for_perm(args, clean_perm_new, "relocated")
    rows_new = scan_heads(A_new, inv_new, METHODS, args.control_seeds)

    # ── Compare ──
    strong_orig = {(r["layer"], r["head"], r["method"]): r for r in rows_orig if r["gate"] == "strong_pass"}
    weak_orig = {(r["layer"], r["head"], r["method"]): r for r in rows_orig if r["gate"] == "weak_pass"}

    print(f"\n=== RESULTS ===")
    print(f"Original: {len(strong_orig)} strong, {len(weak_orig)} weak")
    for key, r in sorted(strong_orig.items(), key=lambda x: -x[1]["tau"]):
        l, h, m = key
        r_new = [x for x in rows_new if x["layer"] == l and x["head"] == h and x["method"] == m]
        if r_new:
            rn = r_new[0]
            delta = r["tau"] - rn["tau"]
            print(f"  L{l}H{h} {m}: τ_orig={r['tau']:.3f} → τ_reloc={rn['tau']:.3f} (Δ={delta:+.3f})  gate_new={rn['gate']}")
        else:
            print(f"  L{l}H{h} {m}: τ_orig={r['tau']:.3f} → MISSING")

    # ── Save ──
    report = {
        "original": {"strong": len(strong_orig), "weak": len(weak_orig),
                      "top_tau": max(r["tau"] for r in rows_orig)},
        "relocated": {"strong": sum(1 for r in rows_new if r["gate"] == "strong_pass"),
                       "weak": sum(1 for r in rows_new if r["gate"] == "weak_pass"),
                       "top_tau": max(r["tau"] for r in rows_new)},
        "verdict": "content-bound" if sum(1 for r in rows_new if r["gate"] == "strong_pass") > 0 else
                   "mixed" if sum(1 for r in rows_new if r["gate"] == "weak_pass") > 0 else
                   "position-bound",
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "rows_original.json").write_text(json.dumps(rows_orig, indent=2))
    (out_dir / "rows_relocated.json").write_text(json.dumps(rows_new, indent=2))

    print(f"\nVerdict: {report['verdict']}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
