#!/usr/bin/env python3
"""Strict label-free 65-node None-separated head/method search.

Key difference from ``search_none_separated_65_heads.py``:
  - Extraction uses model-frame coordinates (NO inv_perm during graph construction).
  - inv_perm is applied ONLY posthoc to translate sigma_model → sigma_phys for scoring.
  - This is the true "label-free discovery" protocol.

Protocol:
  1. Extract A_with_none in model-frame: node 0=None, node 1+i=model_block_i.
  2. Build B65 from model-frame A.
  3. CDL rollout → sigma_nodes (model-frame).
  4. Posthoc: sigma_phys[t] = inv_perm[sigma_nodes[t] - 1].
  5. Score tau(sigma_phys, [0..N-1]).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(ROOT / "scripts"))

from clean_training_protocol import expand_model_blocks_to_token_order  # noqa: E402
from none_separated_block_graph import (  # noqa: E402
    build_none_separated_B,
    classify_gate_status,
    combined_discovery_score,
    content_label_permutation_control,
    discovery_metrics,
    entry_shuffled_control,
    rollout_by_method,
)
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_model_vec  # noqa: E402
from scan_collaborator_ckpt_b1 import (  # noqa: E402
    _clean_perm_from_ckpt,
    _load_model,
    _physical_chunks_to_model,
)
from training_utils import BLOCK_LEN, N, SEQ_LEN, load_train_chunks  # noqa: E402


METHODS = ("C-D+L", "L", "C-D", "C+L", "C", "none_edge", "content_out_degree", "content_in_degree_low")


def _posthoc_translate_sigma(sigma_nodes: np.ndarray, inv_perm: np.ndarray) -> np.ndarray:
    """Translate model-frame sigma_nodes to physical-frame content_order.

    sigma_nodes: model-frame node indices from CDL rollout.
                 Node 0 = None, Node 1+i = model block i.
    Returns: content_order in physical block coordinates.
    """
    # sigma_nodes already has None (node 0) removed in rollout_from_none
    # it returns (np.asarray(node_order, dtype=np.int64) - 1) which gives
    # 0-indexed model block indices. Wait - let me check rollout_from_none.
    # rollout_from_none returns node_order - 1, so it's already 0-indexed model block indices.
    # Actually no: the nodes in B65 are 0=None, 1+i=model_block_i.
    # rollout_from_none selects nodes from unselected = list(range(1, N+1)), so it produces
    # node indices in {1..N}. Then it returns (node_order - 1), giving {0..N-1}.
    # These are model block indices because node 1+i = model block i.
    sigma_model_blocks = np.asarray(sigma_nodes, dtype=np.int64)  # model block indices 0..N-1
    inv = np.asarray(inv_perm, dtype=np.int64)
    return inv[sigma_model_blocks]  # physical block indices


def _control_abs_tau_mean(B65: np.ndarray, method: str, inv_perm: np.ndarray, seeds: list[int]) -> float:
    vals = []
    for seed in seeds:
        for Bc in (
            entry_shuffled_control(B65, seed=seed),
            content_label_permutation_control(B65, seed=seed),
        ):
            sigma_model = rollout_by_method(Bc, method)
            sigma_phys = _posthoc_translate_sigma(sigma_model, inv_perm)
            vals.append(abs(discovery_metrics(sigma_phys)["tau_vs_l2r"]))
    return float(np.mean(vals)) if vals else float("nan")


def _extract_A_with_none_lh_model_frame(args) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract (L,H,N,N+1) A_with_none in MODEL-frame coordinates.

    Returns:
        A_lh: (L, H, N, N+1) model-frame A_with_none.
        inv_perm: (N,) inv_perm_model_to_phys for posthoc translation.
        meta: dict.
    """
    ckpt, model, dev = _load_model(args.ckpt, args.device)
    clean_perm = _clean_perm_from_ckpt(ckpt, args.perm_orientation)
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()

    total = int(args.M) * int(args.batch_size)
    chunks_phys = load_train_chunks(n_chunks=total)
    chunks_model = _physical_chunks_to_model(chunks_phys, clean_perm)

    token_orders = torch.empty((total, SEQ_LEN), dtype=torch.long)
    for i in range(total):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(args.seed) + int(i))
        rand_blocks = torch.randperm(N, generator=gen, device="cpu")
        token_orders[i] = expand_model_blocks_to_token_order(rand_blocks.unsqueeze(0), BLOCK_LEN)[0]

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
            attn_batch = torch.stack(attn_list).cpu().numpy()  # (L,B,H,T+1,T+1)
            for bi in range(stop - start):
                sample = attn_batch[:, bi]  # (L,H,T+1,T+1)
                # ── MODEL-FRAME extraction: NO inv_perm ──
                A_lh_sample = _attn_to_A_block_loss_aligned_with_none_model_vec(
                    sample, token_orders[start + bi].numpy(),
                )  # (L,H,N,N+1) in model-frame
                if sums is None:
                    sums = np.zeros_like(A_lh_sample, dtype=np.float64)
                sums += A_lh_sample
                count += 1
            print(f"  [extract model-frame A65] {stop}/{total}", flush=True)
    if sums is None or count == 0:
        raise RuntimeError("no attention maps extracted")
    A_mean = (sums / count).astype(np.float32)
    meta = {
        "ckpt": args.ckpt,
        "protocol": "strict_label_free_65",
        "extraction": "model-frame loss-aligned AR, [None] separate, NO inv_perm during construction",
        "posthoc": "inv_perm applied only to translate sigma_model → sigma_phys for scoring",
        "M": args.M,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "fwd_batch": args.fwd_batch,
        "perm_orientation": args.perm_orientation,
        "shape": list(A_mean.shape),
        "iter_num": ckpt.get("iter_num"),
        "best_val_loss": ckpt.get("best_val_loss"),
    }
    return A_mean, inv_perm, meta


def _write_markdown(out_dir: pathlib.Path, rows: list[dict], meta: dict) -> None:
    strong = [r for r in rows if r["gate_status"] == "strong_pass"]
    weak = [r for r in rows if r["gate_status"] == "weak_pass"]
    best = max(rows, key=lambda r: r["combined_score"]) if rows else None

    def fmt_row(r):
        return (
            f"L{r['layer']}H{r['head']} {r['method']}: first={r['first_block']}, "
            f"phys0_rank={r['phys0_rank']}, tau={r['tau_vs_l2r']:.3f}, "
            f"p4={r['prefix4_overlap']}, destroyed|tau|={r['destroyed_abs_tau_mean']:.3f}, "
            f"gate={r['gate_status']}, score={r['combined_score']:.3f}"
        )

    readme = [
        "# Strict Label-Free 65-node None-separated head/method search",
        "",
        "Protocol: node0=None/BOS, node 1+i = MODEL block i (NOT physical).",
        "inv_perm is NEVER used during extraction or CDL rollout.",
        "It is applied ONLY posthoc to translate sigma_model → sigma_phys for scoring.",
        "",
        f"Config: `{json.dumps(meta, sort_keys=True)}`",
        "",
        f"Rows: {len(rows)}",
        f"Strong pass: {len(strong)}",
        f"Weak pass: {len(weak)}",
        "",
        "Compare with oracle-remapped version in `none_separated_65_head_method_search/`.",
    ]
    (out_dir / "00_README.md").write_text("\n".join(readme) + "\n")

    top = sorted(rows, key=lambda r: r["combined_score"], reverse=True)[:12]
    lines = ["# Top candidates (strict label-free)", ""]
    if best:
        lines += ["Best combined candidate:", "", f"- {fmt_row(best)}", ""]
    lines += ["## By combined score", ""]
    lines += [f"- {fmt_row(r)}" for r in top]
    lines += ["", "## By tau_vs_l2r", ""]
    lines += [f"- {fmt_row(r)}" for r in sorted(rows, key=lambda r: r["tau_vs_l2r"], reverse=True)[:12]]
    lines += ["", "## By phys0 rank", ""]
    lines += [f"- {fmt_row(r)}" for r in sorted(rows, key=lambda r: (r["phys0_rank"], -r["tau_vs_l2r"]))[:12]]
    (out_dir / "top_candidates.md").write_text("\n".join(lines) + "\n")

    claim = ["# Claim impact (strict label-free)", ""]
    if strong:
        claim += [
            "Strict label-free 65-node discovery has at least one strong-pass head/method.",
            "This means: NO inv_perm during extraction or CDL rollout; inv_perm only for posthoc sigma→phys translation.",
        ]
    elif weak:
        claim += [
            "No strong-pass head/method under strict label-free protocol.",
            "Weak exploratory evidence only. Oracle-remapped (inv_perm during construction) results are separate.",
        ]
    else:
        claim += [
            "No strong-pass or weak-pass head/method under strict label-free protocol.",
            "Current framing: anchored controller evidence remains, but unanchored block-level discovery is not closed under strict protocol.",
        ]
    (out_dir / "claim_impact.md").write_text("\n".join(claim) + "\n")

    boss = [
        "# Boss update summary (strict label-free)",
        "",
        "We ran the 65-node None-separated protocol under STRICT label-free conditions:",
        "- Extraction: model-frame coordinates, NO inv_perm during graph construction.",
        "- inv_perm is used ONLY posthoc to translate sigma_model → sigma_phys.",
        "",
        f"Strong pass heads: {len(strong)}",
        f"Weak pass heads: {len(weak)}",
    ]
    if best:
        boss.append(f"Best candidate: {fmt_row(best)}")
    (out_dir / "boss_update_summary.md").write_text("\n".join(boss) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="/home/admin/ych/nanogpt-learned-order/ckpt/ckpt.pt")
    p.add_argument("--out-dir", default="reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--M", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--fwd-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perm-orientation", choices=["phys_to_model", "model_to_phys"], default="model_to_phys")
    p.add_argument("--methods", nargs="*", default=list(METHODS))
    p.add_argument("--control-seeds", type=int, nargs="*", default=list(range(20)))
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    A_lh, inv_perm, meta = _extract_A_with_none_lh_model_frame(args)
    np.save(out_dir / "A_with_none_lh_mean_MODEL_FRAME.npy", A_lh)
    np.save(out_dir / "inv_perm.npy", inv_perm)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    rows = []
    L, H = A_lh.shape[:2]
    for layer in range(L):
        for head in range(H):
            B65_model = build_none_separated_B(A_lh[layer, head])  # model-frame B65
            for method in args.methods:
                sigma_model = rollout_by_method(B65_model, method)  # model-frame sigma
                sigma_phys = _posthoc_translate_sigma(sigma_model, inv_perm)  # ← posthoc translate
                metrics = discovery_metrics(sigma_phys)  # score against physical L2R
                destroyed = _control_abs_tau_mean(B65_model, method, inv_perm, args.control_seeds)
                gate = classify_gate_status(metrics, destroyed)
                score = combined_discovery_score(metrics, destroyed)
                rows.append({
                    "layer": layer,
                    "head": head,
                    "method": method,
                    **{k: metrics[k] for k in (
                        "first_block", "first_is_phys0", "phys0_rank", "tau_vs_l2r",
                        "prefix4_overlap", "prefix8_overlap",
                    )},
                    "abs_tau": abs(float(metrics["tau_vs_l2r"])),
                    "destroyed_abs_tau_mean": destroyed,
                    "destroyed_tau_gap": float(metrics["tau_vs_l2r"]) - destroyed,
                    "combined_score": score,
                    "gate_status": gate,
                    "order_first16": " ".join(str(x) for x in metrics["order_first16"]),
                })

    tsv_path = out_dir / "all_head_methods_strict_label_free_65.tsv"
    fieldnames = [
        "layer", "head", "method", "first_block", "first_is_phys0", "phys0_rank",
        "tau_vs_l2r", "abs_tau", "prefix4_overlap", "prefix8_overlap",
        "destroyed_abs_tau_mean", "destroyed_tau_gap", "combined_score",
        "gate_status", "order_first16",
    ]
    with tsv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "all_head_methods_strict_label_free_65.json").write_text(json.dumps(rows, indent=2))
    _write_markdown(out_dir, rows, meta)

    strong = [r for r in rows if r["gate_status"] == "strong_pass"]
    weak = [r for r in rows if r["gate_status"] == "weak_pass"]
    best = max(rows, key=lambda r: r["combined_score"]) if rows else None
    discovery = "pass" if strong else ("weak" if weak else "fail")

    print("=== Strict Label-Free 65-node Head/Method Search ===")
    print(f"Protocol: model-frame extraction, inv_perm ONLY posthoc for sigma→phys translation")
    print("Strong pass heads:")
    for r in strong:
        print(f"  L{r['layer']}H{r['head']} {r['method']} tau={r['tau_vs_l2r']:.6f} first={r['first_block']} p4={r['prefix4_overlap']}")
    print("Weak pass heads:")
    for r in weak:
        print(f"  L{r['layer']}H{r['head']} {r['method']} tau={r['tau_vs_l2r']:.6f} phys0_rank={r['phys0_rank']}")
    if best:
        print(
            f"Best candidate: "
            f"L{best['layer']}H{best['head']} {best['method']} first={best['first_block']} "
            f"phys0_rank={best['phys0_rank']} tau={best['tau_vs_l2r']:.6f} "
            f"p4={best['prefix4_overlap']} gate={best['gate_status']}"
        )
    print(f"Strict label-free discovery: {discovery}")
    print(f"Saved: {tsv_path}")


if __name__ == "__main__":
    main()
