#!/usr/bin/env python3
"""Search heads/methods under the 65-node None-separated discovery protocol."""

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
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_vec  # noqa: E402
from scan_collaborator_ckpt_b1 import (  # noqa: E402
    _clean_perm_from_ckpt,
    _load_model,
    _physical_chunks_to_model,
)
from training_utils import BLOCK_LEN, N, SEQ_LEN, load_train_chunks  # noqa: E402


METHODS = ("C-D+L", "L", "C-D", "C+L", "C", "none_edge", "content_out_degree", "content_in_degree_low")


def _control_abs_tau_mean(B65: np.ndarray, method: str, seeds: list[int]) -> float:
    vals = []
    for seed in seeds:
        for Bc in (
            entry_shuffled_control(B65, seed=seed),
            content_label_permutation_control(B65, seed=seed),
        ):
            vals.append(abs(discovery_metrics(rollout_by_method(Bc, method))["tau_vs_l2r"]))
    return float(np.mean(vals)) if vals else float("nan")


def _extract_A_with_none_lh(args) -> tuple[np.ndarray, dict]:
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
                A_lh = _attn_to_A_block_loss_aligned_with_none_vec(
                    sample, token_orders[start + bi].numpy(), inv_perm
                )  # (L,H,N,N+1)
                if sums is None:
                    sums = np.zeros_like(A_lh, dtype=np.float64)
                sums += A_lh
                count += 1
            print(f"  [extract A65] {stop}/{total}", flush=True)
    if sums is None or count == 0:
        raise RuntimeError("no attention maps extracted")
    A_mean = (sums / count).astype(np.float32)
    meta = {
        "ckpt": args.ckpt,
        "M": args.M,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "fwd_batch": args.fwd_batch,
        "perm_orientation": args.perm_orientation,
        "shape": list(A_mean.shape),
        "iter_num": ckpt.get("iter_num"),
        "best_val_loss": ckpt.get("best_val_loss"),
    }
    return A_mean, meta


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
        "# 65-node None-separated head/method search",
        "",
        "Protocol: node0=None/BOS, node 1+i = physical content block i = x_{4i}..x_{4i+3}.",
        "No training was run. Attention is token-level observation aggregated into a block-level graph.",
        "",
        f"Config: `{json.dumps(meta, sort_keys=True)}`",
        "",
        f"Rows: {len(rows)}",
        f"Strong pass: {len(strong)}",
        f"Weak pass: {len(weak)}",
    ]
    (out_dir / "00_README.md").write_text("\n".join(readme) + "\n")

    top = sorted(rows, key=lambda r: r["combined_score"], reverse=True)[:12]
    lines = ["# Top candidates", ""]
    if best:
        lines += ["Best combined candidate:", "", f"- {fmt_row(best)}", ""]
    lines += ["## By combined score", ""]
    lines += [f"- {fmt_row(r)}" for r in top]
    lines += ["", "## By tau_vs_l2r", ""]
    lines += [f"- {fmt_row(r)}" for r in sorted(rows, key=lambda r: r["tau_vs_l2r"], reverse=True)[:12]]
    lines += ["", "## By phys0 rank", ""]
    lines += [f"- {fmt_row(r)}" for r in sorted(rows, key=lambda r: (r["phys0_rank"], -r["tau_vs_l2r"]))[:12]]
    (out_dir / "top_candidates.md").write_text("\n".join(lines) + "\n")

    claim = ["# Claim impact", ""]
    if strong:
        claim += ["Strict block-level None-start discovery has at least one strong-pass head/method."]
    elif weak:
        claim += [
            "No strong-pass head/method found. There is at least one weak-pass candidate, suitable only as exploratory mechanism evidence.",
        ]
    else:
        claim += [
            "No strong-pass or weak-pass head/method found under this search.",
            "Current framing should be: anchored controller evidence remains, but unanchored block-level discovery is not closed.",
        ]
    claim += [
        "",
        "Anchored controller claim: still separate from this gate and can remain if evaluated under its own anchored protocol.",
        "Content-derived anchor heuristic: needed if the goal remains label-free block-order discovery.",
    ]
    (out_dir / "claim_impact.md").write_text("\n".join(claim) + "\n")

    boss = [
        "# Boss update summary",
        "",
        "We tightened the protocol to a 65-node graph with None as a separate start node.",
        "This search asks whether any head/readout can select physical block0 and roll out L2R without folding None into block0.",
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
    p.add_argument("--out-dir", default="reports/collaborator_ckpt_b1_scan_20260616/none_separated_65_head_method_search")
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
    A_lh, meta = _extract_A_with_none_lh(args)
    np.save(out_dir / "A_with_none_lh_mean.npy", A_lh)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    rows = []
    L, H = A_lh.shape[:2]
    for layer in range(L):
        for head in range(H):
            B65 = build_none_separated_B(A_lh[layer, head])
            for method in args.methods:
                sigma = rollout_by_method(B65, method)
                metrics = discovery_metrics(sigma)
                destroyed = _control_abs_tau_mean(B65, method, args.control_seeds)
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

    tsv_path = out_dir / "all_head_methods_none_separated_65.tsv"
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
    (out_dir / "all_head_methods_none_separated_65.json").write_text(json.dumps(rows, indent=2))
    _write_markdown(out_dir, rows, meta)

    strong = [r for r in rows if r["gate_status"] == "strong_pass"]
    weak = [r for r in rows if r["gate_status"] == "weak_pass"]
    best = max(rows, key=lambda r: r["combined_score"])
    discovery = "pass" if strong else ("weak" if weak else "fail")
    print("=== 65-node None-separated Head/Method Search Complete ===")
    print("Strong pass heads:")
    for r in strong:
        print(f"  L{r['layer']}H{r['head']} {r['method']} tau={r['tau_vs_l2r']:.6f}")
    print("Weak pass heads:")
    for r in weak:
        print(f"  L{r['layer']}H{r['head']} {r['method']} tau={r['tau_vs_l2r']:.6f} phys0_rank={r['phys0_rank']}")
    print(
        "Best candidate: "
        f"L{best['layer']}H{best['head']} {best['method']} first={best['first_block']} "
        f"phys0_rank={best['phys0_rank']} tau={best['tau_vs_l2r']:.6f} "
        f"p4={best['prefix4_overlap']} gate={best['gate_status']}"
    )
    print(f"Current discovery claim: {discovery}")
    print("Recommended Wednesday framing:")
    if discovery == "pass":
        print("  strict 65-node None-start discovery has a passing head/method; present as head-specific evidence.")
    elif discovery == "weak":
        print("  no strong discovery; weak exploratory evidence only, anchored controller remains separate.")
    else:
        print("  anchored controller can be discussed; unanchored block-level discovery is not closed.")
    print(f"Saved: {tsv_path}")


if __name__ == "__main__":
    main()
