"""Canonical strict 65-node None-separated re-analysis on runs/handoff_overnight.

Replaces the model-frame + identity-reveal readout the 3/A/5 arc used (tautological
at init, wrong carrier L1) with the project's sealed strict 65-node protocol:
random reveal + posthoc-inv physical scoring + method C-D+L + destroyed controls.

See docs/superpowers/specs/2026-06-28-canonical-reanalysis-design.md and
reports/strict_65node_discovery_ckpt_verification_20260617/.
"""
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch

_BLOCK_DIR = pathlib.Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"
if str(_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(_BLOCK_DIR))

from neural_readout.extract_b import _load_model_and_chunks  # noqa: E402
from clean_training_protocol import expand_model_blocks_to_token_order  # noqa: E402
from training_utils import N, SEQ_LEN, BLOCK_LEN  # noqa: E402
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_vec  # noqa: E402
from none_separated_block_graph import (  # noqa: E402
    build_none_separated_B, rollout_by_method, discovery_metrics,
    classify_gate_status, combined_discovery_score,
    entry_shuffled_control, content_label_permutation_control)


# ── Task 1: canonical per-ckpt 65-node scan ──────────────────────────────────

def random_reveal_orders(n, seed):
    """Per-sample random reveal orders, seeded seed+i (canonical convention)."""
    orders = np.empty((n, SEQ_LEN), dtype=np.int64)
    for i in range(n):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) + int(i))
        rb = torch.randperm(N, generator=g)
        orders[i] = expand_model_blocks_to_token_order(rb.unsqueeze(0), BLOCK_LEN)[0].numpy()
    return orders


def _destroyed_floor(B65, method, control_seeds):
    vals = []
    for s in control_seeds:
        for Bc in (entry_shuffled_control(B65, seed=s),
                   content_label_permutation_control(B65, seed=s)):
            vals.append(abs(discovery_metrics(rollout_by_method(Bc, method))["tau_vs_l2r"]))
    return float(np.mean(vals)) if vals else 0.0


@torch.no_grad()
def canonical_scan(ckpt_path, M=8, batch_size=8, sampling_seed=0,
                   methods=("C-D+L", "L", "none_edge"), control_seeds=(0, 1, 2, 3, 4),
                   device="cpu", ablate=None):
    """Strict 65-node None-separated scan -> rows per (layer,head,method).

    ablate=(layer, heads) optionally mean-ablates those heads' attn.c_proj output
    during the forward (for C2).
    """
    total = M * batch_size
    model, chunks, clean_perm, dev, _ = _load_model_and_chunks(
        ckpt_path, total, seed=sampling_seed, device=device, split="train")
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    orders = random_reveal_orders(total, sampling_seed)

    handle = None
    if ablate is not None:
        from analyses.path_patch_handoff import mean_ablation_prehook
        a_layer, a_heads = ablate
        n_head = model.transformer.h[a_layer].attn.n_head
        handle = model.transformer.h[a_layer].attn.c_proj.register_forward_pre_hook(
            mean_ablation_prehook(a_heads, n_head))

    try:
        A_acc = None
        for start in range(0, total, batch_size):
            bs = min(batch_size, total - start)
            tok = chunks[start:start + bs].to(dev)
            po = torch.from_numpy(orders[start:start + bs]).to(dev)
            _, _, attn_list = model.forward_fn(tok, po, return_attentions=True)
            attn = torch.stack(attn_list, 0).cpu().numpy()       # (L,bs,H,257,257)
            for bi in range(bs):
                A = _attn_to_A_block_loss_aligned_with_none_vec(
                    attn[:, bi], orders[start + bi], inv_perm)    # (L,H,64,65)
                A_acc = A.astype(np.float64) if A_acc is None else A_acc + A
    finally:
        if handle is not None:
            handle.remove()

    A_lh = A_acc / total
    rows = []
    L, H = A_lh.shape[:2]
    for layer in range(L):
        for head in range(H):
            B65 = build_none_separated_B(A_lh[layer, head])
            for method in methods:
                m = discovery_metrics(rollout_by_method(B65, method))
                dz = _destroyed_floor(B65, method, control_seeds)
                rows.append({"layer": layer, "head": head, "method": method,
                             "tau_vs_l2r": float(m["tau_vs_l2r"]),
                             "abs_tau": abs(float(m["tau_vs_l2r"])),
                             "first_block": int(m["first_block"]),
                             "phys0_rank": int(m["phys0_rank"]),
                             "prefix4_overlap": int(m["prefix4_overlap"]),
                             "prefix8_overlap": int(m["prefix8_overlap"]),
                             "destroyed_abs_tau_mean": dz,
                             "gate_status": classify_gate_status(m, dz),
                             "combined_score": combined_discovery_score(m, dz)})
    return rows


# ── Task 2: multi-sampling-seed aggregation + carrier ────────────────────────

def scan_aggregated(ckpt_path, K=3, primary="C-D+L", **kw):
    """Run canonical_scan for sampling_seed in range(K); aggregate per
    (layer,head,method) tau mean/std + strong_frac; pick the primary-method carrier."""
    by_key = defaultdict(list)
    gate_by_key = defaultdict(list)
    floors = []
    for s in range(K):
        for r in canonical_scan(ckpt_path, sampling_seed=s, **kw):
            k = (r["layer"], r["head"], r["method"])
            by_key[k].append(r["tau_vs_l2r"])
            gate_by_key[k].append(r["gate_status"])
            floors.append(r["destroyed_abs_tau_mean"])
    per = {}
    for k, taus in by_key.items():
        per[k] = {"tau_mean": float(np.mean(taus)), "tau_std": float(np.std(taus)),
                  "strong_frac": float(np.mean([g == "strong_pass" for g in gate_by_key[k]]))}
    prim = {k: v for k, v in per.items() if k[2] == primary}
    best_k = max(prim, key=lambda k: abs(prim[k]["tau_mean"]))
    strong = sorted({(l, h) for (l, h, m), v in prim.items() if v["strong_frac"] >= 0.5})
    return {"per": per, "best_head": (best_k[0], best_k[1]), "best_method": primary,
            "best_tau": abs(prim[best_k]["tau_mean"]), "strong_pass_heads": strong,
            "destroyed_floor_mean": float(np.mean(floors))}


# ── Task 3: per-seed ckpt sweep (C1 emergence baseline) ──────────────────────

import csv as _csv  # noqa: E402
import json as _json  # noqa: E402

DEFAULT_STEPS = tuple(range(0, 10001, 1000))


def sweep_seed(seed, root="runs/handoff_overnight", steps=DEFAULT_STEPS, K=3, out_dir=None):
    out = pathlib.Path(out_dir or f"runs/canonical_reanalysis/seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    by_step = {}
    for st in steps:
        agg = scan_aggregated(f"{root}/seed{seed}/ckpt_step{st}.pt", K=K)
        by_step[str(st)] = {"best_head": list(agg["best_head"]), "best_method": agg["best_method"],
                            "best_tau": agg["best_tau"], "n_strong": len(agg["strong_pass_heads"]),
                            "strong_pass_heads": [list(h) for h in agg["strong_pass_heads"]],
                            "destroyed_floor": agg["destroyed_floor_mean"]}
    with open(out / "strict65_sweep.tsv", "w", newline="") as f:
        w = _csv.writer(f, delimiter="\t")
        w.writerow(["step", "n_strong", "best_head", "best_method", "best_tau", "destroyed_floor"])
        for st in steps:
            b = by_step[str(st)]
            w.writerow([st, b["n_strong"], f"L{b['best_head'][0]}H{b['best_head'][1]}",
                        b["best_method"], round(b["best_tau"], 4), round(b["destroyed_floor"], 4)])
    summary = {"seed": seed, "by_step": by_step}
    _json.dump(summary, open(out / "sweep.json", "w"), indent=2, default=float)
    return summary
