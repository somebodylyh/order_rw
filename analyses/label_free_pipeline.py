#!/usr/bin/env python3
"""Label-free head selection + order clustering + cluster readout training.

Stage A: per-head label-free graph-intrinsic scores (no physical labels).
Stage B: top-k selection + Kendall-based order clustering.
Stage C: cluster-mean readout training (FlattenReadout/NodewiseReadout).

Protocol boundary (inv_perm rule):
    Selection:        model-frame, NO inv_perm  ← label-free
    Clustering:       model-frame, NO inv_perm  ← label-free
    Teacher:          model-frame, NO inv_perm  ← label-free
    Readout training: model-frame, NO inv_perm  ← label-free
    Evaluation:       posthoc inv_perm → τ_vs_physical  ← oracle metric ONLY
"""

from __future__ import annotations

import argparse, json, pathlib, sys
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from scipy.stats import kendalltau

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))

from training_utils import SEQ_LEN, N, BLOCK_LEN
from clean_training_protocol import expand_model_blocks_to_token_order
from neural_readout.extract_b import _load_model_and_chunks
from none_separated_block_graph import build_none_separated_B, rollout_by_method
from per_head_order_scan import (
    _attn_to_A_block_loss_aligned_with_none_model_vec as _A_model_vec,
)


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _random_probe_orders(batch_size: int, seed: int, device: str) -> torch.Tensor:
    rows = []
    for b in range(batch_size):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) * 100_000_000 + b)
        blocks = torch.randperm(N, generator=g)
        rows.append(expand_model_blocks_to_token_order(blocks.unsqueeze(0), BLOCK_LEN)[0])
    return torch.stack(rows).to(device)


# ═══════════════════════════════════════════════════════════════════════════
# Stage A: Label-free head scores
# ═══════════════════════════════════════════════════════════════════════════

def _B65_to_pairwise_pref(B65: np.ndarray) -> np.ndarray:
    """B65 (65,65) → pairwise preference P(i before j) for content blocks 1..64.

    B65[src, tgt] is interpreted as src→tgt attention.  Removing the None
    token (index 0), we treat the (64,64) content-block submatrix.  A simple
    heuristic: P(i before j) = B[i, j] / (B[i, j] + B[j, i] + eps).
    """
    Bc = B65[1:, 1:].copy()  # (64, 64) content blocks
    Bc_sym = Bc + Bc.T + 1e-8
    P = Bc / Bc_sym  # (64, 64), P[i, j] + P[j, i] ≈ 1
    return P


def _pairwise_confidence(B65: np.ndarray) -> float:
    """S_conf = mean_{i,j} |P(i before j) - 0.5|. High → decisive preferences."""
    P = _B65_to_pairwise_pref(B65)
    return float(np.mean(np.abs(P - 0.5)))


def _cycle_rate(B65: np.ndarray) -> float:
    """Fraction of 3-cycles in the tournament induced by argmax(P > 0.5)."""
    P = _B65_to_pairwise_pref(B65)
    pref = (P > 0.5).astype(np.int8)  # (64, 64)
    n = pref.shape[0]
    cycles = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = pref[i, j], pref[j, k], pref[k, i]
                if a == b == c == 1:  # i→j→k→i
                    cycles += 1
                elif a == b == c == 0:  # i←j←k←i (equivalent, reverse direction)
                    cycles += 1
                total += 1
    return cycles / max(total, 1)


def _none_mass(B65: np.ndarray) -> float:
    """Fraction of total attention mass on None/BOS node (row or col)."""
    total = np.abs(B65).sum()
    none_row = np.abs(B65[0, :]).sum()
    none_col = np.abs(B65[:, 0]).sum()
    return float((none_row + none_col - np.abs(B65[0, 0])) / max(total, 1e-8))


def _reveal_robustness(
    model, idx_batch, head_idx, clean_perm, device, n_reveal=10, seed=0
) -> float:
    """S_reveal: mean Kendall tau between rollout orders under different reveal orders.

    For a single (layer, head), run n_reveal forward passes with different
    random probe orders, rollout C-D+L, compute mean pairwise tau.
    """
    layer, head = head_idx
    orders = []
    for ri in range(n_reveal):
        probe = _random_probe_orders(1, seed * 1000 + ri, str(device))
        dev = torch.device(device)
        model.eval()
        with torch.no_grad():
            _, _, attn_list = model.forward_fn(idx_batch, probe.to(dev), return_attentions=True)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            attn = attn_list[layer][0, head].cpu().numpy()  # (257, 257) for batch item 0
            probe_np = probe[0].cpu().numpy()

        # Build model-frame A65, then B65, then rollout
        A65 = _A_model_vec(attn, probe_np)  # (64, 65)
        B65 = build_none_separated_B(A65)
        order = rollout_by_method(B65, "C-D+L")  # model-block indices 0..63
        orders.append(order)

    # Mean pairwise tau across reveal-order pairs
    taus = []
    for i in range(n_reveal):
        for j in range(i + 1, n_reveal):
            tau, _ = kendalltau(orders[i], orders[j])
            taus.append(tau)
    return float(np.mean(taus))


def stage_a_canonical_scan(
    ckpt_path: str,
    M: int = 40,
    batch_size: int = 8,
    seed: int = 0,
    device: str = "cuda:0",
) -> list[dict]:
    """Label-free head selection via canonical 65-node scan with batch-mean.

    Uses batch-mean aggregation (M*batch_size per-sample graphs → M batch-mean B)
    and destroyed-control gate to classify heads as strong/weak/fail.
    Entirely label-free — no physical labels used.
    """
    from canonical_reanalysis import canonical_scan

    rows = canonical_scan(ckpt_path, M=M, batch_size=batch_size,
                          sampling_seed=seed, device=device,
                          methods=["C-D+L"])
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Stage B: Clustering
# ═══════════════════════════════════════════════════════════════════════════

def stage_b_cluster(
    ckpt_path: str,
    top_heads: list[tuple[int, int]],
    M: int = 20,
    batch_size: int = 4,
    seed: int = 0,
    device: str = "cuda:0",
) -> dict:
    """Cluster selected heads by Kendall tau between their CDL rollout orders.

    Returns dict mapping cluster_id → list of (layer, head, order, weight).
    """
    total = M * batch_size
    model, chunks, clean_perm, dev, _ci = _load_model_and_chunks(
        ckpt_path, total, seed, device, "train"
    )
    del clean_perm

    idx_batch = chunks[:1].to(dev)
    probe = _random_probe_orders(1, seed, str(dev))
    model.eval()
    with torch.no_grad():
        _, _, attn_list = model.forward_fn(idx_batch, probe, return_attentions=True)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    probe_np = probe[0].cpu().numpy()

    # Rollout each selected head
    head_orders = {}
    for layer, head in top_heads:
        attn = attn_list[layer][0, head].cpu().numpy()
        A65 = _A_model_vec(attn, probe_np)
        B65 = build_none_separated_B(A65)
        order = rollout_by_method(B65, "C-D+L")
        head_orders[(layer, head)] = order

    # Compute pairwise Kendall similarity
    k = len(top_heads)
    sim = np.zeros((k, k))
    for i, hi in enumerate(top_heads):
        for j, hj in enumerate(top_heads):
            tau, _ = kendalltau(head_orders[hi], head_orders[hj])
            sim[i, j] = tau

    # Simple greedy clustering: group heads with |tau| > 0.5
    threshold = 0.5
    clusters = []
    assigned = set()
    for i, hi in enumerate(top_heads):
        if i in assigned:
            continue
        cluster = [hi]
        assigned.add(i)
        for j, hj in enumerate(top_heads):
            if j in assigned:
                continue
            if abs(sim[i, j]) >= threshold:
                cluster.append(hj)
                assigned.add(j)
        clusters.append(cluster)

    # Map to dict
    result = {}
    for cid, cluster in enumerate(clusters):
        result[cid] = {
            "heads": [(l, h) for (l, h) in cluster],
            "orders": {f"L{l}H{h}": head_orders[(l, h)].tolist() for (l, h) in cluster},
            "size": len(cluster),
        }

    print(f"\n  Clusters ({k} heads, threshold={threshold}):")
    for cid, info in result.items():
        heads_str = ", ".join(f"L{l}H{h}" for (l, h) in info["heads"])
        print(f"    C{cid}: [{heads_str}] (size={info['size']})")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Stage C: Build cluster teacher + train readout
# ═══════════════════════════════════════════════════════════════════════════

def stage_c_build_teacher_and_train(
    ckpt_path: str,
    cluster: list[tuple[int, int]],
    M: int = 500,
    batch_mean_size: int = 8,
    seed: int = 0,
    device: str = "cuda:0",
    out_dir: str = "",
    epochs: int = 40,
    lr: float = 3e-4,
) -> dict:
    """Build cluster-mean B, pairwise teacher, train FlattenReadout."""
    from batch_readout.train_offline import train, _eval_split
    from batch_readout.eval_frozen_phase2 import _load_g_beta
    from batch_readout.eval_metrics import kendall_tau_batch, pairwise_acc
    from batch_readout.integration_hook import _build_from_config
    from batch_readout.model import FlattenReadout
    from none_separated_block_graph import discovery_metrics

    total = M * batch_mean_size
    model, chunks, clean_perm, dev, _ci = _load_model_and_chunks(
        ckpt_path, total, seed, device, "train"
    )
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    del clean_perm

    # ── Extract per-sample B for cluster heads, average across heads ──
    B_single = np.zeros((total, 65, 65), dtype=np.float32)
    for bi in range(total):
        idx_batch = chunks[bi:bi+1].to(dev)
        probe = _random_probe_orders(1, seed * 100 + bi, str(dev))
        model.eval()
        with torch.no_grad():
            _, _, attn_list = model.forward_fn(idx_batch, probe, return_attentions=True)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
        probe_np = probe[0].cpu().numpy()

        # Mean over cluster heads
        B_sum = np.zeros((65, 65), dtype=np.float64)
        for layer, head in cluster:
            attn = attn_list[layer][0, head].cpu().numpy()
            A65 = _A_model_vec(attn, probe_np)
            B65 = build_none_separated_B(A65)
            B_sum += B65
        B_single[bi] = (B_sum / len(cluster)).astype(np.float32)

        if (bi + 1) % max(1, total // 5) == 0:
            print(f"  [extract] {bi+1}/{total} per-sample graphs")

    # ── Batch-mean ──
    B_grouped = B_single.reshape(M, batch_mean_size, 65, 65).mean(axis=1)
    B_mean = B_grouped.astype(np.float32)

    # ── Cluster teacher: rollout each batch-mean, build pairwise teacher ──
    sigma_T = np.zeros((M, 64), dtype=np.int64)
    teacher_pairwise = np.zeros((M, 64, 64), dtype=np.float32)
    teacher_confidence = np.zeros((M, 64, 64), dtype=np.float32)

    for m in range(M):
        order = rollout_by_method(B_mean[m], "C-D+L")
        sigma_T[m] = order
        # Pairwise teacher: P(i before j) = 1 if i before j in order
        for i in range(64):
            for j in range(64):
                if i == j:
                    continue
                teacher_pairwise[m, i, j] = 1.0 if np.where(order == i)[0] < np.where(order == j)[0] else 0.0
                teacher_confidence[m, i, j] = 1.0  # full confidence for consensus order

    # ── Train/val/test split ──
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    train_n = int(round(M * 0.8))
    val_n = int(round(M * 0.1))
    train_idx = perm[:train_n]
    val_idx = perm[train_n:train_n + val_n]
    test_idx = perm[train_n + val_n:]

    # ── Train FlattenReadout (64×64 input, no None node) ──
    B_content = B_mean[:, 1:, 1:]  # (M, 64, 64) — strip None token

    # Simple FlattenReadout
    readout = FlattenReadout(N=64, hidden=(1024, 256))
    readout.train()
    optimizer = torch.optim.AdamW(readout.parameters(), lr=lr, weight_decay=1e-2)

    B_train_t = torch.from_numpy(B_content[train_idx]).float()
    T_pair_t = torch.from_numpy(teacher_pairwise[train_idx]).float()

    best_val_acc = 0.0
    best_state = None
    for ep in range(epochs):
        readout.train()
        total_loss = 0.0
        n_batches = 0
        perm_ep = torch.randperm(len(train_idx))
        for start in range(0, len(train_idx), 32):
            idx = perm_ep[start:start+32]
            B_batch = B_train_t[idx]
            T_batch = T_pair_t[idx]
            scores = readout(B_batch)  # (bs, 64)
            # Pairwise BCE loss
            loss = 0.0
            for bi in range(len(idx)):
                s = scores[bi]
                t = T_batch[bi]
                # For each pair (i,j): BCE(s_i - s_j, t_ij)
                s_diff = s.unsqueeze(1) - s.unsqueeze(0)  # (64, 64)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(s_diff, t, reduction="mean")
                loss = loss + bce
            loss = loss / len(idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Val
        readout.eval()
        with torch.no_grad():
            B_val_t = torch.from_numpy(B_content[val_idx]).float()
            val_accs = []
            for bi in range(len(val_idx)):
                scores = readout(B_val_t[bi].unsqueeze(0))
                z = scores[0]
                # Pairwise accuracy vs teacher
                T_val = teacher_pairwise[val_idx[bi]]
                pred = (z.unsqueeze(1) > z.unsqueeze(0)).float()
                acc = (pred == torch.from_numpy(T_val).float()).float().mean().item()
                val_accs.append(acc)
            val_acc = np.mean(val_accs)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in readout.state_dict().items()}
            best_ep = ep + 1

        if (ep + 1) % 5 == 0:
            print(f"  [{ep+1:>3}/{epochs}] train_loss={total_loss/n_batches:.4f} val_acc={val_acc:.4f} best={best_val_acc:.4f} (ep{best_ep})")

    readout.load_state_dict(best_state)
    readout.eval()

    # ── Posthoc evaluate τ_vs_physical ──
    tau_physical = []
    for m in range(M):
        B = torch.from_numpy(B_content[m]).float().unsqueeze(0)
        with torch.no_grad():
            scores = readout(B)[0]
        sigma_model = scores.argsort(descending=True).cpu().numpy()
        sigma_phys = inv_perm[sigma_model]
        tau, _ = kendalltau(sigma_phys, np.arange(64))
        tau_physical.append(tau)

    tau_mean = float(np.mean(tau_physical))
    print(f"\n  τ_vs_physical (posthoc): {tau_mean:.4f} (mean over {M} samples)")
    print(f"  best val pairwise acc: {best_val_acc:.4f}")

    # Save
    out_path = pathlib.Path(out_dir) / "label_free_readout.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": readout.state_dict(), "config": {"model_name": "flatten", "N": 64, "hidden": (1024, 256)}}, str(out_path))
    print(f"  saved → {out_path}")

    return {"tau_physical_mean": tau_mean, "val_acc": best_val_acc, "out_path": str(out_path)}


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Label-free head selection + cluster readout")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--M-score", type=int, default=20, help="M for Stage A scoring")
    p.add_argument("--batch-score", type=int, default=4, help="batch for A scoring")
    p.add_argument("--M-train", type=int, default=200, help="M for Stage C training")
    p.add_argument("--batch-train", type=int, default=8, help="batch_mean for C training")
    p.add_argument("--n-reveal", type=int, default=10, help="reveal pairs for robustness")
    p.add_argument("--topk", type=int, default=4, help="top-k heads to cluster")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out-dir", default="reports/label_free_pipeline")
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ═══ Stage A: Canonical scan (batch-mean + destroyed-control gate) ═══
    print("=" * 60)
    print("STAGE A: Canonical 65-node scan (label-free, batch-mean)")
    print("=" * 60)
    rows = stage_a_canonical_scan(
        args.ckpt, M=args.M_score, batch_size=args.batch_score,
        seed=args.seed, device=args.device,
    )

    # Select strong_pass heads
    strong = [r for r in rows if r.get("gate_status") == "strong_pass"]
    strong = sorted(strong, key=lambda r: abs(r["tau_vs_l2r"]), reverse=True)
    print(f"\n  strong_pass heads: {len(strong)}")
    for r in strong:
        print(f"    L{r['layer']}H{r['head']}: τ={r['tau_vs_l2r']:.4f}")

    if not strong:
        print("  No strong_pass heads found — using top-4 by |tau|")
        all_heads = sorted(rows, key=lambda r: abs(r["tau_vs_l2r"]), reverse=True)
        top_heads = [(r["layer"], r["head"]) for r in all_heads[:args.topk]]
    else:
        top_heads = [(r["layer"], r["head"]) for r in strong[:args.topk]]

    print(f"\n  Selected heads: {[f'L{l}H{h}' for l,h in top_heads]}")

    (out_dir / "stage_a_canonical_scan.json").write_text(
        json.dumps(rows, indent=1, default=str)
    )

    # ═══ Stage B: Cluster ═══
    print("\n" + "=" * 60)
    print("STAGE B: Order clustering")
    print("=" * 60)
    clusters = stage_b_cluster(
        args.ckpt, top_heads, M=args.M_score, batch_size=args.batch_score,
        seed=args.seed, device=args.device,
    )

    # Pick largest cluster
    best_cid = max(clusters, key=lambda cid: clusters[cid]["size"])
    best_cluster = clusters[best_cid]
    print(f"\n  Selected cluster C{best_cid}: {best_cluster['heads']}")

    (out_dir / "stage_b_clusters.json").write_text(
        json.dumps({str(k): v for k, v in clusters.items()}, indent=1)
    )

    # ═══ Stage C: Teacher + Readout ═══
    print("\n" + "=" * 60)
    print("STAGE C: Cluster readout training")
    print("=" * 60)
    result = stage_c_build_teacher_and_train(
        args.ckpt,
        cluster=best_cluster["heads"],
        M=args.M_train,
        batch_mean_size=args.batch_train,
        seed=args.seed,
        device=args.device,
        out_dir=str(out_dir),
        epochs=args.epochs,
        lr=args.lr,
    )

    print("\n" + "=" * 60)
    print(f"FINAL: τ_vs_physical = {result['tau_physical_mean']:.4f}")
    print(f"  val_pairwise_acc = {result['val_acc']:.4f}")
    print("=" * 60)

    (out_dir / "stage_c_result.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
