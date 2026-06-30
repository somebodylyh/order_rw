#!/usr/bin/env python3
"""Uniform Label-Free Cluster Discovery v1.

Protocol:
  Stage A: per-head split-half stability + reveal-invariance → head scoring
           head-head Kendall clustering → select best cluster
  Stage B: cluster-mean B → pairwise teacher → FlattenReadout training
  Stage C: posthoc physical τ evaluation (ONLY here use physical labels)

ALL layer, ALL head, NO hard-coded layer/head.
Physical labels used ONLY for final evaluation.
"""

from __future__ import annotations

import argparse, json, pathlib, sys
import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

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


def _random_probe_orders(batch_size: int, seed: int, device) -> torch.Tensor:
    rows = []
    for b in range(batch_size):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) * 100_000_000 + b)
        blocks = torch.randperm(N, generator=g)
        rows.append(expand_model_blocks_to_token_order(blocks.unsqueeze(0), BLOCK_LEN)[0])
    return torch.stack(rows).to(device)


def _extract_all_B65(model, chunks, total, seed, dev, n_reveal=4):
    """Forward under n_reveal random probe orders, extract B65 for all heads.
    Returns: B65_by_reveal[ri][layer][head] = (total, 65, 65) float32
    """
    L, H_per = 4, 8
    B65_by_reveal = []
    all_probes = []

    for ri in range(n_reveal):
        probe_seed = seed * 1000 + ri
        B65_ri = [[None for _ in range(H_per)] for _ in range(L)]
        B_sum = {(l, h): np.zeros((total, 65, 65), dtype=np.float64) for l in range(L) for h in range(H_per)}

        # Batch forward
        fwd_batch = 32
        for start in range(0, total, fwd_batch):
            end = min(start + fwd_batch, total)
            bs = end - start
            probe = _random_probe_orders(bs, probe_seed * 100 + start, dev)
            model.eval()
            with torch.no_grad():
                _, _, attn_list = model.forward_fn(
                    chunks[start:end].to(dev), probe, return_attentions=True
                )
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
            probe_np = probe.cpu().numpy()
            if ri == 0:
                all_probes.append(probe_np)

            for li in range(L):
                attn_batch = attn_list[li].cpu().numpy()  # (bs, 8, 257, 257)
                for hi in range(H_per):
                    for bi in range(bs):
                        A65 = _A_model_vec(attn_batch[bi, hi], probe_np[bi])
                        B_sum[(li, hi)][start + bi] = build_none_separated_B(A65)
            del attn_list

        for li in range(L):
            for hi in range(H_per):
                B65_ri[li][hi] = B_sum[(li, hi)].astype(np.float32)

        B65_by_reveal.append(B65_ri)
        print(f"  reveal {ri+1}/{n_reveal} done", flush=True)

    all_probes_np = np.concatenate(all_probes, axis=0)[:total] if all_probes else None
    return B65_by_reveal


def _pairwise_confidence(B65):
    Bc = B65[1:, 1:].copy()
    Bc_sym = Bc + Bc.T + 1e-8
    P = Bc / Bc_sym
    return float(np.mean(np.abs(P - 0.5)))


def _cycle_rate(B65):
    P = _pairwise_pref(B65)
    pref = (P > 0.5).astype(np.int8)
    n = pref.shape[0]
    cycles = 0
    total = max(1, n * (n - 1) * (n - 2) // 6)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if pref[i, j] == pref[j, k] == pref[k, i] == 1:
                    cycles += 1
                elif pref[i, j] == pref[j, k] == pref[k, i] == 0:
                    cycles += 1
    return cycles / total


def _pairwise_pref(B65):
    Bc = B65[1:, 1:]
    Bc_sym = Bc + Bc.T + 1e-8
    return Bc / Bc_sym


def _none_leakage(B65):
    total = np.abs(B65).sum()
    if total < 1e-8:
        return 1.0
    none_row = np.abs(B65[0, :]).sum()
    none_col = np.abs(B65[:, 0]).sum()
    return float((none_row + none_col - np.abs(B65[0, 0])) / total)


# ═══════════════════════════════════════════════════════════════════════════
# Stage A: Uniform label-free head scoring + clustering
# ═══════════════════════════════════════════════════════════════════════════

def stage_a_uniform_discovery(ckpt_path, M=100, batch_size=8, n_reveal=8,
                               seed=0, device="cuda:0"):
    """Uniform label-free head discovery with split-half + cluster-first.

    Returns: best_cluster [(layer, head), ...], all_scores, clusters
    """
    total = M * batch_size
    dev = torch.device(device)
    model, chunks, clean_perm, dev_actual, _ci = _load_model_and_chunks(
        ckpt_path, total, seed, str(dev), "train"
    )
    print(f"  M={M} batch_size={batch_size} total={total} n_reveal={n_reveal}")

    # ═══ Extract B65 for all reveals ═══
    B65_by_reveal = _extract_all_B65(model, chunks, total, seed, dev_actual, n_reveal)

    # ═══ Per-head scoring ═══
    L, H_per = 4, 8
    head_scores = []

    for layer in range(L):
        for head in range(H_per):
            # Split-half: use reveals 0-3 for split A, 4-7 for split B
            mid = n_reveal // 2

            # Split A: mean B65 over reveals, then batch-mean over samples, then rollout
            B_A_reveals = np.mean([B65_by_reveal[ri][layer][head] for ri in range(mid)], axis=0)  # (total, 65, 65)
            B_A = B_A_reveals.mean(axis=0)  # (65, 65)
            sigma_A = rollout_by_method(B_A, "C-D+L")

            # Split B: same
            B_B_reveals = np.mean([B65_by_reveal[ri][layer][head] for ri in range(mid, n_reveal)], axis=0)
            B_B = B_B_reveals.mean(axis=0)
            sigma_B = rollout_by_method(B_B, "C-D+L")

            # S_split: cross-split rollout stability
            s_split, _ = kendalltau(sigma_A, sigma_B)
            s_split = float(max(0, s_split))  # clip negative

            # S_reveal: pairwise tau across all reveals
            orders = []
            for ri in range(n_reveal):
                B_ri = B65_by_reveal[ri][layer][head].mean(axis=0)  # (65, 65)
                orders.append(rollout_by_method(B_ri, "C-D+L"))
            reveal_taus = []
            for i in range(n_reveal):
                for j in range(i + 1, n_reveal):
                    tau, _ = kendalltau(orders[i], orders[j])
                    reveal_taus.append(max(0, tau))
            s_reveal = float(np.mean(reveal_taus))

            # S_conf, S_cycle, S_none from overall mean B
            B_mean = np.mean([B65_by_reveal[ri][layer][head].mean(axis=0) for ri in range(n_reveal)], axis=0)
            s_conf = _pairwise_confidence(B_mean)
            s_cycle = 1.0 - _cycle_rate(B_mean)
            s_none = _none_leakage(B_mean)

            # Composite score (no physical labels).
            # No none_leakage penalty — None node is part of the CDL mechanism.
            lam_cycle = 2.0
            total_score = s_split + s_reveal + s_conf + lam_cycle * s_cycle

            head_scores.append({
                "layer": layer, "head": head,
                "s_split": s_split, "s_reveal": s_reveal,
                "s_conf": s_conf, "s_cycle": s_cycle, "s_none": s_none,
                "total": total_score,
                "sigma": sigma_A,  # keep for clustering
            })

    # ═══ Select top-1 by label-free score (no clustering needed) ═══
    head_scores.sort(key=lambda x: x["total"], reverse=True)
    best_head = head_scores[0]
    best_cluster = {
        "cid": 0, "score": best_head["total"], "size": 1,
        "heads": [(best_head["layer"], best_head["head"])],
        "mean_head_score": best_head["total"],
        "internal_agreement": 1.0,
    }
    cluster_results = [best_cluster]
    sim = np.zeros((1, 1))

    cluster_results.sort(key=lambda c: c["score"], reverse=True)
    best_cluster = cluster_results[0] if cluster_results else None

    return best_cluster, head_scores, cluster_results, sim


# ═══════════════════════════════════════════════════════════════════════════
# Stage B+C: Cluster teacher + FlattenReadout training
# ═══════════════════════════════════════════════════════════════════════════

def stage_bc_train_readout(ckpt_path, cluster_heads, M=200, batch_size=8,
                            seed=0, device="cuda:0", out_dir="", epochs=40, lr=3e-4):
    """Build cluster-mean B dataset, train FlattenReadout, evaluate physical τ."""
    from batch_readout.model import FlattenReadout
    from batch_readout.integration_hook import _build_from_config

    total = M * batch_size
    dev = torch.device(device)
    model, chunks, clean_perm, dev_actual, _ci = _load_model_and_chunks(
        ckpt_path, total, seed, str(dev), "train"
    )
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()

    # Extract cluster-mean B for all samples
    B_content = np.zeros((total, 64, 64), dtype=np.float32)  # strip None for readout
    fwd_batch = 32
    for start in range(0, total, fwd_batch):
        end = min(start + fwd_batch, total)
        bs = end - start
        probe = _random_probe_orders(bs, seed * 100 + start, dev_actual)
        model.eval()
        with torch.no_grad():
            _, _, attn_list = model.forward_fn(
                chunks[start:end].to(dev_actual), probe, return_attentions=True
            )
            if dev_actual.type == "cuda":
                torch.cuda.synchronize(dev_actual)
        probe_np = probe.cpu().numpy()
        for bi in range(bs):
            B_sum = np.zeros((65, 65), dtype=np.float64)
            for layer, head in cluster_heads:
                attn = attn_list[layer][bi, head].cpu().numpy()
                A65 = _A_model_vec(attn, probe_np[bi])
                B65 = build_none_separated_B(A65)
                B_sum += B65
            B_mean = (B_sum / len(cluster_heads)).astype(np.float32)
            B_content[start + bi] = B_mean[1:, 1:]  # strip None
        del attn_list
        if (end % (fwd_batch * 10) == 0) or end == total:
            print(f"  [extract] {end}/{total}", flush=True)

    # ═══ Per-sample teacher (NO batch-mean collapse) ═══
    # Each sample gets its own CDL teacher → preserves sample diversity
    total_samples = B_content.shape[0]  # total (not M)
    sigma_T_per = np.zeros((total_samples, 64), dtype=np.int64)
    T_pair_per = np.zeros((total_samples, 64, 64), dtype=np.float32)
    for si in range(total_samples):
        B65_s = np.zeros((65, 65), dtype=np.float32)
        B65_s[1:, 1:] = B_content[si]
        sigma_T_per[si] = rollout_by_method(B65_s, "C-D+L")
        o = sigma_T_per[si]
        for i in range(64):
            for j in range(64):
                if i != j:
                    T_pair_per[si, i, j] = 1.0 if np.where(o == i)[0] < np.where(o == j)[0] else 0.0

    # Teacher diagnostics
    unique_teachers = len(set(tuple(s) for s in sigma_T_per))
    taus_vs_l2r = []
    for si in range(min(total_samples, 100)):
        tau_t, _ = kendalltau(sigma_T_per[si], np.arange(64))
        taus_vs_l2r.append(tau_t)
    tau_mean_t = np.mean(taus_vs_l2r)
    tau_std_t = np.std(taus_vs_l2r)
    print(f"  Teacher: {total_samples} samples, unique={unique_teachers}, "
          f"τ_vs_L2R mean={tau_mean_t:.4f} std={tau_std_t:.4f}", flush=True)
    if unique_teachers <= 1 or (abs(tau_mean_t) > 0.99 and tau_std_t < 0.01):
        print(f"  WARNING: teacher collapsed to near-constant L2R!", flush=True)

    # Split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_samples)
    train_n = int(total_samples * 0.8)
    val_idx = perm[train_n:train_n + int(total_samples * 0.1)]
    train_idx = perm[:train_n]

    # Train FlattenReadout with per-sample teacher
    readout = FlattenReadout(N=64, hidden=(1024, 256)).to(dev_actual)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=lr, weight_decay=1e-2)
    B_all = torch.from_numpy(B_content).float().to(dev_actual)
    T_all = torch.from_numpy(T_pair_per).float().to(dev_actual)

    best_acc = 0.0
    best_state = None
    for ep in range(epochs):
        readout.train()
        perm_ep = torch.randperm(len(train_idx))
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(train_idx), 32):
            idx = perm_ep[start:start+32]
            scores = readout(B_all[train_idx[idx]])  # (bs, 64)
            t = T_all[train_idx[idx]]  # (bs, 64, 64) — PER-SAMPLE
            s_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # (bs, 64, 64)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                s_diff, t, reduction="mean"
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Val
        readout.eval()
        with torch.no_grad():
            scores_val = readout(B_all[val_idx])
            s_diff_v = scores_val.unsqueeze(2) - scores_val.unsqueeze(1)
            pred = (s_diff_v > 0).float()
            acc = (pred == T_all[val_idx]).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in readout.state_dict().items()}

        if (ep + 1) % 10 == 0:
            print(f"  [{ep+1:>3}/{epochs}] loss={total_loss/n_batches:.4f} val_acc={acc:.4f} best={best_acc:.4f}", flush=True)

    readout.load_state_dict(best_state)
    readout.eval()

    # ═══ Input sanity: real B vs noise/zero/shuffle ═══
    with torch.no_grad():
        # Real B
        scores_real = readout(B_all[:100])
        sigma_real = scores_real.argsort(dim=1, descending=True).cpu().numpy()
        # Random noise B
        B_noise = torch.randn(100, 64, 64).to(dev_actual)
        scores_noise = readout(B_noise)
        sigma_noise = scores_noise.argsort(dim=1, descending=True).cpu().numpy()
        # Zero B
        B_zero = torch.zeros(100, 64, 64).to(dev_actual)
        scores_zero = readout(B_zero)
        sigma_zero = scores_zero.argsort(dim=1, descending=True).cpu().numpy()

    taus_real, taus_noise, taus_zero = [], [], []
    sigma_lists = [sigma_real, sigma_noise, sigma_zero]
    names = ["real", "noise", "zero"]
    for name, sigmas in zip(names, sigma_lists):
        for si in range(min(100, len(sigmas))):
            tau_s, _ = kendalltau(inv_perm[sigmas[si]], np.arange(64))
            locals()[f"taus_{name}"].append(abs(tau_s))
        print(f"  Sanity {name} B: |τ_phys| mean={np.mean(locals()[f'taus_{name}']):.4f} "
              f"output_unique={len(set(tuple(s) for s in sigmas[:50]))}", flush=True)

    tau_mean_final = float(np.mean(taus_real))
    delta_input = float(np.mean(taus_real) - np.mean(taus_noise))
    print(f"  Input gap Δ(real-noise): {delta_input:.4f}", flush=True)

    # Save
    out_path = pathlib.Path(out_dir) / "label_free_readout.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": readout.state_dict(),
                 "config": {"model_name": "flatten", "N": 64, "hidden": (1024, 256)}},
               str(out_path))

    return {"tau_physical": tau_mean_final, "val_acc": best_acc, "delta_input": delta_input, "path": str(out_path)}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--M-score", type=int, default=100)
    p.add_argument("--batch-score", type=int, default=8)
    p.add_argument("--n-reveal", type=int, default=8)
    p.add_argument("--M-train", type=int, default=200)
    p.add_argument("--batch-train", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="reports/uniform_label_free_v1")
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ═══ Stage A: Uniform discovery ═══
    print("=" * 60)
    print("STAGE A: Uniform Label-Free Cluster Discovery")
    print("=" * 60)
    best_cluster, head_scores, clusters, sim = stage_a_uniform_discovery(
        args.ckpt, M=args.M_score, batch_size=args.batch_score,
        n_reveal=args.n_reveal, seed=args.seed, device=args.device,
    )

    print(f"\n  Best cluster C{best_cluster['cid']}: {best_cluster['heads']}")
    print(f"  score={best_cluster['score']:.4f} size={best_cluster['size']}")
    print(f"\n  All clusters:")
    for c in sorted(clusters, key=lambda c: c["score"], reverse=True):
        print(f"    C{c['cid']}: {c['heads']} score={c['score']:.4f} size={c['size']}")

    # Top-10 per-head scores for reference
    print(f"\n  Top-10 per-head scores:")
    for s in sorted(head_scores, key=lambda x: x["total"], reverse=True)[:10]:
        print(f"    L{s['layer']}H{s['head']}: "
              f"split={s['s_split']:.3f} reveal={s['s_reveal']:.3f} "
              f"conf={s['s_conf']:.3f} cycle={s['s_cycle']:.3f} "
              f"→ S={s['total']:.3f}")

    # ═══ Stage B+C: Train + eval ═══
    print("\n" + "=" * 60)
    print("STAGE B+C: Cluster Readout Training")
    print("=" * 60)
    result = stage_bc_train_readout(
        args.ckpt, best_cluster["heads"],
        M=args.M_train, batch_size=args.batch_train,
        seed=args.seed, device=args.device,
        out_dir=str(out_dir), epochs=args.epochs,
    )

    print(f"\n  τ_vs_physical = {result['tau_physical']:.4f}")
    print(f"  val_pairwise_acc = {result['val_acc']:.4f}")

    # Save all
    (out_dir / "stage_a_scores.json").write_text(json.dumps({
        "best_cluster": {k: v for k, v in best_cluster.items() if k != "sim"},
        "all_clusters": [{k: v for k, v in c.items()} for c in clusters],
        "head_scores": [{k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
                         for k, v in s.items() if k != "sigma"}
                        for s in head_scores],
    }, indent=1, default=str))
    (out_dir / "stage_bc_result.json").write_text(json.dumps(result, indent=1))

    print("\nDONE.")


if __name__ == "__main__":
    main()
