"""Multi-checkpoint B-source comparison + CDL teacher component ablation.

Runs across checkpoints (5k, 10k, 20k, 40k, 60k) to trace how teacher
component contributions evolve as attention crystallizes.

Each ckpt: extract L0H0 single-head + L0 top4-head mean B.
Each B: run key teacher ablations + one-shot baselines.
Final: trend table showing τ(L2R) across steps.
"""
from __future__ import annotations

import sys, os, time, json, glob, re
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AOGPT_DIR = os.path.join(_SCRIPT_DIR, "..", "block_lo_arm_order_network")
sys.path.insert(0, _AOGPT_DIR)

from attn_order_teacher import teacher_components, rollout_order, _softmax, _entropy

N_BLOCKS = 64
L2R = np.arange(N_BLOCKS, dtype=np.int64)
K_SEEDS = 20
TAU_T = 1.0


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════

def kendall_tau(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a); conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0: continue
            if da * db > 0: conc += 1
            elif da * db < 0: disc += 1
    d = conc + disc
    return (conc - disc) / d if d > 0 else 0.0


def pairwise_agreement(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a); same = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0: continue
            total += 1
            if da * db > 0: same += 1
    return same / total if total > 0 else 1.0


# ═══════════════════════════════════════════════════════════════════
# Phase 1: B Extraction
# ═══════════════════════════════════════════════════════════════════

def _A_to_B(A):
    B = np.asarray(A.T, dtype=np.float64).copy()
    np.fill_diagonal(B, 0.0)
    return B


def extract_key_Bs(ckpt_path, device="cuda:0", seed=42, n_samples=200):
    """Extract L0H0 single-head and L0 top4-head mean B from a ckpt.

    Returns dict with keys 'L0H0' and 'L0 top4', values = (64,64) float64.
    """
    import torch
    from per_head_order_scan import extract_per_head_and_heavy_A
    from neural_readout.extract_b import _load_model_and_chunks

    print(f"  [load] {ckpt_path}", flush=True)
    model, chunks, clean_perm, dev, chunk_index = _load_model_and_chunks(
        ckpt_path=ckpt_path, M=n_samples, seed=seed, device=device, split="train")

    A_lh, _ = extract_per_head_and_heavy_A(
        model, chunks, clean_perm, dev, seed=seed, fwd_batch=64, none_mode="b0")
    # (n, L, H, N, N)
    _, L, H, N, _ = A_lh.shape

    # L0H0 single-head
    A_L0H0 = A_lh[:, 0, 0].mean(axis=0)
    B_L0H0 = _A_to_B(A_L0H0)

    # L0 top4 order-head mean: screen each L0 head by τ(CDL, L2R)
    head_taus = {}
    for h in range(H):
        Ah = A_lh[:, 0, h].mean(axis=0)
        Bh = _A_to_B(Ah)
        o = rollout_order(Bh, tau_T=TAU_T, seed=seed, mode="C-D+L", standardize=True)
        head_taus[(0, h)] = kendall_tau(o, L2R)
    top4 = sorted(head_taus.items(), key=lambda x: -x[1])[:4]
    top4_heads = [h for (l, h), _ in top4]
    A_top4 = np.mean([A_lh[:, 0, h].mean(axis=0) for h in top4_heads], axis=0)
    B_top4 = _A_to_B(A_top4)

    del model, A_lh
    if device != "cpu":
        torch.cuda.empty_cache()

    return {"L0H0": B_L0H0, "L0 top4": B_top4}


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Teacher Rollout Engine
# ═══════════════════════════════════════════════════════════════════

def rollout_custom(B, seeds, w_c=0.0, w_d=0.0, w_l=0.0):
    orders, ents = [], []
    for s in seeds:
        Bn = np.asarray(B, dtype=np.float64)
        N = Bn.shape[0]; rng = np.random.default_rng(s)
        S, U, last = [], list(range(N)), None; order = []
        for _ in range(N):
            if len(U) == 1:
                v = U[0]
            else:
                C, D, L, cand = teacher_components(Bn, S, U, last)
                q = w_c * C + w_d * D + w_l * L
                qq = q.copy()
                qq = (qq - qq.mean()) / (qq.std() + 1e-9)
                p = _softmax(qq, TAU_T)
                v = int(cand[rng.choice(len(cand), p=p)])
                ents.append(_entropy(p))
            order.append(v); S.append(v); U.remove(v); last = v
        orders.append(np.asarray(order, dtype=np.int64))
    return np.stack(orders), float(np.mean(ents)) if ents else 0.0


def rollout_one_shot(B, seeds, score_fn):
    scores = score_fn(np.asarray(B, dtype=np.float64))
    orders, ents = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        U = list(range(len(scores))); order = []
        for _ in range(len(scores)):
            if len(U) == 1: v = U[0]
            else:
                q = np.array([scores[u] for u in U], dtype=np.float64)
                q = (q - q.mean()) / (q.std() + 1e-9)
                p = _softmax(q, TAU_T)
                v = int(U[rng.choice(len(U), p=p)])
                ents.append(_entropy(p))
            order.append(v); U.remove(v)
        orders.append(np.asarray(order, dtype=np.int64))
    return np.stack(orders), float(np.mean(ents)) if ents else 0.0


def score_rowsum(B):  return B.sum(axis=1)
def score_colsum(B):  return B.sum(axis=0)
def score_readiness(B): return B.sum(axis=1) - B.sum(axis=0)


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Run Ablation on one B
# ═══════════════════════════════════════════════════════════════════

# Key teachers only (trimmed from full set for speed across ckpts)
KEY_TEACHERS = [
    # (name, kind, params)
    ("full (C−D+L)",    "rollout",  dict(w_c=1.0, w_d=-1.0, w_l=1.0)),
    ("no-S (−D+L)",     "rollout",  dict(w_c=0.0, w_d=-1.0, w_l=1.0)),
    ("no-U (C+L)",      "rollout",  dict(w_c=1.0, w_d= 0.0, w_l=1.0)),
    ("no-local (C−D)",  "rollout",  dict(w_c=1.0, w_d=-1.0, w_l=0.0)),
    ("−D only",          "rollout",  dict(w_c=0.0, w_d=-1.0, w_l=0.0)),
    ("wrong-D (C+D+L)", "rollout",  dict(w_c=1.0, w_d= 1.0, w_l=1.0)),
    ("C only",          "rollout",  dict(w_c=1.0, w_d= 0.0, w_l=0.0)),
    ("L only",          "rollout",  dict(w_c=0.0, w_d= 0.0, w_l=1.0)),
    ("rowsum ↓",        "oneshot",  score_rowsum),
    ("colsum ↓",        "oneshot",  score_colsum),
    ("readiness ↓",     "oneshot",  score_readiness),
    ("L2R (oracle)",    "constant", L2R),
    ("random",          "random",   None),
]


def run_ablation_on_B(B):
    base_seeds = list(range(10000, 10000 + K_SEEDS))

    # full CDL reference
    full_orders, full_ent = rollout_custom(B, base_seeds, w_c=1.0, w_d=-1.0, w_l=1.0)
    full_tau = np.mean([kendall_tau(o, L2R) for o in full_orders])
    full_tau_std = np.std([kendall_tau(o, L2R) for o in full_orders])

    tau_per_sample = np.array([kendall_tau(o, L2R) for o in full_orders])
    non_l2r_mask = tau_per_sample < 0.95
    n_nlr = non_l2r_mask.sum()

    results = []
    for name, kind, params in KEY_TEACHERS:
        t0 = time.time()
        if kind == "rollout":
            orders, ent = rollout_custom(B, base_seeds, **params)
        elif kind == "oneshot":
            orders, ent = rollout_one_shot(B, base_seeds, params)
        elif kind == "constant":
            orders = np.tile(params, (K_SEEDS, 1)); ent = 0.0
        elif kind == "random":
            orders = np.stack([np.random.default_rng(s).permutation(N_BLOCKS)
                               for s in base_seeds]); ent = np.log(N_BLOCKS)

        tau_mean = np.mean([kendall_tau(o, L2R) for o in orders])
        tau_std  = np.std([kendall_tau(o, L2R) for o in orders])
        agree    = np.mean([pairwise_agreement(o, full_orders[i])
                            for i, o in enumerate(orders)])
        agree_nlr = (np.mean([pairwise_agreement(orders[i], full_orders[i])
                              for i in range(K_SEEDS) if non_l2r_mask[i]])
                     if n_nlr > 0 else float("nan"))

        results.append(dict(name=name, tau_vs_l2r=tau_mean, tau_std=tau_std,
                            agree_full=agree, agree_nlr=agree_nlr, entropy=ent,
                            time=time.time() - t0))

    return results, full_tau, full_tau_std, n_nlr


# ═══════════════════════════════════════════════════════════════════
# Trend Report
# ═══════════════════════════════════════════════════════════════════

def print_trend_table(all_data):
    """all_data: [(step, B_source, results, full_tau), ...]"""

    # Group by teacher
    teacher_names = [t[0] for t in KEY_TEACHERS]
    steps = sorted(set(d[0] for d in all_data))
    sources = sorted(set(d[1] for d in all_data))

    print("\n" + "=" * 130)
    print("TREND: τ(teacher, L2R) across checkpoints")
    print("=" * 130)

    # One table per B source
    for src in sources:
        header = f"{'Teacher':<24s}"
        for s in steps:
            header += f"  {s:>8s}"
        print(f"\n── {src} ──")
        print(header)
        print("-" * (24 + 2 + len(steps) * 10))

        for tname in teacher_names:
            row = f"{tname:<24s}"
            for s in steps:
                match = [d for d in all_data if d[0] == s and d[1] == src]
                if match:
                    r = next((r for r in match[0][2] if r["name"] == tname), None)
                    if r:
                        row += f"  {r['tau_vs_l2r']:+8.4f}"
                    else:
                        row += f"  {'N/A':>8s}"
                else:
                    row += f"  {'—':>8s}"
            print(row)
        print()

    # Key question table: is D still load-bearing at 60k?
    print("=" * 130)
    print("Δτ vs full CDL: does −D remain load-bearing?")
    print("=" * 130)
    for src in sources:
        print(f"\n── {src} ──")
        hdr = f"{'Checkpoint':<12s}  {'full τ':>8s}  {'Δ(no-S)':>8s}  {'Δ(no-U)':>8s}  {'Δ(no-L)':>8s}  {'Δ(−D only)':>10s}  {'Δ(colsum)':>10s}"
        print(hdr)
        print("-" * len(hdr))
        for s in steps:
            match = [d for d in all_data if d[0] == s and d[1] == src]
            if not match: continue
            results, full_tau = match[0][2], match[0][3]
            deltas = {}
            for ab_name in ["no-S (−D+L)", "no-U (C+L)", "no-local (C−D)", "−D only", "colsum ↓"]:
                r = next((r for r in results if r["name"] == ab_name), None)
                deltas[ab_name] = r["tau_vs_l2r"] - full_tau if r else float("nan")

            def tag(d):
                if np.isnan(d): return "?"
                return "LOAD-BEARING" if d < -0.03 else ("IMPROVES" if d > 0.03 else "negligible")

            print(f"{s:<12s}  {full_tau:+8.4f}  {deltas['no-S (−D+L)']:+8.4f}  "
                  f"{deltas['no-U (C+L)']:+8.4f}  {deltas['no-local (C−D)']:+8.4f}  "
                  f"{deltas['−D only']:+10.4f}  {deltas['colsum ↓']:+10.4f}")
            print(f"{'':>12s}  {'':>8s}  {tag(deltas['no-S (−D+L)']):>8s}  "
                  f"{tag(deltas['no-U (C+L)']):>8s}  {tag(deltas['no-local (C−D)']):>8s}  "
                  f"{tag(deltas['−D only']):>10s}  {tag(deltas['colsum ↓']):>10s}")
    print("=" * 130)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main(ckpt_dir, steps, device="cuda:0", output_dir=None):
    all_data = []  # [(step, source, results, full_tau)]

    for step in steps:
        ckpt_path = os.path.join(ckpt_dir, f"ckpt_step{step}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[skip] {ckpt_path} not found")
            continue

        print(f"\n{'█'*60}")
        print(f"█  STEP {step}")
        print(f"{'█'*60}")

        variants = extract_key_Bs(ckpt_path, device=device)

        for src_label, B in variants.items():
            print(f"  [{src_label}] μ={B.mean():.4f} σ={B.std():.4f}  — running ablation...", flush=True)
            results, full_tau, full_tau_std, n_nlr = run_ablation_on_B(B)
            print(f"    full τ(L2R) = {full_tau:.4f}±{full_tau_std:.4f}  nonL2R={n_nlr}/{K_SEEDS}")
            for r in results:
                print(f"      {r['name']:<22s} τ={r['tau_vs_l2r']:+.4f}±{r['tau_std']:.4f}")
            all_data.append((step, src_label, results, full_tau))

    # Print trend
    print_trend_table(all_data)

    # Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out = {"steps": list(steps), "data": []}
        for step, src, results, full_tau in all_data:
            out["data"].append({"step": step, "source": src, "full_tau": full_tau,
                                "results": results})
        out_path = os.path.join(output_dir, "multi_ckpt_ablation.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=_json_default)
        print(f"\nSaved to {out_path}")

    return all_data


def _json_default(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    raise TypeError(f"not serializable: {type(obj)}")


if __name__ == "__main__":
    default_dir = os.path.join(_AOGPT_DIR, "probe_results", "clean_base_random_perm")
    default_steps = [5000, 10000, 20000, 40000, 60000]
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "cdl_teacher_ablation")

    # Allow step override: python script.py [output_dir] [step1,step2,...]
    steps = default_steps
    if len(sys.argv) > 2:
        steps = [int(s) for s in sys.argv[2].split(",")]

    main(default_dir, steps, output_dir=out_dir)
