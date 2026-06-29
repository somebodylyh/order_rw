"""Paper-ready attention diagnostic: CDL τ_vs_L2R per head across three training regimes.

Solidifies the quick script into a reproducible paper control:
  shuffled-L2R (fixed wrong order) vs random-order vs ori-L2R (natural L2R)

Outputs:
  - Console table: best / top-3 mean / top-5 mean τ per model
  - JSON dump: full per-(L,H) τ matrices for heatmap plotting
  - Per-head ablation trace for best head of each model
"""
from __future__ import annotations
import sys, os, time, json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AOGPT_DIR = os.path.join(_SCRIPT_DIR, "..", "block_lo_arm_order_network")
sys.path.insert(0, _AOGPT_DIR)

from attn_order_teacher import teacher_components, rollout_order, _softmax, _entropy

N_BLOCKS = 64
L2R = np.arange(N_BLOCKS, dtype=np.int64)
TAU_T = 1.0
M_SAMPLES = 200
SEED = 42

OUT_DIR = os.path.join(_SCRIPT_DIR, "attention_diagnostic_20260609")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

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


def _A_to_B(A):
    B = np.asarray(A.T, dtype=np.float64).copy()
    np.fill_diagonal(B, 0.0)
    return B


# ═══════════════════════════════════════════════════════════════════════
# Extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_per_head_B(ckpt_path, device="cuda:0", n_samples=M_SAMPLES):
    import torch
    from per_head_order_scan import extract_per_head_and_heavy_A
    from neural_readout.extract_b import _load_model_and_chunks

    t0 = time.time()
    print(f"  [load] {ckpt_path}", flush=True)
    model, chunks, clean_perm, dev, chunk_index = _load_model_and_chunks(
        ckpt_path=ckpt_path, M=n_samples, seed=SEED, device=device, split="train")

    A_lh, _ = extract_per_head_and_heavy_A(
        model, chunks, clean_perm, dev, seed=SEED, fwd_batch=64, none_mode="b0")
    n, L, H, N, _N = A_lh.shape
    assert N == _N == N_BLOCKS

    B_per_head = {}
    for l in range(L):
        for h in range(H):
            A_mean = A_lh[:, l, h].mean(axis=0)
            B_per_head[(l, h)] = _A_to_B(A_mean)

    del model, A_lh
    if device != "cpu":
        torch.cuda.empty_cache()
    print(f"  [done] {L}×{H} heads in {time.time()-t0:.1f}s", flush=True)
    return B_per_head, L, H


# ═══════════════════════════════════════════════════════════════════════
# CDL teacher scan
# ═══════════════════════════════════════════════════════════════════════

def compute_tau_matrix(B_per_head, L, H):
    """Return (L×H) τ matrix + sorted flat list."""
    tau_mat = np.zeros((L, H))
    for (l, h), B in B_per_head.items():
        order = rollout_order(B, tau_T=TAU_T, seed=SEED, mode="C-D+L", standardize=True)
        tau_mat[l, h] = kendall_tau(order, L2R)
    # sorted flat list (descending)
    sorted_flat = sorted([(l, h, float(tau_mat[l, h])) for l in range(L) for h in range(H)],
                         key=lambda x: -x[2])
    return tau_mat, sorted_flat


def top_k_stats(sorted_flat, ks=(1, 3, 5)):
    """Return {k: mean_tau_of_top_k} for each k in ks."""
    taus = [t for _, _, t in sorted_flat]
    return {k: float(np.mean(taus[:k])) for k in ks}


def compute_ablations(B_per_head, sorted_flat):
    """On best head: C, L, C-D, C+L, C-D+L, -D only τ."""
    best_l, best_h, _ = sorted_flat[0]
    B = B_per_head[(best_l, best_h)]
    ablations = {}
    for mode in ["C", "L", "C-D", "C+L", "C-D+L"]:
        o = rollout_order(B, tau_T=TAU_T, seed=SEED, mode=mode, standardize=True)
        ablations[mode] = float(kendall_tau(o, L2R))
    # -D only
    Bn = np.asarray(B, dtype=np.float64)
    N = Bn.shape[0]; rng = np.random.default_rng(SEED)
    S, U, last = [], list(range(N)), None; order = []
    for _ in range(N):
        if len(U) == 1: v = U[0]
        else:
            _, D, _, cand = teacher_components(Bn, S, U, last)
            q = -D; q = (q - q.mean()) / (q.std() + 1e-9)
            p = _softmax(q, TAU_T)
            v = int(cand[rng.choice(len(cand), p=p)])
        order.append(v); S.append(v); U.remove(v); last = v
    ablations["-D only"] = float(kendall_tau(np.array(order, dtype=np.int64), L2R))
    return (best_l, best_h), ablations


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    device = "cuda:0"

    BASE = "/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/probe_results"
    ckpts = {
        "shuffled-L2R":   f"{BASE}/shuffled_l2r_continuous_jun05/ckpt_step50000.pt",
        "random-order":   f"{BASE}/random_baseline_continuous_jun08_seed2/ckpt_step50000.pt",
        "ori-L2R":        f"{BASE}/l2r_continuous_jun05/ckpt_step50000.pt",
    }

    all_data = {}

    for name, ckpt in ckpts.items():
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        B_per_head, L, H = extract_per_head_B(ckpt, device=device)
        tau_mat, sorted_flat = compute_tau_matrix(B_per_head, L, H)
        stats = top_k_stats(sorted_flat)
        (best_l, best_h), ablations = compute_ablations(B_per_head, sorted_flat)

        # Print summary
        print(f"  Best head:  L{best_l}H{best_h}  τ={sorted_flat[0][2]:+.4f}")
        print(f"  Top-1:      {stats[1]:+.4f}")
        print(f"  Top-3 mean: {stats[3]:+.4f}")
        print(f"  Top-5 mean: {stats[5]:+.4f}")
        print(f"  All-head mean: {float(tau_mat.mean()):+.4f} ± {float(tau_mat.std()):.4f}")
        print(f"  Ablations (best head L{best_l}H{best_h}):")
        for mode in ["C-D+L", "C-D", "C", "L", "-D only"]:
            v = ablations.get(mode, float('nan'))
            print(f"    {mode:10s}  τ={v:+.4f}")

        # Per-head table
        print(f"\n  τ matrix ({L}×{H}):")
        header = "     " + " ".join(f"H{h:1d}     " for h in range(H))
        print(header)
        for l in range(L):
            row = f"  L{l} " + " ".join(f"{tau_mat[l,h]:+.4f}" for h in range(H))
            print(row)

        all_data[name] = {
            "tau_matrix": tau_mat.tolist(),
            "sorted_heads": [(l, h, float(t)) for l, h, t in sorted_flat],
            "stats": {str(k): v for k, v in stats.items()},
            "best_head": [best_l, best_h],
            "best_tau": float(sorted_flat[0][2]),
            "ablations": ablations,
            "L": L, "H": H,
        }

    # ── Paper table ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PAPER TABLE: CDL τ_vs_L2R per training regime")
    print(f"{'='*70}")
    print(f"  {'Model':<20s} {'Training Order':<20s} {'Best τ':>8s} {'Top-3 mean':>10s} {'Top-5 mean':>10s} {'All mean':>10s}")
    print(f"  {'-'*20} {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    training_labels = {
        "shuffled-L2R": "fixed wrong order",
        "random-order": "random permutation",
        "ori-L2R": "natural L2R",
    }
    for name in ["shuffled-L2R", "random-order", "ori-L2R"]:
        d = all_data[name]
        s = d["stats"]
        print(f"  {name:<20s} {training_labels[name]:<20s} "
              f"{d['best_tau']:>+8.4f} {s['3']:>+10.4f} {s['5']:>+10.4f} "
              f"{float(np.mean(d['tau_matrix'])):>+10.4f}")

    # ── Save JSON ──────────────────────────────────────────────────
    json_path = os.path.join(OUT_DIR, "cdl_tau_diagnostic.json")
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ── Save heatmap CSV (easy import to matplotlib) ───────────────
    for name, d in all_data.items():
        slug = name.lower().replace("-", "_")
        csv_path = os.path.join(OUT_DIR, f"tau_heatmap_{slug}.csv")
        np.savetxt(csv_path, np.array(d["tau_matrix"]), fmt="%.4f", delimiter=",",
                   header=",".join(f"H{h}" for h in range(d["H"])), comments="")
        print(f"Saved: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
