"""Cross-ckpt unsupervised head selector: random-order checkpoints only.

seed2: 5k, 10k, 20k, 30k, 40k, 50k
seed1: 5k, 10k

M=500 for stable split-half CDL. 3-fold CDL consistency as S_consistency.
"""
import sys, os, time, json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AOGPT_DIR = os.path.join(_SCRIPT_DIR, "..", "block_lo_arm_order_network")
sys.path.insert(0, _AOGPT_DIR)

from attn_order_teacher import rollout_order
from per_head_order_scan import extract_per_head_and_heavy_A
from neural_readout.extract_b import _load_model_and_chunks

N_BLOCKS = 64; L2R = np.arange(N_BLOCKS); TAU_T = 1.0; M = 500; K = 3


def kendall_tau(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a); conc = disc = 0
    for i in range(n):
        for j in range(i+1, n):
            da, db = a[i]-a[j], b[i]-b[j]
            if da==0 and db==0: continue
            if da*db>0: conc+=1
            elif da*db<0: disc+=1
    d=conc+disc; return (conc-disc)/d if d>0 else 0.0


def _A_to_B(A):
    B=np.asarray(A.T,dtype=np.float64).copy(); np.fill_diagonal(B,0.0); return B


def kfold_consistency(B_list, K=3, seed=42):
    """K-fold CDL consistency: mean pairwise τ across K folds.

    Splits M B matrices into K folds, runs CDL on each fold-mean B,
    then computes mean τ across the K orders. Higher = more stable CDL output.
    """
    M = len(B_list)
    if M < K * 10:
        return 0.0
    fold_size = M // K
    rng = np.random.default_rng(seed)
    idx = rng.permutation(M)

    orders = []
    for k in range(K):
        fold_idx = idx[k*fold_size : (k+1)*fold_size]
        B_fold = _A_to_B(np.mean([B_list[i] for i in fold_idx], axis=0))
        o = rollout_order(B_fold, tau_T=TAU_T, seed=seed+k, mode="C-D+L", standardize=True)
        orders.append(o)

    taus = []
    for i in range(K):
        for j in range(i+1, K):
            taus.append(kendall_tau(orders[i], orders[j]))
    return float(np.mean(taus))


def row_conc_from_B(B):
    B=np.asarray(B,dtype=np.float64); N=B.shape[0]
    concs=[]
    for i in range(N):
        row=np.abs(B[i].copy()); s=row.sum()
        if s<1e-12: concs.append(0.0)
        else:
            p=row/s; p=p[p>1e-12]; H=-np.sum(p*np.log(p))
            concs.append(1.0-H/max(np.log(N-1),1e-12))
    return float(np.mean(concs))


def scan_checkpoint(ckpt_path, label, device="cuda:0"):
    print(f"\n  {label} ...", end=" ", flush=True)
    t0 = time.time()
    model, chunks, clean_perm, dev, ci = _load_model_and_chunks(
        ckpt_path=ckpt_path, M=M, seed=42, device=device, split="train")
    A_lh, _ = extract_per_head_and_heavy_A(
        model, chunks, clean_perm, dev, seed=42, fwd_batch=64, none_mode="b0")
    n, L, H, N, _ = A_lh.shape

    results = []
    for l in range(L):
        for h in range(H):
            B_mean = _A_to_B(A_lh[:,l,h].mean(axis=0))
            tau_o = kendall_tau(
                rollout_order(B_mean, tau_T=TAU_T, seed=42, mode="C-D+L", standardize=True),
                L2R)
            sc = kfold_consistency([_A_to_B(A_lh[i,l,h]) for i in range(n)], K=K, seed=42)
            rc = row_conc_from_B(B_mean)
            results.append({"head": f"L{l}H{h}", "tau": tau_o, "sc": sc, "rc": rc})

    del model, A_lh
    import torch
    if device!="cpu": torch.cuda.empty_cache()

    # stats
    from scipy.stats import spearmanr
    tau_arr = np.array([r["tau"] for r in results])
    sc_arr = np.array([r["sc"] for r in results])
    rc_arr = np.array([r["rc"] for r in results])
    rho_sc, p_sc = spearmanr(tau_arr, sc_arr)
    rho_rc, p_rc = spearmanr(tau_arr, rc_arr)

    # best heads
    best_oracle = max(results, key=lambda x: x["tau"])
    best_sc = max(results, key=lambda x: x["sc"])
    best_rc = max(results, key=lambda x: x["rc"])
    sc_correct = best_sc["head"] == best_oracle["head"]

    # top-3 recall
    oracle_top3 = set(r["head"] for r in sorted(results, key=lambda x: -x["tau"])[:3])
    sc_top3 = set(r["head"] for r in sorted(results, key=lambda x: -x["sc"])[:3])
    sc_top3_recall = len(oracle_top3 & sc_top3)

    elapsed = time.time()-t0
    print(f"{elapsed:.0f}s  oracle={best_oracle['head']} τ={best_oracle['tau']:+.3f}  "
          f"sc={best_sc['head']} sc={best_sc['sc']:+.3f} {'✓' if sc_correct else '✗'}  "
          f"ρ_sc={rho_sc:+.3f} p={p_sc:.3f}  recall@3={sc_top3_recall}/3", flush=True)

    return {"label": label, "best_oracle": best_oracle["head"], "best_tau": best_oracle["tau"],
            "best_sc": best_sc["head"], "best_sc_val": best_sc["sc"],
            "sc_correct": sc_correct, "sc_top3_recall": sc_top3_recall,
            "rho_sc": rho_sc, "p_sc": p_sc, "rho_rc": rho_rc, "p_rc": p_rc,
            "heads": results}


def main():
    device = "cuda:0"
    BASE = "/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/probe_results"

    ckpts = [
        (f"{BASE}/random_baseline_continuous_jun08_seed2/ckpt_step5000.pt",  "s2 5k"),
        (f"{BASE}/random_baseline_continuous_jun08_seed2/ckpt_step10000.pt", "s2 10k"),
        (f"{BASE}/random_baseline_continuous_jun08_seed2/ckpt_step20000.pt", "s2 20k"),
        (f"{BASE}/random_baseline_continuous_jun08_seed2/ckpt_step30000.pt", "s2 30k"),
        (f"{BASE}/random_baseline_continuous_jun08_seed2/ckpt_step40000.pt", "s2 40k"),
        (f"{BASE}/random_baseline_continuous_jun08_seed2/ckpt_step50000.pt", "s2 50k"),
        (f"{BASE}/random_baseline_continuous_jun09_seed1/ckpt_step5000.pt",  "s1 5k"),
        (f"{BASE}/random_baseline_continuous_jun09_seed1/ckpt_step10000.pt", "s1 10k"),
    ]

    print(f"Scanning {len(ckpts)} checkpoints (M={M}, K={K}-fold)")

    summaries = []
    for ckpt, label in ckpts:
        s = scan_checkpoint(ckpt, label, device)
        summaries.append(s)

    # ── Summary table ────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  SUMMARY: Unsupervised head selector (K={K}-fold CDL consistency)")
    print(f"{'='*80}")
    print(f"  {'ckpt':>10s}  {'oracle':>6s} {'τ':>7s}  {'SC best':>6s} {'SC':>7s}  "
          f"{'ρ_sc':>7s} {'p_sc':>6s} {'ρ_rc':>7s}  top-1 {'recall@3'}")
    print(f"  {'-'*10}  {'-'*6} {'-'*7}  {'-'*6} {'-'*7}  {'-'*7} {'-'*6} {'-'*7}  ----- {'--------'}")

    n_correct = 0
    total_recall = 0
    for s in summaries:
        ck = "✓" if s["sc_correct"] else "✗"
        print(f"  {s['label']:>10s}  {s['best_oracle']:>6s} {s['best_tau']:>+7.3f}  "
              f"{s['best_sc']:>6s} {s['best_sc_val']:>+7.3f}  "
              f"{s['rho_sc']:>+7.3f} {s['p_sc']:>6.3f} {s['rho_rc']:>+7.3f}  "
              f"  {ck}     {s['sc_top3_recall']}/3")
        if s["sc_correct"]: n_correct += 1
        total_recall += s["sc_top3_recall"]

    n = len(summaries)
    print(f"\n  Top-1 accuracy: {n_correct}/{n}  "
          f"Recall@3: {total_recall}/{n*3}  "
          f"Mean ρ_sc: {np.mean([s['rho_sc'] for s in summaries]):+.3f}  "
          f"Mean ρ_rc: {np.mean([s['rho_rc'] for s in summaries]):+.3f}")

    # Save for later plotting
    out_dir = os.path.join(_SCRIPT_DIR, "attention_diagnostic_20260609")
    with open(os.path.join(out_dir, "selector_cross_ckpt_random.json"), "w") as f:
        json.dump(summaries, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else str(x))
    print(f"\nSaved to {out_dir}/selector_cross_ckpt_random.json")
    print("Done.")


if __name__ == "__main__":
    main()
