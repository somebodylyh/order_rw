"""Cross-seed, cross-checkpoint validation of unsupervised head selector.

Tests: does S_drift/S_forward consistently identify the oracle-best CDL head
without L2R labels, across seeds and training steps?

Checkpoints:
  seed2 @ 10k, 50k  (random baseline)
  seed1 @ 10k       (random baseline)
  shuffled-L2R @ 50k
  ori-L2R @ 50k
"""
import sys, os, time
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AOGPT_DIR = os.path.join(_SCRIPT_DIR, "..", "block_lo_arm_order_network")
sys.path.insert(0, _AOGPT_DIR)

from attn_order_teacher import rollout_order
from per_head_order_scan import extract_per_head_and_heavy_A
from neural_readout.extract_b import _load_model_and_chunks

N_BLOCKS = 64; L2R = np.arange(N_BLOCKS); TAU_T = 1.0; M = 200


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


def compute_unsupervised_signals(A_lh_head):
    """Given (M, N, N) A matrices for one head, return:
    tau_oracle, row_conc, S_forward, S_drift
    """
    M = A_lh_head.shape[0]
    # Oracle: batch-mean B → CDL → τ vs L2R
    B_mean = _A_to_B(A_lh_head.mean(axis=0))
    o = rollout_order(B_mean, tau_T=TAU_T, seed=42, mode="C-D+L", standardize=True)
    tau_oracle = kendall_tau(o, L2R)

    # row_conc
    Bm = np.asarray(B_mean, dtype=np.float64)
    N = Bm.shape[0]
    concs = []
    for i in range(N):
        row = np.abs(Bm[i].copy()); s = row.sum()
        if s < 1e-12: concs.append(0.0)
        else:
            p = row/s; p = p[p>1e-12]
            H = -np.sum(p*np.log(p))
            concs.append(1.0 - H/max(np.log(N-1), 1e-12))
    row_conc = float(np.mean(concs))

    # S_forward: split-half at two different seeds
    if M < 20:
        return tau_oracle, row_conc, 0.0, 0.0
    mid = M//2
    rng = np.random.default_rng(42)
    idx1 = rng.permutation(M)
    idx2 = np.random.default_rng(99).permutation(M)

    def split_tau(idx):
        B1 = _A_to_B(np.mean([_A_to_B(A_lh_head[i]) for i in idx[:mid]], axis=0))
        B2 = _A_to_B(np.mean([_A_to_B(A_lh_head[i]) for i in idx[mid:2*mid]], axis=0))
        o1 = rollout_order(B1, tau_T=TAU_T, seed=42, mode="C-D+L", standardize=True)
        o2 = rollout_order(B2, tau_T=TAU_T, seed=43, mode="C-D+L", standardize=True)
        return kendall_tau(o1, o2)

    s_forward = split_tau(idx1)
    s_drift = split_tau(idx2)
    return tau_oracle, row_conc, s_forward, s_drift


def scan_checkpoint(ckpt_path, label, device="cuda:0"):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    model, chunks, clean_perm, dev, ci = _load_model_and_chunks(
        ckpt_path=ckpt_path, M=M, seed=42, device=device, split="train")
    A_lh, _ = extract_per_head_and_heavy_A(
        model, chunks, clean_perm, dev, seed=42, fwd_batch=64, none_mode="b0")
    n, L, H, N, _ = A_lh.shape
    print(f"  {L}x{H} heads in {time.time()-t0:.1f}s")

    results = []
    for l in range(L):
        for h in range(H):
            tau_o, rc, sf, sd = compute_unsupervised_signals(A_lh[:, l, h])
            results.append({"head": f"L{l}H{h}", "l": l, "h": h,
                            "tau_oracle": tau_o, "row_conc": rc,
                            "s_forward": sf, "s_drift": sd})

    del model, A_lh
    torch = __import__("torch")
    if device != "cpu": torch.cuda.empty_cache()

    # Rankings
    oracle_best = max(results, key=lambda x: x["tau_oracle"])
    s_drift_best = max(results, key=lambda x: x["s_drift"])
    s_fwd_best = max(results, key=lambda x: x["s_forward"])
    row_conc_best = max(results, key=lambda x: x["row_conc"])

    # Spearman ρ
    from scipy.stats import spearmanr
    oracle_t = np.array([r["tau_oracle"] for r in results])
    sdrift_t = np.array([r["s_drift"] for r in results])
    sfwd_t = np.array([r["s_forward"] for r in results])
    rc_t = np.array([r["row_conc"] for r in results])

    rho_sd, p_sd = spearmanr(oracle_t, sdrift_t)
    rho_sf, p_sf = spearmanr(oracle_t, sfwd_t)
    rho_rc, p_rc = spearmanr(oracle_t, rc_t)

    # Top-1 correctness
    sd_correct = (s_drift_best["head"] == oracle_best["head"])
    sf_correct = (s_fwd_best["head"] == oracle_best["head"])
    rc_correct = (row_conc_best["head"] == oracle_best["head"])

    print(f"  Oracle best:   {oracle_best['head']}  τ={oracle_best['tau_oracle']:+.4f}")
    print(f"  S_drift best:  {s_drift_best['head']}  sd={s_drift_best['s_drift']:+.4f}  {'✓' if sd_correct else '✗'}")
    print(f"  S_forward best:{s_fwd_best['head']}  sf={s_fwd_best['s_forward']:+.4f}  {'✓' if sf_correct else '✗'}")
    print(f"  row_conc best: {row_conc_best['head']}  rc={row_conc_best['row_conc']:+.4f}  {'✓' if rc_correct else '✗'}")
    print(f"  ρ(S_drift,τ)={rho_sd:+.3f} p={p_sd:.3f}  ρ(S_fwd,τ)={rho_sf:+.3f} p={p_sf:.3f}  ρ(row_conc,τ)={rho_rc:+.3f} p={p_rc:.3f}")

    return {
        "label": label,
        "oracle_best": oracle_best["head"], "oracle_tau": oracle_best["tau_oracle"],
        "s_drift_best": s_drift_best["head"], "s_drift_correct": sd_correct,
        "s_forward_best": s_fwd_best["head"], "s_forward_correct": sf_correct,
        "row_conc_best": row_conc_best["head"], "row_conc_correct": rc_correct,
        "rho_sd": rho_sd, "rho_sf": rho_sf, "rho_rc": rho_rc,
        "p_sd": p_sd, "p_sf": p_sf, "p_rc": p_rc,
        "all_heads": results,
    }


def main():
    device = "cuda:0"
    BASE = "/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/probe_results"

    ckpts = [
        (f"{BASE}/random_baseline_continuous_jun08_seed2/ckpt_step10000.pt", "seed2 @ 10k (random)"),
        (f"{BASE}/random_baseline_continuous_jun08_seed2/ckpt_step50000.pt", "seed2 @ 50k (random)"),
        (f"{BASE}/random_baseline_continuous_jun09_seed1/ckpt_step10000.pt", "seed1 @ 10k (random)"),
        (f"{BASE}/shuffled_l2r_continuous_jun05/ckpt_step50000.pt",   "seed2 @ 50k (shuffled-L2R)"),
        (f"{BASE}/l2r_continuous_jun05/ckpt_step50000.pt",            "seed2 @ 50k (ori-L2R)"),
    ]

    all_summaries = []
    for ckpt, label in ckpts:
        summary = scan_checkpoint(ckpt, label, device)
        all_summaries.append(summary)

    # ── Final table ────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  CROSS-CHECKPOINT SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Checkpoint':<28s} {'Oracle':>6s} {'S_drift':>8s} {'S_fwd':>8s} {'row_conc':>10s} {'ρ_sd':>7s} {'ρ_sf':>7s} {'ρ_rc':>7s}")
    print(f"  {'-'*28} {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*7} {'-'*7} {'-'*7}")
    for s in all_summaries:
        sd = "✓" if s["s_drift_correct"] else "✗"
        sf = "✓" if s["s_forward_correct"] else "✗"
        rc = "✓" if s["row_conc_correct"] else "✗"
        print(f"  {s['label']:<28s} {s['oracle_best']:>6s} {s['s_drift_best']+' '+sd:>8s} "
              f"{s['s_forward_best']+' '+sf:>8s} {s['row_conc_best']+' '+rc:>10s} "
              f"{s['rho_sd']:>+7.3f} {s['rho_sf']:>+7.3f} {s['rho_rc']:>+7.3f}")

    # Aggregate stats
    sd_correct = sum(1 for s in all_summaries if s["s_drift_correct"])
    sf_correct = sum(1 for s in all_summaries if s["s_forward_correct"])
    rc_correct = sum(1 for s in all_summaries if s["row_conc_correct"])
    n = len(all_summaries)
    print(f"\n  Top-1 accuracy: S_drift={sd_correct}/{n}  S_forward={sf_correct}/{n}  row_conc={rc_correct}/{n}")

    # Mean ρ
    print(f"  Mean ρ: S_drift={np.mean([s['rho_sd'] for s in all_summaries]):+.3f}  "
          f"S_forward={np.mean([s['rho_sf'] for s in all_summaries]):+.3f}  "
          f"row_conc={np.mean([s['rho_rc'] for s in all_summaries]):+.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
