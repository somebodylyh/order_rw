"""Frame-controlled per-sample oracle-headroom test (text).

Is there context-dependent reveal-order headroom beyond physical L2R? Frame-correct:
all NLLs via order_nll(sigma_PHYSICAL_block, clean_perm) — the validated mapping
phys-block order -> model-token order. Baseline = TRUE physical L2R = arange(64).

Artifact control = cross-sample order transfer: if sigma*_i (best order for sample
i) only helps sample i (not sample j), the gain is sample-specific (real context-
dependence). If sigma*_i helps everyone, it is a generic better-fixed-order /
easy-first artifact, NOT context-dependence.
"""
import json as _json
import pathlib
import numpy as np
from scipy.stats import kendalltau

from analyses.p5_utility_controller import load_p5_ckpt, order_nll, N


def hill_climb(model, idx_row, clean_perm, dev, n_steps=600, seed=0):
    """Random-swap hill climb over PHYSICAL-block orders, objective = order_nll.
    Starts from physical L2R (arange)."""
    rng = np.random.default_rng(seed)
    best = np.arange(N, dtype=np.int64)
    l2r_nll = order_nll(model, idx_row, best, clean_perm, dev)
    best_nll = l2r_nll
    for _ in range(n_steps):
        cand = best.copy()
        i, j = rng.integers(0, N, size=2)
        cand[i], cand[j] = cand[j], cand[i]
        nll = order_nll(model, idx_row, cand, clean_perm, dev)
        if nll < best_nll:
            best, best_nll = cand, nll
    return best, float(best_nll), float(l2r_nll)


def run_oracle_headroom(ckpt_path, M=8, n_steps=600, out_dir="runs/p6/oracle_headroom",
                        tag="seed123_10k", device="cpu"):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    model, chunks, clean_perm, dev = load_p5_ckpt(ckpt_path, M, device=device)
    l2r = np.arange(N, dtype=np.int64)

    sigmas, nll_best, nll_l2r = [], [], []
    for t in range(M):
        s, nb, nl = hill_climb(model, chunks[t:t+1], clean_perm, dev, n_steps=n_steps,
                               seed=t)
        sigmas.append(s); nll_best.append(nb); nll_l2r.append(nl)
        print(f"  seq{t}: L2R={nl:.4f} best={nb:.4f} gain={nl-nb:+.4f}", flush=True)
    nll_best = np.array(nll_best); nll_l2r = np.array(nll_l2r)

    # cross-transfer matrix X[i,j] = NLL of sample i under sigma*_j
    X = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            X[i, j] = (nll_best[i] if i == j else
                       order_nll(model, chunks[i:i+1], sigmas[j], clean_perm, dev))

    own_gain = float(np.mean([nll_l2r[i] - X[i, i] for i in range(M)]))
    off = [nll_l2r[i] - X[i, j] for i in range(M) for j in range(M) if i != j]
    transfer_gain = float(np.mean(off))
    specificity = own_gain - transfer_gain          # >0 => sample-specific

    # frame-correct order similarity (physical-block frame)
    tau_vs_l2r = [float(kendalltau(s, l2r)[0]) for s in sigmas]
    import itertools
    pw = [float(kendalltau(sigmas[i], sigmas[j])[0])
          for i, j in itertools.combinations(range(M), 2)]

    result = {
        "ckpt": ckpt_path, "tag": tag, "M": M, "n_steps": n_steps,
        "per_seq_gain_vs_L2R": [float(nll_l2r[i] - nll_best[i]) for i in range(M)],
        "mean_own_gain_vs_L2R": own_gain,
        "mean_transfer_gain_vs_L2R": transfer_gain,
        "specificity_own_minus_transfer": specificity,
        "tau_best_vs_physL2R_mean": float(np.mean(tau_vs_l2r)),
        "tau_between_best_orders_mean": float(np.mean(pw)),
        "_note": "headroom real & context-dependent IFF own_gain large AND "
                 "specificity>0 (own>>transfer). If own~=transfer => generic better "
                 "fixed order / easy-first artifact, not context-dependence.",
    }
    _json.dump(result, open(out / f"{tag}.json", "w"), indent=2, default=float)
    print(f"\nown_gain={own_gain:+.4f} transfer_gain={transfer_gain:+.4f} "
          f"specificity={specificity:+.4f} | tau_best_vs_L2R={np.mean(tau_vs_l2r):.3f} "
          f"tau_between_best={np.mean(pw):.3f}", flush=True)
    print("ORACLE HEADROOM DONE", flush=True)
    return result


if __name__ == "__main__":
    import sys
    ck = sys.argv[1] if len(sys.argv) > 1 else \
        "runs/handoff_overnight/seed123/ckpt_step10000.pt"
    tag = sys.argv[2] if len(sys.argv) > 2 else "seed123_10k"
    ns = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    run_oracle_headroom(ck, n_steps=ns, tag=tag)
