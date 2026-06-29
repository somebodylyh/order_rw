"""Compare two (N, 16, 16) A-matrix arrays. Used to decide if AOGPT's
attention pattern shifted enough after Round 0 to warrant Round 1.

Metrics:
  - Frobenius norm: ||A_after - A_before||_F (per-seq, then mean)
  - Relative Frobenius: ||delta||_F / ||A_before||_F
  - Top-3 row argmax overlap (per row, per seq, then mean Jaccard)
  - Row-distribution KL: mean over rows and seqs of KL(A_before_row || A_after_row)

Decision rule (printed at end):
  rel_frob > 0.05 AND top3_overlap < 0.7  →  GO Round 1  (exit 0)
  else                                     →  STOP, debug  (exit 2)
"""
import argparse
import numpy as np


def frobenius_diff(A_before, A_after):
    delta = A_after - A_before
    per_seq = np.sqrt(np.sum(delta ** 2, axis=(1, 2)))         # (N,)
    base = np.sqrt(np.sum(A_before ** 2, axis=(1, 2)))          # (N,)
    rel = per_seq / np.maximum(base, 1e-9)
    return float(per_seq.mean()), float(rel.mean())


def top3_row_overlap(A_before, A_after):
    """For each (seq, row), compute |top3_before ∩ top3_after| / 3, then mean."""
    top3_b = np.argsort(-A_before, axis=-1)[..., :3]            # (N, 16, 3)
    top3_a = np.argsort(-A_after, axis=-1)[..., :3]
    N, R, _ = top3_b.shape
    overlaps = []
    for n in range(N):
        for r in range(R):
            inter = len(set(top3_b[n, r].tolist()) & set(top3_a[n, r].tolist()))
            overlaps.append(inter / 3.0)
    return float(np.mean(overlaps))


def row_kl(A_before, A_after, eps=1e-8):
    """Mean KL(P_before || P_after) over rows and seqs.
    A is treated as already row-stochastic (it's softmaxed attention)."""
    P = np.clip(A_before, eps, 1.0)
    P = P / P.sum(axis=-1, keepdims=True)
    Q = np.clip(A_after, eps, 1.0)
    Q = Q / Q.sum(axis=-1, keepdims=True)
    kl = np.sum(P * (np.log(P) - np.log(Q)), axis=-1)            # (N, 16)
    return float(kl.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="path to A_before.npy")
    ap.add_argument("--after", required=True, help="path to A_after.npy")
    ap.add_argument("--rel-frob-threshold", type=float, default=0.05)
    ap.add_argument("--top3-threshold", type=float, default=0.7)
    args = ap.parse_args()

    A_b = np.load(args.before).astype(np.float32)
    A_a = np.load(args.after).astype(np.float32)

    n = min(A_b.shape[0], A_a.shape[0])
    if A_b.shape[0] != A_a.shape[0]:
        print(f"[warn] shape mismatch: before={A_b.shape}, after={A_a.shape}; "
              f"truncating to {n}")
        A_b, A_a = A_b[:n], A_a[:n]

    abs_frob, rel_frob = frobenius_diff(A_b, A_a)
    top3 = top3_row_overlap(A_b, A_a)
    kl = row_kl(A_b, A_a)

    print(f"A diff report ({n} seqs)")
    print(f"  Frobenius (mean per-seq): {abs_frob:.4f}")
    print(f"  Relative Frobenius:        {rel_frob:.4f}  (threshold > {args.rel_frob_threshold})")
    print(f"  Top-3 row argmax overlap:  {top3:.4f}  (threshold < {args.top3_threshold})")
    print(f"  Mean row KL(before||after):{kl:.4f}")

    go = rel_frob > args.rel_frob_threshold and top3 < args.top3_threshold
    print()
    print(f"DECISION: {'GO Round 1' if go else 'STOP, debug Round 0'}")
    print(f"  reason: rel_frob={'>' if rel_frob > args.rel_frob_threshold else '<='}thr "
          f"AND top3={'<' if top3 < args.top3_threshold else '>='}thr")
    return 0 if go else 2  # 0=go, 2=stop (so shell can branch)


if __name__ == "__main__":
    import sys
    sys.exit(main())
