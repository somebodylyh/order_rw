"""Real-model equivalence + speedup check for the batched per-head extractor.

Loads a real clean_base checkpoint, runs the verbatim batch=1 loop reference
and the batched extractor on the SAME chunks, and reports:
  - max abs diff on A_lh / A_heavy (expected ~fp noise: matmul kernels differ
    across batch sizes, so this is closeness, not bit-equality);
  - whether the downstream per-head tau ranking is unchanged (the only thing
    selection cares about);
  - wall-clock speedup.

Usage:
  python scripts/verify_batched_extractor.py --ckpt <ckpt.pt> --total 64 \
      --fwd-batch 64 --device cuda:0
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch

_PKG = pathlib.Path(__file__).resolve().parent.parent / "block_lo_arm_order_network"
sys.path.insert(0, str(_PKG))

from neural_readout.extract_b import _load_model_and_chunks  # noqa: E402
import per_head_order_scan as phs  # noqa: E402


def _tau_ranking(A_lh, M, batch_size, alpha_dep=0.5):
    """Per-head tau_vs_l2r, sorted head list (layer,head) by |tau| desc."""
    L, H = A_lh.shape[1], A_lh.shape[2]
    rows = []
    for l in range(L):
        for h in range(H):
            B = phs._batch_mean_B(A_lh[:, l, h], M, batch_size)
            sig = phs._orders_from_graphs(B, alpha_dep)
            tau = phs._mean_tau_vs(sig, np.arange(phs.N))
            rows.append((l, h, tau))
    rows.sort(key=lambda r: abs(r[2]), reverse=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--total", type=int, default=64, help="M*batch_size chunks")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--fwd-batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--atol", type=float, default=1e-4)
    args = p.parse_args()

    M = max(1, args.total // args.batch_size)
    total = M * args.batch_size
    print(f"[verify] ckpt={pathlib.Path(args.ckpt).name} total={total} "
          f"(M={M} x bs={args.batch_size}) fwd_batch={args.fwd_batch} dev={args.device}")

    model, chunks, clean_perm, dev, _ = _load_model_and_chunks(
        args.ckpt, total, args.seed, args.device, "train")

    t0 = time.perf_counter()
    A_lh_ref, A_heavy_ref = phs._extract_per_head_and_heavy_A_loop(
        model, chunks, clean_perm, dev, args.seed)
    t_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    A_lh_b, A_heavy_b = phs.extract_per_head_and_heavy_A(
        model, chunks, clean_perm, dev, args.seed, fwd_batch=args.fwd_batch)
    t_batched = time.perf_counter() - t0

    d_lh = float(np.max(np.abs(A_lh_b - A_lh_ref)))
    d_heavy = float(np.max(np.abs(A_heavy_b - A_heavy_ref)))
    print(f"[verify] max|dA_lh|={d_lh:.2e}  max|dA_heavy|={d_heavy:.2e}  (atol={args.atol})")
    print(f"[verify] timing: loop={t_loop:.2f}s  batched={t_batched:.2f}s  "
          f"speedup={t_loop / t_batched:.1f}x")

    rank_ref = _tau_ranking(A_lh_ref, M, args.batch_size)
    rank_b = _tau_ranking(A_lh_b, M, args.batch_size)
    heads_ref = [(l, h) for (l, h, _t) in rank_ref]
    heads_b = [(l, h) for (l, h, _t) in rank_b]
    top1_same = heads_ref[0] == heads_b[0]
    top3_same = set(heads_ref[:3]) == set(heads_b[:3])
    print(f"[verify] tau-rank top1 same={top1_same} ({heads_ref[0]} vs {heads_b[0]}); "
          f"top3 set same={top3_same}")
    print(f"[verify]   loop  top3: {rank_ref[:3]}")
    print(f"[verify]   batch top3: {rank_b[:3]}")

    ok = (d_lh <= args.atol and d_heavy <= args.atol and top1_same)
    print(f"[verify] RESULT: {'PASS' if ok else 'FAIL'} "
          f"(numerically close AND top-head ranking preserved)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
