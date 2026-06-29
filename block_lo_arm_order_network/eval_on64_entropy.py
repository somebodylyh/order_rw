"""
Quantify ON64 effective entropy via K-sample intra-chunk diversity.

For each OOD chunk:
  1. Extract A64 once
  2. Sample K orders from ON at given temperature
  3. Compute pairwise Kendall tau among the K orders -> intra-sample tau

intra_tau == 1.0 -> ON is fully deterministic (T=1 == argmax)
intra_tau == 0.0 -> ON is uniformly random
in between       -> effective entropy of ON

Usage:
    python -u eval_on64_entropy.py \
        --on-ckpt probe_results/on64_from_50k_nn10k/grpo_on64_from_50k_nn10k.bc.pt \
        --skip 10000 --n-chunks 200 --k-samples 5 --temperature 1.0 \
        --device cuda:0
"""

import os
import sys
import time
import argparse
from itertools import combinations

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from scipy.stats import kendalltau
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_train_A import (
    AO_GPT_CKPT,
    SEQ_LEN,
    load_aogpt,
    extract_one_A,
    build_model_to_phys,
)
from train_on_mixed import load_on, sample_on_order
from eval_on64_generalization import load_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--on-ckpt", required=True)
    parser.add_argument("--aogpt-ckpt", default=AO_GPT_CKPT)
    parser.add_argument("--skip", type=int, default=10000)
    parser.add_argument("--n-chunks", type=int, default=200)
    parser.add_argument("--k-samples", type=int, default=5)
    parser.add_argument("--n-blocks", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    N = args.n_blocks
    K = args.k_samples

    print(f"Device: {device}, n_blocks={N}, n_chunks={args.n_chunks}, "
          f"K={K}, temp={args.temperature}", flush=True)

    print("Loading AO-GPT...", flush=True)
    aogpt, bp, _ = load_aogpt(args.aogpt_ckpt, device)
    model_to_phys = build_model_to_phys(bp)

    print("Loading ON64...", flush=True)
    on = load_on(args.on_ckpt, device, num_blocks=N)

    chunks = load_chunks(args.skip, args.n_chunks)
    print(f"Loaded {len(chunks)} OOD chunks", flush=True)

    l2r = np.arange(N)
    intra_taus = []   # per-chunk mean pairwise tau among K samples
    l2r_taus = []     # per-chunk mean tau vs l2r over K samples
    unique_counts = []
    t0 = time.time()

    for i in tqdm(range(len(chunks)), desc="Eval"):
        tokens = torch.tensor(chunks[i], dtype=torch.long)
        A = extract_one_A(aogpt, tokens, bp, model_to_phys, device, n_blocks=N)
        A_t = torch.from_numpy(A).float().unsqueeze(0).to(device)
        # Sample K orders by repeating A across batch
        A_batch = A_t.expand(K, -1, -1).contiguous()
        orders = sample_on_order(on, A_batch, temperature=args.temperature)
        orders_np = orders.cpu().numpy()  # (K, N)

        # Pairwise intra-chunk tau
        pair_taus = []
        for a, b in combinations(range(K), 2):
            tau, _ = kendalltau(orders_np[a], orders_np[b])
            pair_taus.append(tau)
        intra_taus.append(np.mean(pair_taus))

        # vs-l2r tau averaged over K
        l2r_pairs = [kendalltau(orders_np[k], l2r)[0] for k in range(K)]
        l2r_taus.append(np.mean(l2r_pairs))

        # unique-order count among K
        order_strs = {tuple(o.tolist()) for o in orders_np}
        unique_counts.append(len(order_strs))

    intra = np.asarray(intra_taus)
    l2rt = np.asarray(l2r_taus)
    uniq = np.asarray(unique_counts)
    elapsed = time.time() - t0

    print(f"\n=== ON64 Entropy (T={args.temperature}, K={K}, n={len(chunks)}) ===",
          flush=True)
    print(f"Intra-chunk pairwise Kendall tau (1.0=deterministic, 0=random):",
          flush=True)
    print(f"  mean:   {intra.mean():.4f}", flush=True)
    print(f"  std:    {intra.std():.4f}", flush=True)
    print(f"  median: {np.median(intra):.4f}", flush=True)
    print(f"  range:  [{intra.min():.4f}, {intra.max():.4f}]", flush=True)
    print(f"  pct >0.99: {(intra > 0.99).mean()*100:.1f}% "
          f"(near-deterministic)", flush=True)
    print(f"  pct >0.95: {(intra > 0.95).mean()*100:.1f}%", flush=True)
    print(f"  pct >0.80: {(intra > 0.80).mean()*100:.1f}%", flush=True)

    print(f"\nUnique orders out of K={K}:", flush=True)
    print(f"  mean: {uniq.mean():.2f}", flush=True)
    print(f"  pct == 1 (all identical): {(uniq == 1).mean()*100:.1f}%",
          flush=True)
    print(f"  pct == K (all distinct):  {(uniq == K).mean()*100:.1f}%",
          flush=True)

    print(f"\nMean tau vs l2r over K samples:", flush=True)
    print(f"  mean: {l2rt.mean():+.4f} (greedy was +0.3531, T=1 single +0.3571)",
          flush=True)

    print(f"\nElapsed: {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
