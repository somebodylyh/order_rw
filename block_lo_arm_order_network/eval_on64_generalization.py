"""
Evaluate ON64 generalization on OOD wikitext-103 chunks.

Skips the first 10000 chunks (used by ON training data) and evaluates the next
1000 chunks. For each chunk:
  1. AOGPT forward to extract physical-coord A64
  2. ON64 greedy decode -> block order
  3. Kendall tau against ascending l2r [0..63]

Usage:
    python -u eval_on64_generalization.py \
        --on-ckpt probe_results/on64_from_50k_nn10k/grpo_on64_from_50k_nn10k.bc.pt \
        --skip 10000 --n-chunks 1000 --device cuda:0
"""

import os
import sys
import time
import argparse

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


def load_chunks(skip, n_chunks):
    """Load wikitext-103 chunks [skip, skip + n_chunks)."""
    from datasets import Dataset
    from transformers import GPT2TokenizerFast

    TOKENIZER_DIR = os.path.expanduser(
        "~/.cache/huggingface/hub/models--gpt2/snapshots/"
        "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    )
    WIKITEXT_CACHE = os.path.expanduser(
        "~/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/"
        "b08601e04326c79dfdd32d625aee71d232d685c3"
    )
    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)

    target = skip + n_chunks
    chunks = []
    buffer_ids = []
    print(f"Chunking wikitext-103 to reach {target} chunks (skip={skip}, "
          f"need={n_chunks})...", flush=True)
    for shard in [
        "wikitext-train-00000-of-00002.arrow",
        "wikitext-train-00001-of-00002.arrow",
    ]:
        ds = Dataset.from_file(os.path.join(WIKITEXT_CACHE, shard))
        for ex in tqdm(ds, desc=f"Chunking {shard}", unit=" texts"):
            ids = tok.encode(ex["text"])
            if len(ids) < SEQ_LEN:
                buffer_ids.extend(ids)
                while len(buffer_ids) >= SEQ_LEN:
                    chunks.append(buffer_ids[:SEQ_LEN])
                    buffer_ids = buffer_ids[SEQ_LEN:]
                    if len(chunks) >= target:
                        break
            else:
                for start in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN // 2):
                    chunks.append(ids[start:start + SEQ_LEN])
                    if len(chunks) >= target:
                        break
            if len(chunks) >= target:
                break
        if len(chunks) >= target:
            break
    return chunks[skip:target]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--on-ckpt", required=True)
    parser.add_argument("--aogpt-ckpt", default=AO_GPT_CKPT)
    parser.add_argument("--skip", type=int, default=10000,
                        help="Skip first N chunks (ON training set size)")
    parser.add_argument("--n-chunks", type=int, default=1000)
    parser.add_argument("--n-blocks", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy (argmax)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-orders", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    N = args.n_blocks

    print(f"Device: {device}, n_blocks={N}, skip={args.skip}, "
          f"n_chunks={args.n_chunks}, temp={args.temperature}", flush=True)

    print("Loading AO-GPT...", flush=True)
    aogpt, bp, _ = load_aogpt(args.aogpt_ckpt, device)
    model_to_phys = build_model_to_phys(bp)
    print(f"  block_perm[:8]={bp[:8].tolist()}", flush=True)

    print("Loading ON64...", flush=True)
    on = load_on(args.on_ckpt, device, num_blocks=N)
    print(f"  ON params: {sum(p.numel() for p in on.parameters()):,}", flush=True)

    chunks = load_chunks(args.skip, args.n_chunks)
    print(f"Loaded {len(chunks)} OOD chunks", flush=True)

    l2r = np.arange(N)
    taus = []
    orders_out = []
    t0 = time.time()

    for i in tqdm(range(len(chunks)), desc="Eval"):
        tokens = torch.tensor(chunks[i], dtype=torch.long)
        A = extract_one_A(aogpt, tokens, bp, model_to_phys, device, n_blocks=N)
        A_t = torch.from_numpy(A).float().unsqueeze(0).to(device)  # (1, N, N)
        order = sample_on_order(on, A_t, temperature=args.temperature)
        order_np = order[0].cpu().numpy()
        tau, _ = kendalltau(order_np, l2r)
        taus.append(tau)
        orders_out.append(order_np)

    taus = np.asarray(taus)
    elapsed = time.time() - t0
    print(f"\n=== Generalization (skip={args.skip}, n={len(chunks)}, "
          f"temp={args.temperature}) ===", flush=True)
    print(f"Kendall tau vs l2r [0..{N-1}]:", flush=True)
    print(f"  mean: {taus.mean():+.4f}", flush=True)
    print(f"  std:  {taus.std():.4f}", flush=True)
    print(f"  median: {np.median(taus):+.4f}", flush=True)
    print(f"  range: [{taus.min():+.4f}, {taus.max():+.4f}]", flush=True)
    print(f"  pct >0: {(taus > 0).mean() * 100:.1f}% "
          f"(positive correlation with l2r)", flush=True)
    print(f"  pct >0.5: {(taus > 0.5).mean() * 100:.1f}%", flush=True)
    print(f"Elapsed: {elapsed:.0f}s", flush=True)

    if args.save_orders:
        np.savez(args.save_orders, orders=np.stack(orders_out), taus=taus)
        print(f"Saved orders + taus -> {args.save_orders}", flush=True)


if __name__ == "__main__":
    main()
