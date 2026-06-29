"""
Quantify A-matrix drift between 50k AOGPT and the continued-training ckpt.

Steps:
  1. Load 50k AOGPT (frozen reference for model_args + data_permutation).
  2. Load continued-training ckpt's state_dict into a fresh AOGPT (60k weights).
  3. For first N_CHUNKS wikitext chunks:
       - extract A_50k and A_60k (same chunks, same block_perm)
       - record both
  4. Drift metrics:
       - rel_frob: ||A_60k - A_50k||_F / ||A_50k||_F   (per chunk + mean)
       - top3_overlap: per-row top-3 set overlap (per chunk + mean)
  5. ON sensitivity:
       - run ON greedy on A_50k -> order_50
       - run ON greedy on A_60k -> order_60
       - kendall tau(order_50, order_60) per chunk + mean

Usage:
    python -u drift_eval.py \
        --ckpt-50k <path> --ckpt-60k <path> \
        --on-ckpt <path> --n-chunks 200 --device cuda:0
"""

import os
import sys
import time
import argparse
from copy import deepcopy

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from scipy.stats import kendalltau
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AO-GPT-MDM"))

from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig

from extract_train_A import (
    AO_GPT_CKPT,
    SEQ_LEN,
    extract_one_A,
    build_model_to_phys,
)
from train_on_mixed import load_on, sample_on_order
from eval_on64_generalization import load_chunks


def load_aogpt_ref(ckpt_path, device):
    """Load 50k reference: returns model, block_perm, model_args, sd."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid = {k: v for k, v in dict(ckpt["model_args"]).items() if k in sig}
    model = AOGPT(AOGPTConfig(**valid))
    sd = ckpt["model"]
    for k in list(sd.keys()):
        clean = k.replace("_orig_mod.", "")
        if clean != k:
            sd[clean] = sd.pop(k)
    model.load_state_dict(sd)
    model.crop_block_size(SEQ_LEN)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    bp = np.array(ckpt["data_permutation"]["block_perm"], dtype=np.int64)
    return model, bp, valid


def load_continued_aogpt(ckpt_path, model_args, device):
    """Load step-3500 ckpt's weights into a fresh AOGPT with given args."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AOGPT(AOGPTConfig(**model_args))
    sd = ckpt["model"]
    for k in list(sd.keys()):
        clean = k.replace("_orig_mod.", "")
        if clean != k:
            sd[clean] = sd.pop(k)
    model.load_state_dict(sd)
    model.crop_block_size(SEQ_LEN)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def top_k_overlap(A1, A2, k=3):
    """Per-row top-k index set overlap (averaged over rows)."""
    N = A1.shape[0]
    overlaps = []
    for i in range(N):
        s1 = set(np.argsort(-A1[i])[:k].tolist())
        s2 = set(np.argsort(-A2[i])[:k].tolist())
        overlaps.append(len(s1 & s2) / k)
    return float(np.mean(overlaps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-50k", default=AO_GPT_CKPT,
                        help="Reference 50k AOGPT")
    parser.add_argument("--ckpt-60k", required=True,
                        help="Continued-training ckpt (e.g. step 3500 best)")
    parser.add_argument("--on-ckpt", required=True)
    parser.add_argument("--n-chunks", type=int, default=200,
                        help="Number of wikitext chunks to extract A on")
    parser.add_argument("--n-blocks", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-prefix", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    N = args.n_blocks

    print(f"Device: {device}, n_blocks={N}, n_chunks={args.n_chunks}", flush=True)

    print("Loading 50k AOGPT (reference)...", flush=True)
    aogpt50, bp, model_args = load_aogpt_ref(args.ckpt_50k, device)
    model_to_phys = build_model_to_phys(bp)
    print(f"  block_perm[:8]={bp[:8].tolist()}", flush=True)

    print("Loading 60k AOGPT (continued)...", flush=True)
    aogpt60 = load_continued_aogpt(args.ckpt_60k, model_args, device)

    print("Loading ON64...", flush=True)
    on = load_on(args.on_ckpt, device, num_blocks=N)

    chunks = load_chunks(skip=0, n_chunks=args.n_chunks)
    print(f"Loaded {len(chunks)} chunks (in-domain, idx 0..{args.n_chunks-1})",
          flush=True)

    # Use a fixed seed for the random block order inside extract_one_A so both
    # extractions see the same "view"; reset before each chunk.
    rel_frobs = []
    top3_overlaps = []
    top1_overlaps = []
    on_taus = []
    A50_list, A60_list = [], []
    order50_list, order60_list = [], []

    t0 = time.time()
    for i in tqdm(range(len(chunks)), desc="Drift"):
        tokens = torch.tensor(chunks[i], dtype=torch.long)

        # Same random order seed for both extractions
        seed = 1000 + i
        torch.manual_seed(seed)
        A50 = extract_one_A(aogpt50, tokens, bp, model_to_phys, device, n_blocks=N)
        torch.manual_seed(seed)
        A60 = extract_one_A(aogpt60, tokens, bp, model_to_phys, device, n_blocks=N)

        A50_list.append(A50)
        A60_list.append(A60)

        diff_norm = np.linalg.norm(A60 - A50)
        ref_norm = np.linalg.norm(A50) + 1e-12
        rel_frobs.append(diff_norm / ref_norm)
        top3_overlaps.append(top_k_overlap(A50, A60, k=3))
        top1_overlaps.append(top_k_overlap(A50, A60, k=1))

        # ON sensitivity
        A50_t = torch.from_numpy(A50).float().unsqueeze(0).to(device)
        A60_t = torch.from_numpy(A60).float().unsqueeze(0).to(device)
        order50 = sample_on_order(on, A50_t, temperature=0.0)[0].cpu().numpy()
        order60 = sample_on_order(on, A60_t, temperature=0.0)[0].cpu().numpy()
        tau, _ = kendalltau(order50, order60)
        on_taus.append(tau)
        order50_list.append(order50)
        order60_list.append(order60)

    rel = np.asarray(rel_frobs)
    t3 = np.asarray(top3_overlaps)
    t1 = np.asarray(top1_overlaps)
    on_t = np.asarray(on_taus)
    elapsed = time.time() - t0

    print(f"\n=== A drift (50k -> step3500 ckpt, n={len(chunks)} chunks) ===",
          flush=True)
    print(f"rel_frob ||A60-A50||/||A50||:", flush=True)
    print(f"  mean={rel.mean():.4f}, std={rel.std():.4f}, "
          f"median={np.median(rel):.4f}", flush=True)
    print(f"  range=[{rel.min():.4f}, {rel.max():.4f}]", flush=True)
    print(f"top-1 overlap (per-row argmax neighbor preserved):", flush=True)
    print(f"  mean={t1.mean():.4f}", flush=True)
    print(f"top-3 overlap (per-row top-3 set):", flush=True)
    print(f"  mean={t3.mean():.4f}", flush=True)

    print(f"\n=== ON sensitivity to A drift ===", flush=True)
    print(f"Kendall tau(order(A50), order(A60)) (1.0=ON insensitive, 0=ON unstable):",
          flush=True)
    print(f"  mean={on_t.mean():.4f}, std={on_t.std():.4f}, "
          f"median={np.median(on_t):.4f}", flush=True)
    print(f"  range=[{on_t.min():.4f}, {on_t.max():.4f}]", flush=True)
    print(f"  pct >0.9: {(on_t > 0.9).mean()*100:.1f}%", flush=True)
    print(f"  pct >0.7: {(on_t > 0.7).mean()*100:.1f}%", flush=True)
    print(f"  pct >0.5: {(on_t > 0.5).mean()*100:.1f}%", flush=True)
    print(f"\nElapsed: {elapsed:.0f}s", flush=True)

    if args.save_prefix:
        np.savez(
            args.save_prefix + "_drift.npz",
            A50=np.stack(A50_list),
            A60=np.stack(A60_list),
            order50=np.stack(order50_list),
            order60=np.stack(order60_list),
            rel_frob=rel, top3=t3, top1=t1, on_tau=on_t,
        )
        print(f"Saved -> {args.save_prefix}_drift.npz", flush=True)


if __name__ == "__main__":
    main()
