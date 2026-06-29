"""
Diagnostic: measure NLL distribution across random N16 orders per sequence.

Goal: determine if there's a learnable signal — do different orders produce
meaningfully different NLLs for the same sequence?
"""

import os, sys, time
import argparse
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AO-GPT-MDM"))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import GPT2TokenizerFast

from config import Config
from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig

# ── Paths ──
AO_GPT_CKPT = os.path.expanduser(
    "~/ych/nanogpt-learned-order/out/base/permute/seq256/block64/"
    "out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt"
)
WIKITEXT_DIR = os.path.expanduser(
    "~/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/"
    "b08601e04326c79dfdd32d625aee71d232d685c3"
)
TOKENIZER_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)

SEQ_LEN = 256
N16 = 16
SUB_BLOCKS = 4
BLOCK_LEN = 4
N64 = N16 * SUB_BLOCKS


def block_order_to_token_order_np(block_order, block_len=BLOCK_LEN):
    N = len(block_order)
    token_order = np.zeros(N * block_len, dtype=np.int64)
    for i, blk in enumerate(block_order):
        for k in range(block_len):
            token_order[i * block_len + k] = blk * block_len + k
    return token_order


def phys_to_model_idx(idx_phys, inv_perm):
    B, T = idx_phys.shape
    M = len(inv_perm)
    blk_size = T // M
    device = idx_phys.device
    idx_model = torch.zeros_like(idx_phys)
    for s in range(T):
        model_block = inv_perm[s // blk_size].item()
        offset = s % blk_size
        model_pos = model_block * blk_size + offset
        idx_model[:, model_pos] = idx_phys[:, s]
    return idx_model


def load_wikitext_sequences(min_len=SEQ_LEN, max_count=None):
    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    tokens_all = []
    arrow_file = os.path.join(WIKITEXT_DIR, "wikitext-test.arrow")
    ds = Dataset.from_file(arrow_file)
    for ex in ds:
        raw = tok.encode(ex["text"])
        if len(raw) >= min_len:
            tokens_all.append(raw[:min_len])
            if max_count and len(tokens_all) >= max_count:
                break
    return torch.tensor(tokens_all, dtype=torch.long)


def load_aogpt(ckpt_path, device):
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
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    block_perm = torch.tensor(ckpt["data_permutation"]["block_perm"], dtype=torch.long)
    inv_perm = torch.tensor(ckpt["data_permutation"]["inverse_block_perm"], dtype=torch.long)
    return model, block_perm, inv_perm


@torch.no_grad()
def compute_per_seq_nll(model, idx, orders):
    B, T = idx.shape
    device = idx.device
    pos = torch.arange(0, T + 1, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, T)

    tok_emb = model.transformer.wte(idx)
    tok_emb = tok_emb[batch_indices, orders]
    none_emb = model.transformer.wnonee(
        torch.tensor([[0]], device=device)
    ).expand(B, -1, -1)
    tok_emb = torch.cat([none_emb, tok_emb], dim=1)

    pos_emb = model.transformer.wpe(pos).unsqueeze(0).expand(B, -1, -1)
    pos_emb_prefix = pos_emb[:, :1, :]
    pos_emb_postfix = pos_emb[:, 1:, :][batch_indices, orders]
    pos_emb_final = torch.cat([pos_emb_prefix, pos_emb_postfix], dim=1)

    tgt_emb = model.transformer.wtpe(pos[:T]).unsqueeze(0).expand(B, -1, -1)
    tgt_emb_prefix = tgt_emb[batch_indices, orders]
    tgt_emb_postfix = torch.zeros(B, 1, tgt_emb.shape[-1], device=device)
    c = torch.cat([tgt_emb_prefix, tgt_emb_postfix], dim=1)

    targets = idx[batch_indices, orders]
    x = tok_emb + pos_emb_final
    x = model.transformer.drop(x)
    for block in model.transformer.h:
        x = block(x, c)
    x = model.transformer.final_layer(x, c)

    logits = model.lm_head(x)[:, :-1, :].contiguous()
    ce_per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        ignore_index=-1,
    ).reshape(B, T)
    return ce_per_token.mean(dim=1)


def phys_n16_to_token_order(n16_order, block_perm):
    B, N = n16_order.shape
    T = N * SUB_BLOCKS * BLOCK_LEN
    device = n16_order.device
    bp = block_perm.to(device)
    token_order = torch.zeros(B, T, dtype=torch.long, device=device)
    for t in range(N):
        phys_n16_blk = n16_order[:, t]
        for a in range(SUB_BLOCKS):
            phys_n64_blk = phys_n16_blk * SUB_BLOCKS + a
            model_n64_blk = bp[phys_n64_blk]
            for k in range(BLOCK_LEN):
                token_order[:, t * SUB_BLOCKS * BLOCK_LEN + a * BLOCK_LEN + k] = (
                    model_n64_blk * BLOCK_LEN + k
                )
    return token_order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-matrices", default="probe_results/A_n16_direct_100x5.npy")
    parser.add_argument("--n-sequences", type=int, default=50,
                        help="Number of sequences to analyze")
    parser.add_argument("--n-orders", type=int, default=64,
                        help="Random N16 orders per sequence")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--ao-gpt-ckpt", default=AO_GPT_CKPT)
    args = parser.parse_args()

    device = torch.device(args.device)
    Config.set_seed()

    # Load data
    print(f"Loading A matrices from {args.a_matrices}...")
    A_all = np.load(args.a_matrices)
    A_all = A_all[:args.n_sequences]
    print(f"  {A_all.shape[0]} sequences, shape {A_all.shape}")

    print("Loading token sequences...")
    idx_phys = load_wikitext_sequences(max_count=args.n_sequences)
    idx_phys = idx_phys[:args.n_sequences]
    print(f"  {idx_phys.shape}")

    # Load AO-GPT
    print("Loading AO-GPT...")
    aogpt, block_perm, inv_perm = load_aogpt(args.ao_gpt_ckpt, device)
    print("  Done.")

    # Convert to model coordinates
    idx_model = []
    for i in range(idx_phys.shape[0]):
        idx_model.append(phys_to_model_idx(idx_phys[i:i+1], inv_perm))
    idx_all = torch.cat(idx_model, dim=0).to(device)

    # Generate random orders and score
    n_seqs = args.n_sequences
    n_orders = args.n_orders
    M = n_orders

    all_nlls = np.zeros((n_seqs, M), dtype=np.float32)

    print(f"\nScoring {M} random N16 orders × {n_seqs} sequences...")
    print(f"  Total forward passes: {n_seqs * M}")
    t_start = time.time()

    for s in range(n_seqs):
        if s % 10 == 0:
            print(f"  Sequence {s}/{n_seqs}...")

        # Generate all random orders for this sequence at once
        rand_orders = torch.stack([
            torch.randperm(N16, device=device) for _ in range(M)
        ])  # (M, 16)
        token_orders = phys_n16_to_token_order(rand_orders, block_perm)  # (M, 256)

        # Score in micro-batches
        idx_s = idx_all[s:s+1]  # (1, 256)
        nlls_s = []
        micro_batch = 32
        for mb in range(0, M, micro_batch):
            mb_end = min(mb + micro_batch, M)
            idx_mb = idx_s.expand(mb_end - mb, -1)
            tok_mb = token_orders[mb:mb_end]
            nll_mb = compute_per_seq_nll(aogpt, idx_mb, tok_mb)
            nlls_s.append(nll_mb.cpu().numpy())
        all_nlls[s] = np.concatenate(nlls_s)

    elapsed = time.time() - t_start
    print(f"  Done in {elapsed:.0f}s ({elapsed/n_seqs/M:.2f}s per forward)")

    # ── Analysis ──
    print(f"\n{'='*70}")
    print(f"RESULTS: {n_seqs} seqs × {M} random N16 orders")
    print(f"{'='*70}")

    # Per-sequence stats
    per_seq_mean = all_nlls.mean(axis=1)
    per_seq_std = all_nlls.std(axis=1)
    per_seq_min = all_nlls.min(axis=1)
    per_seq_max = all_nlls.max(axis=1)
    per_seq_range = per_seq_max - per_seq_min  # span across M random orders

    print(f"\n--- Per-sequence NLL distribution (across {M} random orders) ---")
    print(f"  Mean NLL:           {per_seq_mean.mean():.4f} ± {per_seq_mean.std():.4f} (across seqs)")
    print(f"  Within-seq std:     {per_seq_std.mean():.4f} ± {per_seq_std.std():.4f}")
    print(f"  Within-seq range:   {per_seq_range.mean():.4f} ± {per_seq_range.std():.4f}")
    print(f"  Min NLL per seq:    {per_seq_min.mean():.4f}")
    print(f"  Max NLL per seq:    {per_seq_max.mean():.4f}")

    # SNR: across-seq variance / within-seq variance
    across_var = per_seq_mean.var()
    within_var = (all_nlls.var(axis=1)).mean()
    snr = across_var / (within_var + 1e-8)
    print(f"\n--- SNR (signal-to-noise) ---")
    print(f"  Across-seq variance: {across_var:.6f}")
    print(f"  Within-seq variance: {within_var:.6f}")
    print(f"  SNR (across/within): {snr:.2f}")
    if snr < 1:
        print(f"  >> Within-seq NLL variance > across-seq: which ORDER matters more than which SEQUENCE")
        print(f"  >> This means there IS a learnable per-sequence ordering signal.")
    else:
        print(f"  >> Across-seq NLL variance > within-seq: which SEQUENCE matters more than which ORDER")
        print(f"  >> Ordering signal is weak relative to sequence difficulty.")

    # Best order vs baseline (mean of random)
    best_vs_mean = per_seq_mean - per_seq_min  # improvement of best over mean
    print(f"\n--- Achievable gain (best random order vs random mean) ---")
    print(f"  Mean improvement:   {best_vs_mean.mean():.4f} NLL")
    print(f"  Max improvement:    {best_vs_mean.max():.4f} NLL")
    print(f"  Min improvement:    {best_vs_mean.min():.4f} NLL")
    print(f"  Fraction seqs with improvement > 0.02: {(best_vs_mean > 0.02).mean():.2%}")
    print(f"  Fraction seqs with improvement > 0.05: {(best_vs_mean > 0.05).mean():.2%}")

    # Best vs worst gap
    print(f"\n--- Best-worst gap (upper bound of what GRPO can achieve) ---")
    print(f"  Mean gap:           {per_seq_range.mean():.4f}")
    print(f"  Fraction seqs gap > 0.05:  {(per_seq_range > 0.05).mean():.2%}")
    print(f"  Fraction seqs gap > 0.10:  {(per_seq_range > 0.10).mean():.2%}")

    # Correlation: A matrix structure vs optimal order
    print(f"\n--- Top/bottom order analysis ---")
    # For each seq, find the top-5 and bottom-5 orders by NLL
    top_orders = []
    bot_orders = []
    for s in range(n_seqs):
        idx_sorted = np.argsort(all_nlls[s])
        top_orders.append(idx_sorted[:5])
        bot_orders.append(idx_sorted[-5:])

    # How often does the same random index appear in top across sequences?
    # (This tests if there's a universal good ordering)
    top_indices = np.concatenate([all_nlls[s][top_orders[s]] for s in range(n_seqs)])
    bot_indices = np.concatenate([all_nlls[s][bot_orders[s]] for s in range(n_seqs)])
    print(f"  Top-5 mean NLL:     {top_indices.mean():.4f}")
    print(f"  Bottom-5 mean NLL:  {bot_indices.mean():.4f}")
    print(f"  Top/Bottom gap:     {bot_indices.mean() - top_indices.mean():.4f}")

    # Save results
    if args.output:
        np.savez(args.output,
                 nlls=all_nlls,
                 per_seq_mean=per_seq_mean,
                 per_seq_std=per_seq_std,
                 per_seq_min=per_seq_min,
                 per_seq_max=per_seq_max,
                 within_var=within_var,
                 across_var=across_var,
                 args=vars(args))
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
