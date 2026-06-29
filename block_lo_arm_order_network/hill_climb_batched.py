"""
Batched Random-Swap Hill Climbing for 64-block AO-GPT Ordering.

Each step: propose 1 random block-swap per sequence, batch-evaluate all candidates,
accept individually if NLL improves.

Usage:
    # Quick test on 8 val seqs (2 min)
    python -u hill_climb_batched.py --n-seqs 8 --steps 4000

    # Full run on 187 test seqs (~5 min single GPU)
    python -u hill_climb_batched.py --n-seqs 0 --steps 4000

    # Train set (500 seqs) for EM loop
    python -u hill_climb_batched.py --split train --n-seqs 500 --steps 4000
"""

import argparse
import os
import sys
import time
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

from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig

# ── Paths ────────────────────────────────────────────────────────────────────
CKPT_PATH = os.path.expanduser(
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
N_BLOCKS = 64
BLOCK_LEN = 4  # tokens per block (256 / 64)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_sequences(split="test", min_len=SEQ_LEN, max_count=None):
    """Load token sequences from wikitext-103."""
    if split == "test":
        arrow_file = os.path.join(WIKITEXT_DIR, "wikitext-test.arrow")
    elif split == "train":
        # Load both shards
        tokens_all = []
        tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
        for shard in ["wikitext-train-00000-of-00002.arrow", "wikitext-train-00001-of-00002.arrow"]:
            arrow_file = os.path.join(WIKITEXT_DIR, shard)
            ds = Dataset.from_file(arrow_file)
            for ex in ds:
                tokens = tok.encode(ex["text"])
                if len(tokens) >= min_len:
                    tokens_all.append(tokens[:min_len])
                    if max_count and len(tokens_all) >= max_count:
                        break
            if max_count and len(tokens_all) >= max_count:
                break
        return torch.tensor(tokens_all, dtype=torch.long)
    else:
        raise ValueError(f"Unknown split: {split}")

    ds = Dataset.from_file(arrow_file)
    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    seqs = []
    for ex in ds:
        tokens = tok.encode(ex["text"])
        if len(tokens) >= min_len:
            seqs.append(tokens[:min_len])
            if max_count and len(seqs) >= max_count:
                break
    return torch.tensor(seqs, dtype=torch.long)


# ── Model Loading ─────────────────────────────────────────────────────────────

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
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    block_perm = torch.tensor(ckpt["data_permutation"]["block_perm"], dtype=torch.long)
    inv_perm = torch.tensor(ckpt["data_permutation"]["inverse_block_perm"], dtype=torch.long)
    return model, block_perm, inv_perm


def phys_to_model_idx(idx_phys, inv_perm):
    """Convert token sequences from physical order to model-coordinate order."""
    B, T = idx_phys.shape
    N64 = len(inv_perm)
    blk_size = T // N64
    idx_model = torch.zeros_like(idx_phys)
    for s in range(T):
        model_block = inv_perm[s // blk_size].item()
        offset = s % blk_size
        model_pos = model_block * blk_size + offset
        idx_model[:, model_pos] = idx_phys[:, s]
    return idx_model


# ── Block-level ↔ Token-level ────────────────────────────────────────────────

def block_order_to_token_order(block_order, block_len=BLOCK_LEN):
    """(N,) block indices → (T,) token indices."""
    N = len(block_order)
    token_order = np.zeros(N * block_len, dtype=np.int64)
    for i, blk in enumerate(block_order):
        for k in range(block_len):
            token_order[i * block_len + k] = blk * block_len + k
    return token_order


def token_order_to_block_order(token_order, block_len=BLOCK_LEN):
    """(T,) token indices → (N,) block indices."""
    return token_order[::block_len] // block_len


# ── Per-sequence NLL from logits ──────────────────────────────────────────────

@torch.no_grad()
def compute_per_seq_nll(model, idx, orders):
    """
    Compute per-sequence NLL from AO-GPT forward pass.

    Unlike model.forward_fn which returns a scalar (mean) loss,
    this computes CE per sequence using the returned logits.

    Args:
        model: frozen AO-GPT
        idx: (B, T) token IDs in model coordinates
        orders: (B, T) token-level order

    Returns:
        nll: (B,) per-sequence NLL
    """
    B, T = idx.shape
    # Reorder idx same way as model.forward_fn does
    batch_indices = torch.arange(B, device=idx.device).unsqueeze(1).expand(-1, T)
    targets = idx[batch_indices, orders]  # (B, T) — reordered tokens

    # Full forward (same as forward_fn but we compute per-seq loss)
    pos = torch.arange(0, T + 1, dtype=torch.long, device=idx.device)

    tok_emb = model.transformer.wte(idx)
    tok_emb = tok_emb[batch_indices, orders]  # shuffle token embeddings
    none_emb = model.transformer.wnonee(
        torch.tensor([[0]], device=idx.device)
    ).expand(B, -1, -1)
    tok_emb = torch.cat([none_emb, tok_emb], dim=1)

    pos_emb = model.transformer.wpe(pos).unsqueeze(0).expand(B, -1, -1)
    pos_emb_prefix = pos_emb[:, :1, :]
    pos_emb_postfix = pos_emb[:, 1:, :][batch_indices, orders]  # shuffle position embeddings
    pos_emb_final = torch.cat([pos_emb_prefix, pos_emb_postfix], dim=1)

    tgt_emb = model.transformer.wtpe(pos[:T]).unsqueeze(0).expand(B, -1, -1)
    tgt_emb_prefix = tgt_emb[batch_indices, orders]  # shuffle target position embeddings
    tgt_emb_postfix = torch.zeros(B, 1, model.transformer.wtpe.weight.shape[1],
                                   device=idx.device)
    c = torch.cat([tgt_emb_prefix, tgt_emb_postfix], dim=1)

    x = tok_emb + pos_emb_final
    x = model.transformer.drop(x)
    for block in model.transformer.h:
        x = block(x, c)
    x = model.transformer.final_layer(x, c)

    logits = model.lm_head(x)  # (B, T+1, V)
    shift_logits = logits[:, :-1, :].contiguous()  # (B, T, V)

    # Per-sequence CE (vectorized)
    ce_per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        targets.reshape(-1),
        reduction='none',
        ignore_index=-1,
    ).reshape(B, T)
    return ce_per_token.mean(dim=1)


# ── Batched Hill Climbing ────────────────────────────────────────────────────

@torch.no_grad()
def batched_hill_climb(model, idx_all, batch_size, n_steps, device, verbose=True):
    """
    Random-swap hill climbing on batched sequences.

    Each step: propose 1 random block-swap per sequence, batch-evaluate,
    accept individually if NLL improves.

    Args:
        model: frozen AO-GPT
        idx_all: (num_seqs, 256) token IDs in model coordinates
        batch_size: GPU batch size for forward pass
        n_steps: number of HC steps per sequence
        device: torch device

    Returns:
        best_orders: (num_seqs, 256) best token-level order found per sequence
        best_nlls: (num_seqs,) best NLL per sequence
        history: list of (step, n_accepted_this_step, running_best_nll_mean)
    """
    num_seqs, T = idx_all.shape
    N = T // BLOCK_LEN  # 64

    # Initialize: L2R ordering + baseline NLL
    l2r_blocks = np.arange(N, dtype=np.int64)
    l2r_tokens = block_order_to_token_order(l2r_blocks)

    current_blocks = np.tile(l2r_blocks[None, :], (num_seqs, 1))  # (S, 64)
    candidate_blocks = np.copy(current_blocks)

    # Compute initial NLLs
    print("Computing baseline NLL (L2R)...", flush=True)
    current_nlls = np.zeros(num_seqs, dtype=np.float32)
    for b_start in range(0, num_seqs, batch_size):
        b_end = min(b_start + batch_size, num_seqs)
        B = b_end - b_start
        idx_b = idx_all[b_start:b_end].to(device)
        orders_b = torch.as_tensor(
            np.tile(l2r_tokens[None, :], (B, 1)), dtype=torch.long, device=device
        )
        nll_b = compute_per_seq_nll(model, idx_b, orders_b)
        current_nlls[b_start:b_end] = nll_b.cpu().numpy()

    best_blocks = np.copy(current_blocks)
    best_nlls = np.copy(current_nlls)
    print(f"  Baseline NLL: mean={current_nlls.mean():.4f}, "
          f"min={current_nlls.min():.4f}, max={current_nlls.max():.4f}", flush=True)

    history = []
    total_accepted = 0
    t_start = time.perf_counter()

    for step in range(1, n_steps + 1):
        # Generate candidate swaps: random i, j for each sequence
        i_vals = np.random.randint(0, N, size=num_seqs)
        j_vals = np.random.randint(0, N, size=num_seqs)
        # Ensure i != j
        for s in range(num_seqs):
            while j_vals[s] == i_vals[s]:
                j_vals[s] = np.random.randint(0, N)

        # Apply swaps to candidate
        candidate_blocks[:] = current_blocks
        for s in range(num_seqs):
            candidate_blocks[s, i_vals[s]], candidate_blocks[s, j_vals[s]] = \
                candidate_blocks[s, j_vals[s]], candidate_blocks[s, i_vals[s]].copy()

        # Convert to token orders
        candidate_tokens = np.zeros((num_seqs, T), dtype=np.int64)
        for s in range(num_seqs):
            candidate_tokens[s] = block_order_to_token_order(candidate_blocks[s])

        # Batch-evaluate all candidates
        candidate_nlls = np.zeros(num_seqs, dtype=np.float32)
        for b_start in range(0, num_seqs, batch_size):
            b_end = min(b_start + batch_size, num_seqs)
            idx_b = idx_all[b_start:b_end].to(device)
            orders_b = torch.as_tensor(candidate_tokens[b_start:b_end], dtype=torch.long, device=device)
            nll_b = compute_per_seq_nll(model, idx_b, orders_b)
            candidate_nlls[b_start:b_end] = nll_b.cpu().numpy()

        # Accept/reject individually
        improved = candidate_nlls < current_nlls
        current_blocks[improved] = candidate_blocks[improved].copy()
        current_nlls[improved] = candidate_nlls[improved]

        # Update best
        new_best = current_nlls < best_nlls
        best_blocks[new_best] = current_blocks[new_best].copy()
        best_nlls[new_best] = current_nlls[new_best].copy()

        n_accepted = improved.sum()
        total_accepted += n_accepted

        if step % 200 == 0 or step == 1:
            elapsed = time.perf_counter() - t_start
            history.append((step, n_accepted, best_nlls.mean()))
            if verbose:
                print(f"  Step {step:5d}/{n_steps} | accepted={n_accepted:4d}/{num_seqs} "
                      f"| best_nll={best_nlls.mean():.4f} | {elapsed:.1f}s",
                      flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"Done. {n_steps} steps, {total_accepted} total accepted "
          f"({total_accepted / n_steps:.1f}/step), {elapsed:.1f}s",
          flush=True)

    # Convert best to token orders for saving
    best_token_orders = np.zeros((num_seqs, T), dtype=np.int64)
    for s in range(num_seqs):
        best_token_orders[s] = block_order_to_token_order(best_blocks[s])

    return best_token_orders, best_nlls, history


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batched Random-Swap Hill Climbing")
    p.add_argument("--split", default="test", choices=["test", "train"])
    p.add_argument("--n-seqs", type=int, default=0,
                   help="Number of sequences (0 = all available)")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--output-dir", default="probe_results")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}, Batch size: {args.batch_size}", flush=True)

    # Load model
    print("Loading AO-GPT...", flush=True)
    model, block_perm, inv_perm = load_aogpt(CKPT_PATH, device)
    print(f"Loaded. Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Load data
    print(f"Loading sequences (split={args.split})...", flush=True)
    idx_phys = load_sequences(split=args.split, min_len=SEQ_LEN, max_count=args.n_seqs or None)
    n_loaded = idx_phys.shape[0]
    print(f"Loaded {n_loaded} sequences", flush=True)

    # Convert to model coordinates
    print("Converting to model coordinates...", flush=True)
    idx_list = [phys_to_model_idx(idx_phys[i:i + 1], inv_perm) for i in range(n_loaded)]
    idx_all = torch.cat(idx_list, dim=0)
    print(f"idx_all: {idx_all.shape}", flush=True)

    # Run batched HC
    n_seqs = idx_all.shape[0]
    print(f"\nRunning batched HC: {n_seqs} seqs × {args.steps} steps, "
          f"batch_size={args.batch_size}", flush=True)
    print(f"Estimated time: ~{n_seqs * args.steps / (1363 * (2 if torch.cuda.device_count() > 1 else 1)):.0f}s",
          flush=True)

    best_orders, best_nlls, history = batched_hill_climb(
        model, idx_all, args.batch_size, args.steps, device
    )

    # Summary
    l2r_nll_init = 0  # computed inside HC, approximate from history
    improvement = best_nlls.mean()
    print(f"\nFinal: best_nll mean={best_nlls.mean():.4f}, "
          f"min={best_nlls.min():.4f}, max={best_nlls.max():.4f}", flush=True)

    # Save
    output_path = Path(args.output_dir) / f"hc_{args.split}_{n_seqs}seqs_{args.steps}steps.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        best_orders=best_orders,
        best_nlls=best_nlls,
        history=np.array(history),
        args=vars(args),
    )
    print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
