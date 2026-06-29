"""
P0 Route A (N=16 direct): Attention Prune → NLL Pair Score → Asymmetric A (16×16).

直接在 N=16 打分: 每个 pair (I,J) 把 I 的 4 个 sub-blocks 放前面、
J 的 4 个 sub-blocks 紧随其后，只跑 240 forward/seq/rep。

用法:
    python p0_n16_direct.py --num_seqs 100 --num_repeats 5 --output probe_results/A_n16.npy
    python p0_n16_direct.py --num_seqs 3 --num_repeats 1 --debug
"""

import os
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'AO-GPT-MDM'))

import argparse, time, random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset
from transformers import GPT2TokenizerFast

from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N16 = 16          # output A matrix dimension
SUB_BLOCKS = 4    # order blocks per N16 block (256/64=4)
N64 = 64          # model's native order blocks
BLOCK_LEN = 4     # tokens per order block
SEQ_LEN = N64 * BLOCK_LEN  # 256

CKPT_PATH = os.path.expanduser(
    "~/ych/nanogpt-learned-order/out/base/permute/seq256/block64/"
    "out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt"
)

WIKITEXT_ARROW = os.path.expanduser(
    "~/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/"
    "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow"
)
TOKENIZER_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)


# ---------------------------------------------------------------------------
# Model & Data
# ---------------------------------------------------------------------------

def load_model_and_perm(ckpt_path, device):
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
    block_perm = torch.tensor(ckpt["data_permutation"]["block_perm"], dtype=torch.long)
    inv_perm = torch.tensor(ckpt["data_permutation"]["inverse_block_perm"], dtype=torch.long)
    return model, block_perm, inv_perm


def load_sequences(min_len=SEQ_LEN, max_count=500):
    ds = Dataset.from_file(WIKITEXT_ARROW)
    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    seqs = []
    for ex in ds:
        ids = tok.encode(ex["text"])
        if len(ids) >= min_len:
            seqs.append(ids[:min_len])
        if len(seqs) >= max_count:
            break
    return seqs


def permute_sequence(seq, block_perm, block_len=BLOCK_LEN):
    T = len(seq)
    seq_2d = seq.view(-1, block_len)
    return seq_2d[block_perm].flatten()


# ---------------------------------------------------------------------------
# N=16 Direct NLL Scoring
# ---------------------------------------------------------------------------

def build_n16_pair_orders(n16_i, n16_j, num_n16=N16, sub_blocks=SUB_BLOCKS,
                          device='cuda'):
    """
    构造 block-level order (N64 坐标): I 的 sub-blocks first, J 的 sub-blocks 其次, random suffix.

    返回: (N64,) tensor in model-block coordinates
    """
    order = []
    # I's sub-blocks (at N64 level)
    for a in range(sub_blocks):
        order.append(n16_i * sub_blocks + a)
    # J's sub-blocks
    for b in range(sub_blocks):
        order.append(n16_j * sub_blocks + b)
    # Random suffix
    used = set(order)
    remaining = [k for k in range(N64) if k not in used]
    random.shuffle(remaining)
    order.extend(remaining)
    return torch.tensor(order, dtype=torch.long, device=device)


def block_orders_to_token_orders(block_orders, block_len=BLOCK_LEN):
    """block_orders: (B, N) → token_orders: (B, T)"""
    B, N = block_orders.shape
    offsets = torch.arange(block_len, device=block_orders.device).view(1, 1, block_len)
    token_orders = block_orders.unsqueeze(-1) * block_len + offsets
    return token_orders.reshape(B, N * block_len)


def score_n16_pair(model, idx, n16_i, n16_j, autocast_ctx,
                   pair_score_k=8, tv_weight=0.3):
    """
    打分: N=16 有向 pair (I→J).

    pair_score_k=8: 看前 8 个 order blocks (= 前 2 个 N=16 blocks = I+J).
    """
    device = idx.device
    block_order = build_n16_pair_orders(n16_i, n16_j).unsqueeze(0)
    token_orders = block_orders_to_token_orders(block_order, block_len=BLOCK_LEN)

    with autocast_ctx:
        logits, _ = model.forward_fn(idx, token_orders, return_attentions=False)

    b, t = idx.shape
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = model.shuffle(idx, token_orders).long()
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=-1,
        reduction='none'
    ).view(b, t)

    # Aggregate to block losses (4-token blocks)
    block_losses = per_token_loss.float().view(b, N64, BLOCK_LEN).mean(dim=-1)  # (1, 64)
    window = block_losses[0, :pair_score_k]  # (k,)

    area = -window.mean()
    tv = 0.0 if window.numel() < 2 else -(window[1:] - window[:-1]).abs().sum()
    return (area + tv_weight * tv).item()


def mine_n16_pairs(model, seq, block_perm, device, dtype,
                   num_repeats=5, pair_score_k=8, tv_weight=0.3):
    """
    对一条序列的所有 240 个 N=16 directed pairs 做 NLL scoring.

    返回: (16, 16) asymmetric score matrix in PHYSICAL coords.
    """
    seq_t = torch.tensor(seq, dtype=torch.long, device=device)
    seq_model = permute_sequence(seq_t, block_perm.to(device)).unsqueeze(0)

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if "cuda" in device else torch.no_grad()

    # Score in model-block coordinates, then map back
    # model_block → phys_block = inv_perm[model_block]
    # n16_i_phys ↔ n16_j_phys: we need model-level indices for ordering
    # phys_n16_I → model_block = block_perm[phys_n16_I * 4]
    # But we want to keep scores in PHYSICAL coords

    scores_phys = np.full((N16, N16), np.nan, dtype=np.float32)

    total_pairs = N16 * (N16 - 1)
    for phys_i in range(N16):
        for phys_j in range(N16):
            if phys_i == phys_j:
                scores_phys[phys_i, phys_j] = np.nan
                continue
            # Map to model coords for ordering
            model_i = phys_i  # Wait, we need to think about this...
            # The block_perm maps phys_block → model_block at N=64 level
            # phys N16 block I → phys N64 blocks [4I, 4I+1, 4I+2, 4I+3]
            # model N64 blocks: block_perm[4I], block_perm[4I+1], ...
            # When building pair order, we need MODEL-block indices
            #
            # Simplification: score in model coords, then map result back to phys coords

    # SIMPLER APPROACH: score in MODEL coords, translate results
    # Initialize score matrix in model coords
    scores_model = np.full((N16, N16), np.nan, dtype=np.float32)

    all_pairs = [(i, j) for i in range(N16) for j in range(N16) if i != j]

    for model_i, model_j in tqdm(all_pairs, desc="  NLL scoring", leave=False):
        rep_scores = []
        for _ in range(num_repeats):
            s = score_n16_pair(model, seq_model, model_i, model_j, autocast_ctx,
                              pair_score_k=pair_score_k, tv_weight=tv_weight)
            rep_scores.append(s)
        scores_model[model_i, model_j] = float(np.mean(rep_scores))

    # Map to physical coords
    # model N16 block I → phys N16 block = inv_perm[I * sub_blocks] // sub_blocks
    # Actually, let me think again...
    # block_perm[phys64] = model64
    # inv_perm[model64] = phys64
    # For N16 level: phys16_I → phys64 blocks [I*4, I*4+1, I*4+2, I*4+3]
    # model64 blocks: block_perm[I*4], block_perm[I*4+1], ...
    # These 4 model64 blocks might NOT be in the same model N16 group!
    # So there's no clean N16-level mapping.
    #
    # SOLUTION: score at N64 level and aggregate, OR
    # accept that model N16 groups don't correspond to phys N16 groups
    #
    # Let's instead define N16 groups at the PHYSICAL level:
    # phys_N16_I = phys_order_blocks [I*4, I*4+1, I*4+2, I*4+3]
    # These are 4 consecutive blocks in physical space
    #
    # For the model to score this pair correctly, we need to tell it
    # to reveal those 8 specific phys blocks first (in the right order),
    # expressing them in MODEL coordinates.
    #
    # phys block p → model block block_perm[p]
    # So phys group I → model blocks [block_perm[I*4], ..., block_perm[I*4+3]]
    #
    # The pair order in model coords:
    # [block_perm[I*4], ..., block_perm[I*4+3], block_perm[J*4], ..., block_perm[J*4+3], random...]

    # REDO with proper phys→model mapping:
    scores_phys = np.full((N16, N16), np.nan, dtype=np.float32)
    bp = block_perm.cpu().numpy()  # phys64 → model64

    for phys_i in range(N16):
        for phys_j in range(N16):
            if phys_i == phys_j:
                continue

            # Build model-level pair order
            model_order = []
            for a in range(SUB_BLOCKS):
                model_order.append(int(bp[phys_i * SUB_BLOCKS + a]))
            for b in range(SUB_BLOCKS):
                model_order.append(int(bp[phys_j * SUB_BLOCKS + b]))
            used = set(model_order)
            remaining = [k for k in range(N64) if k not in used]
            random.shuffle(remaining)
            model_order.extend(remaining)

            block_order = torch.tensor(model_order, dtype=torch.long, device=device).unsqueeze(0)
            token_orders = block_orders_to_token_orders(block_order, block_len=BLOCK_LEN)

            # Score (with repeats)
            rep_scores = []
            for _ in range(num_repeats):
                with autocast_ctx:
                    logits, _ = model.forward_fn(seq_model, token_orders, return_attentions=False)
                b, t = seq_model.shape
                shift_logits = logits[..., :-1, :].contiguous()
                shift_targets = model.shuffle(seq_model, token_orders).long()
                per_token_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=-1, reduction='none'
                ).view(b, t)
                block_losses = per_token_loss.float().view(b, N64, BLOCK_LEN).mean(dim=-1)
                window = block_losses[0, :pair_score_k]
                area = -window.mean()
                tv = 0.0 if window.numel() < 2 else -(window[1:] - window[:-1]).abs().sum()
                rep_scores.append((area + tv_weight * tv).item())

            scores_phys[phys_i, phys_j] = float(np.mean(rep_scores))

    return scores_phys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="P0 Route A: N=16 Direct NLL Scoring")
    p.add_argument("--num_seqs", type=int, default=100)
    p.add_argument("--num_repeats", type=int, default=5)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--output", type=str, default="probe_results/A_n16_direct.npy")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--ckpt", type=str, default=CKPT_PATH,
        help="AOGPT checkpoint path (default: ych Random_CL ckpt)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug:
        args.num_seqs = 3
        args.num_repeats = 1

    print("=" * 60)
    print("P0 Route A: N=16 Direct NLL Scoring")
    print("=" * 60)
    print(f"  CKPT: permute_data=True, {N16}×{SUB_BLOCKS}×{BLOCK_LEN}={SEQ_LEN} tokens")
    print(f"  Pairs: {N16*(N16-1)} directed, {args.num_repeats} repeats each")
    print(f"  Est. time: {args.num_seqs * N16 * (N16-1) * args.num_repeats / 569:.0f}s")

    device = args.device
    dtype = torch.bfloat16

    print("\n[1/4] Loading model + permutation...")
    model, block_perm, inv_perm = load_model_and_perm(args.ckpt, device)
    print(f"  {model.config.n_layer}L/{model.config.n_head}H/{model.config.n_embd}D")

    print("\n[2/4] Loading sequences...")
    sequences = load_sequences(max_count=args.num_seqs)
    print(f"  {len(sequences)} sequences")

    print(f"\n[3/4] NLL scoring ({len(sequences)} seqs × {args.num_repeats} repeats)...")
    t0 = time.time()
    A_list = []
    for s_idx in tqdm(range(len(sequences)), desc="Sequences"):
        scores_phys = mine_n16_pairs(
            model, sequences[s_idx], block_perm, device, dtype,
            num_repeats=args.num_repeats)

        # Convert NaN (diagonal) to -inf, rest to A matrix
        A = np.where(np.isnan(scores_phys), -np.inf, scores_phys).astype(np.float32)
        A_list.append(A)

        if args.debug:
            print(f"\n--- Seq {s_idx} ---")
            finite = A[A > -1e9]
            print(f"  Score range: [{finite.min():.4f}, {finite.max():.4f}]")
            # Asymmetry
            asym = [abs(A[i,j] - A[j,i]) for i in range(N16) for j in range(N16)
                    if i != j and np.isfinite(A[i,j]) and np.isfinite(A[j,i])]
            print(f"  Mean asymmetry: {np.mean(asym):.4f}")
            # Top pairs
            pairs = [(i,j,A[i,j]) for i in range(N16) for j in range(N16)
                     if i != j and np.isfinite(A[i,j])]
            pairs.sort(key=lambda x: -x[2])
            print(f"  Top 10 pairs:")
            for i,j,s in pairs[:10]:
                print(f"    ({i:2d}→{j:2d}) dist={abs(i-j):2d} score={s:.4f}")
            # Non-adj in top-20
            nonadj_top = sum(1 for i,j,_ in pairs[:20] if abs(i-j) > 1)
            print(f"  Non-adj in top 20: {nonadj_top}/20")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({elapsed/len(sequences):.1f}s/seq)")

    A_matrices = np.stack(A_list, axis=0).astype(np.float32)

    # Signal analysis
    print(f"\n[4/4] Signal analysis...")
    adj, far = [], []
    for A in A_matrices:
        for i in range(N16):
            for j in range(N16):
                if i == j or not np.isfinite(A[i,j]): continue
                if abs(i-j) == 1: adj.append(A[i,j])
                elif abs(i-j) >= 4: far.append(A[i,j])
    adj_m, far_m = np.mean(adj), np.mean(far)
    print(f"  Adjacent: mean={adj_m:.4f}")
    print(f"  Far:      mean={far_m:.4f}")
    print(f"  Diff:     {adj_m - far_m:.4f}")
    print(f"  (Zero diff = NLL score decoupled from physical distance)")

    if not args.debug:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        np.save(args.output, A_matrices)
        print(f"\n  Saved to {args.output}")

    print(f"\n  ✓ P0 Route A N=16 done.")


if __name__ == "__main__":
    main()
