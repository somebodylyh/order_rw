"""
P0 Route A: Attention Pruning → NLL Pair Scoring → 有向非对称 A 矩阵.

Stage 1 (Attention Pruning): cosine similarity of [None]-delta profiles → top-K per block
Stage 2 (NLL Scoring): 对每个候选 pair (i,j), 跑 "i first, j second" 的 reveal order,
   计算 early block loss → 有向 score(i→j) ≠ score(j→i)

用法:
    python p0_route_a.py --num_seqs 5 --debug
    python p0_route_a.py --num_seqs 100 --output probe_results/real_A_route_a_n64.npy
"""

import os
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'AO-GPT-MDM'))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset
from transformers import GPT2TokenizerFast

from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig

# ---------------------------------------------------------------------------
# 配置: permute_data=True ckpt, N=64 blocks × 4 tokens = 256
# ---------------------------------------------------------------------------
NUM_BLOCKS = 64
BLOCK_LEN = 4
SEQ_LEN = NUM_BLOCKS * BLOCK_LEN  # 256
PAIR_SCORE_K = 2       # 用前 K 个 block 的 loss 打分
TV_WEIGHT = 0.3        # loss smoothness weight
ATTN_TOP_K = 8         # Stage 1: per-block top-K neighbors to keep

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
    sig_params = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid_args = {k: v for k, v in dict(ckpt["model_args"]).items() if k in sig_params}
    model = AOGPT(AOGPTConfig(**valid_args))
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


def load_sequences(min_seq_len=SEQ_LEN, max_sequences=500):
    ds = Dataset.from_file(WIKITEXT_ARROW)
    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    seqs = []
    for ex in ds:
        ids = tok.encode(ex["text"])
        if len(ids) >= min_seq_len:
            seqs.append(ids[:min_seq_len])
        if len(seqs) >= max_sequences:
            break
    return seqs


# ---------------------------------------------------------------------------
# Token permutation helpers
# ---------------------------------------------------------------------------

def permute_sequence(seq, block_perm, block_len=BLOCK_LEN):
    """seq: (T,) physical order → (T,) model-coord order"""
    T = len(seq)
    seq_2d = seq.view(-1, block_len)  # (N, BL)
    return seq_2d[block_perm].flatten()


def block_orders_to_token_orders(block_orders, block_len=BLOCK_LEN):
    """block_orders: (B, N) → token_orders: (B, T) with L2R within each block"""
    B, N = block_orders.shape
    offsets = torch.arange(block_len, device=block_orders.device).view(1, 1, block_len)
    token_orders = block_orders.unsqueeze(-1) * block_len + offsets
    return token_orders.reshape(B, N * block_len)


# ---------------------------------------------------------------------------
# Stage 1: Attention Pruning (cosine similarity → top-K per block)
# ---------------------------------------------------------------------------

def reorder_attention_from_reveal(attn, orders):
    """attn: (H, T+1, T+1) reveal coords → (H, T+1, T+1) ascending/model coords"""
    H = attn.shape[0]
    r = torch.zeros_like(attn)
    for h in range(H):
        a = attn[h]
        rr = torch.zeros_like(a)
        rr[0, 0] = a[0, 0]
        rr[0, 1:] = a[0, 1:][orders]
        rr[1:, 0] = a[1:, 0][orders]
        rr[1:, 1:] = a[1:, 1:][orders][:, orders]
        r[h] = rr
    return r


def reorder_attention_model_to_physical(attn, inv_perm, block_len=BLOCK_LEN):
    """attn: (H, T+1, T+1) model coords → physical coords"""
    H = attn.shape[0]
    T = attn.shape[1] - 1
    N = T // block_len
    # Build token-level inverse perm
    tok_inv = torch.zeros(T, dtype=torch.long)
    for t in range(T):
        blk = t // block_len
        off = t % block_len
        tok_inv[t] = inv_perm[blk].item() * block_len + off
    r = torch.zeros_like(attn)
    for h in range(H):
        a = attn[h]
        rr = torch.zeros_like(a)
        rr[0, 0] = a[0, 0]
        rr[0, 1:] = a[0, 1:][tok_inv]
        rr[1:, 0] = a[1:, 0][tok_inv]
        rr[1:, 1:] = a[1:, 1:][tok_inv][:, tok_inv]
        r[h] = rr
    return r


def delta_from_uniform(matrix):
    Tp1 = matrix.size(0)
    baseline = torch.zeros_like(matrix)
    for i in range(Tp1):
        baseline[i, :i+1] = 1.0 / (i + 1)
    return matrix - baseline


def extract_attention_similarity_matrix(model, seq, block_perm, inv_perm, device, dtype,
                                         num_blocks=NUM_BLOCKS, block_len=BLOCK_LEN):
    """
    Stage 1: 提取 [None]-delta attention → cosine similarity → 对称相似矩阵。
    用于剪枝（cheap pre-filter）。
    """
    T = num_blocks * block_len
    seq_t = torch.tensor(seq, dtype=torch.long, device=device)
    seq_model = permute_sequence(seq_t, block_perm.to(device), block_len)
    orders = torch.randperm(T, device=device).unsqueeze(0)

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if "cuda" in device else torch.no_grad()
    with torch.no_grad():
        with autocast_ctx:
            _, _, attn_list = model(seq_model.unsqueeze(0), mode=None, orders=orders, return_attentions=True)
    last_attn = attn_list[-1]  # (1, H, T+1, T+1)
    H = last_attn.shape[1]

    # Reveal → model coords → physical coords
    attn_model = reorder_attention_from_reveal(last_attn[0], orders[0])
    attn_phys = reorder_attention_model_to_physical(attn_model, inv_perm, block_len)

    # [None]-delta profile per head: token-to-[None] attention
    # attn[:, 1:, 0] is token→[None], shape (H, T)
    none_delta_per_head = []
    for h in range(H):
        none_raw = attn_phys[h, 1:, 0]  # (T,) token→[None]
        none_delta = none_raw - 1.0 / (torch.arange(T, device=device, dtype=torch.float64) + 1 + 1)
        # ^ uniform baseline: query i (tokens 0..T-1) sees i+2 keys ([None] + 0..i)
        none_delta_per_head.append(none_delta.float())

    # Aggregate to block level: (H, T) → (H, N)
    none_block = torch.stack([d.view(num_blocks, block_len).mean(-1) for d in none_delta_per_head], dim=0)  # (H, N)

    # Profile: (N, H) → cosine similarity → (N, N)
    profiles = none_block.t()  # (N, H)
    profiles_norm = F.normalize(profiles, dim=-1)
    sim_matrix = torch.mm(profiles_norm, profiles_norm.t())  # (N, N) symmetric
    sim_matrix.fill_diagonal_(0.0)
    return sim_matrix.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 2: NLL Pair Scoring
# ---------------------------------------------------------------------------

def compute_per_token_loss(model, idx, token_orders, autocast_ctx):
    """
    调用模型 forward，从 logits 算 per-token loss (B, T).
    model.forward_fn 返回 (logits, loss) or (logits, loss, attn).
    """
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
    return per_token_loss


def score_pair(model, idx, pair, autocast_ctx, num_blocks=NUM_BLOCKS, block_len=BLOCK_LEN,
               pair_score_k=PAIR_SCORE_K, tv_weight=TV_WEIGHT):
    """
    对一对有向 pair (first, second) 打分。

    Reveal order: [first, second, random_suffix...]
    Score = -(mean of first k block losses) - tv_weight * TV(first k block losses)

    返回: scalar float score
    """
    device = idx.device
    first, second = pair
    # Build block order
    blocks = torch.arange(num_blocks, device=device)
    mask = (blocks != first) & (blocks != second)
    remaining = blocks[mask]
    perm = torch.randperm(remaining.numel(), device=device)
    suffix = remaining[perm]
    block_order = torch.cat([torch.tensor([first, second], device=device), suffix], dim=0).unsqueeze(0)

    token_orders = block_orders_to_token_orders(block_order, block_len=block_len)
    per_token_loss = compute_per_token_loss(model, idx, token_orders, autocast_ctx)  # (1, T)

    # Aggregate to block losses
    block_losses = per_token_loss.float().view(1, num_blocks, block_len).mean(dim=-1)  # (1, N)
    window = block_losses[0, :pair_score_k]  # (K,)

    area = -window.mean()
    if window.numel() < 2:
        tv = torch.tensor(0.0, device=device)
    else:
        tv = -(window[1:] - window[:-1]).abs().sum()
    score = area + tv_weight * tv
    return score.item()


def mine_nll_pair_scores(model, seq, block_perm, inv_perm, candidate_pairs, device, dtype,
                          num_blocks=NUM_BLOCKS, block_len=BLOCK_LEN,
                          pair_score_k=PAIR_SCORE_K, tv_weight=TV_WEIGHT,
                          num_repeats=5):
    """
    Stage 2: 对候选 pair 列表做 NLL scoring，多次 repeat 取均值。

    candidate_pairs: list of (i, j) in PHYSICAL coords.
    但模型需要喂 permuted data, orders 是 model-block indices。
    所以: pair 需要从物理坐标映射到模型坐标。

    返回: pair_score_matrix (N, N) in PHYSICAL coords, asymmetric.
    """
    T = num_blocks * block_len
    seq_t = torch.tensor(seq, dtype=torch.long, device=device)
    # Feed data in permuted order (match training)
    seq_model = permute_sequence(seq_t, block_perm.to(device), block_len).unsqueeze(0)  # (1, T)

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if "cuda" in device else torch.no_grad()

    # Map candidate pairs from physical → model coords
    # phys_block i → model_block = block_perm[i]
    model_pairs = [(block_perm[p[0]].item(), block_perm[p[1]].item()) for p in candidate_pairs]

    pair_scores = np.full((num_blocks, num_blocks), -np.inf, dtype=np.float32)

    for (pi, pj), (mi, mj) in zip(candidate_pairs, model_pairs):
        scores = []
        for _ in range(num_repeats):
            s = score_pair(model, seq_model, (mi, mj), autocast_ctx,
                          num_blocks=num_blocks, block_len=block_len,
                          pair_score_k=pair_score_k, tv_weight=tv_weight)
            scores.append(s)
        pair_scores[pi, pj] = float(np.mean(scores))

    return pair_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="P0 Route A: Attention Prune → NLL Score → Asymmetric A")
    p.add_argument("--num_seqs", type=int, default=100)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--output", type=str, default="probe_results/real_A_route_a_n64.npy")
    p.add_argument("--attn_top_k", type=int, default=ATTN_TOP_K)
    p.add_argument("--full_pairs", action="store_true", help="Score ALL pairs (no attention pruning)")
    p.add_argument("--pair_score_k", type=int, default=PAIR_SCORE_K)
    p.add_argument("--num_repeats", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug:
        args.num_seqs = 3

    print("=" * 60)
    print("P0 Route A: Attention Prune → NLL Score → Asymmetric A")
    print("=" * 60)
    print(f"  CKPT: permute_data=True, N={NUM_BLOCKS}, block_len={BLOCK_LEN}")
    print(f"  attn_top_k={args.attn_top_k}, pair_score_k={args.pair_score_k}, num_repeats={args.num_repeats}")
    print(f"  num_seqs={args.num_seqs}, debug={args.debug}")

    device = args.device
    dtype = torch.bfloat16

    print("\n[1/5] Loading model + permutation...")
    model, block_perm, inv_perm = load_model_and_perm(CKPT_PATH, device)
    print(f"  Model: {model.config.n_layer}L/{model.config.n_head}H/{model.config.n_embd}D")

    print("\n[2/5] Loading sequences...")
    sequences = load_sequences(min_seq_len=SEQ_LEN, max_sequences=args.num_seqs)
    print(f"  Loaded {len(sequences)} sequences")

    all_A_matrices = []
    for s_idx in tqdm(range(len(sequences)), desc="Route A"):
        seq = sequences[s_idx]

        # --- Stage 1: Attention Pruning ---
        sim_matrix = extract_attention_similarity_matrix(
            model, seq, block_perm, inv_perm, device, dtype)

        if args.full_pairs:
            candidate_pairs = [(i, j) for i in range(NUM_BLOCKS) for j in range(NUM_BLOCKS) if i != j]
        else:
            # Top-K per block → sparse candidate pairs
            masked = sim_matrix.copy()
            np.fill_diagonal(masked, -np.inf)
            candidate_pairs = set()
            for blk in range(NUM_BLOCKS):
                top_k = min(args.attn_top_k, NUM_BLOCKS - 1)
                neighbors = np.argpartition(-masked[blk], top_k)[:top_k]
                for nb in neighbors:
                    if masked[blk, nb] > -1e9:
                        candidate_pairs.add((int(blk), int(nb)))
            candidate_pairs = sorted(candidate_pairs)

        if args.debug:
            print(f"\n  Seq {s_idx}: Stage 1 found {len(candidate_pairs)} directed pairs "
                  f"(from {NUM_BLOCKS * (NUM_BLOCKS - 1)} possible)")

        # --- Stage 2: NLL Pair Scoring ---
        pair_score_matrix = mine_nll_pair_scores(
            model, seq, block_perm, inv_perm, candidate_pairs, device, dtype,
            num_repeats=args.num_repeats, pair_score_k=args.pair_score_k)

        # For unscored pairs, use a fallback low value
        # (pair_score_matrix already has -inf for unscored)
        # Fill -inf entries with the 10th percentile of scored values
        scored_vals = pair_score_matrix[pair_score_matrix > -1e9]
        if len(scored_vals) > 0:
            fallback = np.percentile(scored_vals, 10)
        else:
            fallback = -10.0
        A = np.where(pair_score_matrix > -1e9, pair_score_matrix, fallback)
        np.fill_diagonal(A, -np.inf)  # no self-loops

        all_A_matrices.append(A.astype(np.float32))

        if args.debug:
            # Stats
            scored = pair_score_matrix[pair_score_matrix > -1e9]
            print(f"  Stage 2 scored: mean={scored.mean():.4f}, std={scored.std():.4f}, "
                  f"range=[{scored.min():.4f}, {scored.max():.4f}]")
            # Asymmetry check
            asym_count = 0
            asym_sum = 0.0
            for i, j in candidate_pairs:
                if pair_score_matrix[j, i] > -1e9:
                    asym_sum += abs(pair_score_matrix[i, j] - pair_score_matrix[j, i])
                    asym_count += 1
            if asym_count > 0:
                print(f"  Mean asymmetry |score(i,j) - score(j,i)|: {asym_sum/asym_count:.4f} (over {asym_count} pairs)")
            # Top pairs
            top_rows = []
            for i in range(NUM_BLOCKS):
                for j in range(NUM_BLOCKS):
                    if i != j and pair_score_matrix[i, j] > -1e9:
                        top_rows.append((i, j, pair_score_matrix[i, j]))
            top_rows.sort(key=lambda x: -x[2])
            print(f"  Top 10 scored pairs (physical coords):")
            for i, j, v in top_rows[:10]:
                dist = abs(i - j)
                print(f"    ({i:2d}→{j:2d}) dist={dist:2d} score={v:.4f}")

            # How many top pairs are adjacent vs non-adjacent?
            adj_top, nonadj_top = 0, 0
            for i, j, v in top_rows:
                if abs(i - j) == 1:
                    adj_top += 1
                else:
                    nonadj_top += 1
            print(f"  Non-adj in top 20: {sum(1 for i,j,v in top_rows[:20] if abs(i-j) > 1)}/20")

    print(f"\n[3/5] Aggregating {len(all_A_matrices)} A matrices...")
    A_matrices = np.stack(all_A_matrices, axis=0).astype(np.float32)
    print(f"  Shape: {A_matrices.shape}, range: [{A_matrices.min():.4f}, {A_matrices.max():.4f}]")

    # Signal analysis: adjacent vs far for NLL scores
    print(f"\n[4/5] Signal analysis...")
    adj_vals, far_vals = [], []
    for A in A_matrices:
        for i in range(NUM_BLOCKS):
            for j in range(NUM_BLOCKS):
                if i == j:
                    continue
                d = abs(i - j)
                if d == 1 and np.isfinite(A[i, j]):
                    adj_vals.append(A[i, j])
                elif d >= 5 and np.isfinite(A[i, j]):
                    far_vals.append(A[i, j])
    if adj_vals and far_vals:
        adj_m, far_m = np.mean(adj_vals), np.mean(far_vals)
        print(f"  Adjacent (|i-j|=1): mean={adj_m:.4f}")
        print(f"  Far (|i-j|≥5):       mean={far_m:.4f}")
        print(f"  Diff:                 {adj_m - far_m:.4f}")

    # Save
    print(f"\n[5/5] Saving...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, A_matrices)
    print(f"  Saved to {args.output}")
    print(f"\n  ✓ P0 Route A done. Ready for DP solver.")


if __name__ == "__main__":
    main()
