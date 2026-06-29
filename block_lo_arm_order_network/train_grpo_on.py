"""
GRPO Training for Order Network (ON).

MVP: Prove ON can learn reward-positive order distributions under aligned action space.

Pipeline:
  1. ON samples K orders per sequence (sequential PL sampling, temp tau)
  2. Frozen AO-GPT scores NLL for each order
  3. Reward = mean_NLL(random_N16_baselines) - NLL(policy_order)
  4. Group-relative advantage + quartile rank loss + entropy + KL

Usage:
    python -u train_grpo_on.py \
      --a-matrices probe_results/A_n16_direct_500x5.npy \
      --on-ckpt probe_results/crossattn_on_best.pt \
      --device cuda:0
"""

import argparse
import copy
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
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import GPT2TokenizerFast

from config import Config
from order_network import CrossAttentionOrderNetwork, masks_to_revealed_bool
from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig

# ── Paths ────────────────────────────────────────────────────────────────────────
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
N64 = 64
BLOCK_LEN = 4   # tokens per atomic N64 block (model granularity)


# ═══════════════════════════════════════════════════════════════════════════════════
# 1. Coordinate Mapping
# ═══════════════════════════════════════════════════════════════════════════════════

def phys_block_order_to_model_token_order(
    block_order, block_perm, n_blocks, block_len=BLOCK_LEN
):
    """
    Convert physical N-block order → model-coordinate token order.

    block_order: (B, n_blocks) — physical block index at each reveal step
    block_perm:  (64,)         — block_perm[phys64] = model64
    n_blocks:    int           — 16 or 64
    Returns:     (B, 256)      — token-level order for AO-GPT forward_fn
    """
    B, N = block_order.shape
    tokens_per_block = SEQ_LEN // n_blocks
    n64_per_step = N64 // n_blocks  # 4 for n_blocks=16, 1 for n_blocks=64
    device = block_order.device
    bp = block_perm.to(device)

    token_order = torch.zeros(B, SEQ_LEN, dtype=torch.long, device=device)
    for t in range(N):
        phys_blk = block_order[:, t]  # (B,)
        for a in range(n64_per_step):
            phys_n64 = phys_blk * n64_per_step + a
            model_n64 = bp[phys_n64]
            for k in range(block_len):
                pos = t * tokens_per_block + a * block_len + k
                token_order[:, pos] = model_n64 * block_len + k
    return token_order


def block_order_to_token_order_np(block_order, block_len=BLOCK_LEN):
    """(N,) numpy block indices → (N*block_len,) numpy token indices."""
    N = len(block_order)
    token_order = np.zeros(N * block_len, dtype=np.int64)
    for i, blk in enumerate(block_order):
        for k in range(block_len):
            token_order[i * block_len + k] = blk * block_len + k
    return token_order


def run_roundtrip_tests(aogpt, idx_model, block_perm, n_blocks):
    """Verify coordinate mapping correctness before training."""
    device = idx_model.device
    bp = block_perm.cpu().numpy()
    tokens_per_block = SEQ_LEN // n_blocks

    # Test 1: phys L2R → token order valid
    phys_l2r = torch.arange(n_blocks, device=device).unsqueeze(0)  # (1, n_blocks)
    token_new = phys_block_order_to_model_token_order(phys_l2r, block_perm, n_blocks)
    vals = token_new[0].tolist()
    assert len(set(vals)) == 256, f"ROUNDTRIP FAIL: duplicate tokens ({len(set(vals))} unique)"
    assert min(vals) == 0 and max(vals) == 255, f"ROUNDTRIP FAIL: out-of-range tokens"
    print(f"  [PASS] Test 1: L2R order valid (256 unique tokens)")

    # Test 2: NLL finite
    nll_val = compute_per_seq_nll(aogpt, idx_model[:1], token_new)
    assert torch.isfinite(nll_val), f"ROUNDTRIP FAIL: NLL is {nll_val.item()}"
    print(f"  [PASS] Test 2: NLL finite ({nll_val.item():.4f})")

    # Test 3: Random orders have all indices 0..255, no duplicates
    for _ in range(4):
        rand_order = torch.stack([torch.randperm(n_blocks, device=device) for _ in range(4)])
        token_rand = phys_block_order_to_model_token_order(rand_order, block_perm, n_blocks)
        for b in range(4):
            vals = token_rand[b].tolist()
            assert len(set(vals)) == 256, f"ROUNDTRIP FAIL: duplicate tokens in seq {b}"
            assert min(vals) == 0 and max(vals) == 255, f"ROUNDTRIP FAIL: out-of-range tokens"
    print("  [PASS] Test 3: random orders valid (0..255, no duplicates)")

    print("  All roundtrip tests passed.\n")


# ═══════════════════════════════════════════════════════════════════════════════════
# 2. Data & Model Loading
# ═══════════════════════════════════════════════════════════════════════════════════

def load_wikitext_sequences(min_len=SEQ_LEN, max_count=None):
    """Load token sequences from wikitext-103 test set (physical order)."""
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


def load_train_chunks(n_chunks):
    """Load token chunks from wikitext-103 TRAIN set (same logic as extract_train_A.py).

    Long texts produce sliding windows (stride=128); short texts accumulate in a
    buffer and get concatenated into 256-token chunks. Deterministic.
    """
    from tqdm import tqdm
    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)

    chunks = []
    buffer_ids = []

    for shard in ["wikitext-train-00000-of-00002.arrow", "wikitext-train-00001-of-00002.arrow"]:
        ds = Dataset.from_file(os.path.join(WIKITEXT_DIR, shard))
        for ex in tqdm(ds, desc=f"Chunking {shard}", unit=" texts"):
            ids = tok.encode(ex["text"])
            if len(ids) < SEQ_LEN:
                buffer_ids.extend(ids)
                while len(buffer_ids) >= SEQ_LEN:
                    chunks.append(torch.tensor(buffer_ids[:SEQ_LEN], dtype=torch.long))
                    buffer_ids = buffer_ids[SEQ_LEN:]
                    if len(chunks) >= n_chunks:
                        break
            else:
                for start in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN // 2):
                    chunks.append(torch.tensor(ids[start:start + SEQ_LEN], dtype=torch.long))
                    if len(chunks) >= n_chunks:
                        break
            if len(chunks) >= n_chunks:
                break
        if len(chunks) >= n_chunks:
            break
    return torch.stack(chunks)  # (n_chunks, 256)


def load_aogpt(ckpt_path, device, freeze=True):
    """Load AO-GPT checkpoint. Returns (model, block_perm, inv_perm)."""
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
    if freeze:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    block_perm = torch.tensor(ckpt["data_permutation"]["block_perm"], dtype=torch.long)
    inv_perm = torch.tensor(ckpt["data_permutation"]["inverse_block_perm"], dtype=torch.long)
    return model, block_perm, inv_perm


def phys_to_model_idx(idx_phys, inv_perm):
    """Convert token sequences from physical order to model-coordinate order."""
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


# ═══════════════════════════════════════════════════════════════════════════════════
# 3. AO-GPT Inference (per-sequence NLL)
# ═══════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_per_seq_nll(model, idx, orders):
    """Per-sequence NLL. Replicates forward_fn but returns (B,) NLL."""
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


# ═══════════════════════════════════════════════════════════════════════════════════
# 4. Sequential ON Sampling
# ═══════════════════════════════════════════════════════════════════════════════════

def sample_orders_from_on(on_model, A, temperature, K, ref_on=None):
    """
    Sample K full orders per sequence from ON policy via sequential PL sampling.

    Returns:
        all_orders:    list[K] of (B, N16) LongTensor
        all_log_probs: list[K] of (B,) FloatTensor  — sum over steps, WITH grad
        all_entropies: list[K] of (B,) FloatTensor  — mean per-step entropy, WITH grad
        kl_states:     list of (visited_mask, last_node, A) for KL computation
                       (from both policy and random trajectories)
    """
    B, N, _ = A.shape
    device = A.device

    all_orders = []
    all_log_probs = []
    all_entropies = []
    kl_states = []

    A_clone = A.clone()  # defensive: avoid view-related issues

    # ── K policy orders ──
    for k in range(K):
        visited_mask = torch.zeros(B, dtype=torch.long, device=device)
        last_node = torch.zeros(B, dtype=torch.long, device=device)
        order_k = []
        log_prob_k = torch.zeros(B, device=device)
        entropy_k = torch.zeros(B, device=device)

        for step in range(N):
            logits = on_model(A_clone, visited_mask, last_node)  # (B, N)
            logits_t = logits / temperature
            logits_t = torch.clamp(logits_t, min=-50.0, max=50.0)
            visited_bool = masks_to_revealed_bool(visited_mask, N)
            logits_t = logits_t.masked_fill(visited_bool, float("-inf"))

            dist = torch.distributions.Categorical(logits=logits_t)
            chosen = dist.sample()  # (B,)
            log_prob_k = log_prob_k + dist.log_prob(chosen)
            entropy_k = entropy_k + dist.entropy()

            order_k.append(chosen)
            # Record KL states: every 4 steps, AND step 0
            if step % 4 == 0 or step == 0:
                kl_states.append((visited_mask.clone(), last_node.clone(), A))

            visited_mask = visited_mask | (1 << chosen)
            last_node = chosen

        all_orders.append(torch.stack(order_k, dim=1))
        all_log_probs.append(log_prob_k)
        all_entropies.append(entropy_k / N)

    # ── 1 random N16 trajectory for KL coverage (same 4-step sampling) ──
    with torch.no_grad():
        visited_mask = torch.zeros(B, dtype=torch.long, device=device)
        last_node = torch.zeros(B, dtype=torch.long, device=device)
        for step in range(N):
            if step % 4 == 0 or step == 0:
                kl_states.append((visited_mask.clone(), last_node.clone(), A))

            logits = on_model(A, visited_mask, last_node)
            visited_bool = masks_to_revealed_bool(visited_mask, N)
            logits = logits.masked_fill(visited_bool, float("-inf"))
            # Use high temperature for random-like coverage
            probs = F.softmax(logits / 5.0, dim=-1)
            chosen = torch.multinomial(probs, 1).squeeze(-1)
            visited_mask = visited_mask | (1 << chosen)
            last_node = chosen

    return all_orders, all_log_probs, all_entropies, kl_states


# ═══════════════════════════════════════════════════════════════════════════════════
# 5. Reward Scoring
# ═══════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def score_orders(aogpt, idx_model, all_orders, block_perm, n_blocks,
                 num_random_baselines=4):
    """
    Score K policy orders + random baselines with frozen AO-GPT.

    Returns:
        policy_nlls:    (K, B)  — NLL for each policy order
        random_nll:     (B,)    — mean NLL over num_random_baselines random orders
        random64_nll:   (B,)    — diagnostic: mean NLL over random N64 orders
                                  (only meaningful for n_blocks=16)
    """
    K = len(all_orders)
    B = all_orders[0].shape[0]
    device = idx_model.device

    # ── Policy orders: process in micro-batches to avoid OOM ──
    max_batch = 32  # max seqs per AO-GPT forward
    policy_nlls_list = []
    for k_start in range(0, K, max_batch // B):
        k_end = min(k_start + max_batch // B, K)
        k_slice = slice(k_start, k_end)
        K_slice = k_end - k_start
        orders_flat = torch.cat(all_orders[k_slice], dim=0)  # (K_slice * B, n_blocks)
        token_orders_flat = phys_block_order_to_model_token_order(
            orders_flat, block_perm, n_blocks
        )
        idx_flat = idx_model.unsqueeze(0).expand(K_slice, -1, -1).reshape(K_slice * B, -1)
        nlls_flat = compute_per_seq_nll(aogpt, idx_flat, token_orders_flat)
        policy_nlls_list.append(nlls_flat.reshape(K_slice, B))
    policy_nlls = torch.cat(policy_nlls_list, dim=0)  # (K, B)

    # ── Random baselines (same action space) ──
    random_nlls_sum = torch.zeros(B, device=device)
    for m in range(num_random_baselines):
        rand_order = torch.stack([torch.randperm(n_blocks, device=device) for _ in range(B)])
        rand_token = phys_block_order_to_model_token_order(rand_order, block_perm, n_blocks)
        nll_m = compute_per_seq_nll(aogpt, idx_model, rand_token)
        random_nlls_sum = random_nlls_sum + nll_m
    random_nll = random_nlls_sum / num_random_baselines

    # ── Random N64 diagnostic (finer-grained, for reference only) ──
    if n_blocks == N64:
        random64_nll = random_nll  # same action space, no finer-grained baseline
    else:
        rand64_nlls = []
        for b in range(B):
            rand64 = torch.randperm(N64, device=device)
            token64 = torch.repeat_interleave(rand64, BLOCK_LEN) * BLOCK_LEN + \
                      torch.arange(BLOCK_LEN, device=device).repeat(N64)
            nll64 = compute_per_seq_nll(aogpt, idx_model[b:b+1], token64.unsqueeze(0))
            rand64_nlls.append(nll64.item())
        random64_nll = torch.tensor(rand64_nlls, device=device)

    return policy_nlls, random_nll, random64_nll


# ═══════════════════════════════════════════════════════════════════════════════════
# 6. Loss Functions
# ═══════════════════════════════════════════════════════════════════════════════════

def compute_grpo_loss(log_probs_stack, rewards_stack):
    """
    GRPO loss with group-relative advantage (normalized per-sequence over K orders).

    log_probs_stack: (K, B) — log π(order_k | A)
    rewards_stack:   (K, B) — R_k = random_nll - policy_nll
    """
    mean_r = rewards_stack.mean(dim=0)  # (B,)
    std_r = rewards_stack.std(dim=0) + 1e-8  # (B,)
    advantages = (rewards_stack - mean_r.unsqueeze(0)) / std_r.unsqueeze(0)  # (K, B)
    return -(advantages * log_probs_stack).mean()


def compute_rank_loss(log_probs_stack, rewards_stack, beta=0.5):
    """
    Quartile pairwise rank loss: top 25% vs bottom 25% mean log prob.

    log_probs_stack: (K, B)
    rewards_stack:   (K, B)
    """
    K, B = rewards_stack.shape
    device = rewards_stack.device
    n_q = max(1, K // 4)

    _, sorted_idx = rewards_stack.sort(dim=0)  # ascending reward
    top_idx = sorted_idx[-n_q:, :]   # (n_q, B)
    bot_idx = sorted_idx[:n_q, :]    # (n_q, B)

    batch_idx = torch.arange(B, device=device).unsqueeze(0).expand(n_q, -1)
    logp_top = log_probs_stack[top_idx, batch_idx].mean()
    logp_bot = log_probs_stack[bot_idx, batch_idx].mean()

    return -F.logsigmoid(beta * (logp_top - logp_bot))


def compute_masked_kl(on_model, ref_on, A, visited_mask, last_node):
    """KL(Categorical(cur_masked) || Categorical(ref_masked))."""
    N = A.shape[1]
    with torch.no_grad():
        ref_logits = ref_on(A, visited_mask, last_node)
    cur_logits = on_model(A, visited_mask, last_node)

    visited_bool = masks_to_revealed_bool(visited_mask, N)
    cur_logits = cur_logits.masked_fill(visited_bool, float("-inf"))
    ref_logits = ref_logits.masked_fill(visited_bool, float("-inf"))

    cur_log_probs = F.log_softmax(cur_logits, dim=-1)
    cur_probs = F.softmax(cur_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # Sanitize visited positions BEFORE subtraction: -inf - (-inf) = NaN.
    # NaN * 0 = NaN, so we must zero out log_probs, not just mask the product.
    cur_log_probs = cur_log_probs.masked_fill(visited_bool, 0.0)
    cur_probs = cur_probs.masked_fill(visited_bool, 0.0)
    ref_log_probs = ref_log_probs.masked_fill(visited_bool, 0.0)

    kl = (cur_probs * (cur_log_probs - ref_log_probs)).sum(dim=-1)  # (B,)
    return kl.mean()


def compute_kl_from_states(on_model, ref_on, kl_states):
    """Average KL over all sampled prefix states."""
    if not kl_states:
        return torch.tensor(0.0, device=next(on_model.parameters()).device)
    kl_sum = 0.0
    for visited_mask, last_node, A in kl_states:
        kl_sum = kl_sum + compute_masked_kl(on_model, ref_on, A, visited_mask, last_node)
    return kl_sum / len(kl_states)


# ═══════════════════════════════════════════════════════════════════════════════════
# 7. Temperature Schedule
# ═══════════════════════════════════════════════════════════════════════════════════

def get_temperature(epoch, args):
    if epoch < 30:
        return args.tau_init
    elif epoch < 60:
        return args.tau_mid
    else:
        return args.tau_final


# ═══════════════════════════════════════════════════════════════════════════════════
# 8. Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_diagnostics(
    on_model, aogpt, A_batch, idx_batch, block_perm, ref_on, tau, K, n_blocks, args
):
    """Compute all diagnostic metrics on a batch. Returns dict."""
    B = A_batch.shape[0]
    device = A_batch.device

    on_model.eval()
    all_orders, all_log_probs, all_entropies, kl_states = sample_orders_from_on(
        on_model, A_batch, tau, K, ref_on
    )
    policy_nlls, random_nll, random64_nll = score_orders(
        aogpt, idx_batch, all_orders, block_perm, n_blocks,
        num_random_baselines=args.num_random_baselines
    )
    on_model.train()

    log_probs_stack = torch.stack(all_log_probs)    # (K, B)
    entropies_stack = torch.stack(all_entropies)    # (K, B)

    rewards = random_nll.unsqueeze(0) - policy_nlls  # (K, B)
    reward_mean = rewards.mean().item()
    reward_std = rewards.std().item()

    # Top-bottom gap: mean reward(top 25%) - mean reward(bottom 25%)
    n_q = max(1, K // 4)
    top_r = rewards.topk(n_q, dim=0).values.mean().item()
    bot_r = rewards.topk(n_q, dim=0, largest=False).values.mean().item()
    top_bottom_gap = top_r - bot_r

    # Entropy
    entropy_step = entropies_stack.mean().item()
    effective_ss = np.exp(entropy_step)

    # KL
    kl_val = compute_kl_from_states(on_model, ref_on, kl_states).item()

    # Unique order ratio
    all_orders_stack = torch.stack(all_orders, dim=0)  # (K, B, n_blocks)
    unique_count = 0
    for b in range(B):
        orders_b = [tuple(all_orders_stack[k, b].tolist()) for k in range(K)]
        unique_count += len(set(orders_b))
    unique_ratio = unique_count / (K * B)

    # Variance decomposition: within-seq vs across-seq
    within_var = rewards.var(dim=0).mean().item()  # mean over sequences
    across_var = rewards.mean(dim=0).var().item()  # var of sequence means
    var_ratio = across_var / (within_var + 1e-8)

    return {
        "policy_nll_mean": policy_nlls.mean().item(),
        "policy_nll_std": policy_nlls.std().item(),
        "random_nll_mean": random_nll.mean().item(),
        "random_nll_std": random_nll.std().item(),
        "random64_nll_mean": random64_nll.mean().item(),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "top_bottom_gap": top_bottom_gap,
        "entropy_step": entropy_step,
        "effective_subset_size": effective_ss,
        "KL_cur_ref": kl_val,
        "unique_order_ratio": unique_ratio,
        "within_seq_reward_var": within_var,
        "across_seq_reward_var": across_var,
        "var_ratio": var_ratio,
        "tau": tau,
    }


def print_diagnostics(d, prefix="train"):
    print(
        f"  [{prefix}] "
        f"policy_nll={d['policy_nll_mean']:.4f}±{d['policy_nll_std']:.4f} | "
        f"rand_nll={d['random_nll_mean']:.4f}±{d['random_nll_std']:.4f} | "
        f"rand64_nll={d['random64_nll_mean']:.4f} | "
        f"reward={d['reward_mean']:.4f}±{d['reward_std']:.4f} | "
        f"gap={d['top_bottom_gap']:.4f} | "
        f"H={d['entropy_step']:.3f} (ess={d['effective_subset_size']:.1f}) | "
        f"KL={d['KL_cur_ref']:.4f} | "
        f"uniq={d['unique_order_ratio']:.2f} | "
        f"var_ratio={d['var_ratio']:.2f} | "
        f"tau={d['tau']:.2f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════════
# 9. BC Pretraining (online teacher forcing from NN paths)
# ═══════════════════════════════════════════════════════════════════════════════════

def _bc_epoch(on_model, A_seq, paths, batch_size, n_blocks, optimizer=None):
    """One epoch of online teacher forcing. optimizer=None → eval mode."""
    B = A_seq.shape[0]
    N = n_blocks
    device = A_seq.device
    training = optimizer is not None

    if training:
        on_model.train()
    else:
        on_model.eval()

    perm = torch.randperm(B, device=device)
    total_loss = 0.0
    total_correct = 0
    n_batches = 0

    for batch_start in range(0, B, batch_size):
        batch_idx = perm[batch_start:batch_start + batch_size]
        A_batch = A_seq[batch_idx]
        path_batch = paths[batch_idx]

        B_cur = A_batch.shape[0]
        visited = torch.zeros(B_cur, dtype=torch.long, device=device)
        last_node = torch.zeros(B_cur, dtype=torch.long, device=device)

        batch_loss = 0.0
        batch_correct = 0

        with torch.set_grad_enabled(training):
            for step in range(N):
                logits = on_model(A_batch, visited, last_node)
                targets = path_batch[:, step]
                batch_loss += F.cross_entropy(logits, targets)
                pred = logits.argmax(dim=-1)
                batch_correct += (pred == targets).sum().item()
                visited = visited | (1 << targets)
                last_node = targets

        loss = batch_loss / N
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(on_model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_correct += batch_correct
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = total_correct / (B * N)
    return avg_loss, accuracy


# ═══════════════════════════════════════════════════════════════════════════════════
# 10. Training Loop
# ═══════════════════════════════════════════════════════════════════════════════════

def train_grpo(args):
    device = torch.device(args.device)
    Config.seed = args.seed
    Config.set_seed()
    print(f"Device: {device}")

    # ── Load data ──
    print("Loading A matrices...")
    A_all = np.load(args.a_matrices)
    num_seqs = A_all.shape[0]
    print(f"  {num_seqs} sequences, shape {A_all.shape}")

    print("Loading token sequences...")
    if args.data_source == "train":
        idx_phys = load_train_chunks(num_seqs)
        idx_phys = idx_phys[:num_seqs]
    else:
        idx_phys = load_wikitext_sequences(max_count=num_seqs)
    idx_phys = idx_phys[:num_seqs]
    print(f"  {idx_phys.shape}")

    # ── Load AO-GPT (frozen scoring) ──
    print("Loading AO-GPT (frozen)...")
    aogpt, block_perm, inv_perm = load_aogpt(args.ao_gpt_ckpt, device, freeze=True)
    n_params = sum(p.numel() for p in aogpt.parameters())
    print(f"  AO-GPT loaded. Params: {n_params:,}")

    # Convert token sequences to model coordinates
    idx_model_list = []
    for i in range(idx_phys.shape[0]):
        idx_model_list.append(phys_to_model_idx(idx_phys[i:i + 1], inv_perm))
    idx_all = torch.cat(idx_model_list, dim=0).to(device)
    print(f"  Model-coordinate tokens: {idx_all.shape}")

    # ── Run roundtrip tests ──
    if not args.skip_tests:
        print("Running roundtrip tests...")
        run_roundtrip_tests(aogpt, idx_all, block_perm, args.n_blocks)

    # ── Load ON (warm start or from scratch) ──
    print("Loading Order Network...")
    n_blocks = args.n_blocks
    if args.on_ckpt:
        on_ckpt = torch.load(args.on_ckpt, map_location=device, weights_only=False)
        sd = on_ckpt["model_state_dict"]
        d_edge = sd["edge_mlp.0.weight"].shape[0]
        d_model = sd["score_mlp.0.weight"].shape[0]
        print(f"  d_edge={d_edge}, d_model={d_model}, "
              f"best_val_acc={on_ckpt.get('best_val_acc', '?')}")
        on_model = CrossAttentionOrderNetwork(num_blocks=n_blocks, d_edge=d_edge, d_model=d_model)
        # Allow loading N=16 ckpt into N=64 model (load matching params only)
        model_sd = on_model.state_dict()
        matched = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        skipped = len(sd) - len(matched)
        model_sd.update(matched)
        on_model.load_state_dict(model_sd)
        if skipped > 0:
            print(f"  Note: skipped {skipped} params (shape mismatch, likely n_blocks differs)")
    else:
        d_edge = args.d_edge
        d_model = args.d_model
        print(f"  Training from scratch, d_edge={d_edge}, d_model={d_model}")
        on_model = CrossAttentionOrderNetwork(num_blocks=n_blocks, d_edge=d_edge, d_model=d_model)
    on_model.to(device)
    on_model.train()
    n_on = sum(p.numel() for p in on_model.parameters() if p.requires_grad)
    print(f"  ON params: {n_on:,}")

    # ── Reference policy (frozen copy) ──
    ref_on = copy.deepcopy(on_model)
    ref_on.eval()
    for p in ref_on.parameters():
        p.requires_grad = False

    # ── Train/val split ──
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(num_seqs)
    val_size = max(1, int(round(num_seqs * args.val_fraction)))
    val_indices = set(perm[:val_size].tolist())
    train_indices = sorted(set(perm[val_size:].tolist()))
    print(f"  Train: {len(train_indices)} seqs, Val: {len(val_indices)} seqs")

    # Pre-load A and idx for fast access
    A_train = torch.as_tensor(A_all[train_indices], dtype=torch.float32, device=device)
    idx_train = idx_all[train_indices]
    A_val = torch.as_tensor(A_all[list(val_indices)], dtype=torch.float32, device=device)
    idx_val = idx_all[list(val_indices)]

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        on_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ── BC Pretraining (online teacher forcing from NN paths) ──
    if args.bc_paths:
        print(f"\n{'='*80}")
        print(f"BC Pretraining: N={n_blocks}, {args.bc_epochs} epochs, "
              f"lr={args.bc_lr}, batch_size={args.batch_size}")
        print(f"{'='*80}\n")

        paths_data = np.load(args.bc_paths)
        paths_all = torch.tensor(paths_data["paths"], dtype=torch.long, device=device)
        paths_train = paths_all[train_indices]
        paths_val = paths_all[list(val_indices)]

        bc_optimizer = torch.optim.AdamW(
            on_model.parameters(), lr=args.bc_lr, weight_decay=args.weight_decay
        )

        t_bc = time.time()
        for epoch in range(1, args.bc_epochs + 1):
            train_loss, train_acc = _bc_epoch(
                on_model, A_train, paths_train, args.batch_size, n_blocks,
                optimizer=bc_optimizer
            )
            val_loss, val_acc = _bc_epoch(
                on_model, A_val, paths_val, args.batch_size, n_blocks,
                optimizer=None
            )
            print(f"  BC {epoch:3d}/{args.bc_epochs} | "
                  f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.3f}")

        bc_elapsed = time.time() - t_bc
        print(f"\n  BC done in {bc_elapsed:.0f}s. "
              f"Final val_acc={val_acc:.3f}")

        # Save BC checkpoint
        bc_output = Path(args.output).with_suffix(".bc.pt")
        torch.save(
            {
                "model_state_dict": {k: v.cpu().clone() for k, v in on_model.state_dict().items()},
                "args": vars(args),
                "bc_val_acc": val_acc,
                "bc_val_loss": val_loss,
            },
            bc_output,
        )
        print(f"  BC ckpt saved: {bc_output}")

        # Update ref_on to BC checkpoint
        ref_on.load_state_dict(on_model.state_dict())
        print("  ref_on ← BC ckpt\n")

    # ── GRPO Training ──
    history = []
    best_val_reward = -float("inf")
    best_epoch = 0
    best_state = None
    n_train = len(train_indices)

    print(f"\n{'='*80}")
    print(f"GRPO Training: N={n_blocks}, {args.epochs} epochs, K={args.K}, "
          f"lr={args.lr}, batch_size={args.batch_size}")
    print(f"  w_grpo={args.w_grpo}, w_rank={args.w_rank}, "
          f"lambda_entropy={args.lambda_entropy}, gamma_kl={args.gamma_kl}")
    print(f"{'='*80}\n")

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        tau = get_temperature(epoch, args)

        # Shuffle training sequences
        epoch_order = list(range(n_train))
        np.random.shuffle(epoch_order)

        on_model.train()
        epoch_loss = 0.0
        epoch_n_batches = 0

        for batch_start in range(0, n_train, args.batch_size):
            batch_idx = epoch_order[batch_start:batch_start + args.batch_size]
            B = len(batch_idx)

            A_batch = A_train[batch_idx]  # (B, n_blocks, n_blocks)
            idx_batch = idx_train[batch_idx]  # (B, 256)

            # 1. Sample K orders with log_probs and entropies (WITH grad)
            all_orders, all_log_probs, all_entropies, kl_states = sample_orders_from_on(
                on_model, A_batch, tau, args.K, ref_on
            )

            # 2. Score orders with frozen AO-GPT (no grad)
            policy_nlls, random_nll, _ = score_orders(
                aogpt, idx_batch, all_orders, block_perm, n_blocks,
                num_random_baselines=args.num_random_baselines
            )

            # 3. Compute rewards
            rewards = random_nll.unsqueeze(0) - policy_nlls  # (K, B)

            # 4. Losses
            log_probs_stack = torch.stack(all_log_probs)   # (K, B)
            entropies_stack = torch.stack(all_entropies)   # (K, B)

            loss_grpo = compute_grpo_loss(log_probs_stack, rewards)
            loss_rank = compute_rank_loss(log_probs_stack, rewards, beta=args.rank_beta)
            entropy = entropies_stack.mean()
            loss_kl = compute_kl_from_states(on_model, ref_on, kl_states)

            loss = (
                args.w_grpo * loss_grpo
                + args.w_rank * loss_rank
                - args.lambda_entropy * entropy
                + args.gamma_kl * loss_kl
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(on_model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_n_batches += 1

        avg_loss = epoch_loss / max(epoch_n_batches, 1)

        # ── Logging ──
        if epoch % args.log_interval == 0 or epoch == 1:
            # Train diagnostics on a random subset for speed
            diag_idx = np.random.choice(n_train, min(32, n_train), replace=False)
            diag_A = A_train[diag_idx]
            diag_idx_t = idx_train[diag_idx]
            train_d = compute_diagnostics(
                on_model, aogpt, diag_A, diag_idx_t, block_perm, ref_on, tau,
                args.K, n_blocks, args
            )

            # Val diagnostics
            n_val = len(list(val_indices))
            val_idx = np.random.choice(n_val, min(32, n_val), replace=False)
            val_d = compute_diagnostics(
                on_model, aogpt, A_val[val_idx], idx_val[val_idx], block_perm,
                ref_on, tau, args.K, n_blocks, args
            )

            elapsed = time.time() - t_start
            print(f"Epoch {epoch:4d}/{args.epochs} | loss={avg_loss:.4f} | "
                  f"time={elapsed:.0f}s")
            print_diagnostics(train_d, "train")
            print_diagnostics(val_d, "  val")

            # Early stopping based on val reward
            if val_d["reward_mean"] > best_val_reward:
                best_val_reward = val_d["reward_mean"]
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in on_model.state_dict().items()}
                print(f"  >>> New best val_reward={best_val_reward:.4f}")

            history.append({
                "epoch": epoch, "loss": avg_loss, "tau": tau,
                "train": train_d, "val": val_d,
            })

    # ── Save ──
    elapsed = time.time() - t_start
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "args": vars(args),
            "history": history,
            "best_val_reward": best_val_reward,
            "best_epoch": best_epoch,
            "training_time_s": elapsed,
        },
        output_path,
    )
    print(f"\nBest val_reward={best_val_reward:.4f} at epoch {best_epoch}")
    print(f"Saved: {output_path} ({elapsed:.0f}s total)")

    return history


# ═══════════════════════════════════════════════════════════════════════════════════
# 10. CLI
# ═══════════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="GRPO Training for Order Network")
    # Paths
    p.add_argument("--a-matrices", default="probe_results/A_train_10k.npy")
    p.add_argument("--on-ckpt", default=None,
                   help="Warm-start checkpoint (N=16 ckpt loaded partially into N=64 model)")
    p.add_argument("--ao-gpt-ckpt", default=AO_GPT_CKPT)
    p.add_argument("--output", default="probe_results/grpo_on_n16.pt")
    # Model
    p.add_argument("--n-blocks", type=int, default=16,
                   help="Number of ON-action blocks (16 or 64)")
    p.add_argument("--d-edge", type=int, default=128,
                   help="Edge MLP hidden dim (used when training from scratch)")
    p.add_argument("--d-model", type=int, default=128,
                   help="Score MLP hidden dim (used when training from scratch)")
    # BC pretraining
    p.add_argument("--bc-paths", default=None,
                   help="Path to NN paths .npz for BC pretraining (skip if None)")
    p.add_argument("--bc-epochs", type=int, default=30,
                   help="Number of BC pretraining epochs")
    p.add_argument("--bc-lr", type=float, default=1e-3,
                   help="BC pretraining learning rate")
    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Sampling
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--num-random-baselines", type=int, default=4)
    # Temperature
    p.add_argument("--tau-init", type=float, default=2.0)
    p.add_argument("--tau-mid", type=float, default=1.0)
    p.add_argument("--tau-final", type=float, default=0.5)
    # Loss weights
    p.add_argument("--w-grpo", type=float, default=0.3)
    p.add_argument("--w-rank", type=float, default=1.0)
    p.add_argument("--lambda-entropy", type=float, default=0.02)
    p.add_argument("--gamma-kl", type=float, default=0.05)
    p.add_argument("--rank-beta", type=float, default=0.5)
    # Data
    p.add_argument("--data-source", default="train", choices=["test", "train"],
                   help="'test' = wikitext-103 test set, 'train' = train set via sliding-window chunks")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    # Logging
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument("--val-interval", type=int, default=5)
    # Debug
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_grpo(args)
