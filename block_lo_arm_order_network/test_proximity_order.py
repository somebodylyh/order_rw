"""
Proximity ordering: symmetrize A → W → max-Ham-path / NN / spectral → NLL.

Tests whether block-level attention proximity encodes useful ordering signal.

Algorithms (all on W = 0.5*(A + A.T)):
  1. Max-Ham DP  — bitmask DP, O(N²·2^N), global optimum on W
  2. NN greedy   — low-degree endpoint start, nearest-neighbor
  3. Spectral    — Laplacian Fiedler vector argsort

A matrix: raw attention from AO-GPT (with [None] signal), 1 forward/seq.
"""

import os, sys, time, argparse
from itertools import permutations

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

# ── Paths ──
AO_GPT_CKPT = os.path.expanduser(
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

SEQ_LEN = 256
N16 = 16
SUB_BLOCKS = 4
BLOCK_LEN = 4
N64 = N16 * SUB_BLOCKS


# ═══════════════════════════════════════════════════════════════════════════════════
# Model & Data
# ═══════════════════════════════════════════════════════════════════════════════════

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


def load_sequences(max_count=200):
    ds = Dataset.from_file(WIKITEXT_ARROW)
    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    seqs = []
    for ex in ds:
        ids = tok.encode(ex["text"])
        if len(ids) >= SEQ_LEN:
            seqs.append(ids[:SEQ_LEN])
        if len(seqs) >= max_count:
            break
    return seqs


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


# ═══════════════════════════════════════════════════════════════════════════════════
# Attention Extraction (with [None] signal)
# ═══════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_block_attention_with_none(model, idx_model, block_perm):
    """
    Extract block-level attention A (16×16) with [None] token signal.
    A[i,j] = mean attention from phys block i to phys block j + 0.1 * attention(i to [None]).
    """
    device = idx_model.device
    bp = block_perm.cpu().numpy()

    rand_n16 = torch.randperm(N16, device=device)
    n64_order = []
    for t in range(N16):
        phys_n16_blk = rand_n16[t].item()
        for a in range(SUB_BLOCKS):
            phys_n64_blk = phys_n16_blk * SUB_BLOCKS + a
            model_n64_blk = int(bp[phys_n64_blk])
            n64_order.append(model_n64_blk)
    token_order = torch.repeat_interleave(
        torch.tensor(n64_order, device=device), BLOCK_LEN
    ) * BLOCK_LEN + torch.arange(BLOCK_LEN, device=device).repeat(N64)
    token_order = token_order.unsqueeze(0)

    logits, _, attn_list = model.forward_fn(
        idx_model.unsqueeze(0), token_order, return_attentions=True
    )
    attn_stack = torch.stack(attn_list).squeeze(1).cpu().numpy()  # (L, H, 257, 257)

    # Select top-4 heads by off-diagonal variance
    L, H = attn_stack.shape[:2]
    head_vars = np.zeros(H)
    for h in range(H):
        content = attn_stack[:, h, 1:, 1:]
        offdiag = content[:, ~np.eye(256, dtype=bool)].reshape(L, 256, 255)
        head_vars[h] = float(np.var(offdiag))
    top_heads = np.argsort(head_vars)[-4:]

    avg_attn = attn_stack[:, top_heads, :, :].mean(axis=(0, 1))  # (257, 257)

    # Build remap: reveal_pos → physical_pos
    reveal_to_model = token_order[0].cpu().numpy()
    bp_i = np.zeros(64, dtype=np.int64)
    for i in range(64):
        bp_i[bp[i]] = i
    model_to_phys = np.zeros(256, dtype=np.int64)
    for model_pos in range(256):
        model_block = model_pos // BLOCK_LEN
        phys_block_64 = bp_i[model_block]
        phys_block_16 = phys_block_64 // SUB_BLOCKS
        offset = model_pos % BLOCK_LEN
        phys_pos = (phys_block_16 * SUB_BLOCKS * BLOCK_LEN +
                    (phys_block_64 % SUB_BLOCKS) * BLOCK_LEN + offset)
        model_to_phys[model_pos] = phys_pos
    reveal_to_phys = model_to_phys[reveal_to_model]

    # Content attention: remap to physical coords, aggregate to block level
    attn_content = avg_attn[1:, 1:]
    attn_phys = np.zeros((256, 256), dtype=np.float32)
    for rq in range(256):
        pq = reveal_to_phys[rq]
        for rk in range(256):
            pk = reveal_to_phys[rk]
            attn_phys[pq, pk] += attn_content[rq, rk]

    A = np.zeros((N16, N16), dtype=np.float32)
    for i in range(N16):
        for j in range(N16):
            isl = slice(i * 16, (i + 1) * 16)
            jsl = slice(j * 16, (j + 1) * 16)
            A[i, j] = attn_phys[isl, jsl].mean()

    # [None] signal: attention TO [None] (col 0, rows 1:)
    none_attn = avg_attn[1:, 0]  # (256,) — each content token's attention to [None]
    none_block = np.zeros(N16, dtype=np.float32)
    for i in range(N16):
        isl = slice(i * 16, (i + 1) * 16)
        none_block[i] = none_attn[isl].mean()
    A += none_block[np.newaxis, :] * 0.1

    np.fill_diagonal(A, 0.0)
    return A, none_block


# ═══════════════════════════════════════════════════════════════════════════════════
# Ordering Algorithms (on W = 0.5*(A + A.T))
# ═══════════════════════════════════════════════════════════════════════════════════

def proximity_matrix(A):
    """Symmetrize to get proximity/affinity matrix."""
    W = 0.5 * (A + A.T)
    np.fill_diagonal(W, 0.0)
    return W


# ── 1. Max Hamiltonian Path DP (theoretical upper bound) ──

def max_ham_path_dp(W):
    """
    Bitmask DP: find Hamiltonian path maximizing sum of W[prev][next].
    Returns (best_path, total_weight).
    """
    N = W.shape[0]
    total = 1 << N
    full = total - 1

    # dp[mask][last] = max weight path ending at `last` with visited set `mask`
    dp = np.full((total, N), -np.inf, dtype=np.float32)
    prev = np.full((total, N), -1, dtype=np.int8)

    # Single-node masks
    for i in range(N):
        dp[1 << i][i] = 0.0

    # Precompute popcount groups
    masks_by_pc = [[] for _ in range(N + 1)]
    for mask in range(total):
        masks_by_pc[bin(mask).count('1')].append(mask)

    for pc in range(1, N):
        for mask in masks_by_pc[pc]:
            in_nodes = [i for i in range(N) if mask & (1 << i)]
            out_nodes = [i for i in range(N) if not (mask & (1 << i))]
            for last in in_nodes:
                cur = dp[mask][last]
                if cur == -np.inf:
                    continue
                for nxt in out_nodes:
                    val = cur + W[last][nxt]
                    new_mask = mask | (1 << nxt)
                    if val > dp[new_mask][nxt]:
                        dp[new_mask][nxt] = val
                        prev[new_mask][nxt] = last

    # Best endpoint
    best_weight = -np.inf
    best_last = -1
    for i in range(N):
        if dp[full][i] > best_weight:
            best_weight = dp[full][i]
            best_last = i

    # Reconstruct
    path = [best_last]
    mask = full
    cur = best_last
    for _ in range(N - 1):
        p = prev[mask][cur]
        path.append(p)
        mask ^= (1 << cur)
        cur = p
    path.reverse()
    return np.array(path), float(best_weight)


# ── 2. Nearest Neighbor Greedy ──

def nn_greedy(W, start):
    """Nearest-neighbor greedy from a given start node."""
    N = W.shape[0]
    remaining = set(range(N))
    remaining.remove(start)
    order = [start]
    cur = start
    while remaining:
        nxt = max(remaining, key=lambda j: W[cur, j])
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return np.array(order)


def nn_best_endpoint(W, n_candidates=2):
    """NN from low-degree endpoint candidates, pick best total weight."""
    deg = W.sum(axis=1)
    candidates = np.argsort(deg)[:n_candidates]

    best_order, best_weight = None, -np.inf
    for start in candidates:
        order = nn_greedy(W, start)
        w = sum(W[order[i], order[i+1]] for i in range(len(order)-1))
        if w > best_weight:
            best_weight = w
            best_order = order
    return best_order, best_weight


# ── 3. Spectral Seriation ──

def spectral_seriation(W):
    """Laplacian Fiedler vector argsort."""
    N = W.shape[0]
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]  # second smallest eigenvector
    order = np.argsort(fiedler)
    return order


# ═══════════════════════════════════════════════════════════════════════════════════
# NLL Scoring
# ═══════════════════════════════════════════════════════════════════════════════════

def n16_to_token_order(n16_order, block_perm):
    bp = block_perm.cpu().numpy()
    n64_order = []
    for phys_n16_blk in n16_order:
        for a in range(SUB_BLOCKS):
            phys_n64_blk = phys_n16_blk * SUB_BLOCKS + a
            model_n64_blk = int(bp[phys_n64_blk])
            n64_order.append(model_n64_blk)
    token_order = np.zeros(N64 * BLOCK_LEN, dtype=np.int64)
    for i, blk in enumerate(n64_order):
        for k in range(BLOCK_LEN):
            token_order[i * BLOCK_LEN + k] = blk * BLOCK_LEN + k
    return token_order


@torch.no_grad()
def compute_nll(model, idx, order):
    device = idx.device
    T = idx.shape[0]
    idx_b = idx.unsqueeze(0)
    order_t = torch.from_numpy(order).to(device).unsqueeze(0)

    pos = torch.arange(0, T + 1, dtype=torch.long, device=device)
    batch_indices = torch.arange(1, device=device).unsqueeze(1).expand(-1, T)

    tok_emb = model.transformer.wte(idx_b)
    tok_emb = tok_emb[batch_indices, order_t]
    none_emb = model.transformer.wnonee(torch.tensor([[0]], device=device)).expand(1, -1, -1)
    tok_emb = torch.cat([none_emb, tok_emb], dim=1)

    pos_emb = model.transformer.wpe(pos).unsqueeze(0)
    pos_emb_prefix = pos_emb[:, :1, :]
    pos_emb_postfix = pos_emb[:, 1:, :][batch_indices, order_t]
    pos_emb_final = torch.cat([pos_emb_prefix, pos_emb_postfix], dim=1)

    tgt_emb = model.transformer.wtpe(pos[:T]).unsqueeze(0)
    tgt_emb_prefix = tgt_emb[batch_indices, order_t]
    tgt_emb_postfix = torch.zeros(1, 1, tgt_emb.shape[-1], device=device)
    c = torch.cat([tgt_emb_prefix, tgt_emb_postfix], dim=1)

    targets = idx_b[batch_indices, order_t]
    x = tok_emb + pos_emb_final
    x = model.transformer.drop(x)
    for block in model.transformer.h:
        x = block(x, c)
    x = model.transformer.final_layer(x, c)

    logits = model.lm_head(x)[:, :-1, :].contiguous()
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="mean", ignore_index=-1,
    )
    return ce.item()


# ═══════════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seqs", type=int, default=50)
    parser.add_argument("--n-random", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Proximity Ordering Test: symmetrize(A) → W → DP/NN/Spectral → NLL")
    print("=" * 60)

    print("\n[1/4] Loading AO-GPT...")
    aogpt, block_perm, inv_perm = load_aogpt(AO_GPT_CKPT, device)
    print(f"  OK ({sum(p.numel() for p in aogpt.parameters()):,} params)")

    print(f"\n[2/4] Loading sequences ({args.n_seqs})...")
    seqs = load_sequences(max_count=args.n_seqs)
    n_seqs = len(seqs)
    print(f"  {n_seqs} sequences")

    idx_phys = torch.tensor(seqs, dtype=torch.long)
    idx_model_list = [phys_to_model_idx(idx_phys[i:i+1].to(device), inv_perm) for i in range(n_seqs)]
    idx_all = torch.cat(idx_model_list, dim=0)

    # ── Process each sequence ──
    methods = ["dp", "nn", "spectral", "random"]
    results = {m: [] for m in methods}
    dp_weights = []
    nn_weights = []
    asymmetry_vals = []

    print(f"\n[3/4] Processing {n_seqs} sequences...")
    t0 = time.time()

    for s in range(n_seqs):
        if s % 10 == 0:
            print(f"  Seq {s}/{n_seqs}...")

        idx_s = idx_all[s]

        # Extract attention A (with [None] signal)
        A, none_block = extract_block_attention_with_none(aogpt, idx_s, block_perm)

        # Symmetrize → W
        W = proximity_matrix(A)
        asymmetry_vals.append(np.mean([abs(A[i,j] - A[j,i]) for i in range(N16) for j in range(N16) if i != j]))

        # DP
        dp_order, dp_w = max_ham_path_dp(W)
        dp_weights.append(dp_w)
        token_dp = n16_to_token_order(dp_order, block_perm)
        results["dp"].append(compute_nll(aogpt, idx_s, token_dp))

        # NN (low-degree endpoint)
        nn_order, nn_w = nn_best_endpoint(W)
        nn_weights.append(nn_w)
        token_nn = n16_to_token_order(nn_order, block_perm)
        results["nn"].append(compute_nll(aogpt, idx_s, token_nn))

        # Spectral
        sp_order = spectral_seriation(W)
        token_sp = n16_to_token_order(sp_order, block_perm)
        results["spectral"].append(compute_nll(aogpt, idx_s, token_sp))

        # Random baseline
        rand_nlls = []
        for _ in range(args.n_random):
            rand_phys = np.random.permutation(N16)
            token_rand = n16_to_token_order(rand_phys, block_perm)
            rand_nlls.append(compute_nll(aogpt, idx_s, token_rand))
        results["random"].append(np.mean(rand_nlls))

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    # ── Report ──
    print(f"\n[4/4] Results")
    print(f"\n{'='*80}")
    print(f"PROXIMITY ORDERING: {n_seqs} seqs")
    print(f"{'='*80}")

    random_mean = np.mean(results["random"])
    print(f"\n  {'Method':<16s} {'Mean NLL':>10s} {'Δ vs random':>12s} {'Win rate':>10s}")
    print(f"  {'-'*55}")

    for name in ["dp", "nn", "spectral"]:
        vals = np.array(results[name])
        delta = random_mean - vals.mean()
        win = np.mean(vals < np.array(results["random"]))
        print(f"  {name:<16s} {vals.mean():10.4f} {delta:+12.4f} {win:10.1%}")

    # Detail: DP vs NN weight ratio
    dp_w_mean = np.mean(dp_weights)
    nn_w_mean = np.mean(nn_weights)
    print(f"\n  DP total weight (mean): {dp_w_mean:.6f}")
    print(f"  NN total weight (mean): {nn_w_mean:.6f}")
    print(f"  NN/DP ratio: {nn_w_mean/max(dp_w_mean, 1e-8):.2%}")

    # Asymmetry
    print(f"\n  Mean |A[i,j] - A[j,i]|: {np.mean(asymmetry_vals):.6f}")
    print(f"  (closer to 0 → proximity assumption justified)")

    # Per-sequence oracle
    all_vals = np.column_stack([results["dp"], results["nn"], results["spectral"]])
    best_idx = np.argmin(all_vals, axis=1)
    best_nlls = all_vals[np.arange(n_seqs), best_idx]
    best_delta = random_mean - best_nlls.mean()
    best_win = np.mean(best_nlls < np.array(results["random"]))
    print(f"\n  Oracle (per-seq best method): Δ={best_delta:+.4f}, win={best_win:.1%}")

    # VERDICT
    dp_delta = random_mean - np.mean(results["dp"])
    print(f"\n  {'='*80}")
    if dp_delta > 0.01:
        print(f"  ✓ DP beats random (Δ={dp_delta:+.4f}) → proximity signal EXISTS")
        nn_delta = random_mean - np.mean(results["nn"])
        if nn_delta > 0.005:
            print(f"  ✓ NN also beats random (Δ={nn_delta:+.4f}) → greedy works")
            print(f"  → ON can learn the gap, GRPO has signal to exploit")
        else:
            print(f"  ✗ NN doesn't beat random (Δ={nn_delta:+.4f})")
            print(f"  → greedy fails, DP-NN gap = {dp_delta - nn_delta:+.4f}")
            print(f"  → ON needs to learn beyond greedy → GRPO is the right approach")
    else:
        print(f"  ✗ DP doesn't beat random (Δ={dp_delta:+.4f})")
        print(f"  → PROXIMITY HYPOTHESIS REJECTED")
        print(f"  → attention proximity doesn't encode ordering signal for this model")
    print(f"  {'='*80}")


if __name__ == "__main__":
    main()
