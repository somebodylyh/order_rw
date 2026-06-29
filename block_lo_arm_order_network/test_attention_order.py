"""
Test: does raw attention A[i,j] encode ordering signal?

Extracts block-level attention (1 forward/seq), computes heuristic orders,
scores NLL against random baseline.

Heuristics:
  1. indegree:   most-depended-on first  (max Σ_i A[i,j])
  2. outdegree:  least-depending first   (min Σ_j A[i,j])
  3. netflow:    net source first        (min out-in)
  4. greedy-in:  seq. pick max indegree among remaining
  5. greedy-out: seq. pick min outdegree among remaining
"""

import os, sys, time, argparse

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
# Attention Extraction
# ═══════════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_block_attention(model, idx_model, block_perm):
    """
    Extract raw asymmetric block-level attention from ONE random-order forward pass.
    Returns: (16, 16) float32, A[i,j] = mean attention from phys block i to phys block j.
    """
    device = idx_model.device
    bp = block_perm.cpu().numpy()

    # Single random N16 order → N64 → token order
    rand_n16 = torch.randperm(N16, device=device)
    # Convert to N64 model-coord order
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
    token_order = token_order.unsqueeze(0)  # (1, 256)

    # Forward with attention
    logits, _, attn_list = model.forward_fn(
        idx_model.unsqueeze(0), token_order, return_attentions=True
    )
    # attn_list: list[L] of (1, H, 257, 257) — [None] at pos 0, then 256 tokens
    attn_stack = torch.stack(attn_list).squeeze(1).cpu().numpy()  # (L, H, 257, 257)

    # Select top-4 heads by off-diagonal variance
    L, H = attn_stack.shape[:2]
    head_vars = np.zeros(H)
    for h in range(H):
        content = attn_stack[:, h, 1:, 1:]  # (L, 256, 256)
        offdiag = content[:, ~np.eye(256, dtype=bool)].reshape(L, 256, 255)
        head_vars[h] = float(np.var(offdiag))
    top_heads = np.argsort(head_vars)[-4:]

    avg_attn = attn_stack[:, top_heads, :, :].mean(axis=(0, 1))  # (257, 257)

    # Aggregate to block level (physical coords)
    # The attention is in reveal-order coordinates, need to map to physical
    # token_order gives: reveal_pos → model_token
    # We need: phys_block → phys_block
    #
    # Approach: map attention from reveal-order to physical coords using inverse permutation
    reveal_to_model = token_order[0].cpu().numpy()  # reveal_pos → model_token
    model_to_phys = np.zeros(256, dtype=np.int64)
    for model_pos in range(256):
        model_block = model_pos // BLOCK_LEN
        phys_block_64 = int(np.where(bp == model_block)[0][0])
        phys_block_16 = phys_block_64 // SUB_BLOCKS
        offset = model_pos % BLOCK_LEN
        phys_pos = phys_block_16 * SUB_BLOCKS * BLOCK_LEN + (phys_block_64 % SUB_BLOCKS) * BLOCK_LEN + offset
        model_to_phys[model_pos] = phys_pos

    reveal_to_phys = model_to_phys[reveal_to_model]  # reveal_pos → phys_pos

    # Attention matrix in reveal coords (content tokens only)
    attn_content = avg_attn[1:, 1:]  # (256, 256) — reveal_q × reveal_k

    # Remap to physical coords
    attn_phys = np.zeros((256, 256), dtype=np.float32)
    for rq in range(256):
        pq = reveal_to_phys[rq]
        for rk in range(256):
            pk = reveal_to_phys[rk]
            attn_phys[pq, pk] += attn_content[rq, rk]

    # Aggregate to block level: phys block I = tokens [I*16 : I*16+16]
    A = np.zeros((N16, N16), dtype=np.float32)
    for i in range(N16):
        for j in range(N16):
            i_slice = slice(i * 16, (i + 1) * 16)
            j_slice = slice(j * 16, (j + 1) * 16)
            A[i, j] = attn_phys[i_slice, j_slice].mean()

    np.fill_diagonal(A, 0.0)
    return A


# ═══════════════════════════════════════════════════════════════════════════════════
# Heuristic Ordering
# ═══════════════════════════════════════════════════════════════════════════════════

def order_indegree(A):
    """Most-depended-on first: sort by sum of attention TO each block."""
    indeg = A.sum(axis=0)  # (16,) — total attention TO j
    return np.argsort(-indeg)

def order_outdegree(A):
    """Least-depending first: sort by sum of attention FROM each block."""
    outdeg = A.sum(axis=1)  # (16,) — total attention FROM i
    return np.argsort(outdeg)

def order_netflow(A):
    """Net source first: sort by (out - in), ascending."""
    outdeg = A.sum(axis=1)
    indeg = A.sum(axis=0)
    net = outdeg - indeg
    return np.argsort(net)

def order_greedy_indeg(A):
    """Sequentially pick block with max indegree among remaining."""
    N = A.shape[0]
    remaining = set(range(N))
    order = []
    A_copy = A.copy()
    for _ in range(N):
        best = max(remaining, key=lambda i: A_copy[:, i].sum())
        order.append(best)
        remaining.remove(best)
    return np.array(order)

def order_greedy_outdeg(A):
    """Sequentially pick block with min outdegree to remaining."""
    N = A.shape[0]
    remaining = set(range(N))
    order = []
    A_copy = A.copy()
    for _ in range(N):
        best = min(remaining, key=lambda i: sum(A_copy[i, j] for j in remaining if j != i))
        order.append(best)
        remaining.remove(best)
    return np.array(order)


HEURISTICS = {
    "indegree": order_indegree,
    "outdegree": order_outdegree,
    "netflow": order_netflow,
    "greedy-in": order_greedy_indeg,
    "greedy-out": order_greedy_outdeg,
}


# ═══════════════════════════════════════════════════════════════════════════════════
# NLL Scoring
# ═══════════════════════════════════════════════════════════════════════════════════

def n16_to_token_order(n16_order, block_perm):
    """Convert (16,) phys N16 order → (256,) model-coord token order."""
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
    """Single-sequence NLL for a given token order."""
    device = idx.device
    T = idx.shape[0]
    idx_b = idx.unsqueeze(0)  # (1, T)
    order_t = torch.from_numpy(order).to(device).unsqueeze(0)  # (1, T)

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
        reduction="mean",
        ignore_index=-1,
    )
    return ce.item()


# ═══════════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seqs", type=int, default=100)
    parser.add_argument("--n-random", type=int, default=50,
                        help="Random baselines per sequence")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    print("Loading AO-GPT...")
    aogpt, block_perm, inv_perm = load_aogpt(AO_GPT_CKPT, device)
    print(f"  OK ({sum(p.numel() for p in aogpt.parameters()):,} params)")

    print(f"Loading sequences...")
    seqs = load_sequences(max_count=args.n_seqs)
    n_seqs = len(seqs)
    print(f"  {n_seqs} sequences")

    # Convert to model coordinates
    print("Converting to model coords...")
    idx_phys = torch.tensor(seqs, dtype=torch.long)
    idx_model_list = []
    for i in range(n_seqs):
        idx_model_list.append(phys_to_model_idx(idx_phys[i:i+1].to(device), inv_perm))
    idx_all = torch.cat(idx_model_list, dim=0)
    print(f"  {idx_all.shape}")

    # Results
    results = {name: [] for name in HEURISTICS}
    random_nlls_all = []

    print(f"\nProcessing {n_seqs} sequences...")
    t0 = time.time()

    for s in range(n_seqs):
        if s % 20 == 0:
            print(f"  Seq {s}/{n_seqs}...")

        idx_s = idx_all[s:s+1]

        # 1. Extract attention A
        A = extract_block_attention(aogpt, idx_s[0], block_perm)

        # 2. Compute heuristic orders + score NLL
        for name, heuristic in HEURISTICS.items():
            order_phys = heuristic(A)  # (16,) phys N16 block indices
            token_order = n16_to_token_order(order_phys, block_perm)
            nll = compute_nll(aogpt, idx_s[0], token_order)
            results[name].append(nll)

        # 3. Score random baselines
        rand_nlls = []
        for _ in range(args.n_random):
            rand_phys = np.random.permutation(N16)
            token_order = n16_to_token_order(rand_phys, block_perm)
            rand_nlls.append(compute_nll(aogpt, idx_s[0], token_order))
        random_nlls_all.append(np.mean(rand_nlls))

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    # ── Analysis ──
    print(f"\n{'='*70}")
    print(f"RESULTS: {n_seqs} seqs, heuristic orders vs random baseline")
    print(f"{'='*70}")

    random_mean = np.mean(random_nlls_all)
    print(f"\n  Random N16 baseline (mean of {args.n_random}): {random_mean:.4f}")

    print(f"\n  {'Heuristic':<16s} {'Mean NLL':>10s} {'Δ vs random':>12s} {'Win rate':>10s} {'p(beat)':>10s}")
    print(f"  {'-'*60}")

    for name in HEURISTICS:
        vals = np.array(results[name])
        mean_nll = vals.mean()
        delta = random_mean - mean_nll  # positive = better than random
        win_rate = np.mean(np.array(results[name]) < np.array(random_nlls_all))
        p_beat = np.mean(vals < np.array(random_nlls_all))
        print(f"  {name:<16s} {mean_nll:10.4f} {delta:+12.4f} {win_rate:10.1%} {p_beat:10.1%}")

    # Best heuristic per sequence
    print(f"\n--- Per-sequence best heuristic ---")
    all_heuristic = np.column_stack([results[name] for name in HEURISTICS])
    best_idx = np.argmin(all_heuristic, axis=1)
    best_nlls = all_heuristic[np.arange(n_seqs), best_idx]
    best_delta = random_mean - best_nlls.mean()
    best_win = np.mean(best_nlls < np.array(random_nlls_all))
    print(f"  Oracle heuristic: NLL={best_nlls.mean():.4f}, Δ={best_delta:+.4f}, win={best_win:.1%}")

    # Attention asymmetry check
    print(f"\n--- Attention signal quality ---")
    asym_all = []
    indeg_std_all = []
    for s in range(min(5, n_seqs)):
        A = extract_block_attention(aogpt, idx_all[s], block_perm)
        asym = np.mean([abs(A[i,j] - A[j,i]) for i in range(N16) for j in range(N16) if i != j])
        asym_all.append(asym)
        indeg_std_all.append(np.std(A.sum(axis=0)))
    print(f"  Mean asymmetry: {np.mean(asym_all):.6f} (first 5 seqs)")
    print(f"  Indegree std:   {np.mean(indeg_std_all):.6f} (higher = more ordering signal)")


if __name__ == "__main__":
    main()
