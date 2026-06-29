"""
Generate NN greedy paths from attention proximity W for BC pretraining.

For each sequence:
  1. Extract raw attention A (with [None] signal), 1 forward pass
  2. Symmetrize: W = 0.5*(A + A.T)
  3. NN greedy from low-degree endpoint (top-2 candidates, pick best total weight)
  4. Save paths → BC training data via p2_build_training_data

Usage:
    python generate_nn_paths.py --n-seqs 187 --output probe_results/nn_paths.npz
"""

import os, sys, time, argparse

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AO-GPT-MDM"))

import numpy as np
import torch
from datasets import Dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig
from p2_build_training_data import build_training_examples

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
    bp = np.array(ckpt["data_permutation"]["block_perm"], dtype=np.int64)
    inv_perm = np.array(ckpt["data_permutation"]["inverse_block_perm"], dtype=np.int64)
    return model, bp, inv_perm


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
    bs = T // M
    d = idx_phys.device
    out = torch.zeros_like(idx_phys)
    for s in range(T):
        mb = inv_perm[s // bs]
        off = s % bs
        out[:, mb * bs + off] = idx_phys[:, s]
    return out


@torch.no_grad()
def extract_attention_A(model, idx_model, bp):
    """Extract 16x16 block attention A with [None] signal."""
    d = idx_model.device
    rand_n16 = torch.randperm(N16, device=d)
    n64_order = []
    for t in range(N16):
        p = rand_n16[t].item()
        for a in range(SUB_BLOCKS):
            n64_order.append(int(bp[p * SUB_BLOCKS + a]))
    token_order = torch.repeat_interleave(
        torch.tensor(n64_order, device=d), BLOCK_LEN
    ) * BLOCK_LEN + torch.arange(BLOCK_LEN, device=d).repeat(N64)
    token_order = token_order.unsqueeze(0)

    _, _, attn_list = model.forward_fn(
        idx_model.unsqueeze(0), token_order, return_attentions=True
    )
    attn_stack = torch.stack(attn_list).squeeze(1).cpu().numpy()

    L, H = attn_stack.shape[:2]
    head_vars = np.zeros(H)
    for h in range(H):
        content = attn_stack[:, h, 1:, 1:]
        offdiag = content[:, ~np.eye(256, dtype=bool)].reshape(L, 256, 255)
        head_vars[h] = float(np.var(offdiag))
    top_heads = np.argsort(head_vars)[-4:]
    avg_attn = attn_stack[:, top_heads, :, :].mean(axis=(0, 1))

    # Remap reveal→phys
    reveal_to_model = token_order[0].cpu().numpy()
    bp_i = np.zeros(64, dtype=np.int64)
    for i in range(64):
        bp_i[bp[i]] = i
    model_to_phys = np.zeros(256, dtype=np.int64)
    for mp in range(256):
        mb = mp // BLOCK_LEN
        pb64 = bp_i[mb]
        pb16 = pb64 // SUB_BLOCKS
        off = mp % BLOCK_LEN
        model_to_phys[mp] = pb16 * SUB_BLOCKS * BLOCK_LEN + (pb64 % SUB_BLOCKS) * BLOCK_LEN + off
    reveal_to_phys = model_to_phys[reveal_to_model]

    attn_content = avg_attn[1:, 1:]
    attn_phys = np.zeros((256, 256), dtype=np.float32)
    for rq in range(256):
        pq = reveal_to_phys[rq]
        for rk in range(256):
            attn_phys[pq, reveal_to_phys[rk]] += attn_content[rq, rk]

    A = np.zeros((N16, N16), dtype=np.float32)
    for i in range(N16):
        for j in range(N16):
            A[i, j] = attn_phys[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16].mean()

    # [None] signal
    none_attn = avg_attn[1:, 0]
    none_block = np.array([none_attn[i * 16:(i + 1) * 16].mean() for i in range(N16)])
    A += none_block[np.newaxis, :] * 0.1
    np.fill_diagonal(A, 0.0)
    return A


def nn_greedy(W, start):
    """Nearest-neighbor greedy from start node."""
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
    """NN from low-degree endpoints, pick best total weight."""
    deg = W.sum(axis=1)
    candidates = np.argsort(deg)[:n_candidates]
    best_order, best_w = None, -np.inf
    for start in candidates:
        order = nn_greedy(W, start)
        w = sum(W[order[i], order[i + 1]] for i in range(len(order) - 1))
        if w > best_w:
            best_w = w
            best_order = order
    return best_order, best_w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seqs", type=int, default=187)
    parser.add_argument("--output", default="probe_results/nn_paths.npz")
    parser.add_argument("--training-data", default="probe_results/on_training_data_nn.npz")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 60)
    print("NN Path Generation for BC Pretraining")
    print("=" * 60)

    print("\n[1/3] Loading AO-GPT...")
    aogpt, bp, inv_perm = load_aogpt(AO_GPT_CKPT, device)
    print(f"  OK ({sum(p.numel() for p in aogpt.parameters()):,} params)")

    print(f"\n[2/3] Loading sequences...")
    seqs = load_sequences(max_count=args.n_seqs)
    n_seqs = len(seqs)
    print(f"  {n_seqs} sequences")

    inv_perm_t = torch.from_numpy(inv_perm)
    idx_phys = torch.tensor(seqs, dtype=torch.long)
    idx_all = torch.cat([
        phys_to_model_idx(idx_phys[i:i+1].to(device), inv_perm)
        for i in range(n_seqs)
    ], dim=0)
    print(f"  idx_all: {idx_all.shape}")

    print(f"\n[3/3] Extracting attention + NN paths...")
    t0 = time.time()
    paths = np.zeros((n_seqs, N16), dtype=np.int16)
    nn_weights_list = []

    for s in tqdm(range(n_seqs), desc="NN paths"):
        A = extract_attention_A(aogpt, idx_all[s], bp)
        W = 0.5 * (A + A.T)
        np.fill_diagonal(W, 0.0)
        order, w = nn_best_endpoint(W)
        paths[s] = order.astype(np.int16)
        nn_weights_list.append(w)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({elapsed/n_seqs:.1f}s/seq)")

    # Quick stats
    from scipy.stats import kendalltau
    l2r = np.arange(N16)
    taus = [kendalltau(l2r, paths[s])[0] for s in range(n_seqs)]
    print(f"\n  NN τ vs L2R: mean={np.mean(taus):+.4f}, min={np.min(taus):+.4f}, "
          f"max={np.max(taus):+.4f}, >0: {np.mean(np.array(taus)>0)*100:.0f}%")
    print(f"  NN total weight: mean={np.mean(nn_weights_list):.6f}")

    # Save paths
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(args.output, paths=paths.astype(np.int16))
    print(f"\n  Saved NN paths to {args.output}")

    # Build BC training data
    print(f"\n  Building BC training examples...")
    examples = build_training_examples(paths, num_blocks=N16)
    np.savez(args.training_data, **examples)
    print(f"  {len(examples['next_nodes'])} training examples")
    print(f"  Saved to {args.training_data}")

    # Also save A matrices (from attention) for ON input
    print(f"\n  Also computing A matrices for {n_seqs} seqs (same attention extraction)...")
    A_matrices = np.zeros((n_seqs, N16, N16), dtype=np.float32)
    for s in tqdm(range(n_seqs), desc="A matrices"):
        A_matrices[s] = extract_attention_A(aogpt, idx_all[s], bp)
    a_output = args.output.replace(".npz", "_A.npy")
    np.save(a_output, A_matrices)
    print(f"  Saved A matrices to {a_output} ({A_matrices.shape})")


if __name__ == "__main__":
    main()
