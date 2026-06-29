"""
Extract A32 attention matrices from N=64 AO-GPT + generate NN paths.

Pipeline:
  N=64 AO-GPT (frozen extractor)
  → extract attention, aggregate 64→32 (2→1)
  → W = 0.5*(A + A.T), zero diag
  → NN greedy from low-degree endpoints
  → save A32_matrices + NN32_paths

Usage:
    python extract_A32_nn_from_N64.py --n-chunks 10000 --device cuda:0
"""

import os, sys, time, argparse
from pathlib import Path

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AO-GPT-MDM"))

import numpy as np
import torch
from tqdm import tqdm

from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig

# ── Paths ──
AO_GPT_CKPT = os.path.expanduser(
    "~/ych/nanogpt-learned-order/out/base/permute/seq256/block64/"
    "out-wikitext103-seq256-random-b64-permute-block-50000-iters/ckpt.pt"
)
OUTPUT_DIR = Path("probe_results")

SEQ_LEN = 256
N64 = 64
N32 = 32
SUB_BLOCKS = N64 // N32  # 2
BLOCK_LEN = SEQ_LEN // N64  # 4
TOKENS_PER_N32 = SEQ_LEN // N32  # 8


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
    return model, bp


@torch.no_grad()
def extract_A32_from_N64(model, idx_tokens, bp64, device):
    """Extract 32×32 A from N=64 AO-GPT (aggregate 2→1 sub-blocks)."""
    d = device
    rand_n32 = torch.randperm(N32, device=d)
    n64_order = []
    for t in range(N32):
        p = rand_n32[t].item()
        for a in range(SUB_BLOCKS):
            n64_order.append(int(bp64[p * SUB_BLOCKS + a]))
    token_order = torch.repeat_interleave(
        torch.tensor(n64_order, device=d), BLOCK_LEN
    ) * BLOCK_LEN + torch.arange(BLOCK_LEN, device=d).repeat(N64)
    token_order = token_order.unsqueeze(0)

    _, _, attn_list = model.forward_fn(
        idx_tokens.unsqueeze(0).to(d), token_order, return_attentions=True
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

    # Remap reveal → phys coords
    reveal_to_model = token_order[0].cpu().numpy()
    bp64_i = np.zeros(64, dtype=np.int64)
    for i in range(64):
        bp64_i[bp64[i]] = i
    model_to_phys = np.zeros(256, dtype=np.int64)
    for mp in range(256):
        mb = mp // BLOCK_LEN
        pb64 = bp64_i[mb]
        pb32 = pb64 // SUB_BLOCKS
        off = mp % BLOCK_LEN
        model_to_phys[mp] = pb32 * TOKENS_PER_N32 + (pb64 % SUB_BLOCKS) * BLOCK_LEN + off
    reveal_to_phys = model_to_phys[reveal_to_model]

    attn_content = avg_attn[1:, 1:]
    attn_phys = np.zeros((256, 256), dtype=np.float32)
    for rq in range(256):
        pq = reveal_to_phys[rq]
        for rk in range(256):
            attn_phys[pq, reveal_to_phys[rk]] += attn_content[rq, rk]

    A = np.zeros((N32, N32), dtype=np.float32)
    for i in range(N32):
        i_s, i_e = i * TOKENS_PER_N32, (i + 1) * TOKENS_PER_N32
        for j in range(N32):
            j_s, j_e = j * TOKENS_PER_N32, (j + 1) * TOKENS_PER_N32
            A[i, j] = attn_phys[i_s:i_e, j_s:j_e].mean()

    none_attn = avg_attn[1:, 0]
    none_block = np.array([none_attn[i * TOKENS_PER_N32:(i + 1) * TOKENS_PER_N32].mean()
                           for i in range(N32)])
    A += none_block[np.newaxis, :] * 0.1
    np.fill_diagonal(A, 0.0)
    return A


def nn_greedy(W, start):
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
    parser.add_argument("--n-chunks", type=int, default=10000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}, n_chunks={args.n_chunks}", flush=True)

    # ── Load N=64 AO-GPT ──
    print("Loading N=64 AO-GPT (extractor)...", flush=True)
    model, bp = load_aogpt(AO_GPT_CKPT, device)
    print(f"  OK, block_perm[:8]={bp[:8].tolist()}", flush=True)

    # ── Tokenize wikitext-103 train ──
    print("Loading & tokenizing wikitext-103 train...", flush=True)
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
    chunks = []
    buffer_ids = []
    for shard in ["wikitext-train-00000-of-00002.arrow", "wikitext-train-00001-of-00002.arrow"]:
        ds = Dataset.from_file(os.path.join(WIKITEXT_CACHE, shard))
        for ex in ds:
            ids = tok.encode(ex["text"])
            if len(ids) < SEQ_LEN:
                buffer_ids.extend(ids)
                while len(buffer_ids) >= SEQ_LEN:
                    chunks.append(buffer_ids[:SEQ_LEN])
                    buffer_ids = buffer_ids[SEQ_LEN:]
                    if len(chunks) >= args.n_chunks:
                        break
            else:
                for start in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN // 2):
                    chunks.append(ids[start:start + SEQ_LEN])
                    if len(chunks) >= args.n_chunks:
                        break
            if len(chunks) >= args.n_chunks:
                break
        if len(chunks) >= args.n_chunks:
            break
    n_available = min(args.n_chunks, len(chunks))
    print(f"  Got {n_available} chunks", flush=True)

    # ── Extract A32 + NN paths ──
    print(f"Extracting A32 from N=64 model ({n_available} seqs)...", flush=True)
    A_list = []
    nn_paths = np.zeros((n_available, N32), dtype=np.int16)
    t0 = time.time()

    for i in tqdm(range(n_available), desc="A32+NN"):
        tokens = torch.tensor(chunks[i], dtype=torch.long)
        A = extract_A32_from_N64(model, tokens, bp, device)
        A_list.append(A)

        W = 0.5 * (A + A.T)
        np.fill_diagonal(W, 0.0)
        order, _ = nn_best_endpoint(W)
        nn_paths[i] = order.astype(np.int16)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_available - i - 1) / rate
            # Quick diagnostic
            from scipy.stats import kendalltau
            l2r = np.arange(N32)
            taus = [kendalltau(l2r, nn_paths[j])[0] for j in range(max(0, i-100), i+1)]
            print(f"  {i+1}/{n_available} | {rate:.2f} seqs/s | ETA {eta:.0f}s | "
                  f"τ(recent)={np.mean(taus):+.4f}", flush=True)

    elapsed = time.time() - t0

    # ── Save ──
    A_all = np.stack(A_list, axis=0).astype(np.float32)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    a_path = output_dir / "A32_from_N64_10k.npy"
    nn_path = output_dir / "NN32_paths_from_N64_10k.npy"
    tokens_path = output_dir / "A32_from_N64_10k.tokens.npy"

    np.save(a_path, A_all)
    np.save(nn_path, nn_paths)
    np.save(tokens_path, np.array(chunks[:n_available], dtype=np.int32))

    print(f"\nSaved:", flush=True)
    print(f"  {a_path} | shape={A_all.shape}", flush=True)
    print(f"  {nn_path} | shape={nn_paths.shape}", flush=True)
    print(f"  {tokens_path} | shape=({n_available}, {SEQ_LEN})", flush=True)
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

    # ── Final diagnostic ──
    from scipy.stats import kendalltau, spearmanr
    l2r = np.arange(N32)
    taus = [kendalltau(l2r, nn_paths[i])[0] for i in range(n_available)]
    taus = np.array(taus)
    print(f"\n=== NN32 Path Diagnostic ===", flush=True)
    print(f"  τ vs L2R: mean={taus.mean():+.4f}, std={taus.std():.4f}", flush=True)
    print(f"  τ > 0: {(taus>0).mean()*100:.0f}%", flush=True)
    print(f"  τ > 0.3: {(taus>0.3).mean()*100:.0f}%", flush=True)
    print(f"  τ > 0.5: {(taus>0.5).mean()*100:.0f}%", flush=True)


if __name__ == "__main__":
    main()
