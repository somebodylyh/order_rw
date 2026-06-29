"""
Extract N32×N32 attention A matrices in MODEL coordinates.

Uses ych B32 ckpt (block_len=8, num_blocks=32, permute_data=True).
A[i,j] = attention from model block i to model block j.
ON trained on model-coordinate A → orders directly usable by AO-GPT.

Usage:
    python -u extract_train_A32.py --n-chunks 10000 --device cuda:0
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
    "~/ych/nanogpt-learned-order/out/base/permute/seq256/block32/"
    "out-wikitext103-seq256-random-b32-permute-block/ckpt.pt"
)
OUTPUT_DIR = "probe_results"

SEQ_LEN = 256
NUM_BLOCKS = 32  # N32
TOKENS_PER_BLOCK = SEQ_LEN // NUM_BLOCKS  # 8


def load_aogpt(ckpt_path, device):
    """Load frozen AO-GPT. Returns (model, block_perm, inv_perm).

    block_perm[phys_n32] = model_n32  (shape: 32,)
    inv_perm[model_n32] = phys_n32    (shape: 32,)
    """
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
    assert len(bp) == NUM_BLOCKS, f"Expected block_perm shape ({NUM_BLOCKS},), got {bp.shape}"
    return model, bp, inv_perm


def build_token_order(rand_order_phys, block_perm):
    """Build model-coordinate token order from physical random block permutation.

    rand_order_phys: (32,) random permutation of [0..31], physical block indices
    block_perm: (32,) block_perm[phys] = model
    Returns: (256,) token indices in model coordinates, ordered by reveal step
    """
    token_order = torch.zeros(SEQ_LEN, dtype=torch.long)
    for t in range(NUM_BLOCKS):
        phys_block = rand_order_phys[t].item()
        model_block = int(block_perm[phys_block])
        for k in range(TOKENS_PER_BLOCK):
            token_order[t * TOKENS_PER_BLOCK + k] = model_block * TOKENS_PER_BLOCK + k
    return token_order


@torch.no_grad()
def extract_one_A(model, idx_tokens, block_perm, device):
    """Extract 32×32 attention A matrix in MODEL coordinates.

    Returns: (32, 32) float32 numpy array, A[i,j] = model block i → model block j.
    """
    idx = idx_tokens.to(device).unsqueeze(0)  # (1, 256)

    # Random block order in physical coords → token order in model coords
    rand_order = torch.randperm(NUM_BLOCKS, device=device)
    token_order = build_token_order(rand_order, block_perm).to(device).unsqueeze(0)  # (1, 256)

    # Forward with attentions
    _, _, attn_list = model.forward_fn(idx, token_order, return_attentions=True)
    attn_stack = torch.stack(attn_list).squeeze(1).cpu().numpy()  # (L, H, 257, 257)

    L, H = attn_stack.shape[:2]

    # Select top-4 highest-variance heads
    head_vars = np.zeros(H)
    for h in range(H):
        content = attn_stack[:, h, 1:, 1:]  # skip [None] row/col
        offdiag = content[:, ~np.eye(256, dtype=bool)].reshape(L, 256, 255)
        head_vars[h] = float(np.var(offdiag))
    top_heads = np.argsort(head_vars)[-4:]
    avg_attn = attn_stack[:, top_heads, :, :].mean(axis=(0, 1))  # (257, 257)

    # Remap attention from reveal-order → model coordinates
    reveal_to_model = token_order[0].cpu().numpy()  # (256,)
    model_to_reveal = np.zeros(256, dtype=np.int64)
    model_to_reveal[reveal_to_model] = np.arange(256)

    attn_content = avg_attn[1:, 1:]  # (256, 256) content-content attention
    attn_model = attn_content[model_to_reveal][:, model_to_reveal]  # (256, 256) in model coords

    # Aggregate to 32×32 blocks
    A = np.zeros((NUM_BLOCKS, NUM_BLOCKS), dtype=np.float32)
    for i in range(NUM_BLOCKS):
        i_s, i_e = i * TOKENS_PER_BLOCK, (i + 1) * TOKENS_PER_BLOCK
        for j in range(NUM_BLOCKS):
            j_s, j_e = j * TOKENS_PER_BLOCK, (j + 1) * TOKENS_PER_BLOCK
            A[i, j] = attn_model[i_s:i_e, j_s:j_e].mean()

    # [None] signal
    none_attn = avg_attn[1:, 0]
    none_model = none_attn[model_to_reveal]  # remap to model coords
    none_block = np.array([none_model[i * TOKENS_PER_BLOCK:(i + 1) * TOKENS_PER_BLOCK].mean()
                           for i in range(NUM_BLOCKS)])
    A += none_block[np.newaxis, :] * 0.1
    np.fill_diagonal(A, 0.0)

    return A


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-chunks", type=int, default=10000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-tokens", action="store_true", default=True,
                        help="Also save token_ids for training alignment")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"{OUTPUT_DIR}/A_train_n{NUM_BLOCKS}_10k.npy"

    device = torch.device(args.device)
    print(f"Device: {device}, n_blocks={NUM_BLOCKS}, target: {args.n_chunks} chunks", flush=True)

    # ── Load AO-GPT ──
    print("Loading AO-GPT...", flush=True)
    model, block_perm, inv_perm = load_aogpt(AO_GPT_CKPT, device)
    print(f"Model loaded, block_perm[:8]={block_perm[:8].tolist()}, "
          f"inv_perm[:8]={inv_perm[:8].tolist()}", flush=True)

    # ── Load & tokenize wikitext-103 train set ──
    print("Loading & tokenizing wikitext-103 train set...", flush=True)
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

    print("Chunking wikitext-103 train set...", flush=True)
    chunks = []
    buffer_ids = []
    total_texts = 0
    for shard in ["wikitext-train-00000-of-00002.arrow", "wikitext-train-00001-of-00002.arrow"]:
        ds = Dataset.from_file(os.path.join(WIKITEXT_CACHE, shard))
        for ex in tqdm(ds, desc=f"Chunking {shard}", unit=" texts"):
            total_texts += 1
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
    print(f"Processed {total_texts} texts, got {len(chunks)} chunks", flush=True)

    # ── Save token_ids ──
    if args.save_tokens:
        tokens_path = Path(args.output).with_suffix(".tokens.npy")
        tokens_arr = np.array(chunks, dtype=np.int32)
        np.save(tokens_path, tokens_arr)
        print(f"Saved tokens: {tokens_path} | shape={tokens_arr.shape}", flush=True)

    # ── Extract A matrices ──
    n_available = min(args.n_chunks, len(chunks))
    print(f"Extracting {n_available} A matrices ({NUM_BLOCKS}×{NUM_BLOCKS}) in MODEL coordinates...",
          flush=True)

    A_list = []
    t0 = time.time()

    for i in tqdm(range(n_available), desc="Extracting A32"):
        tokens = torch.tensor(chunks[i], dtype=torch.long)
        A = extract_one_A(model, tokens, block_perm, device)
        A_list.append(A)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_available - i - 1) / rate
            print(f"  {i+1}/{n_available} | {rate:.1f} chunks/s | ETA {eta:.0f}s", flush=True)

    # ── Save ──
    A_all = np.stack(A_list, axis=0).astype(np.float32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, A_all)

    elapsed = time.time() - t0
    print(f"Saved: {output_path} | shape={A_all.shape} | "
          f"range=[{A_all.min():.4f}, {A_all.max():.4f}] | "
          f"time={elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
