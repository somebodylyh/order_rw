"""Extract A matrices from AR-trained ckpt and compute NN paths using hooks."""
import os, sys
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, "/home/admin/lyuyuhuan/order-shakespeare/nanoGPT")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from scipy.stats import kendalltau

from model_aogpt import AOGPT, AOGPTConfig

CKPT = "/home/admin/lyuyuhuan/order-shakespeare/nanoGPT/out-original-ar-60k/ckpt.pt"
TOKENIZER_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e")
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/datasets")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 256
N_BLOCKS = 64
BLOCK_LEN = 4
N_SEQS = 200
OUT_DIR = "probe_results/ar60k_analysis"
os.makedirs(OUT_DIR, exist_ok=True)


def make_hook(container):
    """Returns a hook that captures attention weights from the last layer."""
    def hook(module, input, output):
        # module is CausalSelfAttention
        # We need to re-run attention computation to get weights
        # input[0] is x after modulation
        x = input[0]
        B, Tp1, C = x.shape
        T = Tp1 - 1

        c_attn_out = module.c_attn(x)
        qkv = c_attn_out.split(C, dim=2)
        q, k, v = qkv[0], qkv[1], qkv[2]

        n_head = module.n_head
        head_dim = q.shape[-1] // n_head
        q = q.view(B, -1, n_head, head_dim).transpose(1, 2)
        k = k.view(B, -1, n_head, head_dim).transpose(1, 2)

        scale = 1.0 / (head_dim ** 0.5)
        att = (q @ k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.tril(torch.ones(Tp1, Tp1, device=x.device)).view(1, 1, Tp1, Tp1)
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # Token queries (pos 1:) → token keys (pos 1:)
        container["attn"] = att[0, :, 1:, 1:].detach().cpu().float()  # (n_head, T, T)
    return hook


def greedy_nn_path(A):
    N = A.shape[0]
    visited = set()
    path = []
    current = int(np.argmax(A.sum(axis=1)))
    path.append(current)
    visited.add(current)
    for _ in range(N - 1):
        best_j, best_val = None, -1e9
        for j in range(N):
            if j not in visited and A[current, j] > best_val:
                best_val = A[current, j]
                best_j = j
        if best_j is None:
            for j in range(N):
                if j not in visited:
                    best_j = j
                    break
        path.append(best_j)
        visited.add(best_j)
        current = best_j
    return np.array(path)


def load_model():
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    cfg = AOGPTConfig(**ckpt["model_args"])
    print(f"  n_layer={cfg.n_layer}, n_head={cfg.n_head}, n_embd={cfg.n_embd}")
    model = AOGPT(cfg).to(DEVICE)
    sd = ckpt["model"]
    for k in list(sd):
        if k.startswith("_orig_mod."):
            sd[k[10:]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval()
    return model


def main():
    print("Loading AR ckpt...")
    model = load_model()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Register hook on last layer
    attn_container = {}
    last_block = model.transformer.h[-1]
    handle = last_block.attn.register_forward_hook(make_hook(attn_container))

    print("Loading wikitext-103...")
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=CACHE_DIR, split="test")

    def tokenize(ex):
        ids = tokenizer.encode(ex["text"])
        return {"input_ids": ids, "len": len(ids)}

    ds = ds.map(tokenize, remove_columns=["text"])
    ds = ds.filter(lambda x: x["len"] >= SEQ_LEN)
    ds.set_format(type="torch", columns=["input_ids"])

    seqs = []
    for row in ds:
        ids = row["input_ids"][:SEQ_LEN]
        if len(ids) == SEQ_LEN:
            seqs.append(ids)
        if len(seqs) >= N_SEQS:
            break
    print(f"  Collected {N_SEQS} sequences (found {len(seqs)})")
    n_actual = len(seqs)

    l2r = np.arange(N_BLOCKS)
    all_paths = []
    all_taus = []
    all_A = []

    print(f"Extracting attention from {n_actual} sequences...")
    for i in tqdm(range(n_actual)):
        seq = seqs[i].unsqueeze(0).to(DEVICE)

        # Run AR forward to populate hook
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            orders = torch.arange(SEQ_LEN, device=DEVICE).unsqueeze(0)
            model.forward_fn(seq, orders)

        # Get attention from hook
        attn_t2t = attn_container["attn"].mean(dim=0).numpy()  # (T, T) = (256, 256)
        # Symmetrize (causal AR gives lower-triangular, we want undirected)
        attn_sym = 0.5 * (attn_t2t + attn_t2t.T)

        # Aggregate to blocks
        A = np.zeros((N_BLOCKS, N_BLOCKS))
        for bi in range(N_BLOCKS):
            for bj in range(N_BLOCKS):
                si, ei = bi * BLOCK_LEN, (bi + 1) * BLOCK_LEN
                sj, ej = bj * BLOCK_LEN, (bj + 1) * BLOCK_LEN
                A[bi, bj] = attn_sym[si:ei, sj:ej].mean()
        np.fill_diagonal(A, 0.0)
        all_A.append(A)

        path = greedy_nn_path(A)
        all_paths.append(path)
        all_taus.append(kendalltau(l2r, path)[0])

    handle.remove()

    all_paths_np = np.array(all_paths, dtype=np.int16)
    all_taus_np = np.array(all_taus)
    all_A_np = np.array(all_A, dtype=np.float32)

    np.save(f"{OUT_DIR}/AR60k_NN_paths_{n_actual}seq.npy", all_paths_np)
    np.save(f"{OUT_DIR}/AR60k_NN_taus_{n_actual}seq.npy", all_taus_np)
    np.save(f"{OUT_DIR}/AR60k_A_matrices_{n_actual}seq.npy", all_A_np)

    print(f"\n{'='*60}")
    print(f"AR 60k ckpt: NN paths from last-layer attention (causal → symmetrized)")
    print(f"{'='*60}")
    print(f"  N seqs: {n_actual}")
    print(f"  τ vs L2R: mean={all_taus_np.mean():+.4f}, std={all_taus_np.std():.4f}")
    print(f"  median={np.median(all_taus_np):+.4f}")
    print(f"  min={all_taus_np.min():+.4f}, max={all_taus_np.max():+.4f}")
    print(f"  5th pctl={np.percentile(all_taus_np, 5):+.4f}, 95th={np.percentile(all_taus_np, 95):+.4f}")

    print(f"\n  Comparison:")
    print(f"  Random-order 50k:  τ≈+0.36 (NN path vs L2R)")
    print(f"  AR 60k (causal):   τ≈{all_taus_np.mean():+.4f} (NN path vs L2R)")

    # Histogram
    bins = np.linspace(-1, 1, 11)
    hist, _ = np.histogram(all_taus_np, bins=bins)
    print(f"\n  τ distribution:")
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        bar = '#' * int(hist[i] / max(hist) * 40) if max(hist) > 0 else ''
        print(f"    [{lo:+.1f}, {hi:+.1f}): {int(hist[i]):4d} {bar}")

    print(f"\n  Saved: {OUT_DIR}/")


if __name__ == "__main__":
    main()
