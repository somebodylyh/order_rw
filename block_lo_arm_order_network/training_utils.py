"""Minimal dependency module: model, data loading, and constants for Graph-RW training."""

import os
import sys

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import torch
from datasets import Dataset
from transformers import GPT2TokenizerFast

from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig  # noqa: F401

# ── Paths ────────────────────────────────────────────────────────────────────────
WIKITEXT_DIR = os.path.expanduser(
    "~/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/"
    "b08601e04326c79dfdd32d625aee71d232d685c3"
)
TOKENIZER_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)
A_PATH_DEFAULT = os.path.join(_SCRIPT_DIR, "probe_results", "A_train_n64_10k.npy")

# ── Constants ─────────────────────────────────────────────────────────────────────
SEQ_LEN = 256
N = 64              # number of blocks
BLOCK_LEN = 4       # tokens per block (256/64 = 4)


def load_train_chunks(n_chunks=None):
    """Load token chunks from wikitext-103 TRAIN set.

    Long texts produce sliding windows (stride=128), short texts accumulate
    in a buffer and get concatenated into 256-token chunks.

    If n_chunks is None, loads ALL available chunks.
    """
    from tqdm import tqdm

    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)

    chunks = []
    buffer_ids = []
    total_texts = 0
    _unlimited = (n_chunks is None)

    for shard in ["wikitext-train-00000-of-00002.arrow", "wikitext-train-00001-of-00002.arrow"]:
        ds = Dataset.from_file(os.path.join(WIKITEXT_DIR, shard))
        for ex in tqdm(ds, desc=f"Chunking {shard}", unit=" texts"):
            total_texts += 1
            ids = tok.encode(ex["text"])
            if len(ids) < SEQ_LEN:
                buffer_ids.extend(ids)
                while len(buffer_ids) >= SEQ_LEN:
                    chunks.append(torch.tensor(buffer_ids[:SEQ_LEN], dtype=torch.long))
                    buffer_ids = buffer_ids[SEQ_LEN:]
                    if not _unlimited and len(chunks) >= n_chunks:
                        break
            else:
                for start in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN // 2):
                    chunks.append(torch.tensor(ids[start:start + SEQ_LEN], dtype=torch.long))
                    if not _unlimited and len(chunks) >= n_chunks:
                        break
            if not _unlimited and len(chunks) >= n_chunks:
                break
        if not _unlimited and len(chunks) >= n_chunks:
            break
    print(f"Processed {total_texts} texts, got {len(chunks)} chunks", flush=True)
    return torch.stack(chunks)  # (n_chunks, 256)
