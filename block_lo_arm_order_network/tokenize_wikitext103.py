"""
Tokenize all wikitext-103 train set into 256-token chunks.
Saves to probe_results/wikitext103_train_tokens.npy for reuse.

Usage:
    python tokenize_wikitext103.py
"""
import os, sys, time
import numpy as np
from tqdm import tqdm

SEQ_LEN = 256
OUTPUT_DIR = "probe_results"

# Offline paths
TOKENIZER_DIR = os.path.expanduser(
    "~/.cache/huggingface/hub/models--gpt2/snapshots/"
    "607a30d783dfa663caf39e06633721c8d4cfcd7e"
)
WIKITEXT_CACHE = os.path.expanduser(
    "~/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0/"
    "b08601e04326c79dfdd32d625aee71d232d685c3"
)

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main():
    from datasets import Dataset
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained(TOKENIZER_DIR, local_files_only=True)

    chunks = []
    buffer_ids = []
    total_texts = 0
    for shard in ["wikitext-train-00000-of-00002.arrow", "wikitext-train-00001-of-00002.arrow"]:
        ds = Dataset.from_file(os.path.join(WIKITEXT_CACHE, shard))
        for ex in tqdm(ds, desc=f"Tokenizing {shard}", unit=" texts"):
            total_texts += 1
            ids = tok.encode(ex["text"])
            if len(ids) < SEQ_LEN:
                buffer_ids.extend(ids)
                while len(buffer_ids) >= SEQ_LEN:
                    chunks.append(buffer_ids[:SEQ_LEN])
                    buffer_ids = buffer_ids[SEQ_LEN:]
            else:
                for start in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN // 2):
                    chunks.append(ids[start:start + SEQ_LEN])

    print(f"Processed {total_texts} texts, got {len(chunks)} chunks")

    tokens_arr = np.array(chunks, dtype=np.int32)
    output_path = os.path.join(OUTPUT_DIR, "wikitext103_train_tokens.npy")
    np.save(output_path, tokens_arr)
    print(f"Saved: {output_path} | shape={tokens_arr.shape}")


if __name__ == "__main__":
    main()
