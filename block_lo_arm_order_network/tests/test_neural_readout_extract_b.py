"""NR-1 Task 2: tests for per-sample B = A^T extraction.

load_train_chunks() re-tokenizes wikitext-103 every call (~2-3 min) and there
is no on-disk cache, so each "heavy" test costs minutes. We consolidate all
properties that need a real model into a single heavy test, and keep two
fast tests that exercise input validation without loading the model or
wikitext data.

Properties verified by the single heavy test:
  - shape (M, 64, 64), dtype float32, diag = 0
  - bit-reproducibility under fixed (ckpt, M, seed, split)
  - extract_per_sample_B and extract_per_sample_B_with_chunks agree on B
  - chunk_index is sorted, shape (M,), int64
  - split echoed back unchanged
"""
import sys
import pathlib

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


CKPT_5K = str(ROOT / "block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step5000.pt")
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_invalid_split_rejected_fast():
    """No model load, no wikitext load -- fails at the first validation."""
    from neural_readout.extract_b import extract_per_sample_B
    with pytest.raises(ValueError, match="split must be one of"):
        extract_per_sample_B(CKPT_5K, M=4, seed=42, device=DEV, split="val")


def test_M_exceeds_eval_pool_raises_fast():
    """Loads only the ckpt's protocol (no model, no wikitext) -- ~1 sec."""
    from neural_readout.extract_b import extract_per_sample_B
    with pytest.raises(ValueError, match="only has"):
        extract_per_sample_B(CKPT_5K, M=1000, seed=42, device=DEV, split="eval")


def test_extract_b_smoke():
    """Single heavy test: model load + wikitext tokenize + 2 extractions.

    All shape/dtype/diag/reproducibility/chunk_index/split assertions are
    folded together to avoid paying the wikitext tokenization cost twice.
    """
    from neural_readout.extract_b import extract_per_sample_B, extract_per_sample_B_with_chunks

    B1, chunk_index, split = extract_per_sample_B_with_chunks(
        CKPT_5K, M=4, seed=42, device=DEV, split="train"
    )
    # Independent call must reproduce B1 bit-for-bit (same ckpt, M, seed, split).
    B2 = extract_per_sample_B(CKPT_5K, M=4, seed=42, device=DEV, split="train")

    assert B1.shape == (4, 64, 64), B1.shape
    assert B1.dtype == np.float32
    assert np.all(np.diagonal(B1, axis1=1, axis2=2) == 0.0)
    np.testing.assert_array_equal(B1, B2)

    assert chunk_index.shape == (4,)
    assert chunk_index.dtype == np.int64
    assert (chunk_index[1:] >= chunk_index[:-1]).all(), "chunk_index must be sorted"
    assert split == "train"
