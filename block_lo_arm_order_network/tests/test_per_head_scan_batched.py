"""Equivalence test for the batched per-head attention extractor.

The slow path in per_head_order_scan was a batch=1 forward loop (M*batch_size
sequential forwards). The batched version forwards `fwd_batch` chunks at once.
This test pins the batched output to the verbatim loop reference
(`_extract_per_head_and_heavy_A_loop`) using a SYNTHETIC model whose forward is
batch-invariant by construction (each sample's attention is computed only from
its own tokens + token_order). Therefore loop and batched must agree EXACTLY
(np.array_equal) — any difference is a batching/sample-alignment bug.

(Real-model fp-level closeness across batch sizes is checked separately by the
 scripts/verify_batched_extractor.py integration check, not here.)
"""
import pathlib
import sys

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_PKG = _HERE.parent
sys.path.insert(0, str(_PKG))

from training_utils import SEQ_LEN, N  # noqa: E402
import per_head_order_scan as phs  # noqa: E402


class _FakePerm:
    """clean_perm stub exposing identity inv_perm_model_to_phys."""
    def __init__(self):
        self.inv_perm_model_to_phys = torch.arange(N)


class _FakeAOGPT:
    """Synthetic AOGPT whose attention depends ONLY on per-sample inputs, so a
    forward of B samples == B independent forwards (batch-invariant)."""
    def __init__(self, L=2, H=3):
        self.L, self.H = L, H

    def eval(self):
        return self

    def forward_fn(self, idx, orders, return_attentions=False):
        assert return_attentions
        B = idx.shape[0]
        T1 = SEQ_LEN + 1
        attn_list = []
        for l in range(self.L):
            a = torch.zeros(B, self.H, T1, T1)
            for b in range(B):
                # seed from this sample's own content + order so a misaligned
                # (sample <-> token_order) pairing in the batched path changes A.
                s = (int(orders[b, 0].item()) * 131
                     + int(idx[b, 0].item()) * 17 + l + 1)
                g = torch.Generator().manual_seed(s)
                a[b] = torch.rand(self.H, T1, T1, generator=g)
            attn_list.append(a)
        return None, None, attn_list


def _make_chunks(n_chunks, vocab=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n_chunks, SEQ_LEN), generator=g)


def _orig_scalar_A_block(avg_attn, reveal_tokens, inv_perm,
                         seq_len=SEQ_LEN, num_blocks=N, block_len=4):
    """Verbatim pre-vectorization attn257_to_A_block (double-loop scatter-add +
    per-block mean). Reference to pin _attn_to_A_block_vec against, so the
    vectorization can't silently drift from the canonical extract_A_matrices math.
    """
    avg_attn = np.asarray(avg_attn, dtype=np.float32)
    reveal_tokens = np.asarray(reveal_tokens, dtype=np.int64)
    inv_perm = np.asarray(inv_perm, dtype=np.int64)
    model_blocks = reveal_tokens // block_len
    phys_tokens = inv_perm[model_blocks] * block_len + (reveal_tokens % block_len)
    attn_phys = np.zeros((seq_len, seq_len), dtype=np.float32)
    np.add.at(attn_phys, (phys_tokens[:, None], phys_tokens[None, :]), avg_attn[1:, 1:])
    A = np.zeros((num_blocks, num_blocks), dtype=np.float32)
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            A[bi, bj] = attn_phys[bi * block_len:(bi + 1) * block_len,
                                  bj * block_len:(bj + 1) * block_len].mean()
    none_attn = avg_attn[1:, 0]
    none_block = np.array([none_attn[b * block_len:(b + 1) * block_len].mean()
                           for b in range(num_blocks)])
    A += none_block[np.newaxis, :] * 0.1
    np.fill_diagonal(A, 0.0)
    return A


def test_vec_matches_original_scalar():
    """Vectorized core reproduces the original double-loop algorithm (fp-close)."""
    block_len = SEQ_LEN // N
    g = np.random.default_rng(0)
    inv_perm = g.permutation(N)
    # reveal_tokens must be a permutation of [0, SEQ_LEN) (a token_order row).
    reveal = g.permutation(SEQ_LEN)
    for _ in range(5):
        attn = g.random((SEQ_LEN + 1, SEQ_LEN + 1)).astype(np.float32)
        ref = _orig_scalar_A_block(attn, reveal, inv_perm, block_len=block_len)
        got = phs.attn257_to_A_block(attn, reveal, inv_perm)
        assert got.shape == ref.shape
        assert np.allclose(got, ref, atol=1e-6), np.abs(got - ref).max()


def test_vec_leading_dims_match_per_head():
    """Stacked (L,H,..) vec call == per-(l,h) scalar calls, bit-for-bit."""
    block_len = SEQ_LEN // N
    g = np.random.default_rng(1)
    inv_perm = g.permutation(N)
    reveal = g.permutation(SEQ_LEN)
    L, H = 2, 3
    stack = g.random((L, H, SEQ_LEN + 1, SEQ_LEN + 1)).astype(np.float32)
    batched = phs._attn_to_A_block_vec(stack, reveal, inv_perm)
    for l in range(L):
        for h in range(H):
            single = phs.attn257_to_A_block(stack[l, h], reveal, inv_perm)
            assert np.array_equal(batched[l, h], single)


def test_batched_equals_loop_reference():
    model, perm, dev = _FakeAOGPT(), _FakePerm(), torch.device("cpu")
    chunks = _make_chunks(6)
    A_lh_ref, A_heavy_ref = phs._extract_per_head_and_heavy_A_loop(
        model, chunks, perm, dev, seed=0)
    A_lh_b, A_heavy_b = phs.extract_per_head_and_heavy_A(
        model, chunks, perm, dev, seed=0, fwd_batch=4)
    assert A_lh_b.shape == A_lh_ref.shape
    assert np.array_equal(A_lh_b, A_lh_ref)
    assert np.array_equal(A_heavy_b, A_heavy_ref)


def test_fwd_batch_one_equals_loop():
    model, perm, dev = _FakeAOGPT(), _FakePerm(), torch.device("cpu")
    chunks = _make_chunks(5)
    A_lh_ref, A_heavy_ref = phs._extract_per_head_and_heavy_A_loop(
        model, chunks, perm, dev, seed=3)
    A_lh_b, A_heavy_b = phs.extract_per_head_and_heavy_A(
        model, chunks, perm, dev, seed=3, fwd_batch=1)
    assert np.array_equal(A_lh_b, A_lh_ref)
    assert np.array_equal(A_heavy_b, A_heavy_ref)


def test_fwd_batch_larger_than_nchunks():
    model, perm, dev = _FakeAOGPT(), _FakePerm(), torch.device("cpu")
    chunks = _make_chunks(3)
    A_lh_ref, _ = phs._extract_per_head_and_heavy_A_loop(model, chunks, perm, dev, seed=1)
    A_lh_b, _ = phs.extract_per_head_and_heavy_A(
        model, chunks, perm, dev, seed=1, fwd_batch=100)
    assert np.array_equal(A_lh_b, A_lh_ref)


def test_partial_last_group():
    # n_chunks not divisible by fwd_batch -> exercises the tail group
    model, perm, dev = _FakeAOGPT(), _FakePerm(), torch.device("cpu")
    chunks = _make_chunks(7)
    A_lh_ref, A_heavy_ref = phs._extract_per_head_and_heavy_A_loop(
        model, chunks, perm, dev, seed=2)
    A_lh_b, A_heavy_b = phs.extract_per_head_and_heavy_A(
        model, chunks, perm, dev, seed=2, fwd_batch=4)
    assert np.array_equal(A_lh_b, A_lh_ref)
    assert np.array_equal(A_heavy_b, A_heavy_ref)
