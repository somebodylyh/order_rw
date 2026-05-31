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
