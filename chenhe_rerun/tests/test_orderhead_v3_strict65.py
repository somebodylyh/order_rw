import sys, os
import numpy as np
import pytest

CHENHE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CHENHE)
from orderhead_v3.l0_strict65 import build_model_frame_strict65
from orderhead_v3.constants import assert_layout


def _fake_attn(S=2, H=8, T=256, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random((S, H, T + 1, T + 1)).astype(np.float32)
    a /= a.sum(-1, keepdims=True)              # row-stochastic like softmax
    probe = np.stack([rng.permutation(T) for _ in range(S)]).astype(np.int64)
    return a, probe


def test_shape_and_wellformed():
    a, probe = _fake_attn()
    B = build_model_frame_strict65(a, probe)
    assert B.shape == (2, 8, 65, 65)
    assert np.isfinite(B).all()
    assert np.allclose(B[:, :, :, 0], 0.0)                 # no edges INTO None
    diag = B[:, :, np.arange(65), np.arange(65)]
    assert np.allclose(diag, 0.0)                          # zero diagonal


def test_frame_index_semantics():
    # Build attention where, in reveal order, token t attends only to token t-1.
    # "token t attends t-1" => block b-1 -> block b. B is source->target with the
    # self-edge diagonal removed, so block b's only outgoing content edge is to the
    # adjacent NEXT block b+1. The strong edges thus form the identity-adjacent
    # band, proving block index i maps to model-frame block i (no inv_perm scramble):
    # an inv_perm mismatch would put the edge at a permuted (non-adjacent) index.
    S, H, T, BL, NB = 1, 8, 256, 4, 64
    probe = np.arange(T)[None, :].astype(np.int64)         # identity reveal
    a = np.zeros((S, H, T + 1, T + 1), np.float32)
    for t in range(T):
        a[:, :, t + 1, t] = 1.0                            # token t attends token t-1 (+None offset)
    a[:, :, 0, 0] = 1.0
    B = build_model_frame_strict65(a, probe)
    for b in range(1, NB - 1):
        row = B[0, 0, b + 1, 1:]                            # outgoing content edges from block b
        assert row.argmax() == (b + 1), (b, row.argmax())  # -> adjacent next block


def test_layout_guard():
    with pytest.raises(ValueError):
        assert_layout(96, 4, 8)                             # block96 must fail
