"""§3.3 selected-head dataset builder: pure-function tests.

The genuinely new logic vs the existing BR-1 batch builder is *head selection*:
pick one head (l, h) out of the per-head stack A_lh (n, L, H, N, N) and turn it
into the canonical batch-mean graph B = mean_b (A_b^T), diagonal zeroed. The
batch-mean math itself is delegated to the already-tested
`per_head_order_scan._batch_mean_B`, so these tests pin (a) that the right head
is selected and (b) that the canonical convention is preserved end-to-end.

The GPU wiring (`build_selected_head_dataset`) is exercised by a smoke run, not here.
"""
import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_selects_the_requested_head_and_applies_canonical_batchmean():
    """selected_head_B((l,h)) == _batch_mean_B(A_lh[:, l, h]) (canonical B=A^T)."""
    from batch_readout.selected_head_dataset import selected_head_B
    from per_head_order_scan import _batch_mean_B
    rng = np.random.default_rng(0)
    n, L, H, N, bs = 6, 2, 3, 4, 3  # M = 2 batches
    A_lh = rng.standard_normal((n, L, H, N, N)).astype(np.float32)
    out = selected_head_B(A_lh, head=(1, 2), batch_size=bs)
    expected = _batch_mean_B(A_lh[:, 1, 2], n // bs, bs)
    assert out.shape == (2, N, N)
    assert out.dtype == np.float32
    assert np.array_equal(out, expected)


def test_different_heads_give_different_B():
    """Picking a different (l, h) must change the result (real selection, not a stub)."""
    from batch_readout.selected_head_dataset import selected_head_B
    rng = np.random.default_rng(1)
    n, L, H, N, bs = 4, 2, 2, 3, 2
    A_lh = rng.standard_normal((n, L, H, N, N)).astype(np.float32)
    b00 = selected_head_B(A_lh, head=(0, 0), batch_size=bs)
    b11 = selected_head_B(A_lh, head=(1, 1), batch_size=bs)
    assert not np.array_equal(b00, b11)


def test_diagonal_is_exactly_zero():
    """Canonical B has a zeroed diagonal regardless of A's diagonal."""
    from batch_readout.selected_head_dataset import selected_head_B
    n, L, H, N, bs = 4, 1, 1, 3, 2
    A_lh = np.ones((n, L, H, N, N), dtype=np.float32)  # diagonal = 1 everywhere
    out = selected_head_B(A_lh, head=(0, 0), batch_size=bs)
    d = np.arange(N)
    assert np.all(out[:, d, d] == 0.0)
