"""NR-1 Task 3: tests for the CDL-source-start teacher label generator.

Pins four properties:
  1. rank 0 = earliest revealed (load-bearing convention used by loss.py)
  2. pairwise Y direction: Y[i, j] = 1 iff rank[i] < rank[j]
  3. first revealed node equals argmax(readiness) (source-start anchor)
  4. on a toy chain B[i, i+1] = 1, the teacher rolls forward 0, 1, ..., N-1
"""
import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_rank_zero_is_earliest():
    from neural_readout.teacher_labels import generate_teacher_label
    rng = np.random.default_rng(0)
    B = rng.random((64, 64)).astype(np.float32)
    np.fill_diagonal(B, 0.0)
    sigma, rank, _Y = generate_teacher_label(B, alpha_dep=0.5)
    assert rank[sigma[0]] == 0
    assert rank[sigma[-1]] == 63
    # rank is a valid permutation
    assert sorted(rank.tolist()) == list(range(64))


def test_pairwise_direction():
    """Y[i, j] = 1 iff rank[i] < rank[j] (i revealed before j)."""
    from neural_readout.teacher_labels import generate_teacher_label
    rng = np.random.default_rng(1)
    B = rng.random((64, 64)).astype(np.float32)
    np.fill_diagonal(B, 0.0)
    sigma, rank, Y = generate_teacher_label(B, alpha_dep=0.5)
    expected = (rank[:, None] < rank[None, :]).astype(np.uint8)
    np.fill_diagonal(expected, 0)
    np.testing.assert_array_equal(Y, expected)
    # Diagonal is zero
    assert np.all(np.diag(Y) == 0)


def test_source_start_uses_readiness_argmax():
    """First revealed node must equal argmax(out(v) - alpha * in(v))."""
    from neural_readout.teacher_labels import generate_teacher_label
    from attn_order_mlp_policy import readiness_vector
    rng = np.random.default_rng(2)
    B = rng.random((64, 64)).astype(np.float32)
    np.fill_diagonal(B, 0.0)
    expected_start = int(np.argmax(readiness_vector(B, alpha_dep=0.5)))
    sigma, _rank, _Y = generate_teacher_label(B, alpha_dep=0.5)
    assert sigma[0] == expected_start


def test_toy_chain_direction():
    """Forward 'chain' B: i->i+1 with weight 1. With source-start anchor and
    greedy C-D+L the teacher should roll out [0, 1, 2, ..., N-1].
    Node 0 has out=1, in=0 -> readiness 1.0 = max. Then C-D+L greedily walks
    the chain forward via the L (last-step) component.
    """
    from neural_readout.teacher_labels import generate_teacher_label
    N = 8
    B = np.zeros((N, N), dtype=np.float32)
    for i in range(N - 1):
        B[i, i + 1] = 1.0
    sigma, _rank, _Y = generate_teacher_label(B, alpha_dep=0.5)
    np.testing.assert_array_equal(sigma, np.arange(N))


def test_diag_assertion_rejects_nonzero_diag():
    """A non-zero diagonal in B is a usage error: must raise, not silently coerce."""
    from neural_readout.teacher_labels import generate_teacher_label
    import pytest
    B = np.eye(8, dtype=np.float32)
    with pytest.raises(ValueError, match="zero diagonal"):
        generate_teacher_label(B, alpha_dep=0.5)
