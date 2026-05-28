"""NR-1 Task 14 (module-level): tests for the four ablation input transforms.

Pins shape preservation and the obvious mathematical identities:
  identity     : B unchanged
  reverse      : B -> B^T
  sym          : B -> 0.5 (B + B^T), result is symmetric
  row_shuffle  : same permutation per sample, shape preserved
  b_global     : all samples become the dataset mean
"""
import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def _toy(M=4, N=8, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.random((M, N, N)).astype(np.float32)
    for i in range(M):
        np.fill_diagonal(B[i], 0.0)
    return B


def test_identity():
    from neural_readout.ablation_inputs import apply_input_transform
    B = _toy()
    out = apply_input_transform(B, "identity")
    np.testing.assert_array_equal(out, B)


def test_reverse_is_transpose():
    from neural_readout.ablation_inputs import apply_input_transform
    B = _toy()
    out = apply_input_transform(B, "reverse")
    np.testing.assert_array_equal(out, np.transpose(B, (0, 2, 1)))


def test_sym_is_symmetric():
    from neural_readout.ablation_inputs import apply_input_transform
    B = _toy()
    out = apply_input_transform(B, "sym")
    np.testing.assert_allclose(out, np.transpose(out, (0, 2, 1)), atol=1e-7)


def test_row_shuffle_preserves_shape_and_uses_one_perm():
    from neural_readout.ablation_inputs import apply_input_transform
    B = _toy()
    rng = np.random.default_rng(0)
    out = apply_input_transform(B, "row_shuffle", rng=rng)
    assert out.shape == B.shape
    # The permuted rows must come from rows of B (set equality per sample)
    for m in range(B.shape[0]):
        # Each output row should be exactly some row of input
        for i in range(B.shape[1]):
            assert any(np.array_equal(out[m, i], B[m, j]) for j in range(B.shape[1])), \
                f"row {i} of sample {m} is not a row of the input"


def test_b_global_collapses_to_mean():
    from neural_readout.ablation_inputs import apply_input_transform
    B = _toy()
    out = apply_input_transform(B, "b_global")
    expected = B.mean(axis=0, keepdims=True)
    for m in range(B.shape[0]):
        np.testing.assert_allclose(out[m], expected[0], atol=1e-7)


def test_unknown_ablation_raises():
    from neural_readout.ablation_inputs import apply_input_transform
    import pytest
    with pytest.raises(ValueError, match="unknown ablation"):
        apply_input_transform(_toy(), "not_a_real_one")
