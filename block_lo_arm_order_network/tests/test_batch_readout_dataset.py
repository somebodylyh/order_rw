"""BR-1 Task 3: tests for dataset_batch builder + loader.

All tests exercise the pure-numpy _build_from_B_array path with synthetic B.
The end-to-end build_dataset(ckpt_path=...) path is exercised by BR-1 Task 10
smoke run (where the wikitext tokenization cost is paid once anyway).
"""
import sys
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def _make_synthetic_B(M, N, seed):
    """Produce a (M, N, N) float32 array with zero diagonal -- shape only,
    no claim about teacher labels being meaningful."""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((M, N, N)).astype(np.float32)
    for m in range(M):
        np.fill_diagonal(B[m], 0.0)
    return B


def test_invalid_split_fractions_rejected():
    from batch_readout.dataset_batch import _build_from_B_array
    B = _make_synthetic_B(2, 8, 0)
    chunks = np.zeros((2, 1), dtype=np.int64)
    with pytest.raises(ValueError, match="invalid split"):
        _build_from_B_array(B, chunks, seed=0, train_frac=0.6, val_frac=0.5)


def test_B_shape_validation():
    from batch_readout.dataset_batch import _build_from_B_array
    bad_B = np.zeros((4, 8, 16), dtype=np.float32)
    chunks = np.zeros((4, 1), dtype=np.int64)
    with pytest.raises(ValueError, match=r"must be \(M, N, N\)"):
        _build_from_B_array(bad_B, chunks, seed=0)


def test_chunks_mismatch_rejected():
    from batch_readout.dataset_batch import _build_from_B_array
    B = _make_synthetic_B(4, 8, 0)
    chunks = np.zeros((3, 1), dtype=np.int64)
    with pytest.raises(ValueError, match="chunks first dim"):
        _build_from_B_array(B, chunks, seed=0)


def test_split_sizes_and_rank_zero_convention(tmp_path):
    from batch_readout.dataset_batch import _build_from_B_array, load_dataset
    M, N = 20, 8
    B = _make_synthetic_B(M, N, seed=0)
    chunks = np.arange(M * 3, dtype=np.int64).reshape(M, 3)
    out = tmp_path / "ds.npz"
    info = _build_from_B_array(B, chunks, seed=0, train_frac=0.8, val_frac=0.1, out_path=str(out))

    assert info["M_train"] == 16
    assert info["M_val"] == 2
    assert info["M_test"] == 2
    # earliest reveal node has rank 0
    sig = info["sigma_T"]
    rk = info["rank"]
    assert (rk[np.arange(M), sig[:, 0]] == 0).all()

    # round-trip via disk
    ds = load_dataset(str(out))
    for split in ("train", "val", "test"):
        for key in ("B_batch", "sigma_T", "rank", "pairwise_Y", "chunks"):
            assert f"{split}_{key}" in ds, f"missing {split}_{key}"
    assert ds["train_B_batch"].shape == (16, N, N)
    assert ds["train_sigma_T"].shape == (16, N)
    assert ds["train_chunks"].shape == (16, 3)


def test_pairwise_Y_consistency(tmp_path):
    """Y[i, j] == 1 iff rank_i < rank_j (rank-0 = earliest)."""
    from batch_readout.dataset_batch import _build_from_B_array
    B = _make_synthetic_B(5, 8, seed=2)
    chunks = np.zeros((5, 1), dtype=np.int64)
    info = _build_from_B_array(B, chunks, seed=2)
    rk = info["rank"]
    Y = info["pairwise_Y"]
    expected = (rk[:, :, None] < rk[:, None, :]).astype(np.uint8)
    np.fill_diagonal_v = None  # noqa
    for m in range(len(Y)):
        np.fill_diagonal(expected[m], 0)
    np.testing.assert_array_equal(Y, expected)


def test_split_deterministic_in_seed(tmp_path):
    from batch_readout.dataset_batch import _build_from_B_array
    B = _make_synthetic_B(10, 8, seed=0)
    chunks = np.zeros((10, 1), dtype=np.int64)
    a = _build_from_B_array(B, chunks, seed=7)
    b = _build_from_B_array(B, chunks, seed=7)
    np.testing.assert_array_equal(a["train_idx"], b["train_idx"])
    np.testing.assert_array_equal(a["val_idx"], b["val_idx"])
    np.testing.assert_array_equal(a["test_idx"], b["test_idx"])
