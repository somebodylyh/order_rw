"""NR-1 Task 4: tests for dataset save/load round-trip.

Round-trips synthetic (B, sigma, rank, chunk_index, split) through .npz and
verifies all arrays are bit-identical on reload. Does NOT exercise
build_dataset_from_ckpt -- that path is exercised end-to-end by Task 12
(smoke 1k run).
"""
import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_dataset_roundtrip(tmp_path):
    from neural_readout.dataset import save_dataset, load_dataset

    M, N = 8, 64
    rng = np.random.default_rng(0)
    B = rng.random((M, N, N)).astype(np.float32)
    for i in range(M):
        np.fill_diagonal(B[i], 0.0)
    sigma = np.stack([rng.permutation(N) for _ in range(M)]).astype(np.int64)
    rank = np.empty_like(sigma)
    for i in range(M):
        rank[i, sigma[i]] = np.arange(N)
    chunk_index = np.array([10, 30, 100, 250, 300, 400, 500, 700], dtype=np.int64)

    path = tmp_path / "tiny.npz"
    save_dataset(str(path), B, sigma, rank, chunk_index, split="train",
                 meta={"ckpt_path": "/dev/null", "M": 8, "seed": 42, "alpha_dep": 0.5})

    B2, sigma2, rank2, ci2, split2 = load_dataset(str(path))
    np.testing.assert_array_equal(B, B2)
    np.testing.assert_array_equal(sigma, sigma2)
    np.testing.assert_array_equal(rank, rank2)
    np.testing.assert_array_equal(chunk_index, ci2)
    assert split2 == "train"


def test_save_rejects_bad_shapes(tmp_path):
    from neural_readout.dataset import save_dataset
    import pytest
    path = str(tmp_path / "bad.npz")
    rng = np.random.default_rng(0)
    B = rng.random((4, 64, 64)).astype(np.float32)
    bad_sigma = rng.integers(0, 64, size=(4, 32), dtype=np.int64)  # wrong N
    rank = np.zeros((4, 64), dtype=np.int64)
    ci = np.zeros(4, dtype=np.int64)
    with pytest.raises(ValueError, match="sigma/rank must be"):
        save_dataset(path, B, bad_sigma, rank, ci, split="train")


def test_load_missing_keys_raises(tmp_path):
    from neural_readout.dataset import load_dataset
    import pytest
    path = tmp_path / "incomplete.npz"
    np.savez_compressed(path, only_one_key=np.zeros(3))
    with pytest.raises(KeyError, match="missing required keys"):
        load_dataset(str(path))
