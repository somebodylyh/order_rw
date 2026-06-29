"""Tests for adding hard consensus orders to existing g_beta datasets."""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analyses"))

from add_consensus_order_to_gbeta_dataset import (  # noqa: E402
    add_consensus_order,
    add_consensus_order_in_place,
)


def _write_dataset(path, include_ranks=True, include_weights=True):
    payload = {
        "B_raw": np.zeros((3, 2, 65, 65), dtype=np.float32),
        "train_idx": np.array([0], dtype=np.int64),
        "val_idx": np.array([1], dtype=np.int64),
        "test_idx": np.array([2], dtype=np.int64),
    }
    if include_ranks:
        payload["teacher_ranks"] = np.array([
            [[0, 1, 2, 3], [3, 2, 1, 0]],
            [[1, 0, 2, 3], [3, 2, 0, 1]],
            [[3, 2, 1, 0], [0, 1, 2, 3]],
        ], dtype=np.int64)
    if include_weights:
        payload["teacher_weights"] = np.array([
            [0.75, 0.25],
            [0.8, 0.2],
            [0.25, 0.75],
        ], dtype=np.float32)
    np.savez_compressed(path, **payload)
    return set(payload)


def test_add_consensus_order_preserves_existing_fields(tmp_path):
    src = tmp_path / "src.npz"
    dst = tmp_path / "dst.npz"
    original_fields = _write_dataset(src)

    add_consensus_order(str(src), str(dst))

    with np.load(dst, allow_pickle=True) as z:
        assert set(z.files) == original_fields | {"teacher_consensus_order"}
        assert z["teacher_consensus_order"].shape == (3, 4)
        np.testing.assert_array_equal(
            z["teacher_consensus_order"][0], np.array([0, 1, 2, 3]),
        )


@pytest.mark.parametrize("missing", ["ranks", "weights"])
def test_add_consensus_order_requires_teacher_inputs(tmp_path, missing):
    src = tmp_path / "src.npz"
    dst = tmp_path / "dst.npz"
    _write_dataset(
        src,
        include_ranks=missing != "ranks",
        include_weights=missing != "weights",
    )

    with pytest.raises(KeyError, match="teacher_"):
        add_consensus_order(str(src), str(dst))


def test_add_consensus_order_in_place_replaces_source(tmp_path):
    src = tmp_path / "src.npz"
    _write_dataset(src)

    add_consensus_order_in_place(str(src))

    with np.load(src, allow_pickle=True) as z:
        assert "teacher_consensus_order" in z.files
    assert not list(tmp_path.glob("*.tmp.npz"))
