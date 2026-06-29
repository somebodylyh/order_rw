"""Tests for label-free L0 dynamic g_beta dataset builder.

Covers: deterministic split, save/load roundtrip, forbidden-field audit,
and shape/value invariants on the saved .npz.

GPU-dependent integration tests (full extract→teacher→save pipeline)
are excluded from unit tests.
"""

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))
sys.path.insert(0, str(ROOT / "analyses"))

from build_l0_dynamic_gbeta_dataset import (  # noqa: E402
    deterministic_split,
    load_dataset_fields,
    save_dataset,
    select_layer_attention,
    weighted_consensus_order,
)


# ---------------------------------------------------------------------------
# select_layer_attention (layer-aware extraction)
# ---------------------------------------------------------------------------

def _fake_attn_list(n_layers=4, B=2, H=8, Tp1=257):
    rng = np.random.default_rng(0)
    return [
        rng.random((B, H, Tp1, Tp1)).astype(np.float32) for _ in range(n_layers)
    ]


def test_select_layer_attention_single_layer_returns_that_layer():
    attn = _fake_attn_list()
    out = select_layer_attention(attn, [1])
    np.testing.assert_array_equal(out, attn[1])


def test_select_layer_attention_default_l0_matches_legacy_indexing():
    attn = _fake_attn_list()
    out = select_layer_attention(attn, [0])
    np.testing.assert_array_equal(out, attn[0])


def test_select_layer_attention_all_layers_concatenates_on_head_axis():
    attn = _fake_attn_list(n_layers=4, H=8)
    out = select_layer_attention(attn, [0, 1, 2, 3])
    assert out.shape[1] == 4 * 8, "all-layer head dim must be L*H"
    # heads preserve per-layer order: layer l occupies heads [l*H : (l+1)*H]
    for l in range(4):
        np.testing.assert_array_equal(out[:, l * 8:(l + 1) * 8], attn[l])


def test_select_layer_attention_rejects_out_of_range_layer():
    attn = _fake_attn_list(n_layers=4)
    with pytest.raises((IndexError, ValueError)):
        select_layer_attention(attn, [4])


def test_select_layer_attention_rejects_empty_layers():
    attn = _fake_attn_list()
    with pytest.raises(ValueError):
        select_layer_attention(attn, [])


# ---------------------------------------------------------------------------
# deterministic_split
# ---------------------------------------------------------------------------

def test_deterministic_split_is_disjoint():
    split = deterministic_split(20, seed=7)
    joined = np.concatenate([split["train"], split["val"], split["test"]])
    assert len(np.unique(joined)) == 20


def test_deterministic_split_sizes_are_correct():
    for M in (10, 20, 100, 2000):
        split = deterministic_split(M, seed=42)
        total = len(split["train"]) + len(split["val"]) + len(split["test"])
        assert total == M
        # Train ≈ 80%
        assert abs(len(split["train"]) - int(round(M * 0.8))) <= 1
        # Val ≈ 10%
        assert abs(len(split["val"]) - int(round(M * 0.1))) <= 1


def test_deterministic_split_is_reproducible():
    s1 = deterministic_split(50, seed=99)
    s2 = deterministic_split(50, seed=99)
    for k in ("train", "val", "test"):
        np.testing.assert_array_equal(s1[k], s2[k])


def test_deterministic_split_differs_with_seed():
    s1 = deterministic_split(50, seed=0)
    s2 = deterministic_split(50, seed=1)
    # At least one split should differ.
    any_diff = any(
        not np.array_equal(s1[k], s2[k]) for k in ("train", "val", "test")
    )
    assert any_diff, "different seeds should produce different splits"


def test_deterministic_split_indices_are_sorted():
    split = deterministic_split(100, seed=3)
    for k in ("train", "val", "test"):
        assert np.all(np.diff(split[k]) > 0), f"{k} indices not sorted"


# ---------------------------------------------------------------------------
# save_dataset / load_dataset_fields
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(M=4, H=8):
    """Build a minimal synthetic dataset for save/load tests."""
    B_raw = np.random.default_rng(0).random((M, H, 65, 65)).astype(np.float32)
    for m in range(M):
        for h in range(H):
            np.fill_diagonal(B_raw[m, h], 0.0)
            B_raw[m, h, 1:, 0] = 0.0
    teacher = {
        "orders": np.zeros((M, H, 64), dtype=np.int64),
        "ranks": np.zeros((M, H, 64), dtype=np.int64),
        "margin": np.zeros((M, H), dtype=np.float32),
        "destroyed_gap": np.zeros((M, H), dtype=np.float32),
        "agreement": np.zeros((M, H), dtype=np.float32),
        "quality": np.zeros((M, H), dtype=np.float32),
        "weights": np.full((M, H), 1.0 / H, dtype=np.float32),
        "pairwise": np.full((M, 64, 64), 0.5, dtype=np.float32),
        "consensus_order": np.tile(
            np.arange(64, dtype=np.int64), (M, 1),
        ),
    }
    split = deterministic_split(M, seed=0)
    return B_raw, teacher, split


def test_save_and_load_roundtrip(tmp_path):
    B_raw, teacher, split = _make_synthetic_dataset(M=20)
    path = tmp_path / "test_dataset.npz"
    meta = {"source_ckpt": "test", "M": 20, "label_free": True}
    save_dataset(str(path), B_raw, teacher, split, meta)

    with np.load(path, allow_pickle=True) as z:
        assert z["B_raw"].shape == (20, 8, 65, 65)
        assert z["teacher_pairwise"].shape == (20, 64, 64)
        assert z["teacher_weights"].shape == (20, 8)
        assert z["teacher_consensus_order"].shape == (20, 64)
        for row in z["teacher_consensus_order"]:
            np.testing.assert_array_equal(np.sort(row), np.arange(64))
        assert z["train_idx"].shape[0] > 0
        assert z["val_idx"].shape[0] > 0
        assert z["test_idx"].shape[0] > 0


def test_weighted_consensus_order_sorts_weighted_mean_rank():
    ranks = np.array([
        [0, 1, 2, 3],
        [3, 2, 1, 0],
    ])
    weights = np.array([0.75, 0.25])
    order = weighted_consensus_order(ranks, weights)
    np.testing.assert_array_equal(order, np.array([0, 1, 2, 3]))


def test_saved_dataset_has_no_physical_fields(tmp_path):
    B_raw, teacher, split = _make_synthetic_dataset(M=4)
    path = tmp_path / "clean.npz"
    save_dataset(str(path), B_raw, teacher, split, {"test": True})

    fields = load_dataset_fields(str(path))
    forbidden = {"block_perm", "inv_perm", "clean_perm",
                 "physical_order", "l2r_order"}
    leaked = forbidden & set(fields)
    assert not leaked, f"Dataset contains forbidden fields: {sorted(leaked)}"


def test_saved_teacher_pairwise_is_antisymmetric(tmp_path):
    """Even for synthetic data, if pairwise is 0.5 everywhere, it must
    be antisymmetric (0.5 + 0.5 = 1)."""
    B_raw, teacher, split = _make_synthetic_dataset(M=4)
    path = tmp_path / "pairwise.npz"
    save_dataset(str(path), B_raw, teacher, split, {})

    with np.load(path, allow_pickle=True) as z:
        Y = z["teacher_pairwise"]  # (M, 64, 64)
        for m in range(Y.shape[0]):
            np.testing.assert_allclose(
                Y[m] + Y[m].T, np.ones((64, 64)), atol=1e-6,
            )
            np.testing.assert_allclose(
                np.diag(Y[m]), 0.5, atol=1e-6,
            )


def test_saved_teacher_weights_sum_to_one(tmp_path):
    B_raw, teacher, split = _make_synthetic_dataset(M=4)
    path = tmp_path / "weights.npz"
    save_dataset(str(path), B_raw, teacher, split, {})

    with np.load(path, allow_pickle=True) as z:
        w = z["teacher_weights"]  # (M, H)
        np.testing.assert_allclose(w.sum(axis=1), np.ones(w.shape[0]), atol=1e-6)


def test_saved_meta_json_is_valid(tmp_path):
    B_raw, teacher, split = _make_synthetic_dataset(M=4)
    path = tmp_path / "meta.npz"
    meta = {"source_ckpt": "/fake/path", "M": 4, "coordinate_frame": "model"}
    save_dataset(str(path), B_raw, teacher, split, meta)

    with np.load(path, allow_pickle=True) as z:
        meta_loaded = json.loads(str(z["meta_json"]))
        assert meta_loaded["coordinate_frame"] == "model"
        assert meta_loaded["M"] == 4


def test_saved_B_raw_is_finite(tmp_path):
    B_raw, teacher, split = _make_synthetic_dataset(M=4)
    path = tmp_path / "finite.npz"
    save_dataset(str(path), B_raw, teacher, split, {})

    with np.load(path, allow_pickle=True) as z:
        assert np.isfinite(z["B_raw"]).all()


def test_saved_B_raw_diagonal_is_zero(tmp_path):
    B_raw, teacher, split = _make_synthetic_dataset(M=4)
    path = tmp_path / "diag.npz"
    save_dataset(str(path), B_raw, teacher, split, {})

    with np.load(path, allow_pickle=True) as z:
        B = z["B_raw"]
        for m in range(B.shape[0]):
            for h in range(B.shape[1]):
                np.testing.assert_allclose(np.diag(B[m, h]), 0.0, atol=1e-7)


def test_saved_B_raw_content_to_none_column_is_zero(tmp_path):
    B_raw, teacher, split = _make_synthetic_dataset(M=4)
    path = tmp_path / "nonecol.npz"
    save_dataset(str(path), B_raw, teacher, split, {})

    with np.load(path, allow_pickle=True) as z:
        B = z["B_raw"]
        assert np.all(B[..., 1:, 0] == 0.0)


def test_split_indices_are_valid(tmp_path):
    B_raw, teacher, split = _make_synthetic_dataset(M=10)
    path = tmp_path / "split.npz"
    save_dataset(str(path), B_raw, teacher, split, {})

    with np.load(path, allow_pickle=True) as z:
        train = z["train_idx"]
        val = z["val_idx"]
        test = z["test_idx"]
        # No overlap
        all_idx = np.concatenate([train, val, test])
        assert len(np.unique(all_idx)) == len(all_idx)
        # All indices in range
        assert train.max() < 10
        assert val.max() < 10
        assert test.max() < 10


def test_builder_module_has_no_physical_fields_in_save_signature():
    """The save_dataset function must not have physical-coordinate parameters."""
    import inspect
    sig = inspect.signature(save_dataset)
    param_names = set(sig.parameters.keys())
    forbidden = {"inv_perm", "clean_perm", "block_perm", "physical_order"}
    overlap = param_names & forbidden
    assert not overlap, (
        f"save_dataset must not accept physical-coordinate parameters; "
        f"found: {sorted(overlap)}"
    )
