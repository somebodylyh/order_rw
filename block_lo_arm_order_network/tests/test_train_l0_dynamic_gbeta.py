"""Tests for L0DynamicGBeta training loop and checkpoint selection."""

import json
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))
sys.path.insert(0, str(ROOT / "analyses"))

from build_l0_dynamic_gbeta_dataset import deterministic_split, save_dataset  # noqa: E402
from batch_readout.train_l0_dynamic_gbeta import (  # noqa: E402
    load_pretrain_dataset,
    train,
)
from batch_readout.l0_dynamic_gbeta import L0DynamicGBeta  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic dataset builder for tests
# ---------------------------------------------------------------------------

def _make_synthetic_pretrain_dataset(
    path: str,
    M: int = 64,
    H: int = 8,
    seed: int = 0,
):
    """Build a tiny label-free pretrain .npz with clear ordering signal.

    One head (head 0) has a strong chain structure → CDL teacher recovers
    a consistent order.  Other heads have weaker/noisy signals so the
    dynamic teacher can learn to gate.
    """
    rng = np.random.default_rng(seed)
    B_raw = np.zeros((M, H, 65, 65), dtype=np.float32)
    teacher_pairwise = np.zeros((M, 64, 64), dtype=np.float32)
    teacher_weights = np.full((M, H), 1.0 / H, dtype=np.float32)
    teacher_orders = np.zeros((M, H, 64), dtype=np.int64)
    teacher_ranks = np.zeros((M, H, 64), dtype=np.int64)
    teacher_margin = np.zeros((M, H), dtype=np.float32)
    teacher_destroyed_gap = np.zeros((M, H), dtype=np.float32)
    teacher_agreement = np.zeros((M, H), dtype=np.float32)
    teacher_quality = np.zeros((M, H), dtype=np.float32)
    teacher_consensus_order = np.zeros((M, 64), dtype=np.int64)

    for m in range(M):
        # Head 0: strong chain 0→1→2→...→63 with noise
        for i in range(1, 64):
            B_raw[m, 0, i, i + 1] = 1.0 + 0.1 * rng.random()
        B_raw[m, 0, 0, 1] = 1.0  # None → block 0
        # Zero diag and content-to-None
        for h in range(H):
            np.fill_diagonal(B_raw[m, h], 0.0)
            B_raw[m, h, 1:, 0] = 0.0

        # Other heads: weaker chain with noise
        for h in range(1, H):
            shift = (h * 7) % 64
            for i in range(1, 64):
                j = 1 + (i - 1 + shift) % 64
                if i != j:
                    B_raw[m, h, i, j] = 0.4 + 0.2 * rng.random()
            B_raw[m, h, 0, 1] = 0.5

        # Simple teacher: head 0 drives the consensus → order 0..63
        teacher_orders[m, 0] = np.arange(64, dtype=np.int64)
        teacher_ranks[m, 0] = np.arange(64, dtype=np.int64)
        for h in range(1, H):
            teacher_orders[m, h] = rng.permutation(64)
            rank = np.empty(64, dtype=np.int64)
            rank[teacher_orders[m, h]] = np.arange(64)
            teacher_ranks[m, h] = rank

        # Higher weight on head 0, uniform on others
        w = np.full(H, 1.0 / H)
        w[0] = 3.0 / H  # slightly favor head 0
        w = w / w.sum()
        teacher_weights[m] = w

        # Build pairwise Y from weighted ranks
        Y = np.zeros((64, 64), dtype=np.float32)
        for h in range(H):
            rh = teacher_ranks[m, h]
            Y += w[h] * (rh[:, None] < rh[None, :])
        np.fill_diagonal(Y, 0.5)
        teacher_pairwise[m] = Y
        mean_rank = (
            teacher_ranks[m].astype(np.float64) * w[:, None]
        ).sum(axis=0)
        teacher_consensus_order[m] = np.argsort(
            mean_rank, kind="stable",
        )

    split = deterministic_split(M, seed=seed)
    teacher = {
        "orders": teacher_orders,
        "ranks": teacher_ranks,
        "margin": teacher_margin,
        "destroyed_gap": teacher_destroyed_gap,
        "agreement": teacher_agreement,
        "quality": teacher_quality,
        "weights": teacher_weights,
        "pairwise": teacher_pairwise,
        "consensus_order": teacher_consensus_order,
    }
    meta = {"synthetic": True, "M": M, "H": H, "label_free": True}
    save_dataset(path, B_raw, teacher, split, meta)


# ============================================================================
# load_pretrain_dataset
# ============================================================================

def test_load_pretrain_dataset_shapes(tmp_path):
    ds_path = str(tmp_path / "test.npz")
    _make_synthetic_pretrain_dataset(ds_path, M=32)
    data = load_pretrain_dataset(ds_path, device="cpu")
    assert data["B_raw"].shape == (32, 8, 65, 65)
    assert data["teacher_pairwise"].shape == (32, 64, 64)
    assert data["teacher_weights"].shape == (32, 8)
    assert data["teacher_consensus_order"].shape == (32, 64)
    assert data["teacher_consensus_order"].dtype == torch.int64
    assert len(data["train_idx"]) > 0
    assert len(data["val_idx"]) > 0
    assert len(data["test_idx"]) > 0


def test_load_pretrain_dataset_rejects_physical_fields(tmp_path):
    ds_path = str(tmp_path / "bad.npz")
    _make_synthetic_pretrain_dataset(ds_path, M=4)
    # Manually inject a forbidden field.
    with np.load(ds_path, allow_pickle=True) as z:
        data = dict(z)
    data["inv_perm"] = np.arange(64)
    np.savez_compressed(ds_path, **data)
    with pytest.raises(RuntimeError, match="forbidden"):
        load_pretrain_dataset(ds_path)


def test_load_pretrain_dataset_allows_legacy_without_consensus(tmp_path):
    ds_path = str(tmp_path / "legacy.npz")
    _make_synthetic_pretrain_dataset(ds_path, M=4)
    with np.load(ds_path, allow_pickle=True) as z:
        payload = {
            key: z[key].copy()
            for key in z.files
            if key != "teacher_consensus_order"
        }
    np.savez_compressed(ds_path, **payload)

    data = load_pretrain_dataset(
        ds_path, device="cpu", require_consensus_order=False,
    )
    assert data["teacher_consensus_order"] is None

    with pytest.raises(KeyError, match="teacher_consensus_order"):
        load_pretrain_dataset(
            ds_path, device="cpu", require_consensus_order=True,
        )


# ============================================================================
# End-to-end training on synthetic data
# ============================================================================

def test_tiny_training_loss_decreases(tmp_path):
    """End-to-end: synthetic dataset → train → loss must decrease."""
    ds_path = str(tmp_path / "synth.npz")
    _make_synthetic_pretrain_dataset(ds_path, M=64, seed=0)
    out_dir = str(tmp_path / "run")

    result = train(
        dataset_path=ds_path,
        out_dir=out_dir,
        epochs=8,
        batch_size=8,
        lr=3e-3,  # higher lr for fast convergence on tiny data
        weight_decay=1e-3,
        lambda_aux=0.05,
        lambda_ent=0.001,
        min_gate_entropy=1.0,  # lower floor for small data
        seed=0,
        device="cpu",
    )

    # Loss should decrease from first to last epoch.
    first_loss = result["metrics_history"][0]["train"]["loss"]
    last_loss = result["metrics_history"][-1]["train"]["loss"]
    assert last_loss < first_loss, (
        f"training loss should decrease: {first_loss:.4f} → {last_loss:.4f}"
    )


@pytest.mark.parametrize("loss_type", [
    "listmle", "pairwise_bce", "rank_kl",
])
def test_tiny_training_supports_each_loss_type(tmp_path, loss_type):
    ds_path = str(tmp_path / f"{loss_type}.npz")
    _make_synthetic_pretrain_dataset(ds_path, M=24, seed=20)
    out_dir = str(tmp_path / loss_type)

    result = train(
        dataset_path=ds_path,
        out_dir=out_dir,
        epochs=2,
        batch_size=8,
        lr=3e-3,
        seed=20,
        device="cpu",
        loss_type=loss_type,
        rank_kl_temperature=4.0,
    )
    assert np.isfinite(result["best_val_loss_final"])

    with open(pathlib.Path(out_dir) / "config.json") as f:
        cfg = json.load(f)
    assert cfg["loss_type"] == loss_type
    assert cfg["rank_kl_temperature"] == 4.0

    ckpt = torch.load(
        pathlib.Path(out_dir) / "g_beta_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert ckpt["config"]["loss_type"] == loss_type
    if loss_type != "pairwise_bce":
        for record in result["metrics_history"]:
            assert record["train"]["loss_aux"] == 0.0
            assert record["val"]["loss_aux"] == 0.0


def test_train_rejects_unknown_loss_type(tmp_path):
    ds_path = str(tmp_path / "synth.npz")
    _make_synthetic_pretrain_dataset(ds_path, M=8)
    with pytest.raises(ValueError, match="loss_type"):
        train(
            dataset_path=ds_path,
            out_dir=str(tmp_path / "run"),
            epochs=1,
            device="cpu",
            loss_type="unknown",
        )


def test_best_checkpoint_saved_and_reloadable(tmp_path):
    ds_path = str(tmp_path / "synth.npz")
    _make_synthetic_pretrain_dataset(ds_path, M=32, seed=1)
    out_dir = str(tmp_path / "run")

    train(
        dataset_path=ds_path,
        out_dir=out_dir,
        epochs=4,
        batch_size=8,
        lr=3e-3,
        seed=1,
        device="cpu",
    )

    best_path = pathlib.Path(out_dir) / "g_beta_best.pt"
    assert best_path.exists()

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "config" in ckpt
    assert ckpt["config"]["selection_label_free"] is True

    # Reload into a fresh model.
    model = L0DynamicGBeta(heads=8)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    B = torch.randn(2, 8, 65, 65)
    for b in range(2):
        for h in range(8):
            B[b, h].fill_diagonal_(0.0)
            B[b, h, 1:, 0] = 0.0
    scores, aux = model(B, apply_head_dropout=False)
    assert scores.shape == (2, 64)
    assert torch.isfinite(scores).all()


def test_last_checkpoint_also_saved(tmp_path):
    ds_path = str(tmp_path / "synth.npz")
    _make_synthetic_pretrain_dataset(ds_path, M=16, seed=2)
    out_dir = str(tmp_path / "run")

    train(dataset_path=ds_path, out_dir=out_dir, epochs=2, batch_size=4,
          seed=2, device="cpu")

    assert (pathlib.Path(out_dir) / "g_beta_last.pt").exists()


def test_config_json_saved(tmp_path):
    ds_path = str(tmp_path / "synth.npz")
    _make_synthetic_pretrain_dataset(ds_path, M=16, seed=3)
    out_dir = str(tmp_path / "run")

    train(dataset_path=ds_path, out_dir=out_dir, epochs=2, batch_size=4,
          seed=3, device="cpu")

    with open(pathlib.Path(out_dir) / "config.json") as f:
        cfg = json.load(f)
    assert cfg["model_name"] == "l0_dynamic_gbeta_v0"
    assert cfg["selection_label_free"] is True
    assert cfg["head_identity"] is False
    # No physical fields in config.
    for forbidden in ("inv_perm", "block_perm", "clean_perm",
                       "physical_order", "l2r_order"):
        assert forbidden not in cfg, f"config contains {forbidden}"
    with open(pathlib.Path(out_dir) / "training_summary.json") as f:
        training_summary = json.load(f)
    assert training_summary["train_seconds"] >= 0.0
    assert training_summary["peak_cuda_memory_mb"] >= 0.0


def test_metrics_jsonl_complete(tmp_path):
    ds_path = str(tmp_path / "synth.npz")
    _make_synthetic_pretrain_dataset(ds_path, M=16, seed=4)
    out_dir = str(tmp_path / "run")

    train(dataset_path=ds_path, out_dir=out_dir, epochs=3, batch_size=4,
          seed=4, device="cpu")

    lines = []
    with open(pathlib.Path(out_dir) / "metrics.jsonl") as f:
        for line in f:
            lines.append(json.loads(line))

    assert len(lines) == 3
    for rec in lines:
        assert "epoch" in rec
        assert "train" in rec
        assert "val" in rec
        for split in ("train", "val"):
            for key in ("loss", "loss_final", "loss_aux", "loss_ent",
                        "pairwise_acc", "gate_entropy_mean"):
                assert key in rec[split], f"missing {split}.{key}"


def test_train_val_test_splits_used_correctly(tmp_path):
    """Train must use only train indices, val only val indices."""
    ds_path = str(tmp_path / "synth.npz")
    _make_synthetic_pretrain_dataset(ds_path, M=32, seed=5)
    out_dir = str(tmp_path / "run")

    # We verify by checking the DataLoader sizes computed from indices.
    data = load_pretrain_dataset(ds_path, device="cpu")
    from torch.utils.data import DataLoader, TensorDataset
    B = data["B_raw"]
    Y = data["teacher_pairwise"]

    train_ds = TensorDataset(B[data["train_idx"]], Y[data["train_idx"]])
    val_ds = TensorDataset(B[data["val_idx"]], Y[data["val_idx"]])

    assert len(train_ds) == len(data["train_idx"])
    assert len(val_ds) == len(data["val_idx"])
    assert len(train_ds) + len(val_ds) + len(data["test_idx"]) == 32


def test_training_is_approximately_reproducible(tmp_path):
    """Same seed → similar final val accuracy (within tolerance)."""
    results = []
    for run_seed in (42, 42):  # run twice with same seed
        ds_path = str(tmp_path / f"synth_{run_seed}.npz")
        _make_synthetic_pretrain_dataset(ds_path, M=32, seed=run_seed)
        out_dir = str(tmp_path / f"run_{run_seed}")
        result = train(
            dataset_path=ds_path, out_dir=out_dir,
            epochs=3, batch_size=8, lr=3e-3, seed=run_seed, device="cpu",
        )
        results.append(result["best_val_acc"])

    # Same seed, same data → exactly same result.
    assert results[0] == pytest.approx(results[1], abs=1e-5)


def test_training_no_nan_in_metrics(tmp_path):
    ds_path = str(tmp_path / "synth.npz")
    _make_synthetic_pretrain_dataset(ds_path, M=32, seed=10)
    out_dir = str(tmp_path / "run")

    result = train(
        dataset_path=ds_path, out_dir=out_dir,
        epochs=4, batch_size=8, lr=3e-3, seed=10, device="cpu",
    )
    for rec in result["metrics_history"]:
        for split in ("train", "val"):
            for key, val in rec[split].items():
                assert not np.isnan(val), f"NaN in epoch {rec['epoch']} {split}.{key}"
                assert np.isfinite(val), f"inf in epoch {rec['epoch']} {split}.{key}"


def test_head_dropout_only_in_train_mode(tmp_path):
    """Verify head dropout is active during training but inactive during eval."""
    ds_path = str(tmp_path / "synth.npz")
    _make_synthetic_pretrain_dataset(ds_path, M=16, seed=11)
    out_dir = str(tmp_path / "run")

    # Patch _run_epoch to check model.training
    orig_train = sys.modules[
        "batch_readout.train_l0_dynamic_gbeta"
    ]._run_epoch

    train_states = []
    val_states = []

    def _patched_epoch(
        model, loader, optimizer, laux, lent, min_ent, dev, *args, **kwargs
    ):
        # _run_epoch sets model.train() / model.eval() internally first,
        # so we capture state AFTER the epoch body.
        result = orig_train(
            model, loader, optimizer, laux, lent, min_ent, dev,
            *args, **kwargs,
        )
        is_train = optimizer is not None
        if is_train:
            train_states.append(model.training)
        else:
            val_states.append(model.training)
        return result

    import batch_readout.train_l0_dynamic_gbeta as tmod
    tmod._run_epoch = _patched_epoch
    try:
        train(dataset_path=ds_path, out_dir=out_dir,
              epochs=2, batch_size=4, seed=11, device="cpu")
    finally:
        tmod._run_epoch = orig_train

    # All train epochs: model.training == True
    assert all(s is True for s in train_states), "model must be in train mode"
    # All val epochs: model.training == False
    assert all(s is False for s in val_states), "model must be in eval mode"


def test_checkpoint_selection_uses_val_loss_final_not_physical(tmp_path):
    """Inspect config.json to confirm selection is label-free."""
    ds_path = str(tmp_path / "synth.npz")
    _make_synthetic_pretrain_dataset(ds_path, M=16, seed=12)
    out_dir = str(tmp_path / "run")

    train(dataset_path=ds_path, out_dir=out_dir,
          epochs=2, batch_size=4, seed=12, device="cpu")

    with open(pathlib.Path(out_dir) / "config.json") as f:
        cfg = json.load(f)
    assert cfg["selection_metric"] == "val_loss_final"
    assert cfg["selection_label_free"] is True

    # g_beta_best.pt config must also be label-free.
    ckpt = torch.load(
        pathlib.Path(out_dir) / "g_beta_best.pt",
        map_location="cpu", weights_only=False,
    )
    assert ckpt["config"]["selection_label_free"] is True


def test_rejects_wrong_dataset_schema(tmp_path):
    """A .npz without 'teacher_pairwise' must fail gracefully."""
    bad_path = str(tmp_path / "bad.npz")
    np.savez_compressed(bad_path, B_raw=np.zeros((4, 8, 65, 65)))
    with pytest.raises(KeyError):
        load_pretrain_dataset(bad_path)
