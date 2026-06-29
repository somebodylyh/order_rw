"""Tests for objective-aware L0DynamicGBeta evaluation."""

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))
sys.path.insert(0, str(ROOT / "analyses"))

from batch_readout.train_l0_dynamic_gbeta import train  # noqa: E402
from eval_l0_dynamic_gbeta import evaluate_checkpoint  # noqa: E402
from tests.test_train_l0_dynamic_gbeta import (  # noqa: E402
    _make_synthetic_pretrain_dataset,
)


@pytest.mark.parametrize("loss_type", [
    "listmle", "pairwise_bce", "rank_kl",
])
def test_evaluation_reports_common_order_metrics(tmp_path, loss_type):
    dataset = str(tmp_path / "dataset.npz")
    _make_synthetic_pretrain_dataset(dataset, M=24, seed=30)
    train_dir = str(tmp_path / "train")
    train(
        dataset_path=dataset,
        out_dir=train_dir,
        epochs=1,
        batch_size=8,
        lr=3e-3,
        seed=30,
        device="cpu",
        loss_type=loss_type,
    )

    summary = evaluate_checkpoint(
        dataset_path=dataset,
        ckpt_path=str(pathlib.Path(train_dir) / "g_beta_best.pt"),
        device="cpu",
        out_dir=str(tmp_path / "eval"),
    )

    assert summary["loss_type"] == loss_type
    for split in ("val", "test"):
        normal = summary["results"][split]["normal"]
        for key in (
            "primary_loss",
            "kendall_tau",
            "pairwise_acc",
            "prefix8",
            "prefix16",
            "gate_entropy_mean",
        ):
            assert key in normal


def test_legacy_checkpoint_defaults_to_pairwise_bce(tmp_path):
    dataset = str(tmp_path / "dataset.npz")
    _make_synthetic_pretrain_dataset(dataset, M=16, seed=31)
    train_dir = str(tmp_path / "train")
    train(
        dataset_path=dataset,
        out_dir=train_dir,
        epochs=1,
        batch_size=8,
        seed=31,
        device="cpu",
    )
    ckpt_path = pathlib.Path(train_dir) / "g_beta_best.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint["config"].pop("loss_type", None)
    legacy_path = tmp_path / "legacy.pt"
    torch.save(checkpoint, legacy_path)

    summary = evaluate_checkpoint(
        dataset_path=dataset,
        ckpt_path=str(legacy_path),
        device="cpu",
        out_dir=str(tmp_path / "eval"),
    )
    assert summary["loss_type"] == "pairwise_bce"
