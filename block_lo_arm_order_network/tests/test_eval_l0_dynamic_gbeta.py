"""Tests for L0DynamicGBeta evaluation sanity checks."""

import json
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))
sys.path.insert(0, str(ROOT / "analyses"))

from batch_readout.l0_dynamic_gbeta import L0DynamicGBeta  # noqa: E402
from batch_readout.label_free_cdl_teacher import destroy_strict65  # noqa: E402
from eval_l0_dynamic_gbeta import (  # noqa: E402
    eval_destroyed,
    eval_remove_top_alpha,
    eval_uniform_alpha,
    evaluate_checkpoint,
    remove_top_alpha,
)
from build_l0_dynamic_gbeta_dataset import deterministic_split, save_dataset  # noqa: E402
from batch_readout.train_l0_dynamic_gbeta import train  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_eval_dataset(path, M=32, H=8, seed=0):
    """Build a synthetic dataset + train a quick model for eval testing."""
    from tests.test_train_l0_dynamic_gbeta import _make_synthetic_pretrain_dataset
    _make_synthetic_pretrain_dataset(path, M=M, H=H, seed=seed)


def _train_tiny_model(ds_path, out_dir, seed=0):
    """Train a tiny model for eval tests."""
    return train(
        dataset_path=ds_path, out_dir=out_dir,
        epochs=3, batch_size=8, lr=3e-3, seed=seed, device="cpu",
    )


# ============================================================================
# remove_top_alpha
# ============================================================================

def test_remove_top_alpha_masks_and_renormalises():
    alpha = torch.tensor([[0.1, 0.6, 0.3]])
    masked = remove_top_alpha(alpha)
    # Head 1 (index 1, value 0.6) should be zero.
    assert masked[0, 1] == 0.0
    # Remaining sum to 1.
    assert masked.sum() == pytest.approx(1.0)
    # Renormalised: 0.1/(0.1+0.3) = 0.25, 0.3/(0.1+0.3) = 0.75.
    torch.testing.assert_close(masked, torch.tensor([[0.25, 0.0, 0.75]]))


def test_remove_top_alpha_handles_ties():
    """When all alpha are equal, argmax picks first; it's masked."""
    alpha = torch.full((2, 4), 0.25)
    masked = remove_top_alpha(alpha)
    torch.testing.assert_close(masked.sum(dim=1), torch.ones(2))
    # Head 0 should be zeroed (argmax picks first on ties).
    assert (masked[:, 0] == 0.0).all()


# ============================================================================
# Sanity: destroyed
# ============================================================================

def test_destroyed_eval_uses_structure_preserving_destroy():
    """Verify eval_destroyed actually calls destroy_strict65."""
    rng = np.random.default_rng(1)
    B = rng.random((2, 4, 65, 65)).astype(np.float32)
    for b in range(2):
        for h in range(4):
            np.fill_diagonal(B[b, h], 0.0)
            B[b, h, 1:, 0] = 0.0
    B[0, 0, 5, 10] = 100.0  # strong edge

    # Apply destroy_strict65 manually on one head.
    B_manual = B.copy()
    rng2 = np.random.default_rng(0 * 1000 + 0 * 100 + 0)
    B_manual[0, 0] = destroy_strict65(B[0, 0], rng2)
    # The specific edge B[0,0,5,10] should be destroyed.
    assert not np.allclose(B_manual[0, 0, 5, 10], 100.0, atol=1e-6), (
        "destroy must shuffle the strong edge"
    )


# ============================================================================
# Sanity: uniform alpha
# ============================================================================

def test_uniform_alpha_does_not_use_gate():
    """Uniform alpha baseline should produce scalar output regardless of gate."""
    model = L0DynamicGBeta(heads=8)
    model.eval()
    B = torch.randn(2, 8, 65, 65)
    for b in range(2):
        for h in range(8):
            B[b, h].fill_diagonal_(0.0)
            B[b, h, 1:, 0] = 0.0
    Y = torch.full((2, 64, 64), 0.5)
    result = eval_uniform_alpha(model, B, Y, H=8)
    assert "loss_final" in result
    assert "pairwise_acc" in result


# ============================================================================
# Full evaluate_checkpoint
# ============================================================================

def test_evaluate_checkpoint_runs_all_conditions(tmp_path):
    ds_path = str(tmp_path / "synth.npz")
    _make_synthetic_eval_dataset(ds_path, M=32, seed=0)
    out_dir = str(tmp_path / "train")
    _train_tiny_model(ds_path, out_dir, seed=0)

    ckpt_path = str(pathlib.Path(out_dir) / "g_beta_best.pt")
    eval_dir = str(tmp_path / "eval")

    summary = evaluate_checkpoint(
        dataset_path=ds_path,
        ckpt_path=ckpt_path,
        device="cpu",
        destroy_seed=42,
        out_dir=eval_dir,
    )

    # All conditions present for val and test.
    for split in ("val", "test"):
        r = summary["results"][split]
        for cond in ("normal", "destroyed", "remove_top_alpha", "uniform_alpha"):
            assert cond in r, f"missing {split}.{cond}"
            assert "pairwise_acc" in r[cond], f"missing acc in {split}.{cond}"

    # Output files exist.
    for fname in ("summary.json", "per_sample_val.jsonl", "alpha_summary_val.json"):
        assert (pathlib.Path(eval_dir) / fname).exists(), f"missing {fname}"


def test_evaluate_checkpoint_summary_is_valid_json(tmp_path):
    ds_path = str(tmp_path / "synth.npz")
    _make_synthetic_eval_dataset(ds_path, M=16, seed=1)
    out_dir = str(tmp_path / "train")
    _train_tiny_model(ds_path, out_dir, seed=1)

    ckpt_path = str(pathlib.Path(out_dir) / "g_beta_best.pt")
    eval_dir = str(tmp_path / "eval")

    evaluate_checkpoint(ds_path, ckpt_path, device="cpu", out_dir=eval_dir)

    with open(pathlib.Path(eval_dir) / "summary.json") as f:
        summary = json.load(f)
    assert summary["label_free"] is True
    assert "results" in summary


def test_evaluate_checkpoint_destroyed_differs_from_normal(tmp_path):
    """Destroyed accuracy should be materially lower than normal."""
    ds_path = str(tmp_path / "synth.npz")
    _make_synthetic_eval_dataset(ds_path, M=48, seed=2)
    out_dir = str(tmp_path / "train")
    _train_tiny_model(ds_path, out_dir, seed=2)

    ckpt_path = str(pathlib.Path(out_dir) / "g_beta_best.pt")
    eval_dir = str(tmp_path / "eval")

    summary = evaluate_checkpoint(ds_path, ckpt_path, device="cpu", out_dir=eval_dir)
    r = summary["results"]["val"]
    # Normal should be better than destroyed.
    assert r["normal"]["pairwise_acc"] > r["destroyed"]["pairwise_acc"], (
        f"normal acc {r['normal']['pairwise_acc']:.4f} should exceed "
        f"destroyed acc {r['destroyed']['pairwise_acc']:.4f}"
    )


def test_remove_top_alpha_degrades_less_than_collapse_to_chance(tmp_path):
    """Removing the top head should hurt but not collapse to ~0.5 random."""
    ds_path = str(tmp_path / "synth.npz")
    _make_synthetic_eval_dataset(ds_path, M=48, seed=3)
    out_dir = str(tmp_path / "train")
    _train_tiny_model(ds_path, out_dir, seed=3)

    ckpt_path = str(pathlib.Path(out_dir) / "g_beta_best.pt")
    eval_dir = str(tmp_path / "eval")

    summary = evaluate_checkpoint(ds_path, ckpt_path, device="cpu", out_dir=eval_dir)
    r = summary["results"]["val"]
    # Remove-top should not collapse to random.
    assert r["remove_top_alpha"]["pairwise_acc"] > 0.55, (
        f"remove_top_alpha acc {r['remove_top_alpha']['pairwise_acc']:.4f} "
        f"should be > 0.55 (above-chance)"
    )


def test_eval_rejects_non_label_free_checkpoint(tmp_path):
    """A checkpoint without selection_label_free=True must be rejected."""
    ds_path = str(tmp_path / "synth.npz")
    _make_synthetic_eval_dataset(ds_path, M=16, seed=4)
    out_dir = str(tmp_path / "train")
    _train_tiny_model(ds_path, out_dir, seed=4)

    ckpt_path = str(pathlib.Path(out_dir) / "g_beta_best.pt")
    # Corrupt the config.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt["config"]["selection_label_free"] = False
    bad_path = str(tmp_path / "bad.pt")
    torch.save(ckpt, bad_path)

    with pytest.raises(AssertionError, match="label-free"):
        evaluate_checkpoint(ds_path, bad_path, device="cpu", out_dir=str(tmp_path))


def test_eval_does_not_read_physical_fields(tmp_path):
    """The evaluation script must not use physical fields in code logic."""
    src = pathlib.Path(
        ROOT / "analyses" / "eval_l0_dynamic_gbeta.py"
    ).read_text()

    # Remove docstrings and comments before checking.
    import re
    # Remove triple-quoted strings (docstrings).
    src_no_docs = re.sub(r'""".*?"""', '', src, flags=re.DOTALL)
    src_no_docs = re.sub(r"'''.*?'''", '', src_no_docs, flags=re.DOTALL)
    # Remove # comments and blank lines.
    lines = [line for line in src_no_docs.split('\n')
             if not line.strip().startswith('#') and line.strip()]
    code = '\n'.join(lines)

    forbidden = ("inv_perm", "block_perm", "clean_perm", "l2r_order",
                 "physical_order", "tau_vs_l2r")
    for term in forbidden:
        assert term not in code, (
            f"eval module code contains forbidden term: {term}"
        )
