"""NR-1 Task 11: smoke tests for the frozen-theta NLL diagnostic module.

Algorithmic correctness lives in the spec definition itself (mean per-token
NLL under a given physical-block reveal order). Here we verify only the
operational properties:

  - module imports and exposes compute_frozen_nll_gap
  - asking for a non-existent g_beta checkpoint raises a clear error
    (post-selection-only contract)
"""
import sys
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_module_exposes_entrypoint():
    import importlib
    mod = importlib.import_module("neural_readout.eval_frozen_nll")
    assert hasattr(mod, "compute_frozen_nll_gap"), \
        "neural_readout.eval_frozen_nll.compute_frozen_nll_gap entrypoint missing"
    assert callable(mod.compute_frozen_nll_gap)


def test_missing_g_beta_raises_clear_error(tmp_path):
    from neural_readout.eval_frozen_nll import compute_frozen_nll_gap
    fake_ckpt = tmp_path / "ckpt_step5000.pt"
    fake_ckpt.write_text("not a real ckpt")
    fake_g_beta = tmp_path / "g_beta_best.pt"  # deliberately not created
    fake_dataset = tmp_path / "ds.npz"
    fake_dataset.write_text("not a real dataset either")
    with pytest.raises(FileNotFoundError, match="post-selection diagnostic"):
        compute_frozen_nll_gap(str(fake_ckpt), str(fake_g_beta), str(fake_dataset))
