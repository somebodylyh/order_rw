"""BR-1 Task 13: wrapper API contract.

The full NLL pass is heavy (loads a real AOGPT ckpt + wikitext) so it is
exercised via scripts/run_br1_phase2_frozen.sh, not pytest. This test only
locks down the compute_arms() signature used by that runner.
"""
import importlib
import inspect
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def test_module_exposes_compute_arms():
    mod = importlib.import_module("batch_readout.eval_frozen_phase2")
    assert hasattr(mod, "compute_arms")
    sig = inspect.signature(mod.compute_arms)
    expected = {"ckpt_path", "g_beta_path", "dataset_path", "arms", "tau", "seed", "device"}
    assert set(sig.parameters.keys()) == expected
