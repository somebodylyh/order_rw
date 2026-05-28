"""BR-1 Task 9 (selection guard): train_offline must NEVER reach AR-NLL.

If any of the forbidden symbols below appears in batch_readout.train_offline's
source, the test fails. Mirrors neural_readout's selection-policy lock-in
(NR-1 spec §1 / §10) for the BR-1 line.
"""
import importlib
import inspect
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


FORBIDDEN_IN_TRAIN_OFFLINE = [
    "eval_frozen_phase2",        # the BR-1 frozen-NLL wrapper
    "compute_frozen_nll_gap",    # NR-1's NLL helper
    "neural_readout.eval_frozen_nll",
    "forward_fn",                # AOGPT AR forward; only AR-NLL callers use it
]


def test_train_offline_does_not_reference_nll_paths():
    mod = importlib.import_module("batch_readout.train_offline")
    src = inspect.getsource(mod)
    for f in FORBIDDEN_IN_TRAIN_OFFLINE:
        assert f not in src, (
            f"selection-policy guard: {f!r} must NOT appear in batch_readout.train_offline "
            "(see NR-1 spec §1 / §10 — AR-NLL is post-selection diagnostic only)"
        )


def test_train_offline_imports_only_matching_metrics():
    """Defensive belt-and-braces: the eval_split helper must use exactly the
    matching-metrics set, not anything from the frozen-NLL stack."""
    mod = importlib.import_module("batch_readout.train_offline")
    src = inspect.getsource(mod)
    assert "from batch_readout.eval_metrics import" in src
    assert "from batch_readout.eval_frozen_phase2" not in src
