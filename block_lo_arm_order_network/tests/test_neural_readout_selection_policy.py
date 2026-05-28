"""NR-1 Task 10: selection-policy enforcement test (spec §1, §8).

NR-1 forbids any use of frozen-theta NLL as a training signal, label, or hard
gate. The training script (`neural_readout.train_nr1`) must NEVER import
`neural_readout.eval_frozen_nll`. This test fails on purpose if a future
developer adds such an import — the assertion is the contract that keeps
NR-1's supervision boundary intact.

If you are reading this because the test failed: remove the import. NLL is
NR-7 only. See spec §1 supervision boundary.
"""
import sys
import pathlib
import importlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))


def _purge_neural_readout_modules():
    for mod in list(sys.modules):
        if mod.startswith("neural_readout"):
            del sys.modules[mod]


def test_train_nr1_does_not_import_frozen_nll():
    """Selection policy: train_nr1 must not transitively import eval_frozen_nll."""
    _purge_neural_readout_modules()
    importlib.import_module("neural_readout.train_nr1")
    leaked = [k for k in sys.modules if k.startswith("neural_readout.eval_frozen_nll")]
    assert not leaked, (
        "neural_readout.train_nr1 must NOT import neural_readout.eval_frozen_nll. "
        "NR-1 spec §1 / §8 forbids NLL as a training signal or selection criterion."
    )


def test_train_nr1_exposes_entrypoint():
    _purge_neural_readout_modules()
    mod = importlib.import_module("neural_readout.train_nr1")
    assert hasattr(mod, "train_nr1"), "neural_readout.train_nr1.train_nr1 entrypoint missing"
    assert callable(mod.train_nr1)
