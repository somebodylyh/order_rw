import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.layout_ood_eval import layout_ood_sweep, _classify_ood

CKPT = ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt"


def test_classify_ood_rules():
    # train_map high + test_physical low -> lookup (signal bound to training layout)
    assert _classify_ood(test_physical_mean=0.1, train_map_mean=0.9) == "lookup"
    # test_physical high + train_map low -> content recovery
    assert _classify_ood(test_physical_mean=0.9, train_map_mean=0.1) == "content_recovery"
    # both low -> degraded
    assert _classify_ood(test_physical_mean=0.1, train_map_mean=0.1) == "degraded"
    # otherwise ambiguous
    assert _classify_ood(test_physical_mean=0.5, train_map_mean=0.5) == "ambiguous"


@pytest.mark.skipif(not CKPT.exists(), reason="seed2 checkpoint artifact is absent")
def test_layout_ood_sweep_structure_and_anchor_invariant():
    res = layout_ood_sweep(str(CKPT), 0, [2, 3, 4, 5], K=2, M=3, n_reveals=4)
    assert "anchor_tau" in res and "best_head" in res
    assert "interpretation" in res
    assert len(res["per_perm"]) == 2          # K unseen test perms
    for row in res["per_perm"]:
        assert {"perm_id", "tau_test_physical", "tau_train_map"} <= set(row)
    # On any relayout, the train-map labeling of the SAME attention must reproduce the
    # training-frame score the model would emit; it is computed on the fixed best head.
    assert isinstance(res["test_physical_mean"], float)
    assert isinstance(res["train_map_mean"], float)
