import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import sample_scaffold, block_b_features

CKPT = str(ROOT / "runs/handoff_overnight/seed123/ckpt_step10000.pt")


def test_scaffold_shapes_and_sigma_is_permutation():
    sc = sample_scaffold(CKPT, M=3, n_reveals=4)
    assert len(sc["B"]) == 3 and len(sc["sigma_B"]) == 3
    assert sc["B"][0].shape == (65, 65)
    sig = sc["sigma_B"][0]
    assert sorted(sig.tolist()) == list(range(64))      # a permutation of 64 blocks
    feat = block_b_features(sc["B"][0])
    assert feat.shape == (64, 130)
