"""Part-1 step-0 tau table under full/zero-wpe/zero-wtpe/zero-both."""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.position_prior_decomp import part1_arms  # noqa: E402


def test_part1_arms_shapes_and_difference():
    arms = part1_arms(2, root=str(ROOT / "runs/handoff_overnight"), n_batches=2)
    for k in ("full", "zero_wpe", "zero_wtpe", "zero_both"):
        assert arms[k].shape == (4, 8)
    assert not np.allclose(arms["full"], arms["zero_both"])
