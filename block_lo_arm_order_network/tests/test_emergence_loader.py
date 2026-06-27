"""Emergence tau-trajectory loader + winner determination."""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.emergence_characterization import load_tau_trajectory, winner  # noqa: E402


def test_load_trajectory_shape_and_order():
    t = load_tau_trajectory(2, root=str(ROOT / "runs/handoff_overnight"))
    assert t["steps"][0] == 0 and t["steps"][-1] == 10000
    assert t["steps"].shape == (51,)
    assert t["tau"].shape == (51, 4, 8)
    assert np.all(np.diff(t["steps"]) > 0)  # sorted


def test_winner_seed2_is_L1_strong():
    t = load_tau_trajectory(2, root=str(ROOT / "runs/handoff_overnight"))
    w = winner(t, 2)
    assert w["winning_layer"] == 1
    assert w["tier"] == "strong"
    assert set(w["winner_heads"]) == {0, 3, 5, 7}


def test_winner_seed42_is_L0_weak():
    t = load_tau_trajectory(42, root=str(ROOT / "runs/handoff_overnight"))
    w = winner(t, 42)
    assert w["winning_layer"] == 0
    assert w["tier"] == "weak"  # no strong head anywhere
    assert len(w["winner_heads"]) >= 1
