"""A3: schedule/loss overlay + correlational classification."""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.emergence_characterization import load_eval_curve, schedule_loss_overlay  # noqa: E402


def test_load_eval_curve_columns():
    ec = load_eval_curve(2, root=str(ROOT / "runs/handoff_overnight"))
    assert ec["step"][0] == 0 and len(ec["step"]) == 51
    assert "lr" in ec and "val_model_order" in ec


def test_classification_intrinsic_when_lr_flat_and_loss_smooth():
    ec = {"step": np.arange(6) * 200, "lr": np.full(6, 1e-3),
          "val_train_objective": np.linspace(6, 5, 6),
          "val_model_order": np.linspace(6, 5, 6)}
    ev = {"onset": 200, "midpoint": 600, "completion": 1000}
    out = schedule_loss_overlay(ec, ev)
    assert out["classification"] == "intrinsic (no alignment)"
