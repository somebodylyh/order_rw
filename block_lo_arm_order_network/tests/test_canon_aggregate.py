"""Multi-sampling-seed aggregation + L0 carrier identification."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.canonical_reanalysis import scan_aggregated  # noqa: E402


def test_aggregate_seed2_10k_carrier_is_L0():
    agg = scan_aggregated(str(ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt"), K=2)
    assert agg["best_tau"] > 0.9
    assert agg["best_head"][0] == 0  # (layer, head); layer 0
    assert len(agg["strong_pass_heads"]) > 0
