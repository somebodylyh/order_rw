"""Pillar-3 frozen carrier-set config must match the step-10000 tau table.

These sets are the load-bearing intervention targets for path patching; they are
frozen (no runtime selection) and must reproduce exactly from the real tau_table
under the spec thresholds (strong |tau|>=0.95, weak 0.60<=|tau|<0.95, null =
3 smallest-|tau| heads excluding any weak/strong carrier).
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # import analyses.* as namespace package

from analyses.handoff_carrier_config import (  # noqa: E402
    CARRIER_SETS,
    load_carrier_sets,
    verify_against_tau_table,
)


def test_frozen_sets_match_spec():
    s2 = load_carrier_sets(2)
    assert s2[1]["strong"] == [0, 3, 5, 7]      # seed2 L1 strong
    assert s2[0]["strong"] == []                 # no strong L0 anywhere
    s123 = load_carrier_sets(123)
    assert s123[1]["strong"] == [0, 5, 6, 7]     # seed123 L1 strong (H0=0.96>=0.95)
    s42 = load_carrier_sets(42)
    assert all(s42[L]["strong"] == [] for L in range(4))  # seed42 no strong head
    # null disjoint from weak/strong in every (seed, layer)
    for seed in (2, 42, 123):
        for _L, tiers in load_carrier_sets(seed).items():
            assert set(tiers["null"]).isdisjoint(tiers["weak"])
            assert set(tiers["null"]).isdisjoint(tiers["strong"])


def test_verify_against_real_tau_table():
    for seed in (2, 42, 123):
        npz = (ROOT / "runs/handoff_overnight" / f"seed{seed}"
               / "attention_trajectory/raw/step_010000/tau_table.npz")
        verify_against_tau_table(seed, str(npz))
