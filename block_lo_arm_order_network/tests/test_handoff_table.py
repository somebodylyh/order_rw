"""Multiplicity-collapse readout (count of strong heads before/after) and the
Stage-1 table writer schema.
"""
import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.path_patch_handoff import (  # noqa: E402
    multiplicity_collapse,
    consensus_tau,
    write_stage1_table,
)


def test_multiplicity_collapse_counts():
    before = np.zeros((4, 8))
    after = np.zeros((4, 8))
    before[1, [0, 3, 5, 7]] = 1.0   # 4 strong before
    after[1, [0]] = 0.97            # 1 strong after
    mc = multiplicity_collapse(before, after, layer=1)
    assert mc["n_strong_before"] == 4
    assert mc["n_strong_after"] == 1


def test_consensus_tau_is_carrier_set_mean():
    tau = np.zeros((4, 8))
    tau[1, [0, 3, 5, 7]] = [1.0, 0.8, 0.6, 0.4]
    assert consensus_tau(tau, layer=1, carrier_heads=[0, 3, 5, 7]) == 0.7
    assert np.isnan(consensus_tau(tau, layer=1, carrier_heads=[]))


def test_table_writer_has_required_columns(tmp_path):
    p = tmp_path / "s1.csv"
    write_stage1_table([{
        "seed": 2, "stage": "1a", "intervention": "full", "target_layer": 1,
        "target_heads": "[0,3,5,7]", "mean_tau_before": 1.0, "mean_tau_after": 0.4,
        "n_strong_before": 4, "n_strong_after": 0, "delta_global_tau": -0.3,
    }], str(p))
    rows = list(csv.DictReader(open(p)))
    assert rows[0]["n_strong_after"] == "0"
    assert "delta_global_tau" in rows[0]
