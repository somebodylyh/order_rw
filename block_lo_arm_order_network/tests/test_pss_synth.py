# tests/test_pss_synth.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import (
    synthetic_content_invariant, synthetic_content_randomized, valid_edge_mask)
from none_separated_block_graph import rollout_by_method, discovery_metrics

def test_synthetic_baselines_shapes():
    inv = synthetic_content_invariant(5)
    rnd = synthetic_content_randomized(5)
    assert len(inv) == 5 and len(rnd) == 5
    assert np.allclose(inv[0], inv[4])                 # content-invariant: identical
    assert not np.allclose(rnd[0], rnd[4])             # randomized: differ

def test_order_lookup_tautology():
    # any B that rolls out to tau_physical=1 has sigma_model == arange in physical frame
    # -> order-level lookup similarity is tautologically 1; documents why E2-order was removed
    from analyses.physical_signal_source import synthetic_ascending_B
    B = synthetic_ascending_B()
    sigma = rollout_by_method(B, "C-D+L")
    assert discovery_metrics(sigma)["tau_vs_l2r"] == 1.0
    assert list(sigma) == list(range(len(sigma)))      # equals the lookup order exactly
