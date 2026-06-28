# tests/test_pss_predictor.py
import pathlib, sys
import numpy as np
import pytest
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import (
    slot_only_r2, valid_edge_mask, synthetic_content_invariant, synthetic_content_randomized)

def test_predictor_high_on_invariant_low_on_random():
    m = valid_edge_mask(65)
    invariant = synthetic_content_invariant(8)
    randomized = synthetic_content_randomized(40)
    invariant_raw = slot_only_r2(invariant, m, normalize=False)
    invariant_norm = slot_only_r2(invariant, m, normalize=True)
    randomized_raw = slot_only_r2(randomized, m, normalize=False)
    randomized_norm = slot_only_r2(randomized, m, normalize=True)
    assert invariant_raw > 0.95 and invariant_norm > 0.95
    assert randomized_raw < 0.1
    # Total-entry normalized R² calibrates near 0.70; this is not by itself
    # strong evidence for a fixed map because the matched random null is high.
    assert 0.6 < randomized_norm < 0.8


@pytest.mark.parametrize("n_train", [0, -1, 4, 5, 1.5, True])
def test_predictor_rejects_invalid_split(n_train):
    m = valid_edge_mask(65)
    with pytest.raises(ValueError, match="n_train"):
        slot_only_r2(synthetic_content_invariant(4), m, n_train=n_train)


def test_predictor_requires_at_least_two_matrices():
    m = valid_edge_mask(65)
    with pytest.raises(ValueError, match="at least 2"):
        slot_only_r2(synthetic_content_invariant(1), m)


def test_predictor_accepts_valid_explicit_split():
    m = valid_edge_mask(65)
    result = slot_only_r2(synthetic_content_randomized(6), m, n_train=2)
    assert np.isfinite(result)
