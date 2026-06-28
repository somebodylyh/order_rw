# tests/test_pss_variance.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import (
    cross_text_variance, pairwise_similarity, within_text_noise_floor,
    content_variance, valid_edge_mask,
    synthetic_content_invariant, synthetic_content_randomized)

def test_variance_and_similarity_calibration():
    m = valid_edge_mask(65)
    inv = synthetic_content_invariant(8); rnd = synthetic_content_randomized(8)
    assert cross_text_variance(inv, m) < 1e-9               # invariant -> ~0
    assert cross_text_variance(rnd, m) > cross_text_variance(inv, m)
    assert pairwise_similarity(inv, m) > 0.99               # invariant -> ~1
    assert pairwise_similarity(rnd, m) < pairwise_similarity(inv, m)

def test_within_text_floor_and_content_variance():
    m = valid_edge_mask(65)
    # identical halves -> zero floor; identical B_list -> zero content variance
    inv = synthetic_content_invariant(6)
    assert within_text_noise_floor(inv, inv, m) < 1e-9
    assert content_variance(inv, inv, inv, m) < 1e-9
    # if cross-text variance is entirely sampling noise (floor == cross var),
    # content_variance clamps to 0
    rnd = synthetic_content_randomized(6)
    cv = content_variance(rnd, rnd, rnd, m)               # floor uses same-as-cross here
    assert cv >= 0.0
