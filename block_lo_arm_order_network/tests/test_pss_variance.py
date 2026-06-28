# tests/test_pss_variance.py
import pathlib, sys
import numpy as np
import pytest
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


def test_centered_noise_floor_matches_full_estimator_and_removes_common_half_bias():
    rng = np.random.default_rng(123)
    M, n = 20_000, 3
    m = valid_edge_mask(n)
    common_half_shift = np.zeros((n, n))
    common_half_shift[m] = np.linspace(-50.0, 50.0, num=m.sum())
    half_a, half_b, full = [], [], []
    for _ in range(M):
        noise_a = np.zeros((n, n)); noise_b = np.zeros((n, n))
        noise_a[m] = rng.normal(size=m.sum())
        noise_b[m] = rng.normal(size=m.sum())
        a = common_half_shift + noise_a
        b = -common_half_shift + noise_b
        half_a.append(a); half_b.append(b); full.append(0.5 * (a + b))

    floor = within_text_noise_floor(half_a, half_b, m, normalize=False)
    full_variance = cross_text_variance(full, m, normalize=False)
    assert floor == pytest.approx(full_variance, rel=0.03)
    assert content_variance(full, half_a, half_b, m, normalize=False) < 0.03


def test_content_variance_retains_real_between_text_signal():
    rng = np.random.default_rng(456)
    M, n = 8_000, 3
    m = valid_edge_mask(n)
    half_a, half_b, full = [], [], []
    for _ in range(M):
        content = np.zeros((n, n)); a_noise = np.zeros((n, n)); b_noise = np.zeros((n, n))
        content[m] = rng.normal(scale=2.0, size=m.sum())
        a_noise[m] = rng.normal(scale=0.2, size=m.sum())
        b_noise[m] = rng.normal(scale=0.2, size=m.sum())
        a = content + a_noise; b = content + b_noise
        half_a.append(a); half_b.append(b); full.append(0.5 * (a + b))

    assert content_variance(full, half_a, half_b, m, normalize=False) > 3.5


@pytest.mark.parametrize(
    ("B_list", "mask", "message"),
    [
        ([], np.ones((2, 2), dtype=bool), "B_list must not be empty"),
        ([np.zeros((2, 2))], np.zeros((2, 2), dtype=bool), "mask must select at least one value"),
        ([np.zeros((2, 2))], np.ones(2, dtype=bool), "mask must be a 2D boolean array"),
        ([np.zeros((2, 2))], np.ones((2, 2), dtype=int), "mask must be a 2D boolean array"),
        ([np.zeros((3, 3))], np.ones((2, 2), dtype=bool), r"B_list\[0\] shape"),
        ([np.array([[np.nan, 0.0], [0.0, 0.0]])], np.ones((2, 2), dtype=bool),
         "non-finite values"),
    ],
)
def test_stack_input_contracts(B_list, mask, message):
    with pytest.raises(ValueError, match=message):
        cross_text_variance(B_list, mask, normalize=False)


def test_stack_rejects_mixed_matrix_shapes():
    m = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match=r"B_list\[1\] shape"):
        cross_text_variance([np.zeros((2, 2)), np.zeros((1, 4))], m, normalize=False)


def test_noise_floor_rejects_unequal_half_lists():
    m = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="equal length"):
        within_text_noise_floor([np.zeros((2, 2))], [], m, normalize=False)
