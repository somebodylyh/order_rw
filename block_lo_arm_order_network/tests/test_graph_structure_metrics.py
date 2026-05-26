import numpy as np, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from graph_structure_metrics import (sharpness, row_entropy, topk_mass,
                                      spectral_gap, structure_vs_null)


def test_sharpness_higher_for_peaked_rows():
    N = 8
    peaked = np.zeros((N, N))
    for i in range(N):
        j = (i + 1) % N
        peaked[i, j] = 1.0
    flat = np.ones((N, N)); np.fill_diagonal(flat, 0.0)
    assert sharpness(peaked) > sharpness(flat)


def test_row_entropy_max_for_uniform():
    N = 8
    flat = np.ones((N, N)); np.fill_diagonal(flat, 0.0)
    assert np.isclose(row_entropy(flat), np.log(N - 1), atol=1e-6)


def test_topk_mass_in_unit_interval():
    rng = np.random.default_rng(0)
    B = np.abs(rng.standard_normal((10, 10))); np.fill_diagonal(B, 0.0)
    mk = topk_mass(B, k=3)
    assert 0.0 <= mk <= 1.0


def test_structure_vs_null_flags_peaked_above_shuffled():
    N = 10
    peaked = np.zeros((N, N))
    for i in range(N):
        peaked[i, (i + 1) % N] = 10.0
        peaked[i, (i - 1) % N] = 1.0
    res = structure_vs_null(peaked, metric=sharpness, n_shuffle=50, seed=0)
    assert res["value"] > res["null_mean"] + 2 * res["null_std"]
    assert res["z"] > 2.0
