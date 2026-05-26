import numpy as np, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from graph_normalize import (offdiag_zscore, shift_nonneg, offdiag_corr,
                             residualize, matched_random_residual, _offdiag_mask)


def _sym(rng, N):
    A = rng.standard_normal((N, N)); A = (A + A.T) / 2; np.fill_diagonal(A, 0.0); return A


def test_offdiag_zscore_zero_mean_unit_std_offdiag():
    rng = np.random.default_rng(1); B = _sym(rng, 6)
    Z = offdiag_zscore(B)
    m = _offdiag_mask(6)
    assert np.allclose(Z[m].mean(), 0.0, atol=1e-9)
    assert np.allclose(Z[m].std(), 1.0, atol=1e-9)
    assert np.allclose(np.diag(Z), 0.0)


def test_shift_nonneg_min_is_zero_and_order_preserved():
    rng = np.random.default_rng(2); B = _sym(rng, 6)
    S = shift_nonneg(offdiag_zscore(B))
    m = _offdiag_mask(6)
    assert S[m].min() == 0.0
    assert np.all(S[m] >= 0.0)
    assert np.allclose(np.diag(S), 0.0)
    base = offdiag_zscore(B)
    assert np.array_equal(np.argsort(base[m]), np.argsort(S[m]))


def test_offdiag_corr_self_is_one():
    rng = np.random.default_rng(3); B = _sym(rng, 8)
    assert np.isclose(offdiag_corr(B, B, method="pearson"), 1.0)
    assert np.isclose(offdiag_corr(B, B, method="spearman"), 1.0)


def test_residualize_removes_basis_correlation():
    rng = np.random.default_rng(4); basis = _sym(rng, 10)
    noise = _sym(rng, 10)
    target = 3.0 * basis + 0.5 * noise
    resid = residualize(target, basis)
    assert abs(offdiag_corr(resid, basis, method="pearson")) < 1e-6
    assert np.allclose(np.diag(resid), 0.0)


def test_matched_random_preserves_offdiag_multiset_and_diag0():
    rng = np.random.default_rng(5); B = _sym(rng, 7)
    M = matched_random_residual(B, seed=123)
    m = _offdiag_mask(7)
    assert np.allclose(np.sort(M[m]), np.sort(B[m]))
    assert np.allclose(np.diag(M), 0.0)
    assert not np.allclose(M, B)
