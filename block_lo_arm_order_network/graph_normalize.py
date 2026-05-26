"""Graph normalization, position residualization, off-diag correlation, matched control.
corr/residualize/mix use off-diag z-score; C-D+L readout uses shift-to-nonnegative."""
import numpy as np


def _offdiag_mask(N):
    return ~np.eye(N, dtype=bool)


def offdiag_zscore(B):
    B = np.asarray(B, dtype=np.float64).copy()
    m = _offdiag_mask(B.shape[0])
    vals = B[m]
    std = vals.std()
    std = std if std > 0 else 1.0
    out = np.zeros_like(B)
    out[m] = (vals - vals.mean()) / std
    return out


def shift_nonneg(B):
    """Shift off-diag entries so min off-diag = 0 (preserves order; diag stays 0)."""
    B = np.asarray(B, dtype=np.float64).copy()
    m = _offdiag_mask(B.shape[0])
    out = np.zeros_like(B)
    out[m] = B[m] - B[m].min()
    return out


def offdiag_corr(A, B, method="pearson"):
    a = np.asarray(A, dtype=np.float64)[_offdiag_mask(A.shape[0])]
    b = np.asarray(B, dtype=np.float64)[_offdiag_mask(B.shape[0])]
    if method == "spearman":
        a = np.argsort(np.argsort(a)).astype(np.float64)
        b = np.argsort(np.argsort(b)).astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def residualize(target, basis):
    """OLS residual of target off-diag on [1, basis off-diag]; diag stays 0."""
    target = np.asarray(target, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    N = target.shape[0]
    m = _offdiag_mask(N)
    y = target[m]
    x = basis[m]
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid_vals = y - X @ beta
    out = np.zeros_like(target)
    out[m] = resid_vals
    return out


def matched_random_residual(B, seed=0):
    """Off-diagonal permutation of B preserving the off-diag value multiset, diag 0.
    Symmetric input -> symmetric matched control (permute upper triangle, mirror)."""
    B = np.asarray(B, dtype=np.float64)
    N = B.shape[0]
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(N, k=1)
    upper = B[iu].copy()
    rng.shuffle(upper)
    out = np.zeros_like(B)
    out[iu] = upper
    out = out + out.T
    return out
