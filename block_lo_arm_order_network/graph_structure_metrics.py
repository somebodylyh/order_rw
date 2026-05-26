"""Structure metrics for a block graph and comparison against a shuffled null."""
import numpy as np
from graph_normalize import matched_random_residual, _offdiag_mask


def _rows_offdiag(B):
    N = B.shape[0]
    out = []
    for i in range(N):
        out.append(np.delete(B[i], i))
    return np.asarray(out, dtype=np.float64)            # (N, N-1)


def sharpness(B):
    """Mean over rows of max / mean of |off-diag| entries (peak-to-mean ratio)."""
    R = np.abs(_rows_offdiag(B))
    mean = R.mean(axis=1)
    mean = np.where(mean == 0, 1.0, mean)
    return float((R.max(axis=1) / mean).mean())


def row_entropy(B):
    """Mean Shannon entropy of normalized non-negative off-diag rows."""
    R = np.abs(_rows_offdiag(B))
    s = R.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    P = R / s
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(P > 0, P * np.log(P), 0.0).sum(axis=1)
    return float(ent.mean())


def topk_mass(B, k=4):
    """Mean fraction of |off-diag| row mass in the top-k entries."""
    R = np.abs(_rows_offdiag(B))
    s = R.sum(axis=1)
    s = np.where(s == 0, 1.0, s)
    topk = np.sort(R, axis=1)[:, -k:].sum(axis=1)
    return float((topk / s).mean())


def spectral_gap(B):
    """Gap between the two largest-magnitude eigenvalues of the symmetrized |B|."""
    M = np.abs((B + B.T) / 2.0)
    w = np.linalg.eigvalsh(M)
    w = np.sort(np.abs(w))[::-1]
    return float(w[0] - w[1]) if len(w) > 1 else float(w[0])


def structure_vs_null(B, metric=sharpness, n_shuffle=100, seed=0):
    """Compare metric(B) to its distribution over off-diag-shuffled nulls."""
    val = metric(B)
    nulls = [metric(matched_random_residual(B, seed=seed + i)) for i in range(n_shuffle)]
    nulls = np.asarray(nulls)
    mu, sd = float(nulls.mean()), float(nulls.std())
    z = (val - mu) / (sd if sd > 0 else 1.0)
    return {"value": val, "null_mean": mu, "null_std": sd, "z": float(z)}
