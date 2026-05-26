"""Hidden-state relational graph B_H = cos(h_u, h_v), block-level."""
import numpy as np


def cosine_graph(H):
    """H: (n, N, E) per-sample block hidden in physical frame.
    Returns (per_sample (n,N,N), mean (N,N)), cosine similarity, diag zeroed.
    Zero-norm blocks yield cosine 0 (not NaN)."""
    H = np.asarray(H, dtype=np.float64)
    norm = np.linalg.norm(H, axis=2, keepdims=True)          # (n,N,1)
    safe = np.where(norm == 0.0, 1.0, norm)
    Hn = H / safe                                            # unit vectors; zero rows stay zero
    per_sample = np.einsum("nie,nje->nij", Hn, Hn)          # (n,N,N) cosine
    N = per_sample.shape[1]
    eye = np.eye(N, dtype=bool)
    per_sample[:, eye] = 0.0
    mean = per_sample.mean(axis=0)
    mean[eye] = 0.0
    return per_sample, mean
