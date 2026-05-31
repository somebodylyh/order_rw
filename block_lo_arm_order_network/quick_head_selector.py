"""Quick head selector — cheap O(N^2) per-head order scores as a top-k
pre-selector for the order-specialized attention head of a text AOGPT run.

Positioning (spec docs/superpowers/specs/2026-05-30-quick-head-selector-design.md):
    A^{l,h} -> cheap score(l,h) -> top-k candidate heads -> CDL validation -> hook
The cheap score is a PRE-SELECTOR, not a final oracle.

Red lines (spec §2): L2R tau is a diagnostic microscope only (used to validate /
sign-calibrate the cheap score). It is NOT a training label and does NOT enter the
readout, MLP teacher, order generation, NLL, or reward. The teacher remains the
attention-derived CDL pseudo-order.

Conventions: per head take physical block graph A (N,N); B = A.T (zero diagonal);
readiness r(v) = out(v) - alpha*in(v) with out = B.sum(1), in = B.sum(0), alpha=0.5
(matches attn_order_mlp_policy.readiness_vector used by generate_teacher_label).
"""
import numpy as np
from scipy.stats import spearmanr

ALPHA_DEP = 0.5


def head_cheap_scores(A, alpha_dep=ALPHA_DEP, eps=1e-8):
    """Cheap order scores for ONE per-head physical block graph A (N, N).

    Returns dict {C1, C2, C3, C4} of raw floats (signs NOT calibrated):
        C1 signed readiness-position correlation: corr(r(v), index)
        C2 signed flow drift: mean over rows of weighted displacement under B
        C3 readiness spread: std(r)                       [dead-head filter]
        C4 asymmetry strength: ||B - B.T||_F / ||B||_F    [directionality filter]
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    B = A.T.copy()
    np.fill_diagonal(B, 0.0)

    out = B.sum(axis=1)
    inn = B.sum(axis=0)
    r = out - alpha_dep * inn
    idx = np.arange(n, dtype=np.float64)

    if np.std(r) < eps:
        c1 = 0.0
    else:
        c1 = float(np.corrcoef(r, idx)[0, 1])

    # C2: per-row weighted mean displacement (j - v) under B, averaged over rows
    v_idx = np.arange(n, dtype=np.float64)
    disp = idx[None, :] - v_idx[:, None]        # (n, n): j - v
    row_w = B.sum(axis=1)
    nz = row_w > eps
    drift_rows = (B * disp).sum(axis=1)
    c2 = float((drift_rows[nz] / row_w[nz]).mean()) if nz.any() else 0.0

    c3 = float(np.std(r))

    num = float(np.linalg.norm(B - B.T))
    den = float(np.linalg.norm(B)) + eps
    c4 = num / den

    return {"C1": c1, "C2": c2, "C3": c3, "C4": c4}


def cheap_head_scores(A_lh, alpha_dep=ALPHA_DEP):
    """Cheap scores for all heads.

    A_lh: (L, H, N, N) batch-mean per-head physical A, or (n_chunks, L, H, N, N)
          which is averaged over chunks first.
    Returns dict {C1, C2, C3, C4: (L, H) float arrays} (raw, signs NOT calibrated).
    """
    A_lh = np.asarray(A_lh, dtype=np.float64)
    if A_lh.ndim == 5:
        A_lh = A_lh.mean(axis=0)
    elif A_lh.ndim != 4:
        raise ValueError(f"A_lh must be 4D or 5D, got {A_lh.shape}")
    L, H = A_lh.shape[:2]
    out = {k: np.zeros((L, H), dtype=np.float64) for k in ("C1", "C2", "C3", "C4")}
    for l in range(L):
        for h in range(H):
            s = head_cheap_scores(A_lh[l, h], alpha_dep=alpha_dep)
            for k in out:
                out[k][l, h] = s[k]
    return out


def calibrate_sign(raw, expensive_tau):
    """Choose a global sign so the score is POSITIVELY aligned with expensive tau.

    raw, expensive_tau: (L, H) arrays. Returns (sign, calibrated) where sign in
    {+1.0, -1.0} and calibrated = sign * raw. Spec §3.1: positive cheap score is
    DEFINED to correlate positively with expensive tau_vs_l2r on the calibration
    split. Degenerate inputs (constant raw / too few pairs) return (+1, raw).
    """
    raw = np.asarray(raw, dtype=np.float64)
    tau = np.asarray(expensive_tau, dtype=np.float64)
    rf, tf = raw.ravel(), tau.ravel()
    mask = ~(np.isnan(rf) | np.isnan(tf))
    if mask.sum() < 3 or np.std(rf[mask]) < 1e-12 or np.std(tf[mask]) < 1e-12:
        return 1.0, raw.copy()
    rho, _ = spearmanr(rf[mask], tf[mask])
    sign = -1.0 if (rho is not None and not np.isnan(rho) and rho < 0) else 1.0
    return sign, sign * raw
