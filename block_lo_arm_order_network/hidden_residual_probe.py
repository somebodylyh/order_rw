# hidden_residual_probe.py
"""Phase-1 hidden-residual diagnostic: probes + order-effect metrics (pure numpy/sklearn)."""
import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import cross_val_score

def representation_probe(H, labels, kind="classification", seed=0):
    """Cross-validated probe score for hidden H -> labels, with a shuffled-hidden control.
    kind='classification' -> accuracy; 'regression' -> R^2."""
    H = np.asarray(H, dtype=np.float64); labels = np.asarray(labels)
    if kind == "classification":
        est = LogisticRegression(max_iter=500, multi_class="auto")
        scoring = "accuracy"
    else:
        est = RidgeCV(alphas=np.logspace(-3, 3, 13))
        scoring = "r2"
    score = float(cross_val_score(est, H, labels, cv=5, scoring=scoring).mean())
    rng = np.random.default_rng(seed)
    H_shuf = H[rng.permutation(len(H))]
    control = float(cross_val_score(est, H_shuf, labels, cv=5, scoring=scoring).mean())
    return {"score": score, "shuffled_control": control, "kind": kind}

def _r2_flat(X, y, seed=0):
    y_flat = np.asarray(y, np.float64).reshape(-1)
    X_flat = np.ascontiguousarray(np.asarray(X, np.float64)).reshape(len(y_flat), -1)
    est = RidgeCV(alphas=np.logspace(-3, 3, 13))
    return float(cross_val_score(est, X_flat, y_flat, cv=5, scoring="r2").mean())

def build_causal_features(h_state, cand_emb, W=None, seed=0):
    """Causal head input per candidate: [emb(v), h_S_t, emb(v) ⊙ (W h_S_t)].
    h_state:(n,E) shared across candidates; cand_emb:(n,k,P). Returns (n*k, P+E+P)."""
    h_state = np.asarray(h_state, np.float64); cand_emb = np.asarray(cand_emb, np.float64)
    n, k, P = cand_emb.shape; E = h_state.shape[1]
    if W is None:
        rng = np.random.default_rng(seed); W = rng.normal(0, 1 / np.sqrt(E), (E, P))
    Wh = h_state @ W                                  # (n, P)
    h_b = np.broadcast_to(h_state[:, None, :], (n, k, E))
    inter = cand_emb * Wh[:, None, :]                 # (n,k,P) candidate-state interaction
    feats = np.concatenate([cand_emb, h_b, inter], axis=2)  # (n,k,P+E+P)
    return feats.reshape(n * k, P + E + P)

def residual_probe(r, H_oracle, phi_global, pos_id, seed=0):
    """1b: predict residual target r (n,k) from baselines and oracle hidden.
    Returns R^2 for B0 (mean), B1 (global phi), B2 (pos/id), oracle ([phi_global, h_v])."""
    r = np.asarray(r, np.float64); n, k = r.shape
    y = r.reshape(-1)
    R2_B0 = 0.0  # mean-only predictor has R^2 = 0 by definition on held-out
    R2_B1 = _r2_flat(np.asarray(phi_global).reshape(n * k, -1), y, seed)
    R2_B2 = _r2_flat(np.asarray(pos_id).reshape(n * k, -1), y, seed)
    Xo = np.concatenate([np.asarray(phi_global), np.asarray(H_oracle)], axis=2).reshape(n * k, -1)
    R2_oracle = _r2_flat(Xo, y, seed)
    return {"R2_B0": R2_B0, "R2_B1": R2_B1, "R2_B2": R2_B2, "R2_oracle": R2_oracle}

def residual_probe_causal(r, causal_feats, seed=0):
    """1b causal head: R^2 of predicting r from build_causal_features output."""
    r = np.asarray(r, np.float64)
    return {"R2_causal": _r2_flat(causal_feats, r.reshape(-1), seed)}
