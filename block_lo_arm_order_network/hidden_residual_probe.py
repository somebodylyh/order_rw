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
