# tests/test_hidden_residual_probe.py
import numpy as np
from hidden_residual_probe import representation_probe

def test_probe_beats_control_when_hidden_encodes_label():
    rng = np.random.default_rng(0)
    n, E = 400, 8
    labels = rng.integers(0, 4, size=n)
    onehot = np.eye(4)[labels]
    H = onehot @ rng.normal(0, 1, (4, E)) + rng.normal(0, 0.05, (n, E))  # hidden encodes label
    res = representation_probe(H, labels, kind="classification", seed=0)
    assert res["score"] > res["shuffled_control"] + 0.2
    assert res["score"] > 0.8

def test_probe_no_signal_when_hidden_random():
    rng = np.random.default_rng(1)
    n, E = 400, 8
    labels = rng.integers(0, 4, size=n)
    H = rng.normal(0, 1, (n, E))  # no relation
    res = representation_probe(H, labels, kind="classification", seed=0)
    assert abs(res["score"] - res["shuffled_control"]) < 0.15

def test_residual_probe_oracle_beats_baselines_when_hidden_carries_r():
    from hidden_residual_probe import residual_probe, build_causal_features
    rng = np.random.default_rng(2)
    n, E, k = 600, 6, 4  # n samples, hidden dim E, k candidates per sample
    # planted: r depends on a hidden direction (per-sample), NOT on global/pos features
    w = rng.normal(0, 1, E)
    H = rng.normal(0, 1, (n, k, E))
    r = (H @ w) + rng.normal(0, 0.05, (n, k))
    phi_global = rng.normal(0, 1, (k, 6))            # B1: same across samples -> can't explain r
    phi_global = np.broadcast_to(phi_global, (n, k, 6))
    pos_id = np.broadcast_to(np.eye(k), (n, k, k))   # B2: position/id, same across samples
    res = residual_probe(r, H_oracle=H, phi_global=phi_global, pos_id=pos_id, seed=0)
    assert res["R2_oracle"] > 0.5
    assert res["R2_oracle"] > res["R2_B2"] + 0.2
    assert abs(res["R2_B1"]) < 0.1 and abs(res["R2_B2"]) < 0.1

def test_causal_interaction_features_shape():
    from hidden_residual_probe import build_causal_features
    rng = np.random.default_rng(3)
    n, k, E, P = 5, 4, 6, 4
    h_state = rng.normal(0, 1, (n, E))               # shared per sample
    cand_emb = rng.normal(0, 1, (n, k, P))
    feats = build_causal_features(h_state, cand_emb)
    # [emb (P), h_state broadcast (E), emb ⊙ (W h_state) (P)] -> per (n,k)
    assert feats.shape[0] == n * k and feats.shape[1] == P + E + P
