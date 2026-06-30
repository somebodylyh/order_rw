import pathlib, sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import (   # noqa: E402
    candidate_priority, priority_matrix, direct_nll_routing_loss,
    train_b_only_routing, train_residual_routing, routing_eval, N,
)

CKPT = str(ROOT / "runs/handoff_overnight/seed123/ckpt_step10000.pt")


# ── priority encoding ────────────────────────────────────────────────────────

def test_candidate_priority_identity():
    phys = np.arange(N)
    y = candidate_priority(phys, N)
    assert y[0] == 1.0                       # first revealed -> top priority
    assert y[-1] == 0.0                      # last revealed -> zero priority
    # monotone decreasing with reveal rank
    assert np.all(np.diff(y) < 0)
    np.testing.assert_allclose(y, 1 - np.arange(N) / (N - 1))


def test_candidate_priority_respects_reveal_order():
    sigma = np.arange(N)[::-1].copy()        # block 63 revealed first
    y = candidate_priority(sigma, N)
    assert y[63] == 1.0 and y[0] == 0.0


def test_priority_matrix_alignment():
    rng = np.random.default_rng(0)
    cands = {"a": np.arange(N), "b": rng.permutation(N), "c": rng.permutation(N)}
    labels, Y = priority_matrix(cands)
    assert Y.shape == (3, N)
    for k, lab in enumerate(labels):
        np.testing.assert_allclose(Y[k], candidate_priority(cands[lab], N))


def test_priority_matrix_dedups_identical_orders():
    # phys/local both = identity -> must collapse to one row (avoids softmax tie)
    cands = {"phys": np.arange(N), "local": np.arange(N),
             "rev": np.arange(N)[::-1].copy()}
    labels, Y = priority_matrix(cands)
    assert Y.shape == (2, N)
    assert ("phys" in labels) ^ ("local" in labels)   # exactly one kept
    assert "rev" in labels


# ── loss ─────────────────────────────────────────────────────────────────────

def test_routing_loss_matches_manual():
    rng = np.random.default_rng(1)
    K = 5
    Y = torch.tensor(rng.standard_normal((K, N)), dtype=torch.float32)
    L = torch.tensor(rng.standard_normal(K), dtype=torch.float32)
    z = torch.tensor(rng.standard_normal(N), dtype=torch.float32)
    tau = 0.3
    loss, p = direct_nll_routing_loss(z, Y, L, tau)
    a = Y @ z
    p_ref = torch.softmax(a / tau, dim=0)
    np.testing.assert_allclose(p.detach().numpy(), p_ref.numpy(), rtol=1e-5)
    assert abs(float(loss) - float((p_ref * L).sum())) < 1e-5


def test_routing_loss_permutation_invariant():
    rng = np.random.default_rng(2)
    K = 6
    Y = torch.tensor(rng.standard_normal((K, N)), dtype=torch.float32)
    L = torch.tensor(rng.standard_normal(K), dtype=torch.float32)
    z = torch.tensor(rng.standard_normal(N), dtype=torch.float32)
    perm = torch.randperm(K)
    l0, _ = direct_nll_routing_loss(z, Y, L, 0.2)
    l1, _ = direct_nll_routing_loss(z, Y[perm], L[perm], 0.2)
    assert abs(float(l0) - float(l1)) < 1e-5


def test_routing_loss_tau_to_zero_picks_min():
    # z aligned so a_k = -L_k -> small tau concentrates on argmin L
    rng = np.random.default_rng(3)
    K = 4
    L = torch.tensor(rng.uniform(1.0, 2.0, K), dtype=torch.float32)
    # Y orthonormal-ish so that we can hit a = -L exactly via least squares is hard;
    # instead use one-hot priority per candidate on distinct blocks.
    Y = torch.zeros(K, N)
    for k in range(K):
        Y[k, k] = 1.0
    z = torch.zeros(N)
    for k in range(K):
        z[k] = -float(L[k])                  # a_k = z . y_k = -L_k
    loss_cold, p_cold = direct_nll_routing_loss(z, Y, L, 0.01)
    loss_hot, _ = direct_nll_routing_loss(z, Y, L, 1.0)
    assert float(loss_cold) < float(loss_hot)
    assert abs(float(loss_cold) - float(L.min())) < 0.05
    assert int(torch.argmax(p_cold)) == int(torch.argmin(L))


# ── training (no model needed: L_k precomputed) ──────────────────────────────

def _toy_samples(M=12, seed=0):
    """Synthetic samples: one candidate is systematically better and its priority
    correlates with a learnable B feature, so a router can find it."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(M):
        good = rng.permutation(N)
        bad = rng.permutation(N)
        cands = {"good": good, "bad": bad,
                 "r0": rng.permutation(N), "r1": rng.permutation(N)}
        nll = {"good": 3.0, "bad": 3.4, "r0": 3.3, "r1": 3.35}
        # B feature = the good candidate's priority (so g_B can learn to align z)
        B_feat = np.stack([candidate_priority(good, N)] * 2 +
                          [candidate_priority(bad, N)] * 2, axis=1).astype(np.float32)
        H = rng.standard_normal((N, 8)).astype(np.float32)
        samples.append({"B_feat": B_feat, "H": H, "cands": cands,
                        "nll_by_label": nll})
    return samples


def test_b_only_routing_lowers_expected_nll():
    samples = _toy_samples()
    g_B, hist = train_b_only_routing(samples, tau=0.3, epochs=300, lr=5e-2,
                                     return_history=True)
    assert hist[-1] < hist[0]                 # expected NLL decreased
    # after training, mass concentrates on the good candidate on average
    ev = routing_eval(samples, g_B, tau=0.3, with_model=False)
    assert ev["best_candidate_frac"]["good"] > 0.5


def test_zero_and_mean_h_modes_run():
    samples = _toy_samples()
    g_B = train_b_only_routing(samples, tau=0.3, epochs=50, lr=5e-2)
    h_dim = samples[0]["H"].shape[1]
    for mode in ("real", "shuffle", "zero", "mean"):
        sc = train_residual_routing(samples, g_B, h_dim, tau=0.3, epochs=50,
                                    lr=5e-2, h_mode=mode)
        ev = routing_eval(samples, sc, h_mode=mode, tau=0.3, with_model=False)
        assert np.isfinite(ev["nll_pool"])


# ── integration with real ckpt (free-argsort eval needs the model) ───────────

def test_routing_eval_with_model_smoke():
    if not pathlib.Path(CKPT).exists():
        import pytest
        pytest.skip("ckpt not present")
    from analyses.p5_utility_controller import build_dataset
    samples = build_dataset(CKPT, M=6, n_reveals=4, K_rand=3, K_noisy=3)
    g_B = train_b_only_routing(samples, tau=0.3, epochs=30, lr=5e-2)
    ev = routing_eval(samples, g_B, tau=0.3, with_model=True)
    assert np.isfinite(ev["nll_pool"]) and np.isfinite(ev["nll_free"])
    assert 0.0 <= ev["entropy"]
