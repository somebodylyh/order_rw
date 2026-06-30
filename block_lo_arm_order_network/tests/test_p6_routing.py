import pathlib, sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import (   # noqa: E402
    candidate_priority, priority_matrix, ScaffoldedController, N,
)
from analyses.p6_online_controller import (    # noqa: E402
    cosine_logits, routing_p, routing_loss_from_scores, controller_scores,
    h_mode_list, p6_candidate_pool, eval_soft_expected_nll,
    eval_hard_selected_nll, _yl, train_controller_online, run_b0_sanity,
)

CKPT = str(ROOT / "runs/handoff_overnight/seed123/ckpt_step10000.pt")


# ── Task 2: cosine logits (scale-invariant — risk #2) ────────────────────────

def test_cosine_logits_scale_invariant():
    rng = np.random.default_rng(0)
    Y = torch.tensor(rng.standard_normal((6, N)), dtype=torch.float32)
    z = torch.tensor(rng.standard_normal(N), dtype=torch.float32)
    a1 = cosine_logits(z, Y, tau=0.3)
    a2 = cosine_logits(10.0 * z, Y, tau=0.3)      # ||z|| x10 must not change logits
    np.testing.assert_allclose(a1.numpy(), a2.numpy(), rtol=1e-4, atol=1e-5)


def test_cosine_logits_bounded():
    rng = np.random.default_rng(1)
    Y = torch.tensor(rng.standard_normal((8, N)), dtype=torch.float32)
    z = torch.tensor(rng.standard_normal(N), dtype=torch.float32)
    tau = 0.5
    a = cosine_logits(z, Y, tau=tau)
    assert torch.all((a * tau).abs() <= 1.0 + 1e-4)   # cosine in [-1,1]


# ── Task 3: routing loss (grad sign / detach — risk #1) ──────────────────────

def test_routing_loss_matches_manual():
    rng = np.random.default_rng(2)
    Y = torch.tensor(rng.standard_normal((5, N)), dtype=torch.float32)
    L = torch.tensor(rng.standard_normal(5), dtype=torch.float32)
    z = torch.tensor(rng.standard_normal(N), dtype=torch.float32)
    loss, info = routing_loss_from_scores(z, Y, L, tau=0.3)
    p_ref = routing_p(z, Y, 0.3)
    assert abs(float(loss) - float((p_ref * L).sum())) < 1e-5
    np.testing.assert_allclose(info["p"].detach().numpy(), p_ref.detach().numpy(),
                               rtol=1e-5)


def test_routing_loss_grad_pushes_to_low_nll():
    rng = np.random.default_rng(3)
    Y = torch.eye(4, N)                            # candidate k -> block k priority
    L = torch.tensor([3.0, 3.4, 3.3, 3.35])
    z = torch.zeros(N, requires_grad=True)
    p0 = routing_p(z.detach(), Y, 0.3)
    opt = torch.optim.SGD([z], lr=1.0)
    for _ in range(50):
        opt.zero_grad()
        loss, _ = routing_loss_from_scores(z, Y, L, tau=0.3)
        loss.backward(); opt.step()
    p1 = routing_p(z.detach(), Y, 0.3)
    kbest = int(torch.argmin(L))
    assert p1[kbest] > p0[kbest]                   # mass moved to lowest-NLL candidate


def test_Lk_constant_wrt_controller():
    # L is a plain tensor -> no graph; controller grad only via p_k
    L = torch.tensor([3.0, 3.5, 3.2])
    assert L.requires_grad is False
    Y = torch.eye(3, N)
    z = torch.zeros(N, requires_grad=True)
    loss, _ = routing_loss_from_scores(z, Y, L, tau=0.3)
    loss.backward()
    assert z.grad is not None and L.grad is None


# ── Task 5: detach_h config (risk #1) ────────────────────────────────────────

def test_detach_h_blocks_gradient_into_H():
    sc = ScaffoldedController(_tiny_gB(), h_dim=8, alpha_init=0.5)
    B = torch.zeros(N, 4)
    Y = torch.eye(4, N); L = torch.tensor([3.0, 3.4, 3.3, 3.35])
    # detach: H is the model's hidden state; gradient must NOT flow into it
    H = torch.randn(N, 8, requires_grad=True)
    z = controller_scores(sc, B, H, detach_h=True)
    routing_loss_from_scores(z, Y, L, 0.3)[0].backward()
    assert H.grad is None
    # no detach: gradient flows into H
    H2 = torch.randn(N, 8, requires_grad=True)
    sc.zero_grad()
    z2 = controller_scores(sc, B, H2, detach_h=False)
    routing_loss_from_scores(z2, Y, L, 0.3)[0].backward()
    assert H2.grad is not None


def _tiny_gB():
    from analyses.p5_utility_controller import BOnlyController
    return BOnlyController(b_dim=4, hidden=8)


# ── Task 4: controller modes / h_mode_list ───────────────────────────────────

def test_h_mode_list():
    Hs = [np.full((N, 3), i, dtype=np.float32) for i in range(4)]
    assert h_mode_list(Hs, "real") is Hs
    z = h_mode_list(Hs, "zero")
    assert all(np.all(h == 0) for h in z)
    m = h_mode_list(Hs, "mean")
    assert all(np.allclose(h, 1.5) for h in m)     # mean of 0,1,2,3
    sh = h_mode_list(Hs, "shuffle")
    assert all(not np.allclose(sh[i], Hs[i]) for i in range(4))   # j != i


# ── Task 1: candidate pool ───────────────────────────────────────────────────

def test_p6_candidate_pool():
    rng = np.random.default_rng(5)
    B65 = rng.standard_normal((65, 65))
    sigma_B = rng.permutation(N)
    pool = p6_candidate_pool(B65, sigma_B, rng, K=6)
    for key in ("sigma_cdl_b", "sigma_b_multihead", "phys", "reverse_phys"):
        assert key in pool
    labels, Y = priority_matrix(pool)              # dedup happens here
    assert Y.shape[0] == len(labels) >= 5
    np.testing.assert_allclose(Y[labels.index("phys")] if "phys" in labels else Y[0],
                               candidate_priority(pool["phys"], N), atol=0) \
        if "phys" in labels else None


# ── B0 frozen sanity (real ckpt, CPU) ────────────────────────────────────────

def test_train_controller_online_lowers_loss():
    samples = _toy_samples()
    Ys, Ls, labs = _yl(samples)
    _, hist = train_controller_online(samples, Ys, Ls, mode="b_only",
                                      tau=0.3, epochs=200, lr=5e-2,
                                      return_history=True)
    assert hist[-1] < hist[0]


def test_b0_sanity_smoke():
    if not pathlib.Path(CKPT).exists():
        import pytest; pytest.skip("ckpt absent")
    r = run_b0_sanity(CKPT, M=6, heads=[1, 2, 3, 4], epochs=40, n_reveals=4)
    a = r["arms"]
    assert "b_only" in a and "real" in a
    assert np.isfinite(a["b_only"]["nll_hard_selected"])
    assert np.isfinite(a["b_only"]["nll_fixed_order"])
    assert 0.0 <= a["b_only"]["entropy"]
    assert r["loss_down"]["b_only"]                 # loss decreased


def test_b2_step_updates_model_and_controller():
    # the gradient test the user asked for: one joint step must move BOTH the
    # AO-GPT params and the controller params (model grad via L_k, controller via p_k)
    if not pathlib.Path(CKPT).exists():
        import pytest; pytest.skip("ckpt absent")
    import copy
    from analyses.p6_online_controller import (
        load_p5_ckpt, _set_trainable, make_b2_controller, build_live_samples,
        b2_routing_step, random_reveal_orders,
    )
    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, 4, device="cpu")
    _set_trainable(model)
    inv = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    reveals = random_reveal_orders(2, 0)
    samples = build_live_samples(model, chunks, clean_perm, dev, [0, 1], reveals, inv,
                                 0, [1, 2, 3, 4], 1, 4, np.random.default_rng(0))
    ctrl = make_b2_controller(2 * (N + 1), samples[0]["H"].shape[1], "real",
                              alpha_init=0.5)
    m0 = copy.deepcopy(next(model.parameters()).detach().clone())
    c0 = copy.deepcopy([p.detach().clone() for p in ctrl.parameters() if p.requires_grad])
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] +
        [p for p in ctrl.parameters() if p.requires_grad], lr=1e-3)
    opt.zero_grad()
    loss, ent = b2_routing_step(model, ctrl, samples, clean_perm, dev, 0.3, 0.0,
                                True, "real")
    loss.backward(); opt.step()
    assert np.isfinite(float(loss)) and np.isfinite(ent)
    assert not torch.allclose(m0, next(model.parameters()).detach())   # model moved
    moved = any(not torch.allclose(c0[i], p.detach())
                for i, p in enumerate(p for p in ctrl.parameters() if p.requires_grad))
    assert moved                                                        # controller moved


def _toy_samples(M=12, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(M):
        good = rng.permutation(N); bad = rng.permutation(N)
        cands = {"good": good, "bad": bad, "r0": rng.permutation(N)}
        nll = {"good": 3.0, "bad": 3.4, "r0": 3.3}
        B_feat = np.stack([candidate_priority(good, N)] * 2 +
                          [candidate_priority(bad, N)] * 2, axis=1).astype(np.float32)
        H = rng.standard_normal((N, 8)).astype(np.float32)
        samples.append({"B_feat": B_feat, "H": H, "cands": cands,
                        "nll_by_label": nll})
    return samples
