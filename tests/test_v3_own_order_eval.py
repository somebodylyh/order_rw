"""TDD for v3_own_order_eval: own-order val (headline/selection metric) and the
L2R order-transfer anti-gaming diagnostic."""
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "block_lo_arm_order_network")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

CKPT = "runs/handoff_overnight/seed123/ckpt_step10000.pt"
_N = 64


# ── mocks (mirror test_v3_group_probe fixtures) ────────────────────────────────

class _MockBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward_fn(self, idx, token_order):
        # deterministic loss that depends on the token order actually revealed
        L2R_token = torch.arange(token_order.shape[1], device=token_order.device)
        dist = (token_order.float() - L2R_token.float().unsqueeze(0)).abs().sum()
        loss = 3.0 + 0.01 * dist / token_order.numel()
        return None, loss


class _MockOrderHead(torch.nn.Module):
    """scores depend on B so different groups yield different orders; the default
    B (i*I) gives a descending-index consensus, i.e. a REVERSED order != L2R."""
    def __init__(self):
        super().__init__()
        self.gbeta = torch.nn.Linear(1, 1)

    def scores(self, A, per_sample):
        # Real OrderHeadModule.scores returns a TENSOR (rows, N) (not a tuple);
        # scores(...)[0] then indexes the first row. arange -> argsort(desc) reverses.
        base = torch.arange(_N, dtype=torch.float32, device=A.device)
        return base.unsqueeze(0)


class _MockPerm:
    block_perm_phys_to_model = torch.arange(_N)
    inv_perm_model_to_phys = torch.arange(_N)


class _MockWrap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _MockBackbone()
        self.order_head = _MockOrderHead()
        self.clean_perm = _MockPerm()
        self.inv_perm = np.arange(_N, dtype=np.int64)  # identity remap for test

    def extract_B(self, idx_batch, probe):
        M = idx_batch.shape[0]
        A = torch.zeros(M, _N + 1, _N + 1)
        for i in range(M):
            A[i, 1:, 1:] = torch.eye(_N) * float(i + 1)
        return A

    def extract_B65(self, idx_batch, probe):
        # strict65 (None retained), zero diagonal — used by the order_fn path.
        M = idx_batch.shape[0]
        B = torch.abs(torch.randn(M, _N + 1, _N + 1))
        for i in range(M):
            B[i].fill_diagonal_(0.0)
        return B


# ── tests ──────────────────────────────────────────────────────────────────────

def test_own_order_val_loss_reveals_under_policy_order(monkeypatch):
    """own_order_val_loss must reveal each group under ITS OWN policy order
    (argsort of group-mean-B scores, model->phys remapped), mean over groups —
    NOT the fixed L2R order."""
    import analyses.v3_own_order_eval as oe
    from analyses.v3_group_credit import group_ids_for
    from analyses.p5_utility_controller import order_nll

    monkeypatch.setattr(oe, "N", _N, raising=False)

    wrap = _MockWrap()
    held = torch.zeros(16, _N, dtype=torch.long)
    m, seed, dev = 8, 0, torch.device("cpu")

    got = oe.own_order_val_loss(wrap, held, m=m, seed=seed, device="cpu")

    # expected: replicate the intended semantics (policy order per group)
    from batch_readout.hook_order_provider import random_probe_token_orders
    probe = random_probe_token_orders(held.shape[0], seed, 0, dev)
    A = wrap.extract_B(held, probe).detach()
    groups = group_ids_for(held.shape[0], m)
    per_group = []
    for g in groups:
        gi = torch.as_tensor(g, dtype=torch.long, device=dev)
        Bg = A.index_select(0, gi)
        scores_g = wrap.order_head.scores(Bg, per_sample=False)[0]
        sig_model = torch.argsort(scores_g.detach(), descending=True,
                                  stable=True).cpu().numpy().astype(np.int64)
        sig_phys = wrap.inv_perm[sig_model]
        row = torch.stack([held[i] for i in g])
        per_group.append(order_nll(wrap.backbone, row, sig_phys, wrap.clean_perm, dev))
    expected = float(np.mean(per_group))

    assert np.isfinite(got)
    assert abs(got - expected) < 1e-6, f"got {got}, expected policy-order mean {expected}"

    # and it must NOT equal the L2R-order value (policy order here is reversed)
    l2r = np.arange(_N, dtype=np.int64)
    l2r_vals = []
    for g in groups:
        row = torch.stack([held[i] for i in g])
        l2r_vals.append(order_nll(wrap.backbone, row, l2r, wrap.clean_perm, dev))
    assert abs(got - float(np.mean(l2r_vals))) > 1e-6, "own-order collapsed to L2R"


def test_order_transfer_reports_own_l2r_and_diff(monkeypatch):
    """order_transfer returns own-order val, the L2R-transfer val on the SAME
    backbone, and own_over_l2r = l2r_transfer_val - own_order_val (>0 => the
    policy's own order helps this backbone; the anti-gaming diagnostic)."""
    import analyses.v3_own_order_eval as oe

    monkeypatch.setattr(oe, "N", _N, raising=False)

    wrap = _MockWrap()
    held = torch.zeros(16, _N, dtype=torch.long)

    r = oe.order_transfer(wrap, held, m=8, seed=0, device="cpu")

    for k in ("own_order_val", "l2r_transfer_val", "own_over_l2r"):
        assert k in r, f"missing key {k!r}"
        assert np.isfinite(r[k]), f"{k} non-finite: {r[k]}"

    # arithmetic contract
    assert abs(r["own_over_l2r"] - (r["l2r_transfer_val"] - r["own_order_val"])) < 1e-9

    # own_order_val must match the standalone headline function
    solo = oe.own_order_val_loss(wrap, held, m=8, seed=0, device="cpu")
    assert abs(r["own_order_val"] - solo) < 1e-9


@pytest.mark.skipif(not pathlib.Path(CKPT).exists(), reason="10k parent ckpt absent")
def test_own_order_and_transfer_with_real_ckpt():
    """Smoke: metrics compute end-to-end on the real 10k frozen-gβ backbone."""
    from analyses.order_head_module import OrderHeadModule, AOGPTWithOrderHead
    from analyses.p5_utility_controller import load_p5_ckpt
    from analyses.p7_gbeta_policy import GBETA_CKPT as GB
    from analyses.v3_own_order_eval import own_order_val_loss, order_transfer

    model, chunks, clean_perm, dev = load_p5_ckpt(CKPT, 32, device="cpu")
    oh = OrderHeadModule(GB, device="cpu")
    wrap = AOGPTWithOrderHead(model, oh, clean_perm, device="cpu")
    held = torch.stack([chunks[i] for i in range(32)])

    own = own_order_val_loss(wrap, held, m=16, seed=0, device="cpu")
    r = order_transfer(wrap, held, m=16, seed=0, device="cpu")

    assert np.isfinite(own) and 0.0 < own < 20.0, f"own-order val out of range: {own}"
    assert abs(r["own_order_val"] - own) < 1e-9
    assert np.isfinite(r["l2r_transfer_val"])
    assert abs(r["own_over_l2r"] - (r["l2r_transfer_val"] - r["own_order_val"])) < 1e-9
    print(f"\n[real ckpt] own={own:.4f} l2r_transfer={r['l2r_transfer_val']:.4f} "
          f"own_over_l2r={r['own_over_l2r']:+.4f}")


def test_group_policy_orders_returns_valid_perms(monkeypatch):
    """group_policy_orders returns G per-group orders, each a valid permutation
    (used to capture the 20k init orders for tau_to_init)."""
    import analyses.v3_own_order_eval as oe
    monkeypatch.setattr(oe, "N", _N, raising=False)
    wrap = _MockWrap()
    held = torch.zeros(16, _N, dtype=torch.long)
    orders = oe.group_policy_orders(wrap, held, m=8, seed=0, device="cpu")
    assert len(orders) == 2  # G = 16 // 8
    for o in orders:
        assert sorted(o.tolist()) == list(range(_N)), "not a valid permutation"


def test_own_order_val_loss_respects_order_fn(monkeypatch):
    """own_order_val_loss must use a provided order_fn instead of gβ argsort
    (this is how the cdl_teacher arm supplies its CDL reveal order)."""
    import analyses.v3_own_order_eval as oe
    from analyses.v3_group_credit import group_ids_for
    from analyses.p5_utility_controller import order_nll
    monkeypatch.setattr(oe, "N", _N, raising=False)

    wrap = _MockWrap()
    held = torch.zeros(16, _N, dtype=torch.long)
    l2r_fn = lambda Bg: np.arange(_N, dtype=np.int64)  # order_fn returns L2R

    got = oe.own_order_val_loss(wrap, held, m=8, seed=0, device="cpu", order_fn=l2r_fn)

    groups = group_ids_for(16, 8)
    l2r = np.arange(_N, dtype=np.int64)
    exp = float(np.mean([
        order_nll(wrap.backbone, torch.stack([held[i] for i in g]), l2r,
                  wrap.clean_perm, torch.device("cpu")) for g in groups]))
    assert abs(got - exp) < 1e-6, "order_fn was not used (own != L2R-under-order_fn)"


def test_cdl_order_fn_returns_valid_perm():
    """cdl_order_fn (greedy CDL rollout on batch-mean strict65 B) returns a valid
    permutation of the 64 content blocks."""
    from analyses.v3_own_order_eval import cdl_order_fn
    rng = np.random.default_rng(0)
    B = np.abs(rng.standard_normal((65, 65))).astype(np.float32)
    np.fill_diagonal(B, 0.0)                    # strict65: zero diagonal
    Bg = torch.tensor(B).unsqueeze(0)          # (1, 65, 65)
    order = cdl_order_fn(Bg)
    assert sorted(int(x) for x in order) == list(range(64)), "not a valid permutation"
