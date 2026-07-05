"""TDD for the Phase-1 evaluator headline switch: val_loss must be OWN-ORDER
val (the RL/selection signal), with L2R kept only as an anti-gaming diagnostic."""
import math
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, "block_lo_arm_order_network")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_N = 64


class _MockBackbone(torch.nn.Module):
    def forward_fn(self, idx, token_order):
        L2R_token = torch.arange(token_order.shape[1], device=token_order.device)
        dist = (token_order.float() - L2R_token.float().unsqueeze(0)).abs().sum()
        return None, 3.0 + 0.01 * dist / token_order.numel()


class _MockOrderHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gbeta = torch.nn.Linear(1, 1)

    def scores(self, A, per_sample):
        # Real OrderHeadModule.scores returns a TENSOR (rows, N); scores(...)[0]
        # indexes the first row.
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
        self.inv_perm = np.arange(_N, dtype=np.int64)

    def extract_B(self, idx_batch, probe):
        M = idx_batch.shape[0]
        A = torch.zeros(M, _N + 1, _N + 1)
        for i in range(M):
            A[i, 1:, 1:] = torch.eye(_N) * float(i + 1)
        return A


def _held(M=16):
    return [torch.zeros(_N, dtype=torch.long) for _ in range(M)]


def test_evaluator_val_loss_is_own_order_with_l2r_diagnostic():
    """For an arm with an OrderHead, val_loss == own_order_val_loss (headline),
    and val_l2r_transfer/own_over_l2r appear as diagnostics only."""
    from analyses.v3_phase1_runner import _build_evaluator
    from analyses.v3_own_order_eval import own_order_val_loss, order_transfer

    held = _held(16)
    wrap = _MockWrap()
    ev = _build_evaluator(held, _MockPerm(), "cpu", m=8)
    out = ev(1, wrap.backbone, wrap)

    held_stack = torch.stack(held)
    expected_own = own_order_val_loss(wrap, held_stack, m=8, seed=0, device="cpu")
    expected_tr = order_transfer(wrap, held_stack, m=8, seed=0, device="cpu")

    # headline
    assert abs(out["val_loss"] - expected_own) < 1e-9, "val_loss is not own-order"
    assert abs(out["ppl"] - math.exp(expected_own)) < 1e-6
    # diagnostics
    assert abs(out["val_l2r_transfer"] - expected_tr["l2r_transfer_val"]) < 1e-9
    assert abs(out["own_over_l2r"] - (out["val_l2r_transfer"] - out["val_loss"])) < 1e-9
    # guards still present
    for k in ("delta_probe_group", "real_beats_controls", "tau_to_l2r", "tau_consensus"):
        assert k in out, f"missing guard key {k!r}"


def test_evaluator_l2r_arm_own_order_equals_l2r():
    """The l2r arm has no OrderHead; its own order IS L2R, so val_loss falls back
    to the fixed-L2R loss and own_over_l2r == 0."""
    from analyses.v3_phase1_runner import _build_evaluator
    from analyses.v3_fixed_order_eval import fixed_order_val_loss

    held = _held(16)
    model = _MockBackbone()
    wrap = SimpleNamespace(backbone=model, order_head=None, clean_perm=_MockPerm())
    ev = _build_evaluator(held, _MockPerm(), "cpu", m=8)
    out = ev(1, model, wrap)

    expected = fixed_order_val_loss(model, held, _MockPerm(), "cpu")
    assert abs(out["val_loss"] - expected) < 1e-9
    assert abs(out["val_l2r_transfer"] - expected) < 1e-9
    assert abs(out["own_over_l2r"]) < 1e-9


def test_evaluator_reports_tau_to_init():
    """With init_orders provided, evaluator reports tau_to_init; when the current
    policy orders equal the init (deterministic mock), tau_to_init == 1.0."""
    from analyses.v3_phase1_runner import _build_evaluator
    from analyses.v3_own_order_eval import group_policy_orders

    held = _held(16)
    wrap = _MockWrap()
    held_stack = torch.stack(held)
    init = group_policy_orders(wrap, held_stack, m=8, seed=0, device="cpu")
    ev = _build_evaluator(held, _MockPerm(), "cpu", m=8, init_orders=init)
    out = ev(1, wrap.backbone, wrap)
    assert "tau_to_init" in out
    assert abs(out["tau_to_init"] - 1.0) < 1e-9


def test_evaluator_tau_to_l2r_uses_physical_frame():
    """tau_to_l2r must compare the PHYSICAL-frame order (inv_perm[model]) to L2R,
    not the raw model-frame order. With a non-identity inv_perm the two differ."""
    from analyses.v3_phase1_runner import _build_evaluator
    import numpy as np
    from scipy.stats import kendalltau
    from analyses.v3_own_order_eval import group_policy_orders, L2R

    held = _held(16)
    wrap = _MockWrap()
    # non-identity model->physical remap (reverse) so model!=physical frame
    wrap.inv_perm = np.arange(_N, dtype=np.int64)[::-1].copy()

    ev = _build_evaluator(held, _MockPerm(), "cpu", m=8)
    out = ev(1, wrap.backbone, wrap)

    cur = group_policy_orders(wrap, torch.stack(held), m=8, seed=0, device="cpu")
    exp_phys = float(np.nanmean([kendalltau(wrap.inv_perm[c], L2R).correlation for c in cur]))
    exp_model = float(np.nanmean([kendalltau(c, L2R).correlation for c in cur]))
    assert abs(out["tau_to_l2r"] - exp_phys) < 1e-9, "tau_to_l2r not in physical frame"
    assert abs(exp_phys - exp_model) > 1e-6, "test degenerate: phys==model frame"
