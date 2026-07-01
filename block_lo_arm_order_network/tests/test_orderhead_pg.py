import pathlib, sys
import torch, pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "block_lo_arm_order_network"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from batch_readout.l0_dynamic_gbeta import L0DynamicGBeta


def test_gbeta_scores_with_grad_shape_and_grad():
    from batch_readout.orderhead_pg import gbeta_scores_with_grad
    m = L0DynamicGBeta(heads=8, nodes=65)
    B = torch.randn(1, 8, 65, 65)              # detached input
    scores = gbeta_scores_with_grad(m, B)
    assert scores.shape == (64,)
    assert scores.requires_grad               # grad flows to gβ params
    scores.sum().backward()
    assert any(p.grad is not None for p in m.parameters())


def test_batch_advantage_is_scalar_and_detached():
    from batch_readout.orderhead_pg import batch_advantage
    from analyses.v3_group_credit import group_ids_for, GroupEMA
    bs, m = 16, 8
    groups = group_ids_for(bs, m)
    ema = GroupEMA(len(groups), 0.9)
    token_losses = torch.rand(bs, 10)          # (Bs, T) per-token CE
    A, ell_i = batch_advantage(token_losses, groups, ema, adv_clip=0.3)
    assert A.dim() == 0                        # scalar advantage for ONE order
    assert not A.requires_grad                 # advantage detached
    assert ell_i.shape == (bs,)
    assert float(A.abs()) <= 0.3 + 1e-6        # clamped


def test_pl_logp_is_grad_connected_to_scores():
    from batch_readout.orderhead_pg import gbeta_scores_with_grad
    from analyses.p7_gbeta_policy import sample_pl
    m = L0DynamicGBeta(heads=8, nodes=65)
    scores = gbeta_scores_with_grad(m, torch.randn(1, 8, 65, 65))
    _order, logp, entropy = sample_pl(scores, tau=1.0)
    logp.backward()
    assert any(p.grad is not None for p in m.parameters())


def test_grad_scores_match_frozen_scores_before_unfreeze():
    """B_PG == B_frozen: same B in => grad and no_grad scores allclose,
    identical argsort (spec B-source + order-frame invariant, gβ-forward level)."""
    from batch_readout.orderhead_pg import gbeta_scores_with_grad
    m = L0DynamicGBeta(heads=8, nodes=65).eval()
    B = torch.randn(1, 8, 65, 65)
    with torch.no_grad():
        s_frozen, _ = m(B, apply_head_dropout=False)
    s_grad = gbeta_scores_with_grad(m, B.detach())
    assert torch.allclose(s_frozen[0], s_grad, atol=1e-6)
    assert torch.equal(s_frozen[0].argsort(descending=True),
                       s_grad.argsort(descending=True))


def test_pg_loss_does_not_touch_backbone():
    """PG grad routes to gβ only; detached B ⇒ no path to the 'backbone'."""
    from batch_readout.orderhead_pg import gbeta_scores_with_grad, batch_advantage
    from analyses.p7_gbeta_policy import sample_pl
    from analyses.v3_group_credit import group_ids_for, GroupEMA
    backbone_p = torch.nn.Parameter(torch.randn(1, 8, 65, 65))
    B_det = backbone_p.detach()                      # spec: B detached before gβ
    m = L0DynamicGBeta(heads=8, nodes=65)
    scores = gbeta_scores_with_grad(m, B_det)
    _order, logp, entropy = sample_pl(scores, tau=1.0)
    groups = group_ids_for(16, 8); ema = GroupEMA(len(groups), 0.9)
    A, _ = batch_advantage(torch.rand(16, 10), groups, ema, 0.3)
    L_pg = -(A * logp) - 3e-3 * entropy
    L_pg.backward()
    assert backbone_p.grad is None                   # backbone untouched
    assert any(p.grad is not None for p in m.parameters())
