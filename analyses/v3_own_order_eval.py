"""Own-order validation loss (Stage 3 headline/selection metric) + the L2R
order-transfer anti-gaming diagnostic.

Stage 3 (PG unfreeze co-adapt) is judged on ``val`` under each arm's OWN policy
reveal order — NOT the fixed L2R order (``val_l2r``/``val_origin_l2r`` are barred
from headline/selection). Because an own-order metric is gameable (a policy can
pick an easy reveal order that lowers teacher-forced NLL without a better
backbone), we ALSO report the L2R order-transfer: the SAME backbone re-evaluated
under the fixed L2R order. If the own-order advantage survives under L2R the
backbone genuinely improved; if it vanishes the gain was order-specific (gaming
suspicion). L2R here is a diagnostic only, never a selection/training signal.
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCK_ROOT = ROOT / "block_lo_arm_order_network"
for path in (ROOT, BLOCK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyses.v3_group_credit import group_ids_for  # noqa: E402
from analyses.p5_utility_controller import order_nll, N  # noqa: E402
from batch_readout.hook_order_provider import random_probe_token_orders  # noqa: E402

L2R = np.arange(N, dtype=np.int64)


def cdl_order_fn(Bg):
    """Greedy CDL (C-D+L) reveal order from a group's strict65 B.

    Batch-means the group's ``(m, 65, 65)`` selected-head B to one ``(65, 65)``
    graph and runs the None-fixed greedy CDL rollout, returning a model-frame
    ``(64,)`` content-block order. This is the cdl_teacher arm's order source —
    the same B that frozen_gβ reads, but the CDL teacher instead of gβ.
    """
    from batch_readout.label_free_cdl_teacher import (
        cdl_rollout_with_standardized_margin,
    )
    B65 = Bg.mean(dim=0).detach().cpu().numpy().astype(np.float64)
    return cdl_rollout_with_standardized_margin(B65)["content_order"].astype(np.int64)


def _group_policy_orders(wrap, held_idx, groups, dev, seed, order_fn=None):
    """Per-group reveal order (model frame). Default = argsort of the group-mean
    64x64 B OrderHead (gβ) scores. If ``order_fn`` is given it maps the group's
    strict65 ``(m, 65, 65)`` B (None retained, via ``extract_B65``) to a model-
    frame ``(64,)`` order instead (e.g. cdl_order_fn — CDL needs the None row)."""
    probe = random_probe_token_orders(held_idx.shape[0], int(seed), 0, dev)
    A = (wrap.extract_B65(held_idx, probe) if order_fn is not None
         else wrap.extract_B(held_idx, probe)).detach()
    orders_model = []
    for g in groups:
        gi = torch.as_tensor(g, dtype=torch.long, device=dev)
        Bg = A.index_select(0, gi)
        if order_fn is None:
            scores_g = wrap.order_head.scores(Bg, per_sample=False)[0]
            sig_model = torch.argsort(scores_g.detach(), descending=True,
                                      stable=True).cpu().numpy().astype(np.int64)
        else:
            sig_model = np.asarray(order_fn(Bg), dtype=np.int64)
        orders_model.append(sig_model)
    return orders_model


@torch.no_grad()
def group_policy_orders(wrap, held_idx, *, m=16, seed=0, device="cpu", order_fn=None):
    """Per-group deterministic reveal orders (model frame). Default gβ argsort; a
    provided ``order_fn`` overrides it (cdl_teacher). Used to capture init orders
    and measure ``tau_to_init`` (policy drift)."""
    dev = torch.device(device)
    groups = group_ids_for(held_idx.shape[0], m)
    return _group_policy_orders(wrap, held_idx, groups, dev, seed, order_fn=order_fn)


@torch.no_grad()
def own_order_val_loss(wrap, held_idx, *, m=16, seed=0, device="cpu", order_fn=None):
    """Mean teacher-forced NLL where each fixed group is revealed under ITS OWN
    reveal order (gβ argsort by default, or ``order_fn`` — e.g. CDL — model->
    physical remapped). Stage 3 headline/selection metric. Lower = better.
    """
    dev = torch.device(device)
    groups = group_ids_for(held_idx.shape[0], m)
    orders_model = _group_policy_orders(wrap, held_idx, groups, dev, seed,
                                        order_fn=order_fn)

    nlls = []
    for g, sig_model in zip(groups, orders_model):
        sig_phys = wrap.inv_perm[sig_model]
        row = torch.stack([held_idx[i] for i in g])
        nlls.append(order_nll(wrap.backbone, row, sig_phys, wrap.clean_perm, dev))
    return float(np.mean(nlls))


@torch.no_grad()
def order_transfer(wrap, held_idx, *, m=16, seed=0, device="cpu", order_fn=None):
    """Anti-gaming order-transfer diagnostic on ONE backbone.

    Evaluates the same held-out set twice: under the policy's OWN per-group order
    and under the fixed L2R order (same grouping). ``own_over_l2r`` > 0 means the
    policy order lowers NLL relative to L2R on this backbone. Comparing this
    diagnostic across arms reveals whether a joint arm's own-order advantage is a
    real backbone improvement (survives L2R transfer) or an easy-order artifact.

    Returns ``{own_order_val, l2r_transfer_val, own_over_l2r}``.
    """
    dev = torch.device(device)
    groups = group_ids_for(held_idx.shape[0], m)

    own = own_order_val_loss(wrap, held_idx, m=m, seed=seed, device=device,
                             order_fn=order_fn)

    l2r_nlls = []
    for g in groups:
        row = torch.stack([held_idx[i] for i in g])
        l2r_nlls.append(order_nll(wrap.backbone, row, L2R, wrap.clean_perm, dev))
    l2r_val = float(np.mean(l2r_nlls))

    return {
        "own_order_val": own,
        "l2r_transfer_val": l2r_val,
        "own_over_l2r": l2r_val - own,
    }


__all__ = ["own_order_val_loss", "order_transfer", "group_policy_orders",
           "cdl_order_fn", "L2R"]
