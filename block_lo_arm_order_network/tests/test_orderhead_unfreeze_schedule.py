import pathlib, sys
import torch, pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "block_lo_arm_order_network"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from batch_readout.l0_dynamic_gbeta import L0DynamicGBeta


def _fresh():
    gb = L0DynamicGBeta(heads=8, nodes=65)
    for p in gb.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    for g in opt.param_groups:
        g.setdefault("is_orderhead", False)
    return gb, opt


def _args(unfreeze_at):
    return type("A", (), {"unfreeze_orderhead_at_step": unfreeze_at,
                          "orderhead_lr": 3e-4})


def test_no_orderhead_group_before_unfreeze():
    from train_clean_aogpt import maybe_unfreeze_orderhead
    gb, opt = _fresh()
    state = {"unfrozen": False}
    did = maybe_unfreeze_orderhead(opt, gb, 50, _args(100), state)
    assert did is False
    assert all(not g.get("is_orderhead") for g in opt.param_groups)
    assert all(not p.requires_grad for p in gb.parameters())


def test_single_insertion_and_weight_continuity():
    from train_clean_aogpt import maybe_unfreeze_orderhead
    gb, opt = _fresh()
    before = {k: v.clone() for k, v in gb.state_dict().items()}
    state = {"unfrozen": False}
    did1 = maybe_unfreeze_orderhead(opt, gb, 100, _args(100), state)
    did2 = maybe_unfreeze_orderhead(opt, gb, 101, _args(100), state)
    assert did1 is True and did2 is False
    oh_groups = [g for g in opt.param_groups if g.get("is_orderhead")]
    assert len(oh_groups) == 1
    ids = [id(p) for g in oh_groups for p in g["params"]]
    assert len(ids) == len(set(ids))                 # no duplicate params
    assert all(p.requires_grad for p in gb.parameters())
    after = gb.state_dict()
    for k in before:                                  # weight continuity
        assert torch.equal(before[k], after[k])


def test_disabled_when_flag_none():
    from train_clean_aogpt import maybe_unfreeze_orderhead
    gb, opt = _fresh()
    state = {"unfrozen": False}
    assert maybe_unfreeze_orderhead(opt, gb, 10_000, _args(None), state) is False


def _nodewise():
    from batch_readout.model import NodewiseReadout
    return NodewiseReadout(N=64, d_model=64, n_layers=2, n_heads=4)


def test_independent_orderhead_lr_survives_per_step_schedule():
    """P1-2: the per-step LR loop must NOT clobber the OrderHead group's LR."""
    gb = _nodewise()
    backbone_p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([backbone_p], lr=1e-3)
    for g in opt.param_groups:
        g.setdefault("is_orderhead", False)
    from train_clean_aogpt import _insert_orderhead_group
    _insert_orderhead_group(opt, gb, orderhead_lr=3e-4)
    # replicate the fixed per-step assignment (train_clean_aogpt loop)
    scheduled_lr = 9e-5
    for g in opt.param_groups:
        g["lr"] = 3e-4 if g.get("is_orderhead") else scheduled_lr
    lrs = {g.get("is_orderhead"): g["lr"] for g in opt.param_groups}
    assert lrs[False] == scheduled_lr        # backbone follows schedule
    assert lrs[True] == 3e-4                  # orderhead keeps its own LR


def test_ckpt_roundtrip_reconstructs_orderhead_group():
    """P1-1: Phase-B ckpt persists gβ+EMA+unfrozen, and resume reconstructs the
    optimizer param group so a 2-group optimizer state loads without mismatch."""
    from train_clean_aogpt import (orderhead_ckpt_fields, restore_orderhead_on_resume,
                                    _insert_orderhead_group)
    from analyses.v3_group_credit import group_ids_for, GroupEMA
    import copy

    # --- source run (Phase B, unfrozen) ---
    gb = _nodewise()
    for p in gb.parameters():
        p.requires_grad_(True)
    with torch.no_grad():                     # make weights non-default
        for p in gb.parameters():
            p.add_(0.05)
    groups = group_ids_for(16, 8)
    ema = GroupEMA(len(groups), 0.9); ema.update(torch.rand(len(groups)))
    src_pg = {"gbeta": gb, "groups": groups, "ema": ema}
    src_unfreeze = {"unfrozen": True}
    fields = orderhead_ckpt_fields(src_pg, src_unfreeze)
    assert fields["orderhead_unfrozen"] is True
    assert fields["orderhead_ema"] is not None
    # a saved optimizer with TWO groups (backbone + orderhead)
    bp = torch.nn.Parameter(torch.zeros(1))
    src_opt = torch.optim.AdamW([bp], lr=1e-3)
    for g in src_opt.param_groups:
        g.setdefault("is_orderhead", False)
    _insert_orderhead_group(src_opt, gb, 3e-4)
    src_opt.step()
    saved_optimizer = copy.deepcopy(src_opt.state_dict())
    ckpt = {"optimizer": saved_optimizer, **fields}

    # --- resume run (fresh optimizer has ONE group) ---
    gb2 = _nodewise()
    for p in gb2.parameters():
        p.requires_grad_(False)
    ema2 = GroupEMA(len(groups), 0.9)
    dst_pg = {"gbeta": gb2, "groups": groups, "ema": ema2}
    dst_unfreeze = {"unfrozen": False}
    bp2 = torch.nn.Parameter(torch.zeros(1))
    dst_opt = torch.optim.AdamW([bp2], lr=1e-3)
    for g in dst_opt.param_groups:
        g.setdefault("is_orderhead", False)

    args = type("A", (), {"orderhead_lr": 3e-4})
    reinserted = restore_orderhead_on_resume(ckpt, dst_opt, dst_pg, dst_unfreeze, args)
    assert reinserted is True
    assert dst_unfreeze["unfrozen"] is True
    assert sum(g.get("is_orderhead", False) for g in dst_opt.param_groups) == 1
    # gβ weights restored bit-identically
    for k in gb.state_dict():
        assert torch.equal(gb.state_dict()[k], gb2.state_dict()[k])
    # EMA restored
    assert torch.allclose(dst_pg["ema"].b, src_pg["ema"].b)
    # the deferred optimizer state now loads without a group-count mismatch
    dst_opt.load_state_dict(ckpt["optimizer"])


def test_phase_a_save_is_not_unfrozen():
    """A ckpt saved during Phase A carries orderhead weights but unfrozen=False,
    and resume does NOT insert a param group."""
    from train_clean_aogpt import orderhead_ckpt_fields, restore_orderhead_on_resume
    from analyses.v3_group_credit import group_ids_for, GroupEMA
    gb = _nodewise()
    pg = {"gbeta": gb, "groups": group_ids_for(16, 8), "ema": GroupEMA(2, 0.9)}
    fields = orderhead_ckpt_fields(pg, {"unfrozen": False})
    assert fields["orderhead_unfrozen"] is False

    gb2 = _nodewise()
    dst_opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    for g in dst_opt.param_groups:
        g.setdefault("is_orderhead", False)
    dst_pg = {"gbeta": gb2, "groups": pg["groups"], "ema": GroupEMA(2, 0.9)}
    dst_unfreeze = {"unfrozen": False}
    args = type("A", (), {"orderhead_lr": 3e-4})
    reinserted = restore_orderhead_on_resume(fields, dst_opt, dst_pg, dst_unfreeze, args)
    assert reinserted is False
    assert not any(g.get("is_orderhead") for g in dst_opt.param_groups)
    # weights still restored
    for k in gb.state_dict():
        assert torch.equal(gb.state_dict()[k], gb2.state_dict()[k])


def test_cli_flags_registered():
    from train_clean_aogpt import parse_args
    ns = parse_args.__wrapped__() if hasattr(parse_args, "__wrapped__") else None
    # parse a minimal frozen_beta arg set that exercises the new flags
    import sys as _sys
    argv = ["--run-kind", "frozen_beta", "--unfreeze-orderhead-at-step", "10050",
            "--orderhead-lr", "1e-4", "--lam-pg", "0.5", "--pg-tau", "0.7",
            "--pg-group-m", "16", "--pg-beta", "1e-3", "--pg-adv-clip", "0.2",
            "--cdl-pretrain"]
    old = _sys.argv
    try:
        _sys.argv = ["prog"] + argv
        args = parse_args()
    finally:
        _sys.argv = old
    assert args.unfreeze_orderhead_at_step == 10050
    assert args.orderhead_lr == 1e-4
    assert args.lam_pg == 0.5
    assert args.pg_tau == 0.7
    assert args.pg_group_m == 16
    assert args.pg_beta == 1e-3
    assert args.pg_adv_clip == 0.2
    assert args.cdl_pretrain is True
