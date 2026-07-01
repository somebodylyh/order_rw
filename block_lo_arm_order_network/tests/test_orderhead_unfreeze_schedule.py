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
