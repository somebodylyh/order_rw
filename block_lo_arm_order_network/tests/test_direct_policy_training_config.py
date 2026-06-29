"""CLI and audit tests for aligned direct attention policies."""

import pathlib
import sys
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_parse_args_accepts_direct_policy(monkeypatch):
    from train_clean_aogpt import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_clean_aogpt.py",
            "--run-kind",
            "direct_policy",
            "--direct-policy",
            "readiness",
        ],
    )

    args = parse_args()

    assert args.run_kind == "direct_policy"
    assert args.direct_policy == "readiness"
    assert args.direct_policy_lambda_dep == 1.0
    assert args.direct_policy_refresh == 10


def test_direct_policy_audit_records_aligned_input_protocol():
    from train_clean_aogpt import direct_policy_audit_metadata

    args = SimpleNamespace(
        run_kind="direct_policy",
        direct_policy="initial_cdl_one_shot",
        direct_policy_lambda_dep=1.0,
        direct_policy_refresh=10,
        batch_mean_probes=4,
    )

    audit = direct_policy_audit_metadata(args)

    assert audit == {
        "policy": "initial_cdl_one_shot",
        "matrix_convention": "B[source,target]",
        "frame": "model-frame strict65",
        "layer_heads": "L0 all-head",
        "probe_aggregation": "mean over 4 probes",
        "head_fusion": "mean over heads",
        "diagonal_handling": "exclude self; strict65 diagonal expected zero",
        "order_direction": "larger score earlier",
        "lambda_dep": 1.0,
        "refresh_every": 10,
        "sequential_cdl_equivalent": False,
    }


def test_direct_policy_audit_is_none_for_other_run_kinds():
    from train_clean_aogpt import direct_policy_audit_metadata

    assert direct_policy_audit_metadata(
        SimpleNamespace(run_kind="frozen_beta")
    ) is None


def test_direct_policy_uses_alpha_schedule():
    from train_clean_aogpt import alpha_for_step

    args = SimpleNamespace(
        run_kind="direct_policy",
        alpha_warmup_start=0,
        alpha_warmup_steps=100,
        alpha_start=0.0,
        alpha_target=0.9,
    )

    assert alpha_for_step(50, 0, args) == 0.45
