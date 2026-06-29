from pathlib import Path

import pytest


def test_rejects_mixed_seed_baselines():
    from analyses.plot_cdl_gbeta_matched import assert_group_seeds

    base = Path("block_lo_arm_order_network/probe_results")
    with pytest.raises(ValueError, match="seed mismatch"):
        assert_group_seeds(
            2,
            [
                base / "l2r_continuous_seed2",
                base / "random_baseline_b1_headscan_seed124",
            ],
        )


def test_default_groups_are_protocol_consistent():
    from analyses.plot_cdl_gbeta_matched import build_default_groups

    for group in build_default_groups():
        protocol_keys = {run.protocol_key for run in group.runs}
        assert len(protocol_keys) == 1, group.title
        for run in group.runs:
            assert run.protocol_key == next(iter(protocol_keys))


def test_split_groups_keep_cdl_and_gbeta_separate():
    from analyses.plot_cdl_gbeta_matched import build_default_groups, split_groups_by_family

    split = split_groups_by_family(build_default_groups())

    assert set(split) == {"cdl", "gbeta"}
    assert split["cdl"]
    assert split["gbeta"]

    for group in split["cdl"]:
        kinds = {run.run_kind for run in group.runs}
        assert "frozen_beta" not in kinds
        assert "cdl_teacher" in kinds

    for group in split["gbeta"]:
        kinds = {run.run_kind for run in group.runs}
        assert "cdl_teacher" not in kinds
        assert "frozen_beta" in kinds


def test_cdl_seed2_run_is_grouped_by_inherited_protocol_not_args_seed():
    from analyses.plot_cdl_gbeta_matched import build_default_groups

    groups = build_default_groups()
    protocol_123 = next(group for group in groups if group.seed == 123)
    protocol_2 = [group for group in groups if group.seed == 2]

    assert not protocol_2
    assert any(
        run.path.name == "cdl_teacher_seed2_from10k_l0h2" and run.seed == 2
        for run in protocol_123.runs
    )
