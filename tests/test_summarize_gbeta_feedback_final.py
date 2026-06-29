from pathlib import Path

import pytest


def test_load_eval_curve_numeric_rows(tmp_path):
    from analyses.summarize_gbeta_feedback_final import load_eval_curve

    path = tmp_path / "eval_curve.tsv"
    path.write_text(
        "step\talpha\ttrain_loss\tval_ori_l2r_block\n"
        "100\t0.0\t4.5\t4.1\n"
        "200\t1.0\t3.5\t3.2\n"
    )

    rows = load_eval_curve(path)

    assert rows == [
        {"step": 100, "alpha": 0.0, "train_loss": 4.5, "val_ori_l2r_block": 4.1},
        {"step": 200, "alpha": 1.0, "train_loss": 3.5, "val_ori_l2r_block": 3.2},
    ]


def test_summarize_run_uses_final_row_and_delta_vs_random(tmp_path):
    from analyses.summarize_gbeta_feedback_final import RunSpec, summarize_run

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "eval_curve.tsv").write_text(
        "step\talpha\ttrain_loss\tval_train_objective\tval_ori_l2r_block\tval_model_order\tval_unstructured_order\tval_rw_order\tval_beta_order\tval_cdl_order\tlr\n"
        "60000\t1.0\t3.1\t3.4\t3.3\t4.4\t4.5\t3.8\t3.4\tnan\t1e-4\n"
    )

    row = summarize_run(
        RunSpec("Frozen g_beta", "frozen_gbeta", run_dir, "note"),
        random_final_ori=3.5,
    )

    assert row["label"] == "Frozen g_beta"
    assert row["family"] == "frozen_gbeta"
    assert row["step"] == 60000
    assert row["val_ori_l2r_block"] == 3.3
    assert row["delta_vs_random_ori"] == pytest.approx(-0.2)
    assert row["note"] == "note"
