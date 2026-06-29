"""Tests for g_beta loss comparison aggregation."""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analyses"))

from summarize_gbeta_loss_comparison import summarize_runs  # noqa: E402


def _write_run(root, name, loss_type, temperature, tau):
    run = root / name
    eval_dir = run / "evaluation"
    eval_dir.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "loss_type": loss_type,
        "rank_kl_temperature": temperature,
    }))
    (run / "training_summary.json").write_text(json.dumps({
        "train_seconds": 12.5,
        "peak_cuda_memory_mb": 321.0,
        "best_val_loss_final": 0.25,
    }))
    (eval_dir / "summary.json").write_text(json.dumps({
        "results": {
            "val": {"normal": {"primary_loss": 0.25}},
            "test": {"normal": {
                "kendall_tau": tau,
                "pairwise_acc": 0.9,
                "prefix8": 0.8,
                "prefix16": 0.85,
            }},
        },
    }))


def test_summarize_runs_writes_json_and_markdown(tmp_path):
    _write_run(tmp_path, "listmle", "listmle", 4.0, 0.98)
    _write_run(tmp_path, "rank_kl_tau4", "rank_kl", 4.0, 0.90)

    summary = summarize_runs(str(tmp_path))

    assert len(summary["runs"]) == 2
    assert (tmp_path / "loss_comparison_summary.json").exists()
    assert (tmp_path / "loss_comparison_summary.md").exists()
    row = next(row for row in summary["runs"] if row["loss_type"] == "listmle")
    assert row["test_tau"] == 0.98
    assert row["train_seconds"] == 12.5
    assert row["peak_cuda_memory_mb"] == 321.0
