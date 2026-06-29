import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import run_phase0
from analyses.plot_p5 import plot_p5

CKPT = str(ROOT / "runs/handoff_overnight/seed123/ckpt_step10000.pt")


def test_run_phase0_writes_outputs(tmp_path):
    r = run_phase0(CKPT, M=8, n_reveals=4, epochs=10, out_dir=str(tmp_path))
    assert (tmp_path / "phase0.json").exists()
    assert "headroom" in r and "gate_pass" in r["headroom"]
    plot_p5(str(tmp_path / "phase0.json"), str(tmp_path))
    assert (tmp_path / "phase0_metrics.png").exists()
