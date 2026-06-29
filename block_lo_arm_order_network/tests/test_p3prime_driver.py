import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.p3prime_causal_verify import run_seed
from analyses.plot_p3prime_causal import plot_p3prime


CKPT_ROOT = ROOT / "runs/handoff_overnight"


@pytest.mark.skipif(not (CKPT_ROOT / "seed2" / "ckpt_step10000.pt").exists(), reason="seed2 checkpoint artifact is absent")
def test_run_seed_writes_outputs_and_plot(tmp_path):
    s = run_seed(2, root=str(CKPT_ROOT), out_dir=str(tmp_path), M=2, n_reveals=2)
    assert (tmp_path / "p3prime.json").exists()
    assert (tmp_path / "p3prime.csv").exists()
    assert s["seed"] == 2 and "verdict" in s
    plot_p3prime(str(tmp_path / "p3prime.json"), str(tmp_path))
    assert (tmp_path / "p3prime_metrics.png").exists()
