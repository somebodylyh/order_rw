"""Canonical 65-node scan on handoff_overnight ckpts: anchor against the sealed
clean_base result (step0 absent, step10000 L0 strong-pass tau~1.0 under C-D+L)."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.canonical_reanalysis import canonical_scan  # noqa: E402


def test_step10000_has_L0_strong_pass_cdl():
    rows = canonical_scan(str(ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt"),
                          M=8, batch_size=8, sampling_seed=0)
    cdl_strong = [r for r in rows if r["method"] == "C-D+L" and r["gate_status"] == "strong_pass"]
    assert any(r["layer"] == 0 for r in cdl_strong)        # carrier is in L0
    assert max(r["abs_tau"] for r in cdl_strong) > 0.95     # tau ~1.0


def test_step0_is_at_floor():
    rows = canonical_scan(str(ROOT / "runs/handoff_overnight/seed2/ckpt_step0.pt"),
                          M=8, batch_size=8, sampling_seed=0)
    strong = [r for r in rows if r["gate_status"] == "strong_pass"]
    assert len(strong) == 0                                 # order absent at init
