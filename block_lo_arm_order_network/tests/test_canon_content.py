"""C4: content-dependence via content_label_permutation_control."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.canonical_reanalysis import content_dependence  # noqa: E402


def test_carrier_is_content_dependent():
    out = content_dependence(str(ROOT / "runs/handoff_overnight/seed2/ckpt_step10000.pt"),
                             carrier_layer=0, carrier_head=2)
    assert out["tau_clean"] > 0.8
    assert out["tau_content_permuted_mean"] < 0.4
    assert out["content_dependent"] is True
