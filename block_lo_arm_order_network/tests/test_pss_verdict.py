"""Per-seed source verdict (B / B+ / C / mixed) from content-variance vs noise floor + R2."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.physical_signal_source import classify_source  # noqa: E402


def test_classify_source_rules():
    # content_variance ~ noise floor (almost all variance is sampling noise) + high R2 -> B
    assert classify_source(content_variance=0.001, noise_floor=0.05, r2=0.95) == "B"
    # content_variance >> floor + low R2 -> C
    assert classify_source(content_variance=0.2, noise_floor=0.02, r2=0.3) == "C"
    # high R2 but content_variance clearly above floor -> B+
    assert classify_source(content_variance=0.06, noise_floor=0.02, r2=0.85) == "B+"
    # ambiguous (R2 in [0.8,0.92) but variance near floor) -> mixed
    assert classify_source(content_variance=0.005, noise_floor=0.05, r2=0.85) == "mixed"
