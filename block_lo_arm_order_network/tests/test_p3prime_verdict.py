import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analyses.p3prime_causal_verify import classify_p3prime


def _b(verdicts):
    return {"per_head": [{"head": i, "verdict": v} for i, v in enumerate(verdicts)]}


def _c(driven):
    return {"per_head": [{"head": i, "content_driven": d} for i, d in enumerate(driven)]}


def test_confirms_bplus():
    b = _b(["base_map_collapses", "base_map_collapses", "intact"])
    c = _c([True, False, False])
    assert classify_p3prime(b, c) == "B+ confirmed"


def test_departure_when_base_map_does_not_collapse():
    b = _b(["mixed_tau_only", "intact"])
    c = _c([True, True])
    assert classify_p3prime(b, c) == "mixed/departure"


def test_departure_when_no_content_residual():
    b = _b(["base_map_collapses", "base_map_collapses"])
    c = _c([False, False])
    assert classify_p3prime(b, c) == "mixed/departure"
