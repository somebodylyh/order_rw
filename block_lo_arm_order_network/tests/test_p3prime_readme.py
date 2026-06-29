import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_readme_has_redlines_and_verdict():
    p = ROOT / "analyses/p3prime_causal_README.md"
    assert p.exists()
    txt = p.read_text()
    assert "does not attempt to convert B+ into C" in txt
    assert "not treated as load-bearing evidence" in txt
    assert "L0 global physical-order carrier" in txt
