import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_readme_has_redlines_and_verdict():
    p = ROOT / "analyses/p3prime_causal_README.md"
    assert p.exists()
    txt = p.read_text()
    assert "does not attempt to convert B+ into C" in txt
    assert "not treated as load-bearing evidence" in txt
    assert "L0 global physical-order carrier" in txt


def test_readme_locks_partial_support_conclusion():
    txt = (ROOT / "analyses/p3prime_causal_README.md").read_text()
    # corrected post-fix headline: partial causal support, not a clean confirmation
    assert "partial causal support" in txt
    assert "not a clean confirmation" in txt
    # the selectivity caveat is why it is not a clean B+ confirmation
    assert "selectivity caveat" in txt
    assert "not carrier-specific" in txt or "not carrier-selective" in txt
    assert "M=12" in txt
    assert "n_reveals=16" in txt


def test_final_synthesis_separates_evidence_levels():
    p = ROOT / "analyses/final_mechanism_synthesis_README.md"
    assert p.exists()
    txt = p.read_text()
    assert "Canonical phenomenon" in txt
    assert "P2: correlational source decomposition" in txt
    assert "P3′: causal probe" in txt
    assert "mixed/departure" in txt
    assert "content-bound recovery" in txt
