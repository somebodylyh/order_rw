"""Tests for label-free candidate set builder.

Protocol: selection logic MUST NOT read oracle tau/L2R/prefix fields.
oracle fields may be copied through for post-hoc reporting only.
"""

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "build_label_free_candidate_sets.py"
    spec = importlib.util.spec_from_file_location("build_label_free_candidate_sets", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_rows(n=8, prefix=None):
    prefix = prefix or {}
    return [
        {**prefix, "layer": 0, "head": i, "method": "L", "structure_score": float(10 - i)}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# select_candidates
# ---------------------------------------------------------------------------

def test_topk_structure_selection_does_not_read_oracle_fields():
    mod = _load_module()
    rows = [
        {"layer": 1, "head": 7, "method": "L", "structure_score": 2.0, "posthoc_tau_vs_l2r": -1.0},
        {"layer": 0, "head": 1, "method": "L", "structure_score": 1.0, "posthoc_tau_vs_l2r": 1.0},
    ]
    out = mod.select_candidates(rows, top_k=1, mode="structure")
    assert out[0]["layer"] == 1
    assert out[0]["head"] == 7


def test_random_selection_is_seeded():
    mod = _load_module()
    rows = [{"layer": 0, "head": i, "method": "L", "structure_score": float(i)} for i in range(8)]
    a = mod.select_candidates(rows, top_k=4, mode="random", seed=123)
    b = mod.select_candidates(rows, top_k=4, mode="random", seed=123)
    assert a == b


def test_structure_mode_respects_topk():
    mod = _load_module()
    rows = _make_rows(8)
    out = mod.select_candidates(rows, top_k=4, mode="structure")
    assert len(out) == 4
    assert [r["head"] for r in out] == [0, 1, 2, 3]


def test_structure_sorts_descending():
    mod = _load_module()
    rows = [
        {"layer": 0, "head": 0, "method": "L", "structure_score": 0.1},
        {"layer": 0, "head": 1, "method": "L", "structure_score": 1.0},
        {"layer": 0, "head": 2, "method": "L", "structure_score": 0.5},
    ]
    out = mod.select_candidates(rows, top_k=3, mode="structure")
    scores = [r["structure_score"] for r in out]
    assert scores == sorted(scores, reverse=True)


def test_structure_skips_nonfinite_scores():
    mod = _load_module()
    rows = [
        {"layer": 0, "head": 0, "method": "L", "structure_score": float("nan")},
        {"layer": 0, "head": 1, "method": "L", "structure_score": float("inf")},
        {"layer": 0, "head": 2, "method": "L", "structure_score": 0.5},
        {"layer": 0, "head": 3, "method": "L", "structure_score": -float("inf")},
    ]
    out = mod.select_candidates(rows, top_k=4, mode="structure")
    assert len(out) == 1
    assert out[0]["head"] == 2


def test_random_selection_different_seeds_differ():
    mod = _load_module()
    rows = [{"layer": 0, "head": i, "method": "L", "structure_score": float(i)} for i in range(64)]
    a = mod.select_candidates(rows, top_k=8, mode="random", seed=0)
    b = mod.select_candidates(rows, top_k=8, mode="random", seed=1)
    assert a != b


def test_random_topk_does_not_exceed_pool():
    mod = _load_module()
    rows = _make_rows(3)
    out = mod.select_candidates(rows, top_k=10, mode="random", seed=0)
    assert len(out) == 3


def test_unknown_mode_raises():
    mod = _load_module()
    rows = [{"layer": 0, "head": 0, "method": "L", "structure_score": 1.0}]
    try:
        mod.select_candidates(rows, top_k=1, mode="oracle")
        assert False, "should have raised"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# build_candidate_sets (programmatic wrapper)
# ---------------------------------------------------------------------------

def test_build_candidate_sets_writes_json(tmp_path):
    mod = _load_module()
    rows = [
        {"layer": 0, "head": i, "method": "L", "structure_score": float(10 - i),
         "posthoc_tau_vs_l2r": 0.5}
        for i in range(8)
    ]
    out_dir = tmp_path / "cs"
    mod.build_candidate_sets(
        rows,
        out_dir=str(out_dir),
        sets=[("top4", "structure", 4), ("random4", "random", 4)],
        seed=42,
    )
    top4 = json.loads((out_dir / "candidate_sets" / "top4.json").read_text())
    random4 = json.loads((out_dir / "candidate_sets" / "random4_seed42.json").read_text())
    assert len(top4["candidates"]) == 4
    assert len(random4["candidates"]) == 4
    assert "posthoc_tau_vs_l2r" in top4["candidates"][0]


def test_random_seed_is_appended_to_random_set_names():
    mod = _load_module()
    out_dir = Path("/tmp/test_clabel_free_sets_seeded")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        rows = _make_rows(8)
        mod.build_candidate_sets(
            rows, out_dir=str(out_dir),
            sets=[("random4", "random", 4)], seed=7,
        )
        path = out_dir / "candidate_sets" / "random4_seed7.json"
        assert path.exists()
    finally:
        import shutil
        shutil.rmtree(str(out_dir), ignore_errors=True)
