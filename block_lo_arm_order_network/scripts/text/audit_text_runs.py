"""Re-audit every text run directory under probe_results/.

Reads each run's config.json (when present) and eval_curve.tsv (when present)
to assemble a fresh registry consistent with docs/text_run_registry.md.

For each run, prints:
  run_name  actual_type  run_kind  alpha_start@last  best_ori_l2r  step@best  notes

This is a side-effect-free audit; it does not modify any run output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def alpha_for_run_kind(run_kind: str) -> str:
    """Mirror alpha_for_step's short-circuit logic."""
    if run_kind in {"graph_rw", "graph_rw_bag"}:
        return "scheduled"
    return "0.0"


def _read_eval_curve(path: Path):
    if not path.exists():
        return None
    rows = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) != len(header):
                continue
            try:
                row = {k: (float(v) if k != "step" else int(float(v))) for k, v in zip(header, cols)}
            except ValueError:
                continue
            rows.append(row)
    return rows


def _last_alpha(rows):
    if not rows:
        return None
    for col in ("alpha", "alpha_target"):
        if col in rows[-1]:
            return rows[-1][col]
    return None


def _best_ori_l2r(rows):
    if not rows:
        return None, None
    key = "val_ori_l2r_block"
    cand = [(r["step"], r[key]) for r in rows if key in r]
    if not cand:
        return None, None
    step, val = min(cand, key=lambda x: x[1])
    return val, step


def classify(run_kind: str | None, last_alpha: float | None) -> str:
    if run_kind is None:
        return "unknown"
    if run_kind == "l2r":
        return "l2r_reference"
    if run_kind == "baseline":
        return "random_perm_baseline"
    if run_kind == "random_continuation":
        return "random_continuation"
    if run_kind in {"graph_rw", "graph_rw_bag"}:
        if last_alpha is None or last_alpha == 0.0:
            return "graph_rw_real_inactive"  # rare; would need investigation
        return "graph_rw_real"
    return f"unknown:{run_kind}"


def audit_run(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    eval_path = run_dir / "eval_curve.tsv"

    info: dict = {
        "run_name": run_dir.name,
        "actual_type": "no_config",
        "run_kind": None,
        "last_alpha": None,
        "best_ori_l2r": None,
        "step_at_best": None,
        "notes": "",
    }

    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            info["notes"] = "bad config.json"
            return info
        args = cfg.get("args", {})
        info["run_kind"] = args.get("run_kind")

    rows = _read_eval_curve(eval_path)
    info["last_alpha"] = _last_alpha(rows) if rows else None
    if rows:
        best_v, best_s = _best_ori_l2r(rows)
        info["best_ori_l2r"] = best_v
        info["step_at_best"] = best_s
    else:
        info["notes"] = (info["notes"] + "; " if info["notes"] else "") + "no eval_curve"

    info["actual_type"] = classify(info["run_kind"], info["last_alpha"])
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe-dir", default=str(_REPO_ROOT / "probe_results"))
    p.add_argument("--output", default="-", help="output tsv path or '-' for stdout")
    args = p.parse_args()

    probe_dir = Path(args.probe_dir)
    if not probe_dir.exists():
        sys.exit(f"probe_results dir not found: {probe_dir}")

    rows = []
    for entry in sorted(probe_dir.iterdir()):
        if not entry.is_dir():
            continue
        rows.append(audit_run(entry))

    cols = ("run_name", "actual_type", "run_kind", "last_alpha", "best_ori_l2r", "step_at_best", "notes")
    out = sys.stdout if args.output == "-" else open(args.output, "w")
    print("\t".join(cols), file=out)
    for r in rows:
        print("\t".join(str(r[c]) if r[c] is not None else "" for c in cols), file=out)
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
