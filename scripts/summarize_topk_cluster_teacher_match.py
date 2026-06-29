#!/usr/bin/env python3
"""Summarise top-k clustered teacher-match results into a matrix table.

Scans ``reports/topk_cluster_teacher_match_20260626/``, collects all
match_result.json files, and writes a summary CSV + markdown report.
Post-hoc oracle tau/L2R fields are read from candidate JSONs but clearly
marked as post-hoc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def collect_rows(root: Path) -> list[dict]:
    """Walk root, collect one row per match_result.json."""
    rows = []
    for result_path in sorted(root.rglob("match_result.json")):
        rel = result_path.relative_to(root)
        parts = list(rel.parts)
        # Expected: <ckpt>/<set_name>/match_<teacher_type>/match_result.json
        # parts[0]=ckpt, parts[1]=set_name, parts[2]=match_<teacher_type>
        match_dir = result_path.parent
        teacher_dir = match_dir.parent  # = <ckpt>/<set_name>

        ckpt = parts[0] if len(parts) >= 1 else "?"
        set_name = parts[1] if len(parts) >= 2 else "?"
        teacher_type = parts[2].replace("match_", "") if len(parts) >= 3 else "?"

        # Load match result
        match = json.loads(result_path.read_text())

        # Load cluster report if available
        cluster_report = {}
        cr_path = teacher_dir / "cluster_report.json"
        if cr_path.exists():
            cluster_report = json.loads(cr_path.read_text())

        # Load candidates for post-hoc oracle fields
        cand_dir = root / ckpt / "candidate_sets"
        cand_path = cand_dir / f"{set_name}.json"
        if not cand_path.exists():
            # random sets have seeded names like random4_seed0.json
            cand_files = sorted(cand_dir.glob(f"{set_name}*.json"))
            if cand_files:
                cand_path = cand_files[0]
        posthoc_taus = []
        if cand_path.exists():
            cand_data = json.loads(cand_path.read_text())
            candidates = cand_data.get("candidates", [])
            for c in candidates:
                if "posthoc_tau_vs_l2r" in c:
                    posthoc_taus.append(c["posthoc_tau_vs_l2r"])

        # Cluster composition
        clusters = cluster_report.get("clusters", [])
        cluster_id = None
        cluster_size = None
        cluster_labels = []
        reverse_contamination = None
        if "cluster" in teacher_type:
            # teacher_cluster0, teacher_cluster1, ...
            try:
                cid = int(teacher_type.replace("teacher_cluster", ""))
                cluster_id = cid
                for cl in clusters:
                    if cl.get("cluster_id") == cid:
                        cluster_size = cl.get("size")
                        cluster_labels = cl.get("labels", [])
                        break
            except ValueError:
                pass

        # reverse contamination: fraction of candidates with negative tau
        if posthoc_taus:
            reverse_contamination = sum(1 for t in posthoc_taus if t < 0) / len(posthoc_taus)

        cov = match.get("confidence_coverage", {})
        rows.append({
            "ckpt": ckpt,
            "candidate_set": set_name,
            "teacher_type": teacher_type,
            "cluster_id": cluster_id,
            "selected_candidates": ", ".join(cluster_labels) if cluster_labels else str(cluster_report.get("candidate_labels", [])),
            "cluster_size": cluster_size,
            "n_candidates": cluster_report.get("n_candidates", None),
            "n_clusters": cluster_report.get("n_clusters", None),
            "val_bce": match.get("val_bce"),
            "val_pairwise_acc": match.get("val_pairwise_acc"),
            "gate_entropy": match.get("gate_entropy"),
            "gate_mean": match.get("gate_mean"),
            "cov_0.05": cov.get("cov_0.05"),
            "cov_0.10": cov.get("cov_0.1"),
            "cov_0.20": cov.get("cov_0.2"),
            "posthoc_tau_mean": float(np.mean(posthoc_taus)) if posthoc_taus else None,
            "posthoc_tau_min": float(np.min(posthoc_taus)) if posthoc_taus else None,
            "reverse_contamination_rate": reverse_contamination,
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="reports/topk_cluster_teacher_match_20260626")
    p.add_argument("--out", default=None, help="If set, write CSV to this path")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root {root} does not exist. Run the matrix first.")
        return

    rows = collect_rows(root)
    if not rows:
        print(f"No match_result.json files found under {root}")
        return

    # Print markdown table
    cols = [
        "ckpt", "candidate_set", "teacher_type", "cluster_id",
        "n_candidates", "n_clusters", "cluster_size",
        "val_bce", "val_pairwise_acc", "gate_entropy",
        "cov_0.05", "cov_0.10", "cov_0.20",
        "posthoc_tau_mean", "posthoc_tau_min", "reverse_contamination_rate",
    ]

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    # Header
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(" --- " for _ in cols) + "|"
    print(header)
    print(sep)
    for r in rows:
        print("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")

    # CSV
    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote CSV → {args.out}")

    # Summary narrative
    print("\n## Gate checks")
    for r in rows:
        issues = []
        if r.get("val_pairwise_acc", 0) is None:
            issues.append("NO_ACC")
        cov10 = r.get("cov_0.10")
        if cov10 is not None and isinstance(cov10, (int, float)) and cov10 < 0.40:
            issues.append(f"LOW_COV(cov0.1={cov10:.3f})")
        if r.get("reverse_contamination_rate", 0) is not None and r.get("reverse_contamination_rate", 0) > 0.5:
            issues.append("HIGH_REVERSE")
        flag = " ⚠️ " + ", ".join(issues) if issues else " ✅"
        print(f"  {r['ckpt']}/{r['candidate_set']}/{r['teacher_type']}: acc={fmt(r.get('val_pairwise_acc'))} {flag}")


if __name__ == "__main__":
    main()
