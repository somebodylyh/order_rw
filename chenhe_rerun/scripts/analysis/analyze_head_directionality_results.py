#!/usr/bin/env python3
"""Offline analysis for continuous raw-attention head directionality."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


EXPORTS = ("with_none", "without_none")
FRAMES = ("current", "original")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def triangle_index(mat: np.ndarray) -> float:
    n = mat.shape[0]
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    l = float(mat[lower].mean())
    u = float(mat[upper].mean())
    return float((l - u) / (l + u + 1e-12))


def load_null_thresholds(report_dir: Path, shuffles: int, seed: int) -> Dict[Tuple[str, str, int, int], Dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    thresholds: Dict[Tuple[str, str, int, int], Dict[str, float]] = {}
    matrix_dir = report_dir / "matrices" / "all_heads_every100"
    for step in range(0, 5001, 500):
        path = matrix_dir / f"step_{step:06d}.npz"
        if not path.exists():
            continue
        npz = np.load(path)
        for export in EXPORTS:
            for frame in FRAMES:
                key = f"{export}_{frame}_exposure"
                if key not in npz:
                    continue
                arr = np.asarray(npz[key], dtype=np.float64)
                n_layer, n_head, n, _ = arr.shape
                for layer in range(n_layer):
                    for head in range(n_head):
                        vals = []
                        mat = arr[layer, head]
                        for _ in range(shuffles):
                            perm = rng.permutation(n)
                            vals.append(abs(triangle_index(mat[perm][:, perm])))
                        vals_np = np.asarray(vals, dtype=np.float64)
                        q95 = float(np.quantile(vals_np, 0.95))
                        mean = float(vals_np.mean())
                        rec = thresholds.setdefault(
                            (export, frame, layer, head),
                            {"max_q95": 0.0, "mean_abs": 0.0, "count": 0.0},
                        )
                        rec["max_q95"] = max(float(rec["max_q95"]), q95)
                        rec["mean_abs"] += mean
                        rec["count"] += 1.0
                        rows.append(
                            {
                                "iter": step,
                                "export_type": export,
                                "frame": frame,
                                "layer": layer,
                                "head": head,
                                "shuffle_abs_t_mean": mean,
                                "shuffle_abs_t_p95": q95,
                                "shuffles": shuffles,
                            }
                        )
    for rec in thresholds.values():
        rec["mean_abs"] = float(rec["mean_abs"]) / max(1.0, float(rec["count"]))
        rec["delta"] = max(0.05, float(rec["max_q95"]))
    write_csv(report_dir / "sanity" / "label_permutation_null" / "results.csv", rows)
    write_json(
        report_dir / "sanity" / "label_permutation_null" / "summary.json",
        {
            "shuffles_per_matrix": shuffles,
            "threshold_rule": "delta = max(0.05, max full-step shuffle |T| p95)",
            "num_thresholds": len(thresholds),
        },
    )
    return thresholds


def group_rows(rows: List[Dict[str, str]], frame: str) -> Dict[Tuple[str, int, int], List[Dict[str, str]]]:
    out: Dict[Tuple[str, int, int], List[Dict[str, str]]] = {}
    for row in rows:
        if row["frame"] != frame:
            continue
        key = (row["export_type"], int(row["layer"]), int(row["head"]))
        out.setdefault(key, []).append(row)
    for value in out.values():
        value.sort(key=lambda r: int(r["iter"]))
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def sign_state(value: float, delta: float) -> int:
    if value > delta:
        return 1
    if value < -delta:
        return -1
    return 0


def classify(values: np.ndarray, pos_frac: np.ndarray, neg_frac: np.ndarray, delta: float) -> Tuple[str, Dict[str, object]]:
    states = np.asarray([sign_state(v, delta) for v in values], dtype=int)
    last = states[-6:] if states.size >= 6 else states
    pos_last = int((last == 1).sum())
    neg_last = int((last == -1).sum())
    final_pos = float(pos_frac[-1])
    final_neg = float(neg_frac[-1])
    sign_changes = int(np.sum((states[1:] * states[:-1]) == -1))
    if pos_last >= 5 and values[-1] > delta and final_pos >= 0.70:
        label = "stable_lower"
    elif neg_last >= 5 and values[-1] < -delta and final_neg >= 0.70:
        label = "stable_upper"
    else:
        early = states[: max(2, min(4, states.size // 2))]
        if states[-1] != 0 and np.any(early == -states[-1]):
            label = "confirmed_flip_candidate"
        elif sign_changes >= 2 or float(np.std(values)) > max(0.10, 2.0 * delta):
            label = "amplitude_oscillation"
        else:
            label = "neutral"
    return label, {
        "states": states.tolist(),
        "sign_changes": sign_changes,
        "final_t": float(values[-1]),
        "mean_t": float(values.mean()),
        "min_t": float(values.min()),
        "max_t": float(values.max()),
        "std_t": float(values.std()),
        "final_positive_fraction": final_pos,
        "final_negative_fraction": final_neg,
        "delta": float(delta),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--shuffles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260619)
    args = parser.parse_args()
    report_dir = args.report_dir
    full_rows = read_csv(report_dir / "metrics" / "full_probe_4096.csv")
    core_rows = read_csv(report_dir / "metrics" / "fixed_core_128.csv")
    thresholds = load_null_thresholds(report_dir, args.shuffles, args.seed)
    full_original = group_rows(full_rows, "original")
    core_original = group_rows(core_rows, "original")
    head_rows: List[Dict[str, object]] = []
    labels_by_export: Dict[Tuple[int, int, str], str] = {}
    for (export, layer, head), rows in sorted(full_original.items()):
        values = np.asarray([float(r["triangle_index_exposure_corrected"]) for r in rows], dtype=np.float64)
        pos_frac = np.asarray([float(r["sample_positive_fraction"]) for r in rows], dtype=np.float64)
        neg_frac = np.asarray([float(r["sample_negative_fraction"]) for r in rows], dtype=np.float64)
        delta = thresholds.get((export, "original", layer, head), {"delta": 0.05})["delta"]
        label, stats = classify(values, pos_frac, neg_frac, float(delta))
        labels_by_export[(layer, head, export)] = label
        core_key = (export, layer, head)
        core_at_full = []
        full_for_core = []
        if core_key in core_original:
            core_map = {int(r["iter"]): float(r["triangle_index_exposure_corrected"]) for r in core_original[core_key]}
            for r in rows:
                it = int(r["iter"])
                if it in core_map:
                    core_at_full.append(core_map[it])
                    full_for_core.append(float(r["triangle_index_exposure_corrected"]))
        head_rows.append(
            {
                "layer": layer,
                "head": head,
                "export_type": export,
                "frame": "original",
                "classification": label,
                "delta": stats["delta"],
                "final_t_exposure": stats["final_t"],
                "mean_t_exposure": stats["mean_t"],
                "min_t_exposure": stats["min_t"],
                "max_t_exposure": stats["max_t"],
                "std_t_exposure": stats["std_t"],
                "sign_changes_full": stats["sign_changes"],
                "final_positive_fraction": stats["final_positive_fraction"],
                "final_negative_fraction": stats["final_negative_fraction"],
                "core_full_pearson_at_full_steps": pearson(np.asarray(core_at_full), np.asarray(full_for_core)),
            }
        )
    comparison_rows: List[Dict[str, object]] = []
    stable_lower: List[str] = []
    stable_upper: List[str] = []
    flips: List[str] = []
    oscillation: List[str] = []
    neutral: List[str] = []
    export_sensitive: List[str] = []
    export_sign_flip: List[str] = []
    for layer in range(4):
        for head in range(8):
            w = labels_by_export.get((layer, head, "with_none"), "missing")
            wo = labels_by_export.get((layer, head, "without_none"), "missing")
            name = f"L{layer}H{head}"
            if w == wo and w == "stable_lower":
                stable_lower.append(name)
                joint = "export_robust_lower"
            elif w == wo and w == "stable_upper":
                stable_upper.append(name)
                joint = "export_robust_upper"
            elif {w, wo} == {"stable_lower", "stable_upper"}:
                export_sign_flip.append(name)
                joint = "export_sign_flip"
            elif w.startswith("stable") or wo.startswith("stable"):
                export_sensitive.append(name)
                joint = "export_sensitive"
            elif "confirmed_flip_candidate" in {w, wo}:
                flips.append(name)
                joint = "flip_candidate"
            elif "amplitude_oscillation" in {w, wo}:
                oscillation.append(name)
                joint = "amplitude_oscillation"
            else:
                neutral.append(name)
                joint = "neutral"
            comparison_rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "with_none_classification": w,
                    "without_none_classification": wo,
                    "joint_classification": joint,
                }
            )
    change_rows: List[Dict[str, object]] = []
    for (export, layer, head), rows in sorted(core_original.items()):
        its = np.asarray([int(r["iter"]) for r in rows], dtype=int)
        vals = np.asarray([float(r["triangle_index_exposure_corrected"]) for r in rows], dtype=np.float64)
        if vals.size < 20:
            continue
        best_idx = 0
        best_score = -1.0
        for idx in range(10, vals.size - 10):
            left = vals[:idx]
            right = vals[idx:]
            score = abs(float(right.mean() - left.mean()))
            if score > best_score:
                best_score = score
                best_idx = idx
        change_rows.append(
            {
                "layer": layer,
                "head": head,
                "export_type": export,
                "frame": "original",
                "method": "max_mean_shift_core128",
                "change_iter": int(its[best_idx]),
                "score_abs_mean_shift": float(best_score),
                "pre_mean": float(vals[:best_idx].mean()),
                "post_mean": float(vals[best_idx:].mean()),
            }
        )
    write_csv(report_dir / "metrics" / "head_summary_by_step.csv", head_rows)
    write_csv(report_dir / "metrics" / "with_without_comparison.csv", comparison_rows)
    write_csv(report_dir / "metrics" / "change_points.csv", change_rows)
    summary_path = report_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update(
        {
            "stable_lower_heads": stable_lower,
            "stable_upper_heads": stable_upper,
            "confirmed_flip_heads": flips,
            "amplitude_oscillation_heads": oscillation,
            "neutral_heads": neutral,
            "export_sensitive_heads": export_sensitive,
            "export_sign_flip_heads": export_sign_flip,
            "label_permutation_null_completed": True,
            "classification_rule": "full 4096 original-frame T_exposure, delta=max(0.05,label-permutation |T| p95); robust requires with_none and without_none same stable class",
        }
    )
    write_json(summary_path, summary)
    write_json(
        report_dir / "analysis_summary.json",
        {
            "stable_lower_heads": stable_lower,
            "stable_upper_heads": stable_upper,
            "confirmed_flip_heads": flips,
            "amplitude_oscillation_heads": oscillation,
            "neutral_heads": neutral,
            "export_sensitive_heads": export_sensitive,
            "export_sign_flip_heads": export_sign_flip,
        },
    )


if __name__ == "__main__":
    main()
