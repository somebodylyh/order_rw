#!/usr/bin/env python3
"""Label-free carrier selector audit from saved 65-node A tensors.

The selector score intentionally does not use physical L2R labels, tau-vs-L2R,
physical-first checks, prefix overlap, or downstream test loss. Those quantities
are only loaded from the oracle JSON for post-hoc evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from typing import Iterable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))

from attn_order_teacher import MODES, teacher_scores  # noqa: E402
from none_separated_block_graph import (  # noqa: E402
    build_none_separated_B,
    content_label_permutation_control,
    discovery_metrics,
    entry_shuffled_control,
    rollout_by_method,
)


def _row_probs(B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    W = np.asarray(B, dtype=np.float64).copy()
    W[:, 0] = 0.0
    W[W < 0.0] = 0.0
    row_sum = W.sum(axis=1, keepdims=True)
    return np.divide(W, row_sum + eps, out=np.zeros_like(W), where=row_sum > eps)


def _entropy(p: np.ndarray) -> float:
    nz = p > 0.0
    return float(-(p[nz] * np.log(p[nz])).sum())


def structure_features(B: np.ndarray, top_k: int) -> dict:
    P = _row_probs(B)
    valid = P.sum(axis=1) > 0.0
    rows = P[valid]
    if rows.size == 0:
        return {
            "row_entropy_mean": float("nan"),
            "row_entropy_norm": float("nan"),
            "topk_mass_mean": float("nan"),
            "sink_argmax_frac": 1.0,
            "col_mass_frac": 1.0,
        }
    ent = np.asarray([_entropy(row) for row in rows], dtype=np.float64)
    max_ent = math.log(max(B.shape[0] - 1, 1))
    k = min(max(int(top_k), 1), rows.shape[1])
    topk = np.sort(rows, axis=1)[:, -k:].sum(axis=1)
    argmax_cols = np.argmax(rows, axis=1)
    _, counts = np.unique(argmax_cols, return_counts=True)
    col_mass = rows.sum(axis=0)
    return {
        "row_entropy_mean": float(ent.mean()),
        "row_entropy_norm": float(ent.mean() / max_ent) if max_ent > 0 else 0.0,
        "topk_mass_mean": float(topk.mean()),
        "sink_argmax_frac": float(counts.max() / rows.shape[0]),
        "col_mass_frac": float(col_mass.max() / col_mass.sum()) if col_mass.sum() > 0 else 1.0,
    }


def rollout_features(B: np.ndarray, method: str, tau_temp: float) -> dict:
    if method not in MODES:
        return {
            "rollout_margin_mean": float("nan"),
            "rollout_margin_min": float("nan"),
            "rollout_entropy_mean": float("nan"),
            "tie_frac": float("nan"),
        }
    N = B.shape[0] - 1
    selected = [0]
    unselected = list(range(1, N + 1))
    last = 0
    margins: list[float] = []
    ents: list[float] = []
    ties = 0
    while unselected:
        scores, candidates = teacher_scores(B, selected, unselected, last, mode=method)
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size >= 2:
            order = np.argsort(-scores)
            margin = float(scores[order[0]] - scores[order[1]])
            if margin <= 1e-12:
                ties += 1
            z = (scores - scores.max()) / max(float(tau_temp), 1e-9)
            probs = np.exp(z)
            probs /= probs.sum()
            ents.append(_entropy(probs))
        else:
            margin = float("inf")
        margins.append(margin)
        node = int(candidates[int(np.argmax(scores))])
        selected.append(node)
        unselected.remove(node)
        last = node
    finite_margins = np.asarray([m for m in margins if np.isfinite(m)], dtype=np.float64)
    return {
        "rollout_margin_mean": float(finite_margins.mean()) if finite_margins.size else float("inf"),
        "rollout_margin_min": float(finite_margins.min()) if finite_margins.size else float("inf"),
        "rollout_entropy_mean": float(np.mean(ents)) if ents else 0.0,
        "tie_frac": float(ties / max(N - 1, 1)),
    }


def destroyed_controls(B: np.ndarray, seeds: Iterable[int]) -> Iterable[np.ndarray]:
    for seed in seeds:
        yield entry_shuffled_control(B, seed=seed)
        yield content_label_permutation_control(B, seed=seed)
        rng = np.random.default_rng(seed)
        out = B.copy()
        content = out[1:, 1:].copy()
        rng.shuffle(content, axis=0)
        out[1:, 1:] = content
        np.fill_diagonal(out, 0.0)
        yield out
        out = B.copy()
        content = out[1:, 1:].copy()
        rng.shuffle(content, axis=1)
        out[1:, 1:] = content
        np.fill_diagonal(out, 0.0)
        yield out


def _zscore(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return [0.0 for _ in values]
    mu = float(arr[finite].mean())
    sd = float(arr[finite].std())
    if sd < 1e-12:
        sd = 1.0
    return [float((x - mu) / sd) if np.isfinite(x) else 0.0 for x in values]


def _oracle_lookup(path: pathlib.Path | None) -> dict[tuple[int, int, str], dict]:
    if path is None:
        return {}
    rows = json.loads(path.read_text())
    return {(int(r["layer"]), int(r["head"]), str(r["method"])): r for r in rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a-npy", required=True, help="Saved A_with_none_lh_mean.npy")
    p.add_argument("--oracle-json", default=None, help="Optional oracle rows for post-hoc reporting only")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--methods", nargs="*", default=["C-D+L", "L"])
    p.add_argument("--control-seeds", type=int, nargs="*", default=list(range(8)))
    p.add_argument("--top-k-mass", type=int, default=4)
    p.add_argument("--tau-temp", type=float, default=1.0)
    p.add_argument("--select-top-k", type=int, default=8)
    p.add_argument("--max-sink-frac", type=float, default=0.35)
    p.add_argument("--max-col-mass-frac", type=float, default=0.35)
    p.add_argument("--max-tie-frac", type=float, default=0.05)
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    A_lh = np.load(args.a_npy)
    oracle = _oracle_lookup(pathlib.Path(args.oracle_json) if args.oracle_json else None)

    raw_rows: list[dict] = []
    L, H = A_lh.shape[:2]
    for layer in range(L):
        for head in range(H):
            B = build_none_separated_B(A_lh[layer, head])
            sf = structure_features(B, args.top_k_mass)
            for method in args.methods:
                rf = rollout_features(B, method, args.tau_temp)
                control_margins = []
                control_ents = []
                control_ties = []
                for Bc in destroyed_controls(B, args.control_seeds):
                    crf = rollout_features(Bc, method, args.tau_temp)
                    control_margins.append(crf["rollout_margin_mean"])
                    control_ents.append(crf["rollout_entropy_mean"])
                    control_ties.append(crf["tie_frac"])
                control_margin = float(np.mean(control_margins))
                control_entropy = float(np.mean(control_ents))
                control_tie = float(np.mean(control_ties))
                row = {
                    "layer": layer,
                    "head": head,
                    "method": method,
                    **sf,
                    **rf,
                    "control_margin_mean": control_margin,
                    "control_entropy_mean": control_entropy,
                    "control_tie_frac": control_tie,
                    "delta_margin": float(rf["rollout_margin_mean"] - control_margin),
                    "delta_entropy": float(control_entropy - rf["rollout_entropy_mean"]),
                    "delta_tie": float(control_tie - rf["tie_frac"]),
                }
                row["passes_lf_gate"] = (
                    row["delta_margin"] > 0.0
                    and row["tie_frac"] <= args.max_tie_frac
                    and row["sink_argmax_frac"] <= args.max_sink_frac
                    and row["col_mass_frac"] <= args.max_col_mass_frac
                )
                post = oracle.get((layer, head, method), {})
                row["posthoc_tau_vs_l2r"] = post.get("tau_vs_l2r")
                row["posthoc_gate_status"] = post.get("gate_status")
                row["posthoc_first_block"] = post.get("first_block")
                raw_rows.append(row)

    for key, out_key in [
        ("delta_margin", "z_delta_margin"),
        ("topk_mass_mean", "z_topk_mass"),
        ("rollout_entropy_mean", "z_rollout_entropy"),
        ("sink_argmax_frac", "z_sink"),
        ("col_mass_frac", "z_col_mass"),
        ("tie_frac", "z_tie"),
    ]:
        for row, z in zip(raw_rows, _zscore([float(r[key]) for r in raw_rows])):
            row[out_key] = z

    for row in raw_rows:
        row["structure_score"] = (
            2.0 * row["topk_mass_mean"]
            - row["row_entropy_norm"]
            - row["sink_argmax_frac"]
            - row["col_mass_frac"]
            - 2.0 * row["tie_frac"]
        )
        row["lf_score"] = (
            row["z_delta_margin"]
            + 0.5 * row["z_topk_mass"]
            - 0.5 * row["z_rollout_entropy"]
            - row["z_sink"]
            - row["z_col_mass"]
            - row["z_tie"]
        )
        if not row["passes_lf_gate"]:
            row["lf_score_gated"] = row["lf_score"] - 1000.0
        else:
            row["lf_score_gated"] = row["lf_score"]

    rows = sorted(raw_rows, key=lambda r: r["lf_score_gated"], reverse=True)
    structure_rows = sorted(raw_rows, key=lambda r: r["structure_score"], reverse=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with (out_dir / "label_free_selector_rows.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "label_free_selector_rows.json").write_text(json.dumps(rows, indent=2))

    top = rows[: max(1, int(args.select_top_k))]
    top_structure = structure_rows[: max(1, int(args.select_top_k))]
    lines = [
        "# Label-Free Selector Audit",
        "",
        "Selection score uses graph-intrinsic features only. Post-hoc tau/gate columns are not used for ranking.",
        "",
        f"- A tensor: `{args.a_npy}`",
        f"- Oracle JSON for post-hoc only: `{args.oracle_json}`",
        f"- Methods: `{', '.join(args.methods)}`",
        f"- Control seeds: `{args.control_seeds}`",
        "",
        "## Top Label-Free Candidates",
        "",
        "| rank | candidate | LF score | gate | delta_margin | entropy | topk_mass | sink | col_mass | posthoc_tau | posthoc_gate |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in enumerate(top, start=1):
        tau = row["posthoc_tau_vs_l2r"]
        tau_s = "" if tau is None else f"{float(tau):.3f}"
        lines.append(
            f"| {i} | L{row['layer']}H{row['head']} {row['method']} | "
            f"{row['lf_score_gated']:.3f} | {row['passes_lf_gate']} | "
            f"{row['delta_margin']:.4g} | {row['row_entropy_norm']:.3f} | "
            f"{row['topk_mass_mean']:.3f} | {row['sink_argmax_frac']:.3f} | "
            f"{row['col_mass_frac']:.3f} | {tau_s} | {row['posthoc_gate_status'] or ''} |"
        )
    lines += [
        "",
        "## Top Structure-Only Candidates",
        "",
        "This ablation uses sharpness/top-k mass/non-sink/tie features only; it does not use destroyed-control margin.",
        "",
        "| rank | candidate | structure score | entropy | topk_mass | sink | col_mass | posthoc_tau | posthoc_gate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in enumerate(top_structure, start=1):
        tau = row["posthoc_tau_vs_l2r"]
        tau_s = "" if tau is None else f"{float(tau):.3f}"
        lines.append(
            f"| {i} | L{row['layer']}H{row['head']} {row['method']} | "
            f"{row['structure_score']:.3f} | {row['row_entropy_norm']:.3f} | "
            f"{row['topk_mass_mean']:.3f} | {row['sink_argmax_frac']:.3f} | "
            f"{row['col_mass_frac']:.3f} | {tau_s} | {row['posthoc_gate_status'] or ''} |"
        )
    strong_hits = [
        row for row in top
        if row.get("posthoc_gate_status") == "strong_pass"
    ]
    structure_strong_hits = [
        row for row in top_structure
        if row.get("posthoc_gate_status") == "strong_pass"
    ]
    lines += [
        "",
        "## Post-Hoc Summary",
        "",
        f"- Top-{len(top)} oracle strong-pass hits: {len(strong_hits)}",
        f"- Structure-only top-{len(top_structure)} oracle strong-pass hits: {len(structure_strong_hits)}",
        f"- Best selected candidate: L{top[0]['layer']}H{top[0]['head']} {top[0]['method']}",
        f"- Best structure-only candidate: L{top_structure[0]['layer']}H{top_structure[0]['head']} {top_structure[0]['method']}",
        "",
        "Reminder: oracle metrics in this section are revealed after selection.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
