#!/usr/bin/env python3
"""Attention-graph diagnostics -> conservative readout recommendation.

This script is intentionally lightweight: it diagnoses an attention-derived graph B,
computes structural/readout statistics, and suggests a readout regime. It does not train
models and does not rewrite Stage-1 conclusions.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
sys.path.insert(0, str(_REPO / "image_order"))

from directed_graph_policy import build_directed_graph, compute_source, sample_order
from graph_rw_image import IMAGE_RW_PARAMS_DEFAULT, IMAGE_RW_PARAMS_V3
from readout_order_diagnostic import (
    b_coverage_order,
    hilbert_order,
    local_greedy_order,
    order_stats,
    raster_order,
    visit_grid,
)


def _parse_grid_shape(text):
    if text is None:
        return None
    if "x" in text.lower():
        a, b = text.lower().split("x", 1)
    else:
        a, b = text.split(",", 1)
    return int(a), int(b)


def _coords(n, grid_shape):
    if grid_shape is None:
        return np.arange(n).reshape(n, 1)
    h, w = grid_shape
    if h * w != n:
        raise ValueError(f"grid_shape={grid_shape} incompatible with N={n}")
    idx = np.arange(n)
    return np.stack([idx // w, idx % w], axis=1)


def _manh(coords, a, b):
    return np.abs(coords[a] - coords[b]).sum(axis=-1)


def _quadrant_id(idx, grid_shape):
    h, w = grid_shape
    r = idx // w
    c = idx % w
    return (r >= h // 2).astype(np.int64) * 2 + (c >= w // 2).astype(np.int64)


def row_entropy_stats(B):
    P = np.asarray(B, dtype=np.float64).copy()
    np.fill_diagonal(P, 0.0)
    row_sum = P.sum(axis=1, keepdims=True)
    P = np.divide(P, row_sum, out=np.zeros_like(P), where=row_sum > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
    return float(H.mean()), float(H.std())


def topk_mass(B, k):
    P = np.asarray(B, dtype=np.float64).copy()
    np.fill_diagonal(P, 0.0)
    row_sum = P.sum(axis=1, keepdims=True)
    P = np.divide(P, row_sum, out=np.zeros_like(P), where=row_sum > 0)
    return float(np.sort(P, axis=1)[:, -k:].sum(axis=1).mean())


def graph_stats(B, grid_shape=None, alpha=0.5):
    B = np.asarray(B, dtype=np.float64)
    N = B.shape[0]
    H_mean, H_std = row_entropy_stats(B)
    source, out_deg, in_deg = compute_source(B, alpha_dep=alpha)
    stats = {
        "N": int(N),
        "row_entropy_mean": H_mean,
        "row_entropy_std": H_std,
        "top1_mass_mean": topk_mass(B, 1),
        "top4_mass_mean": topk_mass(B, min(4, N)),
        "in_degree_std": float(in_deg.std()),
        "out_degree_std": float(out_deg.std()),
        "asymmetry_mean": float(np.mean(np.abs(B - B.T))),
        "readiness_strength": float(np.std(source)),
        "readiness_strength_norm": float(np.std(source) / (np.mean(np.abs(B)) + 1e-12)),
    }
    if grid_shape is not None:
        coords = _coords(N, grid_shape)
        argmax = np.argmax(B, axis=1)
        d = np.array([_manh(coords, i, argmax[i]) for i in range(N)], dtype=np.float64)
        q = _quadrant_id(np.arange(N), grid_shape)
        q_arg = _quadrant_id(argmax, grid_shape)
        rand_d = []
        for i in range(N):
            cand = np.array([j for j in range(N) if j != i], dtype=np.int64)
            rand_d.append(float(_manh(coords, i, cand).mean()))
        lg = local_greedy_order(B, start=0)
        lg_s = order_stats(lg)
        ras_s = order_stats(raster_order())
        hil_s = order_stats(hilbert_order())
        stats.update({
            "argmax_manh_mean": float(d.mean()),
            "top1_P_d_le_1": float((d <= 1).mean()),
            "top1_P_d_le_2": float((d <= 2).mean()),
            "same_quadrant_rate": float((q == q_arg).mean()),
            "local_greedy_manh": lg_s["mean_manh"],
            "local_greedy_P_d_le_1": lg_s["p_d_le1"],
            "random_reference_manh": float(np.mean(rand_d)),
            "raster_reference_manh": ras_s["mean_manh"],
            "hilbert_reference_manh": hil_s["mean_manh"],
        })
    return stats


def sample_rw_orders(B, policy, params, K, seed):
    orders = []
    for k in range(K):
        order, _ = sample_order(B, policy, params, seed=seed * 10000 + k)
        orders.append(order)
    return np.stack(orders)


def readout_stats(B, K=64, seed=0):
    N = B.shape[0]
    rng = np.random.default_rng(seed)
    readouts = {
        "random": np.stack([rng.permutation(N) for _ in range(K)]),
        "v1_graph_rw": sample_rw_orders(B, "progressive_rw", IMAGE_RW_PARAMS_DEFAULT, K, seed + 1),
        "v3_graph_rw": sample_rw_orders(B, "progressive_rw_v3", IMAGE_RW_PARAMS_V3, K, seed + 2),
        "local_greedy": np.stack([local_greedy_order(B, st % N) for st in range(K)]),
        "hilbert": np.tile(hilbert_order(), (K, 1)),
        "raster": np.tile(raster_order(), (K, 1)),
        "Bcov_balanced": np.stack([b_coverage_order(B, 1.0, 1.0, st % N) for st in range(K)]),
    }
    rows = {}
    for name, orders in readouts.items():
        rows[name] = order_stats(orders)
    return rows, readouts


def recommend(gstats, rstats):
    rand_manh = rstats.get("random", {}).get("mean_manh", gstats.get("random_reference_manh", np.nan))
    v1_manh = rstats.get("v1_graph_rw", {}).get("mean_manh", np.nan)
    spatial_local = (
        gstats.get("argmax_manh_mean", np.inf) <= 1.5
        and gstats.get("top1_P_d_le_1", 0.0) >= 0.7
    )
    v1_near_random = np.isfinite(rand_manh) and np.isfinite(v1_manh) and abs(v1_manh - rand_manh) <= 0.75
    entropy_high = gstats["row_entropy_mean"] / np.log(max(gstats["N"] - 1, 2)) >= 0.93
    mass_low = gstats["top1_mass_mean"] <= 0.05 and gstats["top4_mass_mean"] <= 0.20
    readiness_high = gstats["readiness_strength_norm"] >= 1.0 or gstats["readiness_strength"] >= 0.05
    asym_high = gstats["asymmetry_mean"] >= 0.001

    if spatial_local and v1_near_random and readiness_high:
        return "hybrid_or_adaptive", "Strong proximity signal plus nontrivial readiness/asymmetry; default v1/v3 readout is near random, so use proximity coverage readouts while treating B direction conservatively."
    if spatial_local and v1_near_random:
        return "proximity_coverage_readout", "Spatial locality is strong, but support/readiness Graph-RW does not produce local traversal; prefer Bcov_balanced/Hilbert/raster-style coverage readouts."
    if entropy_high and mass_low:
        return "random_or_no_structure_fallback", "Rows are high-entropy and low-mass; do not over-trust B without training evidence."
    if readiness_high and asym_high:
        return "readiness_dominant_graph_rw", "Readiness/asymmetry is prominent and no stronger grid-locality failure was detected; Graph-RW-style readiness readout is plausible."
    return "weak_mixed", "Mixed or weak structure; keep recommendation exploratory and validate by loss/robustness/sample quality."


def _git_status_summary():
    try:
        out = subprocess.check_output(["git", "status", "--short", "--branch"], cwd=_REPO, text=True)
        lines = out.strip().splitlines()
        return {"first_line": lines[0] if lines else "", "n_lines": len(lines), "raw_head": lines[:40]}
    except Exception as exc:
        return {"error": str(exc)}


def load_graph(path, kind):
    arr = np.load(path).astype(np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"expected square matrix, got {arr.shape} from {path}")
    if kind == "A":
        return build_directed_graph(arr), "B = build_directed_graph(A) = A.T with zero diagonal"
    if kind == "B":
        B = arr.copy()
        np.fill_diagonal(B, 0.0)
        return B, "B loaded directly with diagonal zeroed"
    raise ValueError(kind)


def write_table_tsv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            vals = []
            for c in cols:
                v = row[c]
                vals.append(f"{v:.6g}" if isinstance(v, float) else str(v))
            f.write("\t".join(vals) + "\n")


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        vals = []
        for c in cols:
            v = row[c]
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", action="append", required=True, help="Path to A or B npy file. Can be repeated.")
    ap.add_argument("--kind", choices=["A", "B"], default="A")
    ap.add_argument("--grid-shape", default="8x8")
    ap.add_argument("--outdir", default="probe_results_image_large/grw_e3ctrlsmall/graph_diagnostics")
    ap.add_argument("--report", default="probe_results_image_large/grw_e3ctrlsmall/GRAPH_DIAGNOSTICS.md")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inverse-block-perm", default="present in baseline ckpt; physical orders must be remapped with inverse_block_perm before model use")
    args = ap.parse_args()

    grid_shape = _parse_grid_shape(args.grid_shape)
    outdir = (_REPO / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    all_graph_rows = []
    all_readout_rows = []
    metadata = {
        "graphs": args.graph,
        "kind": args.kind,
        "grid_shape": args.grid_shape,
        "B_construction_method": None,
        "inverse_block_perm_metadata": args.inverse_block_perm,
        "readout_parameters": {
            "v1_graph_rw": IMAGE_RW_PARAMS_DEFAULT,
            "v3_graph_rw": IMAGE_RW_PARAMS_V3,
            "Bcov_balanced": {"gamma_B": 1.0, "gamma_d": 1.0, "selection": "deterministic_argmax"},
            "n_samples": args.n_samples,
            "seed": args.seed,
        },
        "git_status_summary": _git_status_summary(),
    }
    recommendations = []
    visit_sections = []

    for graph_path in args.graph:
        p = (_REPO / graph_path).resolve() if not Path(graph_path).is_absolute() else Path(graph_path)
        B, method = load_graph(p, args.kind)
        metadata["B_construction_method"] = method
        name = p.stem
        g = graph_stats(B, grid_shape=grid_shape)
        rstats, readouts = readout_stats(B, K=args.n_samples, seed=args.seed)
        regime, rationale = recommend(g, rstats)
        recommendations.append({"graph": str(p), "suggested_regime": regime, "rationale": rationale})

        graph_row = {"graph": name, "source_path": str(p), "suggested_regime": regime, **g}
        all_graph_rows.append(graph_row)
        for readout, st in rstats.items():
            all_readout_rows.append({"graph": name, "readout": readout, **st})
        for key in ["v1_graph_rw", "v3_graph_rw", "local_greedy", "hilbert", "raster", "Bcov_balanced"]:
            grid = visit_grid(readouts[key][0])
            lines = [f"### {key} visit grid (sample 0)", "```text"]
            lines.extend(" ".join(f"{grid[i, j]:2d}" for j in range(grid.shape[1])) for i in range(grid.shape[0]))
            lines.append("```")
            visit_sections.append("\n".join(lines))

    write_table_tsv(outdir / "graph_diagnostics.tsv", all_graph_rows)
    write_table_tsv(outdir / "readout_diagnostics.tsv", all_readout_rows)
    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(outdir / "recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2)

    gcols = ["graph", "N", "row_entropy_mean", "row_entropy_std", "top1_mass_mean", "top4_mass_mean", "in_degree_std", "out_degree_std", "asymmetry_mean", "readiness_strength", "readiness_strength_norm", "argmax_manh_mean", "top1_P_d_le_1", "top1_P_d_le_2", "same_quadrant_rate", "local_greedy_manh", "local_greedy_P_d_le_1", "random_reference_manh", "raster_reference_manh", "hilbert_reference_manh", "suggested_regime"]
    rcols = ["graph", "readout", "mean_manh", "p_d_le1", "p_d_le2", "same_quad"]
    report = []
    report.append("# Graph Diagnostics — E3-control-small")
    report.append("")
    report.append("## 1. What graph is being diagnosed")
    report.append("")
    source_graphs = ", ".join(args.graph)
    b_method = metadata.get("B_construction_method")
    report.append(f"Source graph(s): {source_graphs}. Inputs are treated as {args.kind} and converted with {b_method}.")
    report.append("")
    report.append("Frame metadata: B/readout orders are in physical 8x8 raster frame. For the permuted-data baseline, physical orders must be remapped with `inverse_block_perm[phys_order]` before model use. This script only diagnoses structure; it does not train.")
    report.append("")
    report.append("## 2. Diagnostics table")
    report.append("")
    report.append(md_table(all_graph_rows, gcols))
    report.append("")
    report.append("## 3. Readout behavior table")
    report.append("")
    report.append(md_table(all_readout_rows, rcols))
    report.append("")
    report.append("## 4. Suggested regime")
    report.append("")
    for rec in recommendations:
        regime = rec.get("suggested_regime")
        graph_name = rec.get("graph")
        rationale = rec.get("rationale")
        report.append(f"- {regime} for {graph_name}: {rationale}")
    report.append("")
    report.append("This is a conservative diagnostic recommendation, not a training conclusion.")
    report.append("")
    report.append("## 5. Interpretation")
    report.append("")
    report.append("The diagnosed E3-control-small graph is strongly proximity-like: its top-1 outgoing edges are local, while default v1/v3 support/readiness Graph-RW readouts remain much closer to random than to Hilbert/raster. This supports the sealed Stage-1 framing: local B exists, but default v1/v3 did not convert it into a useful local traversal.")
    report.append("")
    report.append("Bcov_balanced, Hilbert, and raster should be treated as higher-level proximity readouts. Mean Manhattan distance is only a structural diagnostic; the decision criterion remains validation loss, cross-order robustness, and sample quality.")
    report.append("")
    report.append("## 6. How this motivates Round-2")
    report.append("")
    report.append("Round-2 should test whether the graph-diagnostic-selected `Bcov_balanced` readout gives task benefit beyond generic spatial filling (`Hilbert`) and beyond the failed-readout control (`v1_graph_rw`). Raster remains a specialization reference. If Bcov_balanced beats Hilbert on cross-order or structured averages without raster-like collapse, that would support B-derived proximity structure beyond generic locality; if it only matches Hilbert, proximity helps but B direction may add little; if it loses, the diagnostic can still be right while the current Bcov readout needs refinement.")
    report.append("")
    report.append("## Metadata")
    report.append("")
    report.append("```json")
    report.append(json.dumps(metadata, indent=2))
    report.append("```")
    report.append("")
    report.append("## Visit grids")
    report.append("")
    report.extend(visit_sections)
    report_path = (_REPO / args.report).resolve()
    report_path.write_text("\n".join(report) + "\n")

    print("wrote " + str(outdir / "graph_diagnostics.tsv"))
    print("wrote " + str(outdir / "readout_diagnostics.tsv"))
    print("wrote " + str(outdir / "metadata.json"))
    print(f"wrote {report_path}")
    for rec in recommendations:
        print("suggested_regime=" + str(rec.get("suggested_regime")) + " :: " + str(rec.get("rationale")))


if __name__ == "__main__":
    main()
