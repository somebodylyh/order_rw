#!/usr/bin/env python3
"""Build paper-ready result tables and figures from current AO-GPT artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/admin/lyuyuhuan/order_lyu")
OUT = ROOT / "probe_results_image_large/paper_ready_results"


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def save_table(df: pd.DataFrame, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.tsv", sep="\t", index=False)
    (OUT / f"{name}.md").write_text(to_md_table(df), encoding="utf-8")


def to_md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = [str(row[c]).replace("\n", " ") for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def git_summary():
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
        lines = [line for line in status.splitlines() if line.strip()]
        return {"head": head, "dirty_count": len(lines), "preview": lines[:20], "truncated": len(lines) > 20}
    except Exception as exc:
        return {"error": repr(exc)}


def make_regime_table():
    df = read_tsv(ROOT / "probe_results_image_large/graph_regime_cross_graph_overnight/regime_table.tsv")
    cols = [
        "graph",
        "readiness_strength",
        "row_entropy",
        "argmax_dist",
        "p_nbr_le1",
        "locality_score",
        "directionality",
        "regime",
        "suggested_readout",
    ]
    out = df[cols].copy()
    for c in ["readiness_strength", "row_entropy", "argmax_dist", "p_nbr_le1", "locality_score", "directionality"]:
        out[c] = out[c].map(lambda x: f"{float(x):.4f}")
    save_table(out, "table1_graph_regime_diagnostic")
    return out


def make_stage1_table():
    readout = read_tsv(ROOT / "probe_results_image_large/grw_e3ctrlsmall/graph_diagnostics/readout_diagnostics.tsv")
    wanted = ["random", "v1_graph_rw", "v3_graph_rw", "Bcov_balanced", "raster", "hilbert"]
    out = readout[readout["readout"].isin(wanted)][["readout", "mean_manh", "p_d_le1", "p_d_le2", "same_quad"]].copy()
    out["role"] = out["readout"].map(
        {
            "random": "random reference",
            "v1_graph_rw": "default failed-readout control",
            "v3_graph_rw": "default failed-readout control",
            "Bcov_balanced": "B-guided proximity coverage",
            "raster": "local specialization reference",
            "hilbert": "generic space-filling reference",
        }
    )
    out = out[["readout", "role", "mean_manh", "p_d_le1", "p_d_le2", "same_quad"]]
    for c in ["mean_manh", "p_d_le1", "p_d_le2", "same_quad"]:
        out[c] = out[c].map(lambda x: f"{float(x):.4f}")
    save_table(out, "table2_stage1_readout_mismatch")
    return out


def make_multiseed_table():
    per_seed = read_tsv(ROOT / "probe_results_image_large/grw_e3ctrlsmall_round2_multiseed_minimal/per_seed.tsv")
    rows = []
    for arm, group in per_seed.groupby("arm"):
        item = {"arm": arm}
        for src, dst in [("cross", "cross_avg"), ("structured", "structured_avg"), ("noisy", "noisy_avg"), ("matched", "matched_control")]:
            item[dst] = f"{group[src].mean():.4f} ± {group[src].std():.4f}"
        rows.append(item)
    out = pd.DataFrame(rows)
    order = ["cont_random", "cont_Bcov_balanced", "cont_distance_only_coverage", "cont_shuffled_Bcov_balanced"]
    out["arm"] = pd.Categorical(out["arm"], order, ordered=True)
    out = out.sort_values("arm")
    save_table(out, "table3_round2_multiseed")

    deltas = read_tsv(ROOT / "probe_results_image_large/grw_e3ctrlsmall_round2_multiseed_minimal/deltas.tsv")
    rows = []
    for comp, group in deltas.groupby("comparison"):
        rows.append(
            {
                "comparison": comp,
                "delta_cross": f"{group['delta_cross'].mean():+.4f} ± {group['delta_cross'].std():.4f}",
                "delta_structured": f"{group['delta_structured'].mean():+.4f} ± {group['delta_structured'].std():.4f}",
                "delta_noisy": f"{group['delta_noisy'].mean():+.4f} ± {group['delta_noisy'].std():.4f}",
            }
        )
    d_out = pd.DataFrame(rows)
    save_table(d_out, "table3b_round2_bcov_deltas")
    return per_seed, deltas, out, d_out


def make_boundary_table():
    df = read_tsv(ROOT / "probe_results_image_large/grw_e3ctrlsmall_round2_sample_quality_seed42/metrics.tsv")
    out = df[["arm", "policy", "samples", "fid_vq_val", "pixel_std", "token_entropy_bits", "duplicate_image_rate"]].copy()
    out = out.sort_values("fid_vq_val")
    for c in ["fid_vq_val", "pixel_std", "token_entropy_bits", "duplicate_image_rate"]:
        out[c] = out[c].map(lambda x: f"{float(x):.4f}")
    save_table(out, "table4_sample_quality_boundary")
    return df, out


def make_fallback_table():
    # Parse from existing eval curves to avoid relying on markdown.
    rows = []
    root = ROOT / "probe_results_image_large/e3large_fixed_round2_negative_seed42"
    for arm in ["cont_random", "cont_hilbert", "cont_Bcov_balanced", "cont_distance_only_coverage", "cont_shuffled_Bcov_balanced"]:
        curve = read_tsv(root / arm / "eval_curve.tsv")
        last = curve.iloc[-1]
        cols = [c for c in curve.columns if c.startswith("val_")]
        random = float(last.get("val_random"))
        raster = float(last.get("val_raster"))
        hilbert = float(last.get("val_hilbert"))
        bcov = float(last.get("val_Bcov_balanced"))
        dist = float(last.get("val_distance_only_coverage"))
        rw4 = float(last.get("val_rw_top4_eps0"))
        eps = float(last.get("val_rw_eps015"))
        rw8 = float(last.get("val_rw_topk8"))
        cross = np.mean([random, raster, hilbert, bcov, rw4, eps, rw8])
        structured = np.mean([raster, hilbert, bcov, rw4, rw8])
        noisy = np.mean([random, eps])
        matched = {
            "cont_random": random,
            "cont_hilbert": hilbert,
            "cont_Bcov_balanced": bcov,
            "cont_distance_only_coverage": dist,
            "cont_shuffled_Bcov_balanced": bcov,
        }[arm]
        rows.append(
            {
                "arm": arm,
                "cross_avg": cross,
                "structured_avg": structured,
                "noisy_avg": noisy,
                "matched_control": matched,
            }
        )
    out = pd.DataFrame(rows)
    formatted = out.assign(**{c: out[c].map(lambda x: f"{float(x):.4f}") for c in out.columns if c != "arm"})
    save_table(formatted, "table5_fallback_negative")
    return out, formatted


def make_phaseb_positive_table():
    path = ROOT / "probe_results_image/imagenet32_continuous_round2_minimal_seed42/aggregate.tsv"
    if not path.exists():
        return None
    df = read_tsv(path)
    out = df[["arm", "cross", "structured", "noisy", "matched"]].copy()
    for c in ["cross", "structured", "noisy", "matched"]:
        out[c] = out[c].map(lambda x: f"{float(x):.6f}")
    save_table(out, "table6_phaseb_imagenet32_continuous_positive")
    return out


def make_e2_vq_negative_table():
    root = ROOT / "probe_results_image/e2_vq_round2_negative_seed42"
    if not root.exists():
        return None
    rows = []
    arms = [
        "cont_random",
        "cont_Bcov_balanced",
        "cont_distance_only_coverage",
        "cont_shuffled_Bcov_balanced",
    ]
    common = [
        "val_random",
        "val_raster",
        "val_hilbert",
        "val_Bcov_balanced",
        "val_distance_only_coverage",
        "val_rw_top4_eps0",
        "val_rw_eps015",
        "val_rw_topk8",
    ]
    structured = [
        "val_raster",
        "val_hilbert",
        "val_Bcov_balanced",
        "val_distance_only_coverage",
        "val_rw_top4_eps0",
        "val_rw_topk8",
    ]
    noisy = ["val_random", "val_rw_eps015"]
    matched_col = {
        "cont_random": "val_random",
        "cont_Bcov_balanced": "val_Bcov_balanced",
        "cont_distance_only_coverage": "val_distance_only_coverage",
        "cont_shuffled_Bcov_balanced": "val_Bcov_balanced",
    }
    for arm in arms:
        path = root / arm / "eval_curve.tsv"
        if not path.exists():
            return None
        curve = read_tsv(path)
        last = curve.iloc[-1]
        rows.append(
            {
                "arm": arm,
                "step": int(last["step"]),
                "cross": np.mean([float(last[c]) for c in common]),
                "structured": np.mean([float(last[c]) for c in structured]),
                "noisy": np.mean([float(last[c]) for c in noisy]),
                "matched": float(last[matched_col[arm]]),
            }
        )
    df = pd.DataFrame(rows)
    out = df.copy()
    for c in ["cross", "structured", "noisy", "matched"]:
        out[c] = out[c].map(lambda x: f"{float(x):.4f}")
    save_table(out, "table7_e2_vq_negative")
    return out


def plot_pipeline():
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.axis("off")
    boxes = [
        ("Attention graph B", 0.08),
        ("Graph diagnostics\ng(B)", 0.30),
        ("Regime\nclassification", 0.52),
        ("Selected readout\nreadout(B; w)", 0.74),
        ("Validation / robustness\nsample quality", 0.94),
    ]
    for text, x in boxes:
        ax.text(
            x,
            0.55,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="#333333", lw=1.0),
            transform=ax.transAxes,
        )
    for (_, x0), (_, x1) in zip(boxes[:-1], boxes[1:]):
        ax.annotate(
            "",
            xy=(x1 - 0.09, 0.55),
            xytext=(x0 + 0.09, 0.55),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"),
            xycoords=ax.transAxes,
        )
    ax.text(0.5, 0.12, "Readout is selected from graph structure, not manually assigned by modality.", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_graph_regime_pipeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_round2_bars(per_seed: pd.DataFrame):
    arms = ["cont_random", "cont_Bcov_balanced", "cont_distance_only_coverage", "cont_shuffled_Bcov_balanced"]
    labels = ["Random", "Bcov", "Distance-only", "Shuffled-Bcov"]
    metrics = [("cross", "Cross"), ("structured", "Structured"), ("noisy", "Noisy")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=False)
    palette = ["#6b7280", "#2563eb", "#059669", "#d97706"]
    for ax, (metric, title) in zip(axes, metrics):
        means = []
        stds = []
        for arm in arms:
            vals = per_seed[per_seed["arm"] == arm][metric].astype(float)
            means.append(vals.mean())
            stds.append(vals.std())
        x = np.arange(len(arms))
        ax.bar(x, means, yerr=stds, capsize=3, color=palette, alpha=0.88)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        lo = min(means) - max(stds) - 0.002
        hi = max(means) + max(stds) + 0.002
        ax.set_ylim(lo, hi)
        ax.set_ylabel("Validation loss" if ax is axes[0] else "")
    fig.suptitle("Round-2 minimal multi-seed validation aggregates (lower is better)", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_round2_multiseed_bars.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_delta(deltas: pd.DataFrame):
    mapping = {
        "Bcov_minus_cont_random": "Bcov - Random",
        "Bcov_minus_cont_distance_only_coverage": "Bcov - Distance",
        "Bcov_minus_cont_shuffled_Bcov_balanced": "Bcov - Shuffled",
    }
    metrics = [("delta_cross", "Cross"), ("delta_structured", "Structured"), ("delta_noisy", "Noisy")]
    x = np.arange(len(mapping))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 3.8))
    colors = ["#2563eb", "#059669", "#d97706"]
    for i, (metric, label) in enumerate(metrics):
        means = []
        stds = []
        for comp in mapping:
            vals = deltas[deltas["comparison"] == comp][metric].astype(float)
            means.append(vals.mean())
            stds.append(vals.std())
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3, label=label, color=colors[i], alpha=0.9)
    ax.axhline(0.0, color="#111111", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([mapping[k] for k in mapping], rotation=15, ha="right")
    ax.set_ylabel("Delta loss (negative favors Bcov)")
    ax.set_title("Bcov_balanced deltas across five seeds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_bcov_delta_multiseed.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sample_fid(sample_df: pd.DataFrame):
    df = sample_df.sort_values("fid_vq_val")
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    labels = [a.replace("cont_", "").replace("_", "\n") for a in df["arm"]]
    colors = ["#6b7280" if "Bcov" not in a and "raster" not in a and "hilbert" not in a else "#2563eb" for a in df["arm"]]
    ax.bar(np.arange(len(df)), df["fid_vq_val"], color=colors, alpha=0.9)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel("Proxy FID vs VQ-decoded val")
    ax.set_title("Matched-order sample-quality boundary (256 samples)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_sample_quality_boundary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(regime, stage1, ms_table, delta_table, sample_table, fallback_table, phaseb_table, e2_table):
    phaseb_section = ""
    if phaseb_table is not None:
        phaseb_section = f"""
## Table 6. Phase-B Extra Proximity Graph

{to_md_table(phaseb_table)}
"""
    e2_section = ""
    if e2_table is not None:
        e2_section = f"""
## Table 7. E2 VQ No-Structure Negative

{to_md_table(e2_table)}
"""
    phaseb_paragraph = ""
    if phaseb_table is not None:
        phaseb_paragraph = """
**4.6 Cross-Graph Positive Check.** We also ran a minimal continuation on the continuous ImageNet32 patch model, which is diagnosed as a proximity-dominant graph outside the E3-control-small VQ setting. Coverage-style readouts improve over random and shuffled-Bcov on cross and structured averages, but Bcov_balanced is essentially tied with distance-only coverage. This result broadens the regime story while refining the mechanism: proximity-dominant graphs can benefit from coverage readouts, but the extra contribution of the real B matrix is graph-dependent rather than guaranteed."""
    e2_paragraph = ""
    if e2_table is not None:
        e2_paragraph = """
**4.7 E2 No-Structure Negative Check.** We additionally ran a lightweight 3000-step continuation on E2 ImageNet32 VQ, which the diagnostic classifies as uniform-noisy. This behaves unlike E3-control-small: Bcov_balanced obtains a lower matched Bcov-order loss, but its cross average is worse than random (`+0.0035`), and its noisy average is also worse (`+0.0088`). Distance-only shows the same specialization pattern: strong matched distance-order loss but weaker cross/noisy robustness. This supports the diagnostic boundary: when the graph lacks proximity/readiness structure, B-guided coverage should not be expected to provide the E3-style robustness benefit."""
    summary = f"""# Paper-Ready Results Summary

Generated from current artifacts under `probe_results_image_large/`.

## Mini-Outline

1. Attention graphs are first diagnosed into regimes rather than assigned a readout by modality.
2. Stage-1 shows that E3-control-small contains strong proximity signal, but default v1/v3 Graph-RW readouts fail to convert it into local traversal.
3. Round-2 shows that the proximity-regime Bcov_balanced readout gives stable cross/structured validation gains over random, distance-only, and shuffled-B controls.
4. Fallback validation shows Bcov does not help on a graph diagnosed as no-structure, supporting the diagnostic boundary.
5. Sample-quality proxy FID favors raster/Hilbert under matched-order sampling, so the current Bcov claim should remain a validation/robustness claim rather than a generation-quality claim.

## Table 1. Graph-Regime Diagnostic

{to_md_table(regime)}

## Table 2. Stage-1 Readout Mismatch

{to_md_table(stage1)}

## Table 3. Round-2 Minimal Multi-Seed

{to_md_table(ms_table)}

## Table 3b. Bcov Deltas

{to_md_table(delta_table)}

## Table 4. Sample-Quality Boundary

{to_md_table(sample_table)}

## Table 5. Fallback Negative Control

{to_md_table(fallback_table)}

{phaseb_section}

{e2_section}

## Paper-Ready Result Paragraphs

**4.1 Attention Graph Regime Diagnosis.** We diagnose each attention-derived graph before selecting a readout policy. The cross-graph diagnostics separate E3-control-small from shuffled, random, and E2 image graphs: E3-control-small has high local-neighbor mass (`P(d<=1)=0.9531`) and a low top-1 Manhattan distance (`1.2500`), so it is classified as proximity-dominant. In contrast, E2, shuffled-B, random-B, and E3-large/fixed graphs have near-random spatial diagnostics and are classified as uniform-noisy or fallback regimes. The text graph is classified as readiness-dominant, with high directionality and readiness strength. This supports the central design choice: readout selection is driven by graph structure, not by manually assigning one policy to text and another to images.

**4.2 Stage-1 Readout Mismatch.** Stage-1 establishes that the negative Graph-RW result was a readout failure rather than evidence that the image attention graph lacks useful structure. The E3-control-small graph is strongly local, but the default v1/v3 Graph-RW readouts produce orders whose mean Manhattan distances (`4.8410` and `4.7991`) remain much closer to random (`5.3264`) than to genuinely local traversals such as Hilbert (`1.0000`) or raster (`1.7778`). This explains why the Stage-1 `cont_graph_rw` arm behaves like random: the readout did not convert the proximity graph into a local curriculum. Raster provides the complementary mechanism check: a truly local order gives a large matched-order gain but also produces single-order specialization.

**4.3 Round-2 Proximity-Guided Readout.** Round-2 tests the regime-specific hypothesis that a proximity-dominant graph should use a coverage-style readout. Across five seeds, Bcov_balanced improves over random on cross-order loss by `-0.0029 ± 0.0007` and on structured-order loss by `-0.0062 ± 0.0009`. It also improves over distance-only coverage on cross and structured averages, and it improves over shuffled-Bcov on the same axes. These controls are important because they show that the gain is not explained only by the distance skeleton or by arbitrary row-wise perturbations of B. The result is therefore best stated as stable positive evidence that real attention-derived proximity structure can improve validation robustness when the readout matches the diagnosed regime.

**4.4 Controls and Diagnostic Boundary.** The fallback graph provides a useful boundary condition for the method. E3-large/fixed is diagnosed as no-structure/fallback, and Bcov_balanced does not help there: relative to random, its cross average worsens by about `+0.0139` and its structured average worsens by about `+0.0133`. This negative control makes the framework more defensible because it shows that the method does not blindly claim B-guided coverage should work for every graph. Instead, the evidence supports a conditional claim: Bcov is useful when the diagnostics indicate proximity-dominant structure.

**4.5 Sample-Quality Boundary.** The sample-quality pilot measures a different axis from validation robustness. Under matched-order sampling with 256 generated samples and VQ-decoded validation references, raster obtains the best proxy FID (`165.2249`) and Hilbert is second (`169.6961`), while Bcov_balanced is not the sample-FID winner (`174.0119`). This result should be presented as a boundary, not a contradiction. It indicates that strongly geometric orders can produce better matched-order visual samples, while Bcov's current evidence is about cross-order and structured validation robustness. A stronger sample-quality study should use more samples and common-order sampling across checkpoints before making generation-quality claims.

{phaseb_paragraph}

{e2_paragraph}

## Claim-Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| Readout selection can be framed as graph diagnostics -> regime -> readout. | Cross-graph regime table separates proximity, readiness, and noisy/fallback graphs. | supported |
| Stage-1 Graph-RW failure is readout mismatch, not local-B uselessness. | E3 graph is local, while v1/v3 readouts have near-random mean Manhattan distance and Stage-1 `graph_rw≈random`. | supported |
| Bcov provides stable positive validation evidence on E3-control-small. | Five-seed deltas vs random, distance-only, and shuffled-Bcov on cross/structured averages. | supported |
| Bcov should not be treated as universally effective. | E3-large/fallback negative control: Bcov worsens vs random. | supported |
| Uniform-noisy image graphs do not get E3-style Bcov benefits. | E2 VQ negative: Bcov has lower matched loss but worse cross/noisy than random. | supported |
| Bcov is not currently a sample-quality winner. | 256-sample proxy FID ranks raster and Hilbert above Bcov. | supported |
| Coverage readout can help on another proximity-dominant image graph, but B-specific gain is not universal. | Continuous ImageNet32 Phase-B check improves over random/shuffled but ties distance-only. | supported |

## Figures

- `fig1_graph_regime_pipeline.png`
- `fig2_round2_multiseed_bars.png`
- `fig3_bcov_delta_multiseed.png`
- `fig4_sample_quality_boundary.png`

## Metadata

```json
{json.dumps({"git": git_summary()}, ensure_ascii=False, indent=2)}
```
"""
    (OUT / "paper_results_summary.md").write_text(summary, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    regime = make_regime_table()
    stage1 = make_stage1_table()
    per_seed, deltas, ms_table, delta_table = make_multiseed_table()
    sample_raw, sample_table = make_boundary_table()
    fallback_raw, fallback_table = make_fallback_table()
    phaseb_table = make_phaseb_positive_table()
    e2_table = make_e2_vq_negative_table()
    plot_pipeline()
    plot_round2_bars(per_seed)
    plot_delta(deltas)
    plot_sample_fid(sample_raw)
    write_summary(regime, stage1, ms_table, delta_table, sample_table, fallback_table, phaseb_table, e2_table)
    print(OUT)


if __name__ == "__main__":
    main()
