#!/usr/bin/env python3
"""Summarize existing g_beta/CDL feedback runs into a final report package."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "block_lo_arm_order_network" / "probe_results"
DEFAULT_OUT = ROOT / "reports" / "gbeta_feedback_final_summary"

SUMMARY_COLUMNS = [
    "label",
    "family",
    "path",
    "step",
    "alpha",
    "train_loss",
    "val_train_objective",
    "val_ori_l2r_block",
    "delta_vs_random_ori",
    "val_model_order",
    "val_unstructured_order",
    "val_rw_order",
    "val_beta_order",
    "val_cdl_order",
    "lr",
    "note",
]

PLOT_METRICS = [
    "val_ori_l2r_block",
    "val_train_objective",
    "val_model_order",
    "val_unstructured_order",
]


@dataclass(frozen=True)
class RunSpec:
    label: str
    family: str
    path: Path
    note: str = ""


def default_runs() -> list[RunSpec]:
    return [
        RunSpec(
            "random baseline seed2 60k",
            "random",
            PROBE / "random_baseline_continuous_jun08_seed2_ext60k",
            "primary random-order baseline",
        ),
        RunSpec(
            "L2R seed123 60k",
            "l2r",
            PROBE / "l2r_continuous_seed123",
            "reference AR/L2R run; protocol may differ from seed2 historical runs",
        ),
        RunSpec(
            "frozen g_beta B1 seed2 from20k",
            "frozen_gbeta",
            PROBE / "frozen_beta_b1_seed2_from20000",
            "best clean old-g_beta feedback arm",
        ),
        RunSpec(
            "frozen g_beta B1 seed2 from40k",
            "frozen_gbeta",
            PROBE / "frozen_beta_b1_seed2_from40000",
            "from40k B1 old-g_beta arm",
        ),
        RunSpec(
            "frozen g_beta from20k fixseed",
            "frozen_gbeta",
            PROBE / "frozen_beta_from20k_fixseed",
            "from20k historical/fixseed arm",
        ),
        RunSpec(
            "frozen g_beta from20k seed124",
            "frozen_gbeta",
            PROBE / "frozen_beta_from20k_seed124",
            "from20k seed124 historical arm",
        ),
        RunSpec(
            "frozen g_beta from40k seed124",
            "frozen_gbeta",
            PROBE / "frozen_beta_from40k_seed124",
            "failure/instability case",
        ),
        RunSpec(
            "CDL teacher from20k fixseed",
            "cdl_teacher",
            PROBE / "cdl_teacher_from20k_fixseed",
            "matched teacher-style control",
        ),
        RunSpec(
            "CDL teacher from20k seed124",
            "cdl_teacher",
            PROBE / "cdl_teacher_from20k_seed124",
            "matched teacher-style control",
        ),
        RunSpec(
            "CDL teacher from40k seed124",
            "cdl_teacher",
            PROBE / "cdl_teacher_from40k_seed124",
            "from40k teacher-style control",
        ),
        RunSpec(
            "CDL seed123 from20k L0H2",
            "cdl_teacher",
            PROBE / "cdl_teacher_seed123_from20k_l0h2",
            "seed123 positive control",
        ),
        RunSpec(
            "CDL seed123 from20k L0H2 reverse",
            "reverse_control",
            PROBE / "cdl_teacher_seed123_from20k_l0h2_reverse",
            "reverse-order negative control",
        ),
        RunSpec(
            "CDL seed123 from40k L0H2",
            "cdl_teacher",
            PROBE / "cdl_teacher_seed123_from40k_l0h2",
            "seed123 from40k positive control",
        ),
    ]


def _parse_value(value: str):
    text = (value or "").strip()
    if text == "":
        return float("nan")
    lowered = text.lower()
    if lowered == "nan":
        return float("nan")
    try:
        if "." not in text and "e" not in lowered:
            return int(text)
        return float(text)
    except ValueError:
        return value


def load_eval_curve(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [
            {key: _parse_value(value) for key, value in row.items()}
            for row in reader
        ]


def _safe(row: dict, key: str) -> float:
    value = row.get(key, float("nan"))
    if isinstance(value, str):
        value = _parse_value(value)
    return float(value) if value is not None else float("nan")


def summarize_run(spec: RunSpec, random_final_ori: float | None = None) -> dict:
    curve_path = spec.path / "eval_curve.tsv"
    if not curve_path.exists():
        row = {key: float("nan") for key in SUMMARY_COLUMNS}
        row.update({
            "label": spec.label,
            "family": spec.family,
            "path": str(spec.path),
            "note": f"missing eval_curve.tsv; {spec.note}".strip("; "),
        })
        return row

    rows = load_eval_curve(curve_path)
    final = rows[-1]
    ori = _safe(final, "val_ori_l2r_block")
    delta = (
        ori - float(random_final_ori)
        if random_final_ori is not None and not math.isnan(float(random_final_ori))
        else float("nan")
    )
    out = {
        "label": spec.label,
        "family": spec.family,
        "path": str(spec.path),
        "step": int(final.get("step", -1)),
        "alpha": _safe(final, "alpha"),
        "train_loss": _safe(final, "train_loss"),
        "val_train_objective": _safe(final, "val_train_objective"),
        "val_ori_l2r_block": ori,
        "delta_vs_random_ori": delta,
        "val_model_order": _safe(final, "val_model_order"),
        "val_unstructured_order": _safe(final, "val_unstructured_order"),
        "val_rw_order": _safe(final, "val_rw_order"),
        "val_beta_order": _safe(final, "val_beta_order"),
        "val_cdl_order": _safe(final, "val_cdl_order"),
        "lr": _safe(final, "lr"),
        "note": spec.note,
    }
    return out


def _fmt(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def write_tsv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key, "")) for key in SUMMARY_COLUMNS})


def plot_curves(specs: list[RunSpec], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(10, 5.8))
        for spec in specs:
            curve_path = spec.path / "eval_curve.tsv"
            if not curve_path.exists():
                continue
            rows = load_eval_curve(curve_path)
            xs = [_safe(row, "step") for row in rows]
            ys = [_safe(row, metric) for row in rows]
            ax.plot(xs, ys, marker=".", linewidth=1.4, label=spec.label)
        ax.set_title(metric)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=1, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.png", dpi=140)
        plt.close(fig)


def write_summary_md(rows: list[dict], out_dir: Path) -> None:
    valid = [r for r in rows if not math.isnan(float(r.get("val_ori_l2r_block", float("nan"))))]
    random_rows = [r for r in valid if r["family"] == "random"]
    random_ori = random_rows[0]["val_ori_l2r_block"] if random_rows else float("nan")
    ordered = sorted(valid, key=lambda r: r["val_ori_l2r_block"])
    frozen = [r for r in valid if r["family"] == "frozen_gbeta"]
    cdl = [r for r in valid if r["family"] == "cdl_teacher"]
    reverse = [r for r in valid if r["family"] == "reverse_control"]

    lines = [
        "# gBeta Feedback Final Summary",
        "",
        "## Key Results",
        "",
        f"- Random baseline final `val_ori_l2r_block`: {_fmt(random_ori)}.",
    ]
    if ordered:
        best = ordered[0]
        lines.append(
            f"- Best final `val_ori_l2r_block`: `{best['label']}` = "
            f"{_fmt(best['val_ori_l2r_block'])} "
            f"(delta vs random {_fmt(best['delta_vs_random_ori'])})."
        )
    if frozen:
        best_frozen = min(frozen, key=lambda r: r["val_ori_l2r_block"])
        lines.append(
            f"- Best frozen g_beta arm: `{best_frozen['label']}` = "
            f"{_fmt(best_frozen['val_ori_l2r_block'])} "
            f"(delta vs random {_fmt(best_frozen['delta_vs_random_ori'])})."
        )
    if cdl:
        best_cdl = min(cdl, key=lambda r: r["val_ori_l2r_block"])
        lines.append(
            f"- Best CDL teacher arm: `{best_cdl['label']}` = "
            f"{_fmt(best_cdl['val_ori_l2r_block'])} "
            f"(delta vs random {_fmt(best_cdl['delta_vs_random_ori'])})."
        )
    bad = [r for r in frozen if r["val_ori_l2r_block"] > random_ori]
    if bad:
        lines.append(
            "- Failure/instability case: "
            + "; ".join(
                f"`{r['label']}` = {_fmt(r['val_ori_l2r_block'])}"
                for r in bad
            )
            + "."
        )
    if reverse:
        lines.append(
            "- Reverse control present: "
            + "; ".join(
                f"`{r['label']}` = {_fmt(r['val_ori_l2r_block'])}"
                for r in reverse
            )
            + "."
        )

    lines.extend([
        "",
        "## Claim Boundary",
        "",
        "- This package summarizes existing completed runs; it does not add new training.",
        "- Head-gated g_beta remains smoke-only and is not included in the main evidence claim.",
        "- Treat seed/protocol inheritance carefully; compare matched groups before claiming robustness.",
        "- The from40k seed124 frozen g_beta failure should be diagnosed before making broad stability claims.",
        "",
        "## Files",
        "",
        "- `summary_table.tsv`: final-row metrics for each run.",
        "- `summary.json`: machine-readable copy of the table.",
        "- `val_ori_l2r_block.png`, `val_train_objective.png`, `val_model_order.png`, `val_unstructured_order.png`: curve overlays.",
    ])
    (out_dir / "boss_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = default_runs()
    random_spec = next(spec for spec in specs if spec.family == "random")
    random_rows = load_eval_curve(random_spec.path / "eval_curve.tsv")
    random_final_ori = _safe(random_rows[-1], "val_ori_l2r_block")
    rows = [summarize_run(spec, random_final_ori=random_final_ori) for spec in specs]

    write_tsv(rows, out_dir / "summary_table.tsv")
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    plot_curves(specs, out_dir)
    write_summary_md(rows, out_dir)

    print(f"Wrote {out_dir / 'summary_table.tsv'}")
    print(f"Wrote {out_dir / 'boss_summary.md'}")
    print(f"Wrote curve PNGs to {out_dir}")


if __name__ == "__main__":
    main()
