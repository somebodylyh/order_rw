#!/usr/bin/env python3
"""Plot CDL / g_beta curves grouped by the true config seed.

This replaces the older seed2 plot that mixed `l2r_continuous_seed2` with
`random_baseline_b1_headscan_seed124`. Directory names are not trusted; every
run is validated against its own config.json before plotting.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "block_lo_arm_order_network" / "probe_results"
METRIC = "val_ori_l2r_block"


@dataclass(frozen=True)
class RunCurve:
    label: str
    path: Path
    seed: int
    protocol_key: str
    run_kind: str
    steps_k: List[float]
    values: List[float]
    color: str
    linestyle: str = "-"
    linewidth: float = 2.0
    alpha: float = 1.0

    @property
    def final_step(self) -> float:
        return self.steps_k[-1]

    @property
    def final_value(self) -> float:
        return self.values[-1]


@dataclass(frozen=True)
class PlotGroup:
    title: str
    seed: int
    runs: Sequence[RunCurve]
    note: str = ""


def _read_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing config.json: {cfg_path}")
    with cfg_path.open() as f:
        return json.load(f)


def _read_eval_curve(run_dir: Path, metric: str = METRIC) -> tuple[List[float], List[float]]:
    curve_path = run_dir / "eval_curve.tsv"
    if not curve_path.exists():
        raise FileNotFoundError(f"missing eval_curve.tsv: {curve_path}")
    steps: List[float] = []
    values: List[float] = []
    with curve_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or metric not in reader.fieldnames:
            raise ValueError(f"{curve_path} does not contain metric {metric!r}")
        for row in reader:
            val = row.get(metric)
            if val is None or val == "" or val.lower() == "nan":
                continue
            steps.append(float(row["step"]) / 1000.0)
            values.append(float(val))
    if not steps:
        raise ValueError(f"no plottable rows in {curve_path} for metric {metric!r}")
    return steps, values


def _protocol_key(run_dir: Path) -> str:
    coord_path = run_dir / "coordinate_check.json"
    if not coord_path.exists():
        raise FileNotFoundError(f"missing coordinate_check.json: {coord_path}")
    with coord_path.open() as f:
        coord = json.load(f)
    return json.dumps(coord["block_perm_first16"][:8])


def assert_group_seeds(expected_seed: int, run_dirs: Iterable[Path]) -> None:
    for run_dir in run_dirs:
        cfg = _read_config(Path(run_dir))
        seed = int(cfg.get("args", {}).get("seed"))
        if seed != int(expected_seed):
            raise ValueError(
                f"seed mismatch for {run_dir}: config seed={seed}, group seed={expected_seed}"
            )


def _run(
    label: str,
    dirname: str,
    color: str,
    *,
    linestyle: str = "-",
    linewidth: float = 2.0,
    alpha: float = 1.0,
    base: Path = BASE,
) -> RunCurve:
    path = base / dirname
    cfg = _read_config(path)
    args = cfg.get("args", {})
    steps, values = _read_eval_curve(path)
    return RunCurve(
        label=label,
        path=path,
        seed=int(args.get("seed")),
        protocol_key=_protocol_key(path),
        run_kind=str(args.get("run_kind")),
        steps_k=steps,
        values=values,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
    )


def _maybe_run(label: str, dirname: str, color: str, **kwargs) -> List[RunCurve]:
    path = kwargs.get("base", BASE) / dirname
    if not path.exists():
        return []
    return [_run(label, dirname, color, **kwargs)]


def _group(title: str, seed: int, runs: Sequence[RunCurve], note: str = "") -> PlotGroup:
    if runs:
        protocol_keys = {run.protocol_key for run in runs}
        if len(protocol_keys) != 1:
            raise ValueError(f"protocol/layout mismatch in {title}: {sorted(protocol_keys)}")
    for run in runs:
        pass
    return PlotGroup(title=title, seed=int(seed), runs=list(runs), note=note)


def build_default_groups(base: Path = BASE) -> List[PlotGroup]:
    baseline_kw = {"linewidth": 1.4, "alpha": 0.72}
    return [
        _group(
            "protocol seed=123/layout: node65 B1 closed loop",
            123,
            [
                _run("random baseline 0-50k", "random_baseline_continuous_jun08_seed2", "#bdbdbd", linestyle=(0, (1, 3)), base=base, **baseline_kw),
                _run("random baseline 55-60k", "random_baseline_continuous_jun08_seed2_ext60k", "#9a9a9a", linestyle=(0, (1, 3)), base=base, **baseline_kw),
                _run("ori-L2R reference", "l2r_continuous_seed123", "#5f5f5f", linestyle=(0, (4, 3)), base=base, **baseline_kw),
                *_maybe_run("ori-L2R fresh rerun", "l2r_continuous_seed123_fresh60k_rerun", "#303030", linestyle=(0, (6, 2)), base=base, **baseline_kw),
                _run("CDL from10k L0H2 (args seed=2)", "cdl_teacher_seed2_from10k_l0h2", "#d62728", linewidth=2.4, base=base),
                _run("CDL from20k L0H2 (args seed=2)", "cdl_teacher_seed2_from20k_l0h2", "#1f77b4", linewidth=2.4, base=base),
                _run("CDL from40k L0H2 (args seed=2)", "cdl_teacher_seed2_from40k_l0h2", "#2ca02c", linewidth=2.4, base=base),
                _run("CDL from10k L0H2 (args seed=123)", "cdl_teacher_seed123_from10k_l0h2", "#ff9896", linewidth=2.0, base=base),
                *_maybe_run("CDL from20k L0H2 (args seed=123)", "cdl_teacher_seed123_from20k_l0h2", "#aec7e8", linewidth=2.0, base=base),
                *_maybe_run("CDL reverse from20k L0H2", "cdl_teacher_seed123_from20k_l0h2_reverse", "#111111", linestyle=(0, (2, 2)), linewidth=1.9, base=base),
                *_maybe_run("CDL from40k L0H2 (args seed=123)", "cdl_teacher_seed123_from40k_l0h2", "#98df8a", linewidth=2.0, base=base),
                _run("CDL from20k (args seed=124)", "cdl_teacher_from20k_seed124", "#17becf", linewidth=2.0, base=base),
                _run("CDL from40k (args seed=124)", "cdl_teacher_from40k_seed124", "#bcbd22", linewidth=2.0, base=base),
                _run("g_beta FB from20k (args seed=2)", "frozen_beta_from20k_fixseed", "#8c564b", linewidth=2.0, base=base),
                _run("g_beta FB from20k (args seed=124)", "frozen_beta_from20k_seed124", "#9467bd", linewidth=2.0, base=base),
                _run("g_beta FB from40k (args seed=124)", "frozen_beta_from40k_seed124", "#2ca02c", linewidth=2.0, base=base),
                _run("B1 g_beta FB from20k (args seed=42)", "frozen_beta_b1_seed2_from20000", "#ff7f0e", linewidth=2.0, base=base),
                _run("B1 g_beta FB from40k (args seed=42)", "frozen_beta_b1_seed2_from40000", "#e377c2", linewidth=2.0, base=base),
            ],
            "All curves share the same seed123 block layout; args seed may differ for continuation stochasticity.",
        ),
        _group(
            "protocol seed=42/layout",
            42,
            [
                _run("random baseline (to 50k)", "random_baseline_continuous_jun05", "#9a9a9a", linestyle=(0, (1, 3)), base=base, **baseline_kw),
                _run("ori-L2R reference", "l2r_continuous_seed42", "#5f5f5f", linestyle=(0, (4, 3)), base=base, **baseline_kw),
                _run("CDL B1 from10k (args seed=124)", "cdl_teacher_seed124_from10k_l0h0_b1", "#d62728", linewidth=2.2, base=base),
                *_maybe_run("g_beta FB from10k", "frozen_beta_random_jun05_from10k", "#9467bd", linewidth=2.2, base=base),
                *_maybe_run("g_beta FB from20k", "frozen_beta_random_jun05_from20k", "#1f77b4", linewidth=2.2, base=base),
                *_maybe_run("g_beta FB from40k", "frozen_beta_random_jun05_from40k", "#2ca02c", linewidth=2.2, base=base),
            ],
            "The B1 CDL run labelled seed124 inherits this seed42 layout; random_baseline_b1_headscan_seed124 is a different layout and is not plotted as matched.",
        ),
    ]


def _family_group(group: PlotGroup, family: str) -> Optional[PlotGroup]:
    if family not in {"cdl", "gbeta"}:
        raise ValueError(f"unknown family {family!r}")
    wanted = "cdl_teacher" if family == "cdl" else "frozen_beta"
    family_runs = [run for run in group.runs if run.run_kind == wanted]
    if not family_runs:
        return None
    context_runs = [
        run for run in group.runs
        if run.run_kind in {"baseline", "l2r"}
    ]
    label = "CDL teacher" if family == "cdl" else "g_beta / frozen-beta"
    title = f"protocol seed={group.seed}: {label}"
    return _group(title, group.seed, context_runs + family_runs, group.note)


def split_groups_by_family(groups: Sequence[PlotGroup]) -> Dict[str, List[PlotGroup]]:
    split: Dict[str, List[PlotGroup]] = {"cdl": [], "gbeta": []}
    for group in groups:
        for family in split:
            family_group = _family_group(group, family)
            if family_group is not None:
                split[family].append(family_group)
    return split


def _plot_group(ax, group: PlotGroup) -> None:
    vals: List[float] = []
    for run in group.runs:
        vals.extend(run.values)
        ax.plot(
            run.steps_k,
            run.values,
            label=f"{run.label} [{run.final_step:.0f}k={run.final_value:.3f}]",
            color=run.color,
            linestyle=run.linestyle,
            linewidth=run.linewidth,
            alpha=run.alpha,
        )
        ax.annotate(
            f"{run.final_value:.3f}",
            (run.final_step, run.final_value),
            textcoords="offset points",
            xytext=(6, -8),
            fontsize=8,
            color=run.color,
        )
    ymin = max(3.20, min(vals) - 0.06)
    ymax = min(4.35, max(vals) + 0.08)
    if ymax - ymin < 0.20:
        pad = (0.20 - (ymax - ymin)) / 2.0
        ymin -= pad
        ymax += pad
    ax.set_title(group.title, fontsize=12, fontweight="bold")
    ax.set_xlim(-1, 65)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_xlabel("Training step (k)")
    if group.note:
        ax.text(
            0.0,
            -0.18,
            group.note,
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
            va="top",
        )


def save_plot(groups: Sequence[PlotGroup], out_path: Path, title: str) -> None:
    n = len(groups)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6.2 * rows), squeeze=False)
    for ax, group in zip(axes.ravel(), groups):
        _plot_group(ax, group)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0][0].set_ylabel(f"{METRIC} (CE loss) ↓")
    if rows > 1:
        axes[1][0].set_ylabel(f"{METRIC} (CE loss) ↓")
    handles, labels = [], []
    for ax in axes.ravel()[:n]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    # Keep legends local; global legend is too dense for four seed groups.
    for ax in axes.ravel()[:n]:
        ax.legend(fontsize=8, loc="upper right", frameon=False)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975], pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(groups: Sequence[PlotGroup], out_path: Path) -> None:
    payload = []
    for group in groups:
        payload.append(
            {
                "title": group.title,
                "protocol_seed": group.seed,
                "note": group.note,
                "runs": [
                    {
                        "label": run.label,
                        "path": str(run.path.relative_to(ROOT)),
                        "seed": run.seed,
                        "protocol_seed": group.seed,
                        "protocol_key": run.protocol_key,
                        "run_kind": run.run_kind,
                        "final_step_k": run.final_step,
                        "final_val_ori_l2r_block": run.final_value,
                    }
                    for run in group.runs
                ],
            }
        )
    out_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    groups = build_default_groups()
    split = split_groups_by_family(groups)
    cdl_out = ROOT / "cdl_matched_by_seed.png"
    gbeta_out = ROOT / "gbeta_matched_by_seed.png"
    combined_out = ROOT / "cdl_gbeta_matched_by_seed.png"
    legacy_out = ROOT / "cdl_gbeta_seed2_65node.png"
    summary = ROOT / "cdl_gbeta_matched_by_seed.json"
    save_plot(split["cdl"], cdl_out, "CDL teacher curves grouped by effective protocol/layout")
    save_plot(split["gbeta"], gbeta_out, "g_beta / frozen-beta curves grouped by effective protocol/layout")
    save_plot(groups, combined_out, "CDL / g_beta curves grouped by effective protocol/layout")
    save_plot(groups, legacy_out, "CDL / g_beta curves grouped by effective protocol/layout")
    write_summary(groups, summary)
    print(f"Saved CDL plot: {cdl_out}")
    print(f"Saved g_beta plot: {gbeta_out}")
    print(f"Saved combined plot: {combined_out}")
    print(f"Saved legacy-compatible path: {legacy_out}")
    print(f"Summary: {summary}")
    for group in groups:
        print(f"\n{group.title}")
        for run in group.runs:
            print(
                f"  {run.label:<24} seed={run.seed:<3} {run.final_step:>5.0f}k "
                f"final={run.final_value:.4f}  {run.path.name}"
            )


if __name__ == "__main__":
    main()
