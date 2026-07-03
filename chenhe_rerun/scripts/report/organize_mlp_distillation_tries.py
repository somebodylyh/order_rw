#!/usr/bin/env python3
"""Organize WikiText103 MLP distillation reports by experiment stage."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("Report/language/wikitext103/mlp/distillation")
CLASS_ROOT = ROOT / "tries_by_stage"
CONFIG_ROOT = Path("config/WikiText103/seq256/permute/block64")
LOG_ROOT = Path("Report/logs")


@dataclass(frozen=True)
class Category:
    slug: str
    title: str
    summary: str
    conclusion: str
    tries: tuple[int, ...]


CATEGORIES: tuple[Category, ...] = (
    Category(
        "00_dataset_and_pairwise_student",
        "Offline Dataset And Pairwise Student",
        "Builds/reuses the L0 layer-mean Fiedler dataset and trains the first pairwise/rank MLP students.",
        "try28 passes the strict 0.98 held-out teacher-tau gate and becomes the frozen try28 student.",
        (25, 28),
    ),
    Category(
        "01_frozen_try28_backbone_insertion",
        "Frozen Try28 Backbone Insertion",
        "Loads the offline try28 MLP into AO-GPT and studies no-EMA versus attention-EMA insertion.",
        "The frozen try28 path is viable, but downstream loss is later superseded by the continuous-score and shadow-trained MLP lines.",
        (29, 30, 31, 32),
    ),
    Category(
        "02_continuous_score_mlp_offline",
        "Offline Continuous Score MLP",
        "Switches the target from pairwise/rank order imitation to continuous Fiedler priority score MSE.",
        "try33 learns the score operator cleanly and becomes the source checkpoint for frozen score-MLP downstream tests.",
        (33,),
    ),
    Category(
        "03_frozen_try33_score_mlp_downstream",
        "Frozen Try33 Score MLP Downstream",
        "Runs the offline try33 score MLP as a frozen backbone policy with no EMA, score EMA, and resume/teacher-diagnostic ablations.",
        "This line narrowly improves over early insertion but remains around the 3.44-3.46 final-val band, motivating online shadow training.",
        (44, 45, 46, 47, 48, 49, 50, 51),
    ),
    Category(
        "04_online_shadow_mse_early_ablations",
        "Online Shadow MSE Early Ablations",
        "Trains a random-init score MLP during Random warmup and explores early shadow windows, train/val splits, and attention aggregation.",
        "The early shadow variants expose label/orientation and aggregation issues; they are weaker than the final linear-profile schedule.",
        (52, 53, 54, 55),
    ),
    Category(
        "05_latest_shadow_schedule_seed_sweep",
        "Latest Shadow Schedule Seed Sweep",
        "Uses linear_profile_loss orientation, 10k-18k shadow training, all-train 512-sample shadow items, and the 18k-32k/32k-35k/fixed schedule.",
        "This is the current strongest line: try56-60 are stable across seeds, with try60 best so far.",
        (56, 57, 58, 59, 60),
    ),
)


TRY_INFO: dict[int, dict[str, str]] = {
    25: {
        "title": "Merged Fiedler Dataset And Three-Hidden Pairwise Student",
        "purpose": "Merge try20/try24 L0 layer-mean Fiedler teacher data and train a large pairwise/rank MLP.",
        "conclusion": "Held-out teacher tau reaches 0.979898, narrowly missing the strict 0.980 gate; useful as the main dataset provenance.",
    },
    28: {
        "title": "Smaller Two-Hidden Pairwise Student",
        "purpose": "Reuse the try25 dataset with a smaller [2048,1024] FlatAttentionOrderMLP.",
        "conclusion": "Held-out teacher tau reaches 0.980612 and passes the strict gate; used as the frozen try28 insertion checkpoint.",
    },
    29: {
        "title": "Frozen Try28 No-EMA Insertion, Seed 2027",
        "purpose": "Load frozen try28 MLP into the backbone with no attention/logits EMA and update every step.",
        "conclusion": "Completes 50k with final val about 3.4524; establishes the no-EMA every-step insertion baseline.",
    },
    30: {
        "title": "Frozen Try28 No-EMA Insertion, Seed 2028",
        "purpose": "Seed ablation of try29.",
        "conclusion": "Completes 50k with final val about 3.4380; better than try29 but later score/shadow lines are stronger.",
    },
    31: {
        "title": "Frozen Try28 Attention-EMA Insertion, Seed 2027",
        "purpose": "Restore attention EMA with warmup20k while still refreshing the MLP order every step.",
        "conclusion": "Completes 50k with final val about 3.4460; attention EMA does not clearly solve the downstream gap.",
    },
    32: {
        "title": "Frozen Try28 Attention-EMA Insertion, Seed 2028",
        "purpose": "Seed ablation of try31.",
        "conclusion": "Completes 50k with final val about 3.4259, the strongest frozen try28 insertion run retained here.",
    },
    33: {
        "title": "Continuous Fiedler Score MLP",
        "purpose": "Regenerate continuous Fiedler priority labels from try25 attention and train score MSE.",
        "conclusion": "Real-input score MSE is 0.000249, Pearson is about 0.9995, target tau is 0.9771; accepted as the score-MLP source for downstream.",
    },
    44: {
        "title": "Frozen Try33 Score MLP No-EMA, Seed 2027",
        "purpose": "First downstream seed after accepting the relaxed 0.97 score-MLP gate.",
        "conclusion": "Completes 50k with final val 3.4581; paired with try45 for the initial two-seed gate.",
    },
    45: {
        "title": "Frozen Try33 Score MLP No-EMA, Seed 2028",
        "purpose": "Second seed for the initial frozen try33 downstream check.",
        "conclusion": "Completes 50k with final val 3.4441; try44/45 mean final is about 3.4511, narrowly above the 3.45 target.",
    },
    46: {
        "title": "Frozen Try33 Score MLP No-EMA, Seed 2029",
        "purpose": "Additional no-EMA seed for the frozen try33 score-MLP line.",
        "conclusion": "Completes 50k with final val about 3.4613; no-EMA score-MLP remains around the 3.45 band.",
    },
    47: {
        "title": "Frozen Try33 Score MLP No-EMA, Seed 2030",
        "purpose": "Additional no-EMA seed paired against score-EMA try49.",
        "conclusion": "Completes 50k with final val about 3.4633; does not beat the later shadow-trained schedule.",
    },
    48: {
        "title": "Frozen Try33 Continuous-Score EMA, Seed 2029",
        "purpose": "Maintain EMA over continuous MLP scores before argsort, without attention EMA or discrete-order EMA.",
        "conclusion": "Completes 50k with final val about 3.4340; score EMA helps this seed but is superseded by online shadow training.",
    },
    49: {
        "title": "Frozen Try33 Continuous-Score EMA, Seed 2030",
        "purpose": "Score-EMA seed paired against no-EMA try47; includes a resume after interruption.",
        "conclusion": "Retained logs reach step 41000 with val about 3.4946; this run is incomplete in the retained evidence.",
    },
    50: {
        "title": "Resume10k Frozen Try33 With Teacher Diagnostics",
        "purpose": "Resume from the clean Random 10k checkpoint and compute teacher-vs-MLP diagnostics at each policy refresh.",
        "conclusion": "Kept primarily as a diagnostic design record; no complete retained train log was found in this report tree.",
    },
    51: {
        "title": "Frozen Try33 Score-EMA, Seed 2031",
        "purpose": "One more score-EMA downstream seed for the frozen try33 line.",
        "conclusion": "Completes 50k with final val about 3.4447; still in the same performance band as the frozen try33 family.",
    },
    52: {
        "title": "Online Shadow MSE, 8k-15k Group5 Train5 Val1",
        "purpose": "Train a random-init score MLP inside Random warmup, using grouped attention items and train/val shadow splits.",
        "conclusion": "Completes 50k with final val about 3.4861; useful first shadow-training prototype but not competitive.",
    },
    53: {
        "title": "Online Shadow MSE, 8k-12k Attn1024 Train15 Val3",
        "purpose": "Increase aggregation to 1024 samples and use many train/val items per MLP update.",
        "conclusion": "Completes 50k with final val about 3.5516; weak order recovery motivates label/orientation and schedule changes.",
    },
    54: {
        "title": "Online Shadow MSE, 8k-10k Raw Sample Split",
        "purpose": "Shorten the shadow window and train from fixed raw-sample train/val counts per step.",
        "conclusion": "Completes 50k with final val about 3.6100; this early fixed-sample split is weak.",
    },
    55: {
        "title": "Online Shadow MSE, 8k-12k Seed Repeat",
        "purpose": "Repeat try53-style training with seed 2027.",
        "conclusion": "Cancelled around step 29250; retained partial val is about 3.6813, so it is not a completed result.",
    },
    56: {
        "title": "Latest Shadow Schedule, Seed 2003",
        "purpose": "Use linear_profile_loss direction and Version-B style 18k-32k refresh, 32k-35k fixed transition.",
        "conclusion": "Completes 50k with final val 3.4081 and final original tau 0.9762; first strong latest-schedule run.",
    },
    57: {
        "title": "Latest Shadow Schedule, Seed 2050",
        "purpose": "Same latest schedule as try56 but with 512-sample shadow train items and no val split.",
        "conclusion": "Checkpoint reaches 50k with final/best field 3.3885 and final original tau 0.9921; full train log is not retained in the queue dir.",
    },
    58: {
        "title": "Latest Shadow Schedule, Seed 2051, Policy Attn512",
        "purpose": "Seed sweep with policy input attention refreshed from 512 samples.",
        "conclusion": "Completes 50k with final val 3.3618, best val 3.3469, and final original tau 0.9573.",
    },
    59: {
        "title": "Latest Shadow Schedule, Seed 2052, Policy Attn512",
        "purpose": "Second 512-sample policy-attention seed.",
        "conclusion": "Completes 50k with final val 3.3440, best val 3.3369, and final original tau 0.9950.",
    },
    60: {
        "title": "Latest Shadow Schedule, Seed 2053, Policy Attn1024",
        "purpose": "Return policy input attention to 1024 samples while keeping the latest shadow schedule.",
        "conclusion": "Current best retained MLP run: final val 3.3245, best val 3.3128, final original tau 0.9950.",
    },
}


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    except Exception:
        return None
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except Exception:
        return None


def config_path_for_try(try_id: int) -> Path | None:
    matches = sorted(CONFIG_ROOT.glob(f"attn_mlp_try{try_id}_*.py"))
    return matches[0] if matches else None


def parse_log_metrics() -> dict[int, dict[str, Any]]:
    try_re = re.compile(r"Overriding config with .*attn_mlp_try(\d+)[^/]*\.py")
    step_re = re.compile(r"^step (\d+): train loss ([0-9.]+), val loss ([0-9.]+)")
    view_re = re.compile(r"/runs/([A-Za-z0-9_-]+)")
    current: int | None = None
    rows: dict[int, list[tuple[int, float, float, str]]] = {}
    wandb: dict[int, str] = {}

    candidate_logs = [
        ROOT / f"try_{i}" / f"train_try{i}_cuda1.log" for i in range(1, 100)
    ]
    candidate_logs += [
        LOG_ROOT / "wikitext103_mlp_try31_try32_gpu1_serial_direct_20260626_151545" / "train_serial_gpu1.log",
        LOG_ROOT / "wikitext103_score_mlp_try44_try45_gpu1_serial_wandb_20260626_235328" / "train_serial_gpu1.log",
        LOG_ROOT / "wikitext103_score_mlp_try46_try48_try47_try49_gpu1_serial_20260627_070113" / "train_serial_gpu1.log",
        LOG_ROOT / "wikitext103_score_mlp_try49_resume_gpu1_20260627_202926" / "train.log",
        LOG_ROOT / "wikitext103_score_mlp_try51_gpu1_after_temperature_20260627_223530" / "train_serial_gpu1.log",
        LOG_ROOT / "wikitext103_attn_mlp_try52_group5b_train5val1_gpu1_20260628_161536" / "train_try52_gpu1.log",
        LOG_ROOT / "wikitext103_attn_mlp_try53_seed2036_train8k12k_attn1024_train15val3_gpu1_20260628_223235" / "train_try53_gpu1.log",
        LOG_ROOT / "wikitext103_attn_mlp_try54_try55_when_gpu1_idle_20260629_002230" / "train_try54_gpu1.log",
        LOG_ROOT / "wikitext103_attn_mlp_try54_try55_when_gpu1_idle_20260629_002230" / "train_try55_gpu1.log",
        LOG_ROOT / "wikitext103_attn_mlp_try56_try57_gpu1_after_current_20260629_144933" / "train_try56_gpu1.log",
        LOG_ROOT / "wikitext103_attn_mlp_try58_try59_try60_gpu1_after_current_20260629_223714" / "train_try58_gpu1.log",
        LOG_ROOT / "wikitext103_attn_mlp_try58_try59_try60_gpu1_after_current_20260629_223714" / "train_try59_gpu1.log",
        LOG_ROOT / "wikitext103_attn_mlp_try58_try59_try60_gpu1_after_current_20260629_223714" / "train_try60_gpu1.log",
    ]

    for path in candidate_logs:
        if not path.exists():
            continue
        forced_try = None
        name = path.name
        m_name = re.search(r"try(\d+)", name)
        if m_name and "serial" not in name:
            forced_try = int(m_name.group(1))
        if "try49_resume" in str(path):
            forced_try = 49
        current = forced_try
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = try_re.search(line)
            if m:
                current = int(m.group(1))
            sm = step_re.match(line)
            if sm and current is not None:
                rows.setdefault(current, []).append(
                    (int(sm.group(1)), float(sm.group(2)), float(sm.group(3)), str(path))
                )
            if current is not None and "View run" in line:
                vm = view_re.search(line)
                if vm:
                    wandb[current] = vm.group(1)

    metrics: dict[int, dict[str, Any]] = {}
    for try_id, values in rows.items():
        if not values:
            continue
        best = min(values, key=lambda item: item[2])
        final = max(values, key=lambda item: item[0])
        metrics[try_id] = {
            "best_step": best[0],
            "best_train_loss": best[1],
            "best_val_loss": best[2],
            "final_step": final[0],
            "final_train_loss": final[1],
            "final_val_loss": final[2],
            "log_file": final[3],
        }
        if try_id in wandb:
            metrics[try_id]["wandb_run_id"] = wandb[try_id]
    return metrics


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def rel(path: Path) -> str:
    return str(path)


def ensure_layout() -> None:
    CLASS_ROOT.mkdir(parents=True, exist_ok=True)
    category_by_try = {try_id: category for category in CATEGORIES for try_id in category.tries}
    for try_id, category in category_by_try.items():
        legacy = ROOT / f"try_{try_id}"
        target_dir = CLASS_ROOT / category.slug
        target = target_dir / f"try_{try_id}"
        target_dir.mkdir(parents=True, exist_ok=True)

        if legacy.is_symlink():
            if not target.exists():
                raise RuntimeError(f"{legacy} is a symlink but target {target} is missing")
        elif legacy.exists() and not target.exists():
            legacy.rename(target)
        elif legacy.exists() and target.exists():
            raise RuntimeError(f"Both {legacy} and {target} exist as real paths; refusing to merge")

        if not legacy.exists() and target.exists():
            link_target = os.path.relpath(target, legacy.parent)
            legacy.symlink_to(link_target, target_is_directory=True)


def try_readme(try_id: int, category: Category, log_metrics: dict[int, dict[str, Any]]) -> str:
    info = TRY_INFO[try_id]
    try_dir = CLASS_ROOT / category.slug / f"try_{try_id}"
    summary = read_json(try_dir / "training_summary.json")
    order = last_jsonl(try_dir / "input_attn" / "attn_mlp_policy_order_history.jsonl")
    shadow = last_jsonl(try_dir / "input_attn" / "attn_mlp_shadow_mse_history.jsonl")
    cfg = config_path_for_try(try_id)
    metrics = log_metrics.get(try_id, {})

    lines: list[str] = [
        f"# try_{try_id}: {info['title']}",
        "",
        "## Classification",
        "",
        f"- Stage: `{category.slug}`",
        f"- Stage summary: {category.summary}",
        "",
        "## Purpose",
        "",
        info["purpose"],
        "",
        "## Key Files",
        "",
    ]
    if cfg is not None:
        lines.append(f"- Config: `{rel(cfg)}`")
    if (try_dir / "experiment_design.md").exists():
        lines.append("- Design note: `experiment_design.md`")
    if (try_dir / "downstream_plan.md").exists():
        lines.append("- Downstream plan: `downstream_plan.md`")
    if (try_dir / "training_summary.json").exists():
        lines.append("- Training summary: `training_summary.json`")
    if (try_dir / "input_attn").exists():
        lines.append("- Input/order diagnostics: `input_attn/`")
    if summary and summary.get("checkpoint"):
        lines.append(f"- MLP checkpoint: `{summary.get('checkpoint')}`")
    if metrics.get("log_file"):
        lines.append(f"- Parsed train log: `{metrics['log_file']}`")
    lines.extend(["", "## Metrics Snapshot", ""])

    if summary:
        if "best_val_tau" in summary:
            lines.append(f"- Offline best val teacher tau: `{format_float(summary.get('best_val_tau'), 6)}`")
        if "best_val_score_mse" in summary:
            lines.append(f"- Offline best val score MSE: `{format_float(summary.get('best_val_score_mse'), 6)}`")
        if "best_val_target_tau" in summary:
            lines.append(f"- Offline best val target tau: `{format_float(summary.get('best_val_target_tau'), 6)}`")
        if "passed_tau_gate" in summary:
            lines.append(f"- Passed strict tau gate: `{summary.get('passed_tau_gate')}`")
        if "passed_gate" in summary:
            lines.append(f"- Passed configured gate: `{summary.get('passed_gate')}`")

    if metrics:
        lines.append(
            "- Downstream final: "
            f"step `{metrics.get('final_step')}`, train `{format_float(metrics.get('final_train_loss'))}`, "
            f"val `{format_float(metrics.get('final_val_loss'))}`"
        )
        lines.append(
            "- Downstream best val in parsed log: "
            f"step `{metrics.get('best_step')}`, val `{format_float(metrics.get('best_val_loss'))}`"
        )
        if metrics.get("wandb_run_id"):
            lines.append(f"- W&B run id: `{metrics.get('wandb_run_id')}`")

    if order:
        lines.append(f"- Last cached original tau diagnostic: `{format_float(order.get('cached_tau_original_diagnostic'), 6)}`")
        lines.append(f"- Last cached original Kendall distance: `{format_float(order.get('cached_kendall_distance_original'), 4)}`")
        td = order.get("teacher_diag") or {}
        if td:
            lines.append(f"- Teacher-vs-MLP score pair tau: `{format_float(td.get('score_pair_tau_match'), 6)}`")
            lines.append(f"- Teacher-vs-MLP score MSE: `{format_float(td.get('mse_match'), 6)}`")
        cached = order.get("cached_order_current")
        if isinstance(cached, list):
            lines.append(f"- Cached order current first16: `{cached[:16]}`")

    if shadow:
        lines.append(f"- Shadow MSE stop iter: `{shadow.get('shadow_mse_stop_iter', 'n/a')}`")
        lines.append(f"- Last shadow record iter: `{shadow.get('iter', 'n/a')}`")

    if not summary and not metrics and not order and not shadow:
        lines.append("- No machine-readable summary/log was found in this try directory; read the design/planning notes directly.")

    lines.extend(["", "## Conclusion", "", info["conclusion"], ""])
    if try_id in {49, 50, 55, 57}:
        lines.extend(
            [
                "## Caveat",
                "",
                "This try has incomplete or nonstandard retained evidence. Treat the conclusion above as the best filesystem-backed summary, not as a fully audited final-run claim.",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- Original-frame tau is diagnostic only; it is not a training target or selection rule.",
            "- The root-level `try_XX` path remains a compatibility symlink to this directory.",
            "",
        ]
    )
    return "\n".join(lines)


def category_readme(category: Category, log_metrics: dict[int, dict[str, Any]]) -> str:
    lines = [
        f"# {category.title}",
        "",
        category.summary,
        "",
        "## Stage Conclusion",
        "",
        category.conclusion,
        "",
        "## Tries",
        "",
        "| Try | Title | Key result |",
        "|---|---|---|",
    ]
    for try_id in category.tries:
        info = TRY_INFO[try_id]
        metric = log_metrics.get(try_id)
        result = info["conclusion"]
        if metric:
            result += f" Parsed final val `{format_float(metric.get('final_val_loss'))}`."
        lines.append(f"| [try_{try_id}](try_{try_id}/README.md) | {info['title']} | {result} |")
    lines.append("")
    return "\n".join(lines)


def root_readme(log_metrics: dict[int, dict[str, Any]]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# MLP Distillation",
        "",
        "This directory contains the WikiText103 `seq256/permute/block64` Attn-MLP distillation and downstream insertion reports.",
        "",
        "The report tree is now organized by method stage under `tries_by_stage/`. The old root-level `try_XX` paths are kept as compatibility symlinks so existing configs and notes continue to resolve.",
        "",
        f"Last organized: `{now}`.",
        "",
        "## Current Reading Path",
        "",
        "1. `HISTORY_ARCHIVE_20260624.md` for the pre-cleanup provenance.",
        "2. `tries_by_stage/00_dataset_and_pairwise_student/` for the first pairwise/rank students.",
        "3. `tries_by_stage/02_continuous_score_mlp_offline/try_33/README.md` for the continuous score MLP.",
        "4. `tries_by_stage/05_latest_shadow_schedule_seed_sweep/` for the current strongest online-shadow schedule.",
        "",
        "## Stage Index",
        "",
        "| Stage | Contents | Conclusion |",
        "|---|---|---|",
    ]
    for category in CATEGORIES:
        tries = ", ".join(f"try_{i}" for i in category.tries)
        lines.append(
            f"| [{category.slug}](tries_by_stage/{category.slug}/README.md) | {tries} | {category.conclusion} |"
        )
    lines.extend(
        [
            "",
            "## Current Status",
            "",
            "- Best retained downstream MLP run in this tree: `try_60`, final val `3.3245`, best val `3.3128`.",
            "- Latest-method seed sweep `try56-try60` is the current stable line; earlier frozen-offline MLP insertion runs are historical controls.",
            "- Original-frame tau and `OriginalL2R` remain diagnostic-only throughout this tree.",
            "",
        ]
    )
    return "\n".join(lines)


def write_readmes() -> None:
    log_metrics = parse_log_metrics()
    category_by_try = {try_id: category for category in CATEGORIES for try_id in category.tries}

    for category in CATEGORIES:
        category_dir = CLASS_ROOT / category.slug
        (category_dir / "README.md").write_text(category_readme(category, log_metrics), encoding="utf-8")
        for try_id in category.tries:
            try_dir = category_dir / f"try_{try_id}"
            if try_dir.exists():
                (try_dir / "README.md").write_text(try_readme(try_id, category, log_metrics), encoding="utf-8")

    (ROOT / "README.md").write_text(root_readme(log_metrics), encoding="utf-8")


def main() -> None:
    ensure_layout()
    write_readmes()
    print(f"Organized {len(TRY_INFO)} tries under {CLASS_ROOT}")


if __name__ == "__main__":
    main()
