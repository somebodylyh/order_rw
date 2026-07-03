#!/usr/bin/env python3
"""Batch distribution-teacher diagnostics for seq80 permute random checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "analysis" / "eval_layer_mean_pairwise_max_fiedler_order.py"
DEFAULT_CKPT_DIR = (
    REPO_ROOT
    / "out/base/permute/seq80/block1/"
    / "out-wikitext103-seq80-random-b1-permute-6l8h512-save-attn-ckpts-50000-iters"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "Report/language/wikitext103/order_teacher_distribution/"
    / "seq80_permute_random_6l8h512_ckpt_teacher_1024attn_512loss"
)
DEFAULT_STEPS = "1000,2000,5000,8000,10000,15000,20000,25000"


def parse_steps(text: str) -> list[int]:
    steps = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            steps.append(int(item))
    if not steps:
        raise ValueError("No checkpoint steps requested.")
    return steps


def checkpoint_path(ckpt_dir: Path, step: int) -> Path:
    return ckpt_dir / "checkpoints" / f"ckpt_iter{int(step):07d}.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run layer-mean pairwise-max Fiedler teacher diagnostics over saved checkpoints."
    )
    parser.add_argument("--ckpt_dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=str, default=DEFAULT_STEPS)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--attention_samples", type=int, default=1024)
    parser.add_argument("--attention_batch_size", type=int, default=16)
    parser.add_argument("--loss_samples", type=int, default=512)
    parser.add_argument("--loss_batch_size", type=int, default=16)
    parser.add_argument("--loss_candidate_batch_size", type=int, default=4)
    parser.add_argument(
        "--loss_score",
        type=str,
        default="linear_profile",
        choices=("linear_profile", "exp_profile", "prefix", "full"),
    )
    parser.add_argument("--layers", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--attention_split", type=str, default="train")
    parser.add_argument("--loss_split", type=str, default="train")
    parser.add_argument("--export_type", type=str, default="without_none", choices=("with_none", "without_none"))
    parser.add_argument(
        "--attention_order_mode",
        type=str,
        default="random",
        choices=("random", "current_ar", "original_l2r"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_one(args: argparse.Namespace, step: int, ckpt_path: Path, step_out: Path) -> None:
    done_path = step_out / ".done.json"
    if done_path.exists() and not bool(args.force):
        print(f"[batch-teacher] step={step} already complete: {step_out}", flush=True)
        return
    step_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--ckpt_path",
        str(ckpt_path),
        "--out_dir",
        str(step_out),
        "--attention_split",
        str(args.attention_split),
        "--loss_split",
        str(args.loss_split),
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
        "--export_type",
        str(args.export_type),
        "--attention_order_mode",
        str(args.attention_order_mode),
        "--attention_samples",
        str(int(args.attention_samples)),
        "--attention_batch_size",
        str(int(args.attention_batch_size)),
        "--loss_samples",
        str(int(args.loss_samples)),
        "--loss_batch_size",
        str(int(args.loss_batch_size)),
        "--loss_candidate_batch_size",
        str(int(args.loss_candidate_batch_size)),
        "--loss_score",
        str(args.loss_score),
        "--layers",
        str(args.layers),
        "--write_matrices",
        "--plot",
    ]
    print(f"[batch-teacher] step={step} command: {shlex.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"step={step} failed with exit code {completed.returncode}")
    done_path.write_text(
        json.dumps(
            {
                "step": int(step),
                "ckpt_path": str(ckpt_path),
                "out_dir": str(step_out),
                "command": cmd,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_rows(step: int, step_out: Path) -> list[dict]:
    summary_path = step_out / "summary.json"
    csv_path = step_out / "layer_mean_pairwise_max_fiedler.csv"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            merged = {
                "step": int(step),
                "checkpoint_iter": int(summary.get("checkpoint_iter", -1)),
                "checkpoint_best_val_loss": summary.get("checkpoint_best_val_loss"),
                "ckpt_path": summary.get("ckpt_path"),
            }
            merged.update(row)
            rows.append(merged)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_readme(args: argparse.Namespace, rows: list[dict]) -> None:
    lines = [
        "# Seq80 Permute Random 6L8H512 Checkpoint Teacher Diagnostic",
        "",
        "Method: layer-mean attention -> pairwise-max affinity -> graph Laplacian Fiedler axis -> current-model loss chooses axis direction.",
        "",
        f"- checkpoint dir: `{args.ckpt_dir}`",
        f"- attention samples per checkpoint: `{int(args.attention_samples)}`",
        f"- loss-direction samples per checkpoint: `{int(args.loss_samples)}`",
        f"- loss direction score: `{args.loss_score}`",
        f"- export type: `{args.export_type}`",
        f"- attention order mode: `{args.attention_order_mode}`",
        f"- layers: `{args.layers}`",
        "",
        "Original tau/order are diagnostics only; direction selection uses current-frame model loss.",
        "",
        "| step | layer | raw tau | loss-oriented tau | selected reverse | score gap | input plot | original plot |",
        "| ---: | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for row in rows:
        layer = int(row["layer"])
        step = int(row["step"])
        step_dir = f"ckpt_{step:06d}"
        lines.append(
            "| {step} | L{layer} | {raw:+.6f} | {oriented:+.6f} | {reverse} | {gap:.6f} | "
            "`{input_plot}` | `{original_plot}` |".format(
                step=step,
                layer=layer,
                raw=float(row["fiedler_raw_original_tau"]),
                oriented=float(row["fiedler_loss_oriented_original_tau"]),
                reverse=str(row["fiedler_loss_oriented_selected_reverse"]),
                gap=float(row["fiedler_loss_oriented_score_gap"]),
                input_plot=f"{step_dir}/plots/L{layer}_layermean_pairwise_max_fiedler_input.png",
                original_plot=f"{step_dir}/plots/L{layer}_layermean_pairwise_max_fiedler_true_original.png",
            )
        )
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.out_dir / "batch_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    steps = parse_steps(args.steps)
    for step in steps:
        ckpt_path = checkpoint_path(args.ckpt_dir, step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for step={step}: {ckpt_path}")
        run_one(args, step, ckpt_path, args.out_dir / f"ckpt_{int(step):06d}")

    all_rows: list[dict] = []
    for step in steps:
        all_rows.extend(load_rows(step, args.out_dir / f"ckpt_{int(step):06d}"))
    write_csv(args.out_dir / "all_ckpts_layer_mean_pairwise_max_fiedler.csv", all_rows)
    write_readme(args, all_rows)
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(all_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
