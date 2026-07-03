"""
Run and summarize a score-weight sweep for pair-margin stability probes.

This wrapper calls scripts/analysis/pair_margin_stability_probe.py for two
checkpoints by default:
  - seq256 permuted token-level/block1 checkpoint
  - seq256 permuted block64 checkpoint

It then summarizes each pair_stats.csv by original-frame gap categories so the
effect of the scoring weight can be compared without manually opening every
JSONL file.

Typical usage:
source /home/devbox/project/bin/activate
python scripts/analysis/tv_weight_pair_sweep.py

To only summarize already-produced outputs:
python scripts/analysis/tv_weight_pair_sweep.py --skip_run
"""

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from ast import literal_eval
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_SCRIPT = REPO_ROOT / "scripts" / "analysis" / "pair_margin_stability_probe.py"


DEFAULT_TOKEN_CKPT = (
    "out/base/permute/seq256/block1/"
    "out-wikitext103-seq256-random-b1-permute-block-6-8-256/ckpt.pt"
)
DEFAULT_BLOCK64_CKPT = (
    "out/base/permute/seq256/block64/"
    "out-wikitext103-seq256-random-b64-permute-block"
)


def parse_csv_floats(raw_value):
    return [float(item.strip()) for item in str(raw_value).split(",") if item.strip()]


def parse_csv_strings(raw_value):
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def tv_to_name(value):
    value = float(value)
    prefix = "neg" if value < 0 else "p"
    return f"tv_{prefix}{str(abs(value)).replace('.', 'p')}"


def drop_to_name(value):
    value = float(value)
    prefix = "neg" if value < 0 else "p"
    return f"drop_{prefix}{str(abs(value)).replace('.', 'p')}"


def resolve_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a score-weight sweep on token-level and block64 pair probes."
    )
    parser.add_argument("--out_root", type=Path, default=Path("Report/analysis/tv_weight_sweep"))
    parser.add_argument(
        "--score_mode",
        type=str,
        default="abs_tv",
        choices=["abs_tv", "signed_drop"],
        help=(
            "abs_tv sweeps tv_weight in -mean - weight*abs(diff). "
            "signed_drop sweeps drop_weight in -mean + weight*(loss_first-loss_second)."
        ),
    )
    parser.add_argument("--tv_weights", type=str, default="-0.3,0,0.1,0.3,0.5,1.0")
    parser.add_argument(
        "--drop_weights",
        type=str,
        default="0,0.1,0.2,0.3,0.4,0.5,0.75,1.0",
        help="Weights used when --score_mode signed_drop.",
    )
    parser.add_argument("--skip_run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun probes even if pair_stats.csv exists.")
    parser.add_argument(
        "--parallel_devices",
        type=str,
        default="",
        help=(
            "Optional comma-separated CUDA device ids/names for parallel probe runs, "
            "for example '0,1' or 'cuda:0,cuda:1'. Each child process sees one device."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--num_batches", type=int, default=48)
    parser.add_argument("--pair_score_k", type=int, default=2)
    parser.add_argument("--pair_chunk_size", type=int, default=8)
    parser.add_argument("--mean_margin_threshold", type=float, default=0.02)
    parser.add_argument("--positive_rate_threshold", type=float, default=0.70)
    parser.add_argument("--attn_top_k", type=int, default=8)
    parser.add_argument("--attn_num_batches", type=int, default=24)
    parser.add_argument("--block64_ckpt", type=Path, default=Path(DEFAULT_BLOCK64_CKPT))
    parser.add_argument("--block64_batch_size", type=int, default=32)
    parser.add_argument("--block64_attn_batch_size", type=int, default=32)
    parser.add_argument("--block64_forward_eval_batch_size", type=int, default=64)
    parser.add_argument("--block64_max_directed_pairs", type=int, default=0)
    parser.add_argument("--token_ckpt", type=Path, default=Path(DEFAULT_TOKEN_CKPT))
    parser.add_argument("--token_batch_size", type=int, default=2)
    parser.add_argument("--token_attn_batch_size", type=int, default=16)
    parser.add_argument("--token_forward_eval_batch_size", type=int, default=16)
    parser.add_argument(
        "--token_max_directed_pairs",
        type=int,
        default=3000,
        help="Symmetric cap for token-level directed pairs. 0 means no cap.",
    )
    return parser.parse_args()


def build_probe_command(args, label, ckpt_path, out_dir, score_weight):
    is_token = label == "token_block1"
    batch_size = args.token_batch_size if is_token else args.block64_batch_size
    attn_batch_size = args.token_attn_batch_size if is_token else args.block64_attn_batch_size
    forward_eval_batch_size = (
        args.token_forward_eval_batch_size if is_token else args.block64_forward_eval_batch_size
    )
    max_directed_pairs = (
        args.token_max_directed_pairs if is_token else args.block64_max_directed_pairs
    )
    cmd = [
        sys.executable,
        str(PROBE_SCRIPT),
        "--ckpt_path",
        str(ckpt_path),
        "--out_dir",
        str(out_dir),
        "--batch_size",
        str(int(batch_size)),
        "--num_batches",
        str(int(args.num_batches)),
        "--pair_mining_mode",
        "attention_pruned",
        "--attn_top_k",
        str(int(args.attn_top_k)),
        "--attn_num_batches",
        str(int(args.attn_num_batches)),
        "--attn_batch_size",
        str(int(attn_batch_size)),
        "--pair_score_k",
        str(int(args.pair_score_k)),
        "--score_mode",
        str(args.score_mode),
        "--tv_weight",
        str(float(score_weight) if args.score_mode == "abs_tv" else 0.0),
        "--drop_weight",
        str(float(score_weight) if args.score_mode == "signed_drop" else 0.0),
        "--pair_chunk_size",
        str(int(args.pair_chunk_size)),
        "--forward_eval_batch_size",
        str(int(forward_eval_batch_size)),
        "--max_directed_pairs",
        str(int(max_directed_pairs)),
        "--mean_margin_threshold",
        str(float(args.mean_margin_threshold)),
        "--positive_rate_threshold",
        str(float(args.positive_rate_threshold)),
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
    ]
    return cmd


def run_command(cmd):
    print("[tv-sweep] exec:")
    print("  " + " ".join(shlex.quote(str(part)) for part in cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def run_commands_parallel(commands, devices):
    if not commands:
        return
    if not devices:
        for cmd in commands:
            run_command(cmd)
        return

    pending = list(commands)
    running = []

    def normalize_visible_device(device_name):
        device_name = str(device_name)
        if device_name.startswith("cuda:"):
            return device_name.split(":", 1)[1]
        return device_name

    available_devices = [normalize_visible_device(device_name) for device_name in devices]

    while pending or running:
        while pending and available_devices:
            visible_device = available_devices.pop(0)
            cmd = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = visible_device
            cmd = [str(part) for part in cmd]
            print("[tv-sweep] exec on CUDA_VISIBLE_DEVICES=" + visible_device + ":")
            print("  " + " ".join(shlex.quote(str(part)) for part in cmd))
            process = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
            running.append((process, visible_device, cmd))

        still_running = []
        for process, visible_device, cmd in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, visible_device, cmd))
                continue
            available_devices.append(visible_device)
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
            print(
                "[tv-sweep] finished on CUDA_VISIBLE_DEVICES="
                f"{visible_device}: {' '.join(shlex.quote(str(part)) for part in cmd)}"
            )
        running = still_running

        if pending and not available_devices and running:
            running[0][0].wait()


def parse_list(value):
    if value == "":
        return []
    return literal_eval(value)


def load_pair_rows(csv_path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for key in [
                "mean_margin",
                "median_margin",
                "std_margin",
                "ci95_low",
                "positive_rate",
                "mean_without_top1_margin",
                "trimmed_mean_margin",
                "top10pct_positive_share",
            ]:
                parsed[key] = float(parsed[key])
            for key in ["first_unit", "second_unit", "num_valid_batches", "num_total_batches"]:
                parsed[key] = int(parsed[key])
            for key in ["first_blocks_original", "second_blocks_original"]:
                parsed[key] = parse_list(parsed[key])
            parsed["gap"] = (
                int(parsed["second_blocks_original"][0]) - int(parsed["first_blocks_original"][0])
                if parsed["first_blocks_original"] and parsed["second_blocks_original"]
                else None
            )
            parsed["select_score"] = parsed["mean_margin"] * parsed["positive_rate"]
            parsed["snr"] = (
                parsed["mean_margin"] / parsed["std_margin"]
                if parsed["std_margin"] != 0
                else math.inf
            )
            rows.append(parsed)
    return rows


def disjoint_select(rows, rank_key, mean_margin_threshold):
    candidates = [row for row in rows if row["mean_margin"] > mean_margin_threshold]
    candidates.sort(key=lambda row: row[rank_key], reverse=True)
    used = set()
    selected = []
    for row in candidates:
        if row["first_unit"] in used or row["second_unit"] in used:
            continue
        used.add(row["first_unit"])
        used.add(row["second_unit"])
        selected.append(row)
    return selected


def count_category(rows, category):
    if category == "gap_plus_1":
        return sum(row["gap"] == 1 for row in rows)
    if category == "gap_minus_1":
        return sum(row["gap"] == -1 for row in rows)
    if category == "skip_abs_gt_1":
        return sum(row["gap"] is not None and abs(row["gap"]) > 1 for row in rows)
    raise ValueError(category)


def mean_or_zero(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize_probe(out_dir, mean_margin_threshold):
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "pair_stats.csv"
    if not summary_path.exists() or not csv_path.exists():
        raise FileNotFoundError(f"Missing probe outputs in {out_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    meta = summary.get("run_meta", {})
    rows = load_pair_rows(csv_path)
    high = [row for row in rows if row["mean_margin"] > mean_margin_threshold]
    adj_all = [row for row in rows if row["gap"] == 1]
    num_units = int(meta.get("num_units", 0))
    expected_adjacent = max(0, num_units - 1)
    seen_adjacent = {
        (int(row["first_blocks_original"][0]), int(row["second_blocks_original"][0]))
        for row in adj_all
        if row["first_blocks_original"] and row["second_blocks_original"]
    }
    missing_adjacent = [
        (idx, idx + 1)
        for idx in range(expected_adjacent)
        if (idx, idx + 1) not in seen_adjacent
    ]
    selected_by_margin = disjoint_select(rows, "mean_margin", mean_margin_threshold)
    selected_by_select = disjoint_select(rows, "select_score", mean_margin_threshold)

    return {
        "num_units": num_units,
        "block_order_block_len": int(meta.get("block_order_block_len", 0)),
        "num_scored_pairs": len(rows),
        "num_high_pairs": len(high),
        "high_gap_plus_1": count_category(high, "gap_plus_1"),
        "high_gap_minus_1": count_category(high, "gap_minus_1"),
        "high_skip_abs_gt_1": count_category(high, "skip_abs_gt_1"),
        "high_gap_plus_1_fraction": count_category(high, "gap_plus_1") / len(high) if high else 0.0,
        "adjacent_scored": len(adj_all),
        "adjacent_expected": expected_adjacent,
        "adjacent_missing": len(missing_adjacent),
        "adjacent_low_or_equal_threshold": sum(
            row["mean_margin"] <= mean_margin_threshold for row in adj_all
        ),
        "adjacent_mean_margin": mean_or_zero(row["mean_margin"] for row in adj_all),
        "adjacent_high_mean_margin": mean_or_zero(
            row["mean_margin"] for row in high if row["gap"] == 1
        ),
        "skip_high_max_margin": max(
            [row["mean_margin"] for row in high if row["gap"] is not None and abs(row["gap"]) > 1],
            default=0.0,
        ),
        "reverse_high_max_margin": max(
            [row["mean_margin"] for row in high if row["gap"] == -1],
            default=0.0,
        ),
        "selected_margin_total": len(selected_by_margin),
        "selected_margin_gap_plus_1": count_category(selected_by_margin, "gap_plus_1"),
        "selected_margin_gap_minus_1": count_category(selected_by_margin, "gap_minus_1"),
        "selected_margin_skip_abs_gt_1": count_category(selected_by_margin, "skip_abs_gt_1"),
        "selected_select_total": len(selected_by_select),
        "selected_select_gap_plus_1": count_category(selected_by_select, "gap_plus_1"),
        "selected_select_gap_minus_1": count_category(selected_by_select, "gap_minus_1"),
        "selected_select_skip_abs_gt_1": count_category(selected_by_select, "skip_abs_gt_1"),
    }


def write_comparison_csv(path, rows):
    fieldnames = [
        "label",
        "score_mode",
        "score_weight",
        "tv_weight",
        "drop_weight",
        "num_units",
        "block_order_block_len",
        "num_scored_pairs",
        "num_high_pairs",
        "high_gap_plus_1",
        "high_gap_minus_1",
        "high_skip_abs_gt_1",
        "high_gap_plus_1_fraction",
        "adjacent_scored",
        "adjacent_expected",
        "adjacent_missing",
        "adjacent_low_or_equal_threshold",
        "adjacent_mean_margin",
        "adjacent_high_mean_margin",
        "skip_high_max_margin",
        "reverse_high_max_margin",
        "selected_margin_total",
        "selected_margin_gap_plus_1",
        "selected_margin_gap_minus_1",
        "selected_margin_skip_abs_gt_1",
        "selected_select_total",
        "selected_select_gap_plus_1",
        "selected_select_gap_minus_1",
        "selected_select_skip_abs_gt_1",
        "out_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    out_root = resolve_path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    score_weights = parse_csv_floats(
        args.drop_weights if args.score_mode == "signed_drop" else args.tv_weights
    )
    jobs = [
        ("token_block1", resolve_path(args.token_ckpt)),
        ("block64", resolve_path(args.block64_ckpt)),
    ]

    commands_to_run = []
    comparison_rows = []
    for label, ckpt_path in jobs:
        for score_weight in score_weights:
            weight_dir = (
                drop_to_name(score_weight)
                if args.score_mode == "signed_drop"
                else tv_to_name(score_weight)
            )
            out_dir = out_root / label / weight_dir
            pair_stats_path = out_dir / "pair_stats.csv"
            if not args.skip_run and (args.force or not pair_stats_path.exists()):
                cmd = build_probe_command(args, label, ckpt_path, out_dir, score_weight)
                commands_to_run.append(cmd)
            else:
                print(f"[tv-sweep] using existing output: {out_dir}")

    if commands_to_run:
        parallel_devices = parse_csv_strings(args.parallel_devices)
        run_commands_parallel(commands_to_run, parallel_devices)

    for label, _ckpt_path in jobs:
        for score_weight in score_weights:
            weight_dir = (
                drop_to_name(score_weight)
                if args.score_mode == "signed_drop"
                else tv_to_name(score_weight)
            )
            out_dir = out_root / label / weight_dir
            summary = summarize_probe(out_dir, args.mean_margin_threshold)
            row = {
                "label": label,
                "score_mode": str(args.score_mode),
                "score_weight": float(score_weight),
                "tv_weight": float(score_weight) if args.score_mode == "abs_tv" else "",
                "drop_weight": float(score_weight) if args.score_mode == "signed_drop" else "",
                "out_dir": str(out_dir),
                **summary,
            }
            comparison_rows.append(row)

    comparison_json = out_root / "comparison.json"
    comparison_csv = out_root / "comparison.csv"
    comparison_json.write_text(
        json.dumps(comparison_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_comparison_csv(comparison_csv, comparison_rows)
    print(f"[tv-sweep] saved comparison JSON: {comparison_json}")
    print(f"[tv-sweep] saved comparison CSV : {comparison_csv}")


if __name__ == "__main__":
    main()
