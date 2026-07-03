"""
Probe whether a candidate local token/unit group has internally consistent
directed-pair evidence.

This is an analysis script only. It uses the same pair scoring path as
pair_margin_stability_probe.py, but restricts scoring to all directed pairs
inside one or more small groups. The group order is treated as a candidate
order for analysis: forward adjacent edges are positions i -> i+1, and
forward skip edges are positions i -> j with j > i+1.

Typical usage on a permuted token-level checkpoint:

python scripts/analysis/token_group_consistency_probe.py \
  --ckpt_path out/base/permute/seq80/block1/out-wikitext103-seq80-random-b1-permute/ckpt.pt \
  --out_dir Report/analysis/group_consistency/seq80_auto \
  --auto_original_windows \
  --window_size 4 \
  --num_groups 16 \
  --score_mode signed_drop \
  --drop_weight 0.2

Manual groups are comma-separated units, with semicolons between groups:

python scripts/analysis/token_group_consistency_probe.py \
  --ckpt_path path/to/ckpt.pt \
  --out_dir Report/analysis/group_consistency/manual \
  --groups "44,142,78,179;13,116,19,210" \
  --groups_frame current
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.pair_margin_stability_probe import (  # noqa: E402
    compact_pair_row,
    compute_pair_stats,
    limit_pair_list_symmetric,
    resolve_ckpt_path,
    score_pairs_per_batch,
)
from scripts.benchmark.hierarchical_structured_benchmark import (  # noqa: E402
    build_initial_units,
    build_model,
    get_autocast_context,
    load_block_permutation_from_checkpoint,
    load_checkpoint,
    load_initial_units,
    load_tokens,
    map_blocks_to_original,
    resolve_data_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score all directed pairs inside candidate token/unit groups and "
            "summarize whether adjacent and skip-forward evidence is consistent."
        )
    )
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--initial_units_json", type=Path, default=None)
    parser.add_argument(
        "--groups",
        type=str,
        default="",
        help=(
            "Manual groups, e.g. '1,2,3,4;10,11,12,13'. Values are interpreted "
            "according to --groups_frame."
        ),
    )
    parser.add_argument(
        "--groups_frame",
        type=str,
        default="current",
        choices=["current", "original"],
        help="Frame for --groups. original requires permutation metadata or non-permuted data.",
    )
    parser.add_argument(
        "--auto_original_windows",
        action="store_true",
        help=(
            "Build groups from original-frame contiguous windows, then map them "
            "to current-frame unit ids for scoring. This is for analysis only."
        ),
    )
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--num_groups", type=int, default=16)
    parser.add_argument(
        "--window_stride",
        type=int,
        default=0,
        help="If >0, use deterministic starts spaced by this stride; otherwise sample random starts.",
    )
    parser.add_argument(
        "--start_original",
        type=int,
        default=-1,
        help="Optional first original-frame start for auto windows.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=24)
    parser.add_argument("--pair_score_k", type=int, default=2)
    parser.add_argument(
        "--score_mode",
        type=str,
        default="signed_drop",
        choices=["abs_tv", "signed_drop"],
    )
    parser.add_argument("--tv_weight", type=float, default=0.3)
    parser.add_argument("--drop_weight", type=float, default=0.2)
    parser.add_argument("--pair_chunk_size", type=int, default=8)
    parser.add_argument(
        "--forward_eval_batch_size",
        type=int,
        default=0,
        help="Flattened eval batch size. 0 means batch_size * pair_chunk_size.",
    )
    parser.add_argument("--max_directed_pairs", type=int, default=0)
    parser.add_argument("--mean_margin_threshold", type=float, default=0.02)
    parser.add_argument("--positive_rate_threshold", type=float, default=0.70)
    parser.add_argument(
        "--noncontradiction_margin_floor",
        type=float,
        default=-0.01,
        help="Forward skip edges with mean margin below this are marked contradictory.",
    )
    parser.add_argument(
        "--noncontradiction_positive_rate_floor",
        type=float,
        default=0.45,
        help="Forward skip edges with positive_rate below this are marked contradictory.",
    )
    parser.add_argument("--top_share_threshold", type=float, default=0.50)
    parser.add_argument("--trim_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() else "float32",
        choices=["float32", "float16", "bfloat16"],
    )
    return parser.parse_args()


def parse_manual_groups(raw_groups):
    groups = []
    for group_text in str(raw_groups).split(";"):
        values = [item.strip() for item in group_text.split(",") if item.strip()]
        if not values:
            continue
        group = [int(value) for value in values]
        if len(group) >= 2:
            groups.append(group)
    return groups


def map_original_group_to_current(group_original, inverse_block_perm):
    if inverse_block_perm is None:
        return [int(value) for value in group_original]
    return [int(inverse_block_perm[int(value)].item()) for value in group_original]


def map_current_group_to_original(group_current, block_perm):
    if block_perm is None:
        return [int(value) for value in group_current]
    return [int(block_perm[int(value)].item()) for value in group_current]


def validate_group(group, num_units, label):
    if len(set(group)) != len(group):
        raise ValueError(f"{label} has duplicate unit ids: {group}")
    bad = [value for value in group if value < 0 or value >= num_units]
    if bad:
        raise ValueError(f"{label} has out-of-range unit ids for num_units={num_units}: {bad}")


def build_auto_original_window_groups(args, num_units, inverse_block_perm):
    window_size = max(2, int(args.window_size))
    if window_size > num_units:
        raise ValueError(f"window_size={window_size} exceeds num_units={num_units}")
    max_start = num_units - window_size
    num_groups = max(1, int(args.num_groups))

    if int(args.window_stride) > 0:
        first_start = max(0, int(args.start_original)) if int(args.start_original) >= 0 else 0
        starts = list(range(first_start, max_start + 1, int(args.window_stride)))[:num_groups]
    elif int(args.start_original) >= 0:
        first_start = max(0, min(int(args.start_original), max_start))
        starts = [min(max_start, first_start + offset) for offset in range(num_groups)]
    else:
        rng = np.random.default_rng(int(args.seed) + 909)
        replace = num_groups > max_start + 1
        starts = rng.choice(max_start + 1, size=num_groups, replace=replace).tolist()

    groups = []
    for start in starts:
        group_original = list(range(int(start), int(start) + window_size))
        group_current = map_original_group_to_current(group_original, inverse_block_perm)
        groups.append(
            {
                "source": "auto_original_window",
                "start_original": int(start),
                "group_original": group_original,
                "group_current": group_current,
            }
        )
    return groups


def build_groups(args, num_units, block_perm, inverse_block_perm):
    groups = []
    for group_idx, raw_group in enumerate(parse_manual_groups(args.groups), start=1):
        if args.groups_frame == "original":
            group_original = [int(value) for value in raw_group]
            group_current = map_original_group_to_current(group_original, inverse_block_perm)
        else:
            group_current = [int(value) for value in raw_group]
            group_original = map_current_group_to_original(group_current, block_perm)
        groups.append(
            {
                "source": "manual",
                "manual_group_index": int(group_idx),
                "group_original": group_original,
                "group_current": group_current,
            }
        )

    if bool(args.auto_original_windows):
        groups.extend(build_auto_original_window_groups(args, num_units, inverse_block_perm))

    if not groups:
        raise ValueError("No groups provided. Use --groups and/or --auto_original_windows.")

    for idx, group in enumerate(groups, start=1):
        validate_group(group["group_current"], num_units, f"group {idx} current")
        validate_group(group["group_original"], num_units, f"group {idx} original")
        group["group_id"] = int(idx)
    return groups


def build_intragroup_directed_pairs(groups):
    pairs = []
    seen = set()
    for group in groups:
        values = [int(v) for v in group["group_current"]]
        for first in values:
            for second in values:
                if first == second:
                    continue
                pair = (int(first), int(second))
                if pair in seen:
                    continue
                seen.add(pair)
                pairs.append(pair)
    return pairs


def pair_relation(position_gap):
    if position_gap == 1:
        return "forward_adjacent"
    if position_gap > 1:
        return "forward_skip"
    if position_gap == -1:
        return "reverse_adjacent"
    return "reverse_skip"


def stats_lookup(pair_stats):
    return {
        (int(row["first_unit"]), int(row["second_unit"])): row
        for row in pair_stats
    }


def finite_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key, float("nan"))
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value):
            values.append(value)
    return values


def summarize_rows(rows, desired_sign, args):
    if not rows:
        return {
            "count": 0,
            "mean_margin": float("nan"),
            "median_margin": float("nan"),
            "mean_positive_rate": float("nan"),
            "fraction_sign_consistent": float("nan"),
            "fraction_good": float("nan"),
        }
    margins = finite_values(rows, "mean_margin")
    positive_rates = finite_values(rows, "positive_rate")
    if desired_sign == "positive":
        sign_consistent = [float(row["mean_margin"]) > 0.0 for row in rows]
        good = [
            float(row["mean_margin"]) > float(args.mean_margin_threshold)
            and float(row["positive_rate"]) >= float(args.positive_rate_threshold)
            for row in rows
        ]
    elif desired_sign == "negative":
        sign_consistent = [float(row["mean_margin"]) < 0.0 for row in rows]
        good = [
            float(row["mean_margin"]) < -float(args.mean_margin_threshold)
            and float(row["positive_rate"]) <= 1.0 - float(args.positive_rate_threshold)
            for row in rows
        ]
    else:
        sign_consistent = [
            float(row["mean_margin"]) >= float(args.noncontradiction_margin_floor)
            for row in rows
        ]
        good = [
            float(row["mean_margin"]) >= float(args.noncontradiction_margin_floor)
            and float(row["positive_rate"]) >= float(args.noncontradiction_positive_rate_floor)
            for row in rows
        ]
    return {
        "count": int(len(rows)),
        "mean_margin": float(np.mean(margins)) if margins else float("nan"),
        "median_margin": float(np.median(margins)) if margins else float("nan"),
        "mean_positive_rate": float(np.mean(positive_rates)) if positive_rates else float("nan"),
        "fraction_sign_consistent": float(np.mean(sign_consistent)),
        "fraction_good": float(np.mean(good)),
    }


def build_group_reports(groups, pair_stats, args):
    lookup = stats_lookup(pair_stats)
    group_reports = []
    flat_rows = []
    for group in groups:
        current = [int(v) for v in group["group_current"]]
        original = [int(v) for v in group["group_original"]]
        position_by_unit = {unit_id: pos for pos, unit_id in enumerate(current)}
        pair_rows = []
        for first in current:
            for second in current:
                if first == second:
                    continue
                row = lookup.get((first, second))
                if row is None:
                    continue
                position_gap = int(position_by_unit[second] - position_by_unit[first])
                relation = pair_relation(position_gap)
                out_row = {
                    "group_id": int(group["group_id"]),
                    "source": group["source"],
                    "position_first": int(position_by_unit[first]),
                    "position_second": int(position_by_unit[second]),
                    "position_gap": int(position_gap),
                    "relation": relation,
                    **compact_pair_row(row, keep_margins=False),
                }
                pair_rows.append(out_row)
                flat_rows.append(out_row)

        by_relation = {
            relation: [row for row in pair_rows if row["relation"] == relation]
            for relation in [
                "forward_adjacent",
                "forward_skip",
                "reverse_adjacent",
                "reverse_skip",
            ]
        }
        adjacent_summary = summarize_rows(by_relation["forward_adjacent"], "positive", args)
        skip_summary = summarize_rows(by_relation["forward_skip"], "noncontradictory", args)
        reverse_adjacent_summary = summarize_rows(by_relation["reverse_adjacent"], "negative", args)
        reverse_skip_summary = summarize_rows(by_relation["reverse_skip"], "negative", args)
        group_reports.append(
            {
                **group,
                "group_current": current,
                "group_original": original,
                "num_pair_rows": int(len(pair_rows)),
                "forward_adjacent": adjacent_summary,
                "forward_skip": skip_summary,
                "reverse_adjacent": reverse_adjacent_summary,
                "reverse_skip": reverse_skip_summary,
                "adjacent_chain_good": bool(
                    adjacent_summary["count"] > 0
                    and adjacent_summary["fraction_good"] == 1.0
                ),
                "skip_edges_not_contradictory": bool(
                    skip_summary["count"] == 0
                    or skip_summary["fraction_good"] == 1.0
                ),
                "reverse_adjacent_suppressed": bool(
                    reverse_adjacent_summary["count"] > 0
                    and reverse_adjacent_summary["fraction_good"] == 1.0
                ),
                "pair_rows": pair_rows,
            }
        )
    return group_reports, flat_rows


def write_group_pair_csv(path, rows):
    fieldnames = [
        "group_id",
        "source",
        "relation",
        "position_first",
        "position_second",
        "position_gap",
        "first_unit",
        "second_unit",
        "first_blocks",
        "second_blocks",
        "first_blocks_original",
        "second_blocks_original",
        "mean_score",
        "mean_reverse_score",
        "mean_margin",
        "median_margin",
        "std_margin",
        "ci95_low",
        "ci95_high",
        "positive_rate",
        "trimmed_mean_margin",
        "mean_without_top1_margin",
        "flag_mean_good",
        "flag_suspicious",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, tuple)):
                    value = json.dumps(value, ensure_ascii=False)
                out[key] = value
            writer.writerow(out)


def compact_group_report(row):
    return {key: value for key, value in row.items() if key != "pair_rows"}


def main():
    args = parse_args()
    ckpt_path = resolve_ckpt_path(args.ckpt_path)
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(ckpt_path, args.device)
    model = build_model(checkpoint, args.device)
    autocast_context = get_autocast_context(args.device, args.dtype)
    data_dir = resolve_data_dir(args, checkpoint)
    tokens = load_tokens(data_dir, args.split)
    permutation_state = load_block_permutation_from_checkpoint(
        checkpoint,
        num_blocks=model.num_blocks,
        block_len=model.block_order_block_len,
    )
    block_perm = None if permutation_state is None else permutation_state["block_perm"]
    inverse_block_perm = None if permutation_state is None else permutation_state["inverse_block_perm"]
    token_perm = None if permutation_state is None else permutation_state["token_perm"]

    if args.initial_units_json is not None:
        units = load_initial_units(args.initial_units_json, model.num_blocks)
    else:
        units = build_initial_units(model.num_blocks)

    if not all(len(unit["blocks"]) == 1 for unit in units):
        print(
            "[group-probe] warning: initial units are not all singletons; "
            "group ids refer to current unit ids, not raw token/block ids."
        )

    groups = build_groups(
        args,
        num_units=len(units),
        block_perm=block_perm,
        inverse_block_perm=inverse_block_perm,
    )
    unit_pairs = build_intragroup_directed_pairs(groups)
    unit_pairs = limit_pair_list_symmetric(unit_pairs, args.max_directed_pairs)

    print(
        f"[group-probe] checkpoint={ckpt_path} groups={len(groups)} "
        f"directed_pairs={len(unit_pairs)} batches={int(args.num_batches)}"
    )
    if block_perm is not None:
        first_group = groups[0]
        print(
            "[group-probe] first group current="
            f"{first_group['group_current']} original={first_group['group_original']}"
        )

    batch_score_matrices = score_pairs_per_batch(
        args,
        model=model,
        units=units,
        tokens=tokens,
        unit_pairs=unit_pairs,
        token_perm=token_perm,
        autocast_context=autocast_context,
    )
    pair_stats = compute_pair_stats(
        args,
        units=units,
        unit_pairs=unit_pairs,
        batch_score_matrices=batch_score_matrices,
        block_perm=block_perm,
    )
    group_reports, flat_rows = build_group_reports(groups, pair_stats, args)

    summary = {
        "num_groups": int(len(group_reports)),
        "num_directed_pairs": int(len(unit_pairs)),
        "num_pair_rows": int(len(flat_rows)),
        "num_adjacent_chain_good": int(
            sum(1 for row in group_reports if row["adjacent_chain_good"])
        ),
        "num_skip_edges_not_contradictory": int(
            sum(1 for row in group_reports if row["skip_edges_not_contradictory"])
        ),
        "num_both_adjacent_and_skip_ok": int(
            sum(
                1
                for row in group_reports
                if row["adjacent_chain_good"] and row["skip_edges_not_contradictory"]
            )
        ),
        "mean_forward_adjacent_margin": float(
            np.mean(finite_values(flat_rows_by_relation(flat_rows, "forward_adjacent"), "mean_margin"))
        )
        if finite_values(flat_rows_by_relation(flat_rows, "forward_adjacent"), "mean_margin")
        else float("nan"),
        "mean_forward_skip_margin": float(
            np.mean(finite_values(flat_rows_by_relation(flat_rows, "forward_skip"), "mean_margin"))
        )
        if finite_values(flat_rows_by_relation(flat_rows, "forward_skip"), "mean_margin")
        else float("nan"),
    }
    payload = {
        "run_meta": {
            "ckpt_path": str(ckpt_path),
            "dataset": args.dataset or checkpoint.get("config", {}).get("dataset"),
            "split": str(args.split),
            "initial_units_json": str(args.initial_units_json) if args.initial_units_json else "",
            "batch_size": int(args.batch_size),
            "num_batches": int(args.num_batches),
            "pair_score_k": int(args.pair_score_k),
            "score_mode": str(args.score_mode),
            "tv_weight": float(args.tv_weight),
            "drop_weight": float(args.drop_weight),
            "pair_chunk_size": int(args.pair_chunk_size),
            "forward_eval_batch_size": int(
                args.forward_eval_batch_size
                if int(args.forward_eval_batch_size) > 0
                else int(args.batch_size) * max(1, int(args.pair_chunk_size))
            ),
            "mean_margin_threshold": float(args.mean_margin_threshold),
            "positive_rate_threshold": float(args.positive_rate_threshold),
            "noncontradiction_margin_floor": float(args.noncontradiction_margin_floor),
            "noncontradiction_positive_rate_floor": float(
                args.noncontradiction_positive_rate_floor
            ),
            "block_order_block_len": int(model.block_order_block_len),
            "order_impl": checkpoint.get("model_args", {}).get("order_impl", "block"),
            "permute_data": bool(checkpoint.get("config", {}).get("permute_data", False)),
            "permute_seed": checkpoint.get("config", {}).get("permute_seed", None),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "seed": int(args.seed),
            "note": (
                "Groups are scored in current frame. group_original is analysis-only. "
                "forward_adjacent tests listed group positions i -> i+1; forward_skip "
                "tests i -> j for j > i+1 and is treated as non-contradictory if it "
                "is not reliably negative."
            ),
        },
        "summary": summary,
        "groups": [compact_group_report(row) for row in group_reports],
    }
    if permutation_state is not None:
        payload["permute_map"] = {
            "permute_mode": "block",
            "block_perm": [int(v) for v in block_perm.tolist()],
            "inverse_block_perm": [
                int(v) for v in permutation_state["inverse_block_perm"].tolist()
            ],
        }

    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "group_pair_stats.jsonl").open("w", encoding="utf-8") as handle:
        for row in flat_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_group_pair_csv(out_dir / "group_pair_stats.csv", flat_rows)

    print(f"[group-probe] saved summary to {out_dir / 'summary.json'}")
    print(f"[group-probe] saved group pair stats to {out_dir / 'group_pair_stats.csv'}")
    print(
        "[group-probe] groups with adjacent_chain_good="
        f"{summary['num_adjacent_chain_good']}/{summary['num_groups']}; "
        "skip_not_contradictory="
        f"{summary['num_skip_edges_not_contradictory']}/{summary['num_groups']}; "
        "both="
        f"{summary['num_both_adjacent_and_skip_ok']}/{summary['num_groups']}"
    )


def flat_rows_by_relation(rows, relation):
    return [row for row in rows if row.get("relation") == relation]


if __name__ == "__main__":
    main()
