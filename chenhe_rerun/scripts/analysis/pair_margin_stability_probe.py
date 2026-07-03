"""
Probe whether high mean directed-pair margins are driven by a few batches.

This script intentionally mirrors the scoring path in
scripts/benchmark/hierarchical_structured_benchmark.py, but keeps the
per-batch margin samples instead of only accumulating a mean.

Typical usage:
python scripts/analysis/pair_margin_stability_probe.py \
  --ckpt_path out/base/permute/seq256/block64/out-wikitext103-seq256-random-b64-permute-block \
  --out_dir Report/analysis/pair_margin_stability/seq256_perm_block64 \
  --batch_size 8 \
  --num_batches 48 \
  --pair_mining_mode attention_pruned \
  --attn_top_k 8
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

from scripts.benchmark.hierarchical_structured_benchmark import (  # noqa: E402
    build_initial_units,
    build_random_suffix_orders_for_unit_pairs,
    build_model,
    compute_signal_from_block_losses,
    compute_unit_window_score,
    flatten_unit_order,
    get_autocast_context,
    load_block_permutation_from_checkpoint,
    load_checkpoint,
    load_initial_units,
    load_tokens,
    map_blocks_to_original,
    mine_attention_pruned_unit_candidates,
    resolve_data_dir,
    sample_batch,
    units_are_singleton_blocks,
    evaluate_orders_with_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure per-batch margin stability for the existing directed-pair "
            "scoring mechanism."
        )
    )
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--initial_units_json", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=24)
    parser.add_argument("--pair_score_k", type=int, default=2)
    parser.add_argument(
        "--score_mode",
        type=str,
        default="abs_tv",
        choices=["abs_tv", "signed_drop"],
        help=(
            "abs_tv uses -mean(loss) - tv_weight * abs loss variation. "
            "signed_drop uses -mean(loss) + drop_weight * signed loss drop."
        ),
    )
    parser.add_argument("--tv_weight", type=float, default=0.3)
    parser.add_argument(
        "--drop_weight",
        type=float,
        default=0.0,
        help="Weight for signed_drop mode. Positive rewards earlier loss > later loss.",
    )
    parser.add_argument("--pair_chunk_size", type=int, default=8)
    parser.add_argument(
        "--forward_eval_batch_size",
        type=int,
        default=0,
        help="Flattened eval batch size. 0 means batch_size * pair_chunk_size.",
    )
    parser.add_argument(
        "--pair_mining_mode",
        type=str,
        default="attention_pruned",
        choices=["full", "attention_pruned"],
    )
    parser.add_argument("--attn_top_k", type=int, default=8)
    parser.add_argument("--attn_num_batches", type=int, default=24)
    parser.add_argument("--attn_batch_size", type=int, default=32)
    parser.add_argument("--attn_mode", type=str, default="Random", choices=["AR", "Random"])
    parser.add_argument("--attn_symmetrize", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument(
        "--attn_export_type",
        type=str,
        default="with_none",
        choices=["with_none", "without_none"],
    )
    parser.add_argument("--max_directed_pairs", type=int, default=0)
    parser.add_argument("--export_top_k", type=int, default=200)
    parser.add_argument("--mean_margin_threshold", type=float, default=0.02)
    parser.add_argument("--positive_rate_threshold", type=float, default=0.70)
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


def resolve_ckpt_path(raw_path: Path):
    ckpt_path = raw_path
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "ckpt.pt"
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def finite_float(value, default=None):
    value = float(value)
    if math.isfinite(value):
        return value
    return default


def limit_pair_list_symmetric(pair_list, max_directed_pairs):
    max_directed_pairs = int(max_directed_pairs)
    if max_directed_pairs <= 0 or len(pair_list) <= max_directed_pairs:
        return [(int(i), int(j)) for i, j in pair_list]

    pair_set = {(int(i), int(j)) for i, j in pair_list}
    selected = []
    selected_set = set()
    for i, j in pair_list:
        i, j = int(i), int(j)
        if len(selected) + 2 > max_directed_pairs:
            break
        for pair in ((i, j), (j, i)):
            if pair in pair_set and pair not in selected_set:
                selected.append(pair)
                selected_set.add(pair)
    return selected


def build_unit_pairs(args, model, units, tokens, token_perm, autocast_context):
    if args.pair_mining_mode == "full":
        return [(i, j) for i in range(len(units)) for j in range(len(units)) if i != j], None

    attention_pruning = mine_attention_pruned_unit_candidates(
        model,
        units=units,
        tokens=tokens,
        batch_size=int(args.attn_batch_size),
        num_batches=int(args.attn_num_batches),
        top_k=int(args.attn_top_k),
        mode=str(args.attn_mode),
        token_perm=token_perm,
        device=args.device,
        autocast_context=autocast_context,
        seed=int(args.seed) + 7000,
        symmetrize=str(args.attn_symmetrize),
        export_type=str(args.attn_export_type),
    )
    return attention_pruning["pair_list"], attention_pruning


def compute_signed_drop_from_window(window, drop_weight):
    window = window.float()
    area = -window.mean(dim=-1)
    if window.size(1) < 2:
        signed_drop = torch.zeros_like(area)
    else:
        signed_drop = (window[:, :-1] - window[:, 1:]).sum(dim=-1)
    return area + float(drop_weight) * signed_drop


def compute_signed_drop_from_block_losses(block_losses, early_k, drop_weight):
    block_losses = block_losses.float()
    early_k = max(1, min(int(early_k), block_losses.size(1)))
    return compute_signed_drop_from_window(block_losses[:, :early_k], drop_weight)


def compute_signed_drop_unit_window_score(block_losses, ordered_unit_lengths, window_k, drop_weight):
    scores = []
    cursor = 0
    window_k = max(1, min(int(window_k), len(ordered_unit_lengths)))
    for unit_len in ordered_unit_lengths[:window_k]:
        next_cursor = cursor + int(unit_len)
        scores.append(block_losses[:, cursor:next_cursor].mean(dim=1))
        cursor = next_cursor
    unit_losses = torch.stack(scores, dim=1)
    return compute_signed_drop_from_window(unit_losses, drop_weight)


def compute_local_scores(args, eval_out, units, singleton_units, flat_unit_orders, eval_start, eval_end):
    if singleton_units:
        if args.score_mode == "signed_drop":
            return compute_signed_drop_from_block_losses(
                eval_out["block_losses"],
                early_k=max(2, int(args.pair_score_k)),
                drop_weight=float(args.drop_weight),
            ).cpu()
        return compute_signal_from_block_losses(
            eval_out["block_losses"],
            early_k=max(2, int(args.pair_score_k)),
            tv_weight=float(args.tv_weight),
        )["early_area_plus_tv"].cpu()

    local_scores = []
    local_unit_orders = flat_unit_orders[eval_start:eval_end]
    for row_idx, ordered_units in enumerate(local_unit_orders.tolist()):
        unit_lengths = [len(units[int(unit_idx)]["blocks"]) for unit_idx in ordered_units]
        if args.score_mode == "signed_drop":
            local_scores.append(
                compute_signed_drop_unit_window_score(
                    eval_out["block_losses"][row_idx : row_idx + 1],
                    ordered_unit_lengths=unit_lengths,
                    window_k=int(args.pair_score_k),
                    drop_weight=float(args.drop_weight),
                ).cpu()
            )
        else:
            local_scores.append(
                compute_unit_window_score(
                    eval_out["block_losses"][row_idx : row_idx + 1],
                    ordered_unit_lengths=unit_lengths,
                    window_k=int(args.pair_score_k),
                    tv_weight=float(args.tv_weight),
                ).cpu()
            )
    return torch.cat(local_scores, dim=0)


def score_pairs_per_batch(
    args,
    model,
    units,
    tokens,
    unit_pairs,
    token_perm,
    autocast_context,
):
    num_units = len(units)
    num_pairs = len(unit_pairs)
    pair_chunk_size = max(1, int(args.pair_chunk_size))
    forward_eval_batch_size = (
        int(args.forward_eval_batch_size)
        if int(args.forward_eval_batch_size) > 0
        else int(args.batch_size) * pair_chunk_size
    )
    forward_eval_batch_size = max(1, forward_eval_batch_size)

    rng = np.random.default_rng(int(args.seed))
    generator = torch.Generator(device="cuda" if "cuda" in str(args.device) else "cpu")
    generator.manual_seed(int(args.seed) + 17)
    singleton_units = units_are_singleton_blocks(units)
    batch_score_matrices = []
    total_chunks = max(1, (num_pairs + pair_chunk_size - 1) // pair_chunk_size)

    for batch_idx in range(1, int(args.num_batches) + 1):
        print(f"[probe] scoring batch {batch_idx}/{int(args.num_batches)}")
        idx = sample_batch(tokens, int(args.batch_size), model.config.block_size, rng, args.device)
        if token_perm is not None:
            idx = idx[:, token_perm.to(args.device)]

        batch_score_matrix = torch.full(
            (num_units, num_units),
            float("nan"),
            dtype=torch.float64,
        )
        for chunk_idx, start in enumerate(range(0, num_pairs, pair_chunk_size), start=1):
            chunk_pairs = unit_pairs[start : start + pair_chunk_size]
            pair_orders, pair_unit_orders = build_random_suffix_orders_for_unit_pairs(
                chunk_pairs,
                units=units,
                batch_size=idx.size(0),
                device=args.device,
                generator=generator,
            )
            flat_orders = pair_orders.reshape(idx.size(0) * len(chunk_pairs), model.num_blocks)
            flat_idx = idx.unsqueeze(1).expand(idx.size(0), len(chunk_pairs), idx.size(1)).reshape(
                idx.size(0) * len(chunk_pairs),
                idx.size(1),
            )
            flat_unit_orders = pair_unit_orders.reshape(idx.size(0) * len(chunk_pairs), len(units))

            score_parts = []
            for eval_start in range(0, flat_orders.size(0), forward_eval_batch_size):
                eval_end = min(flat_orders.size(0), eval_start + forward_eval_batch_size)
                eval_out = evaluate_orders_with_metrics(
                    model,
                    flat_idx[eval_start:eval_end],
                    flat_orders[eval_start:eval_end],
                    early_k=max(2, int(args.pair_score_k)),
                    tv_weight=float(args.tv_weight),
                    autocast_context=autocast_context,
                    eval_batch_size=forward_eval_batch_size,
                )
                local_scores = compute_local_scores(
                    args,
                    eval_out=eval_out,
                    units=units,
                    singleton_units=singleton_units,
                    flat_unit_orders=flat_unit_orders,
                    eval_start=eval_start,
                    eval_end=eval_end,
                )
                score_parts.append(local_scores)

            scores = torch.cat(score_parts, dim=0).view(idx.size(0), len(chunk_pairs)).mean(dim=0)
            for local_idx, (first, second) in enumerate(chunk_pairs):
                batch_score_matrix[int(first), int(second)] = float(scores[local_idx].item())

            if (
                chunk_idx == 1
                or chunk_idx % max(1, total_chunks // 5) == 0
                or chunk_idx == total_chunks
            ):
                print(
                    f"[probe] batch {batch_idx}/{int(args.num_batches)} "
                    f"chunk {chunk_idx}/{total_chunks}"
                )
        batch_score_matrices.append(batch_score_matrix)

    return torch.stack(batch_score_matrices, dim=0)


def trimmed_mean(values, trim_fraction):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    trim_n = int(math.floor(values.size * max(0.0, min(float(trim_fraction), 0.49))))
    sorted_values = np.sort(values)
    if trim_n > 0 and sorted_values.size > 2 * trim_n:
        sorted_values = sorted_values[trim_n:-trim_n]
    return float(sorted_values.mean()) if sorted_values.size else float("nan")


def compute_pair_stats(args, units, unit_pairs, batch_score_matrices, block_perm):
    pair_set = {(int(i), int(j)) for i, j in unit_pairs}
    stats = []
    n = batch_score_matrices.size(0)
    for first, second in unit_pairs:
        first = int(first)
        second = int(second)
        if (second, first) not in pair_set:
            continue
        score_values = batch_score_matrices[:, first, second].double().numpy()
        reverse_values = batch_score_matrices[:, second, first].double().numpy()
        valid_mask = np.isfinite(score_values) & np.isfinite(reverse_values)
        if not valid_mask.any():
            continue

        score_values = score_values[valid_mask]
        reverse_values = reverse_values[valid_mask]
        margins = score_values - reverse_values
        valid_n = int(margins.size)
        mean_margin = float(margins.mean())
        std_margin = float(margins.std(ddof=1)) if valid_n > 1 else 0.0
        se_margin = std_margin / math.sqrt(valid_n) if valid_n > 0 else float("nan")
        ci95_low = mean_margin - 1.96 * se_margin if math.isfinite(se_margin) else float("nan")
        ci95_high = mean_margin + 1.96 * se_margin if math.isfinite(se_margin) else float("nan")
        median_margin = float(np.median(margins))
        positive_rate = float(np.mean(margins > 0.0))
        mean_without_top1 = float("nan")
        if valid_n > 1:
            top_idx = int(np.argmax(margins))
            mean_without_top1 = float(np.delete(margins, top_idx).mean())
        positive_mass = np.maximum(margins, 0.0).sum()
        sorted_positive = np.sort(np.maximum(margins, 0.0))[::-1]
        top_count = max(1, int(math.ceil(valid_n * 0.10)))
        top_positive_share = (
            float(sorted_positive[:top_count].sum() / positive_mass)
            if positive_mass > 0
            else float("nan")
        )
        trim_mean = trimmed_mean(margins, args.trim_fraction)
        mean_minus_trimmed = mean_margin - trim_mean if math.isfinite(trim_mean) else float("nan")
        first_blocks = [int(v) for v in units[first]["blocks"]]
        second_blocks = [int(v) for v in units[second]["blocks"]]
        row = {
            "first_unit": first,
            "second_unit": second,
            "first_blocks": first_blocks,
            "second_blocks": second_blocks,
            "first_blocks_original": (
                map_blocks_to_original(first_blocks, block_perm) if block_perm is not None else []
            ),
            "second_blocks_original": (
                map_blocks_to_original(second_blocks, block_perm) if block_perm is not None else []
            ),
            "mean_score": float(score_values.mean()),
            "mean_reverse_score": float(reverse_values.mean()),
            "mean_margin": mean_margin,
            "median_margin": median_margin,
            "std_margin": std_margin,
            "ci95_low": ci95_low,
            "ci95_high": ci95_high,
            "positive_rate": positive_rate,
            "trimmed_mean_margin": trim_mean,
            "mean_minus_trimmed": mean_minus_trimmed,
            "mean_without_top1_margin": mean_without_top1,
            "top10pct_positive_share": top_positive_share,
            "min_margin": float(margins.min()),
            "max_margin": float(margins.max()),
            "num_valid_batches": valid_n,
            "num_total_batches": int(n),
            "margins": [float(v) for v in margins.tolist()],
        }
        row["flag_mean_good"] = bool(mean_margin > float(args.mean_margin_threshold))
        row["flag_unstable_direction"] = bool(
            row["flag_mean_good"]
            and positive_rate < float(args.positive_rate_threshold)
        )
        row["flag_ci_crosses_zero"] = bool(row["flag_mean_good"] and ci95_low <= 0.0)
        row["flag_median_nonpositive"] = bool(row["flag_mean_good"] and median_margin <= 0.0)
        row["flag_trimmed_below_threshold"] = bool(
            row["flag_mean_good"] and trim_mean <= float(args.mean_margin_threshold)
        )
        row["flag_leave_top1_below_threshold"] = bool(
            row["flag_mean_good"]
            and math.isfinite(mean_without_top1)
            and mean_without_top1 <= float(args.mean_margin_threshold)
        )
        row["flag_top_batch_driven"] = bool(
            row["flag_mean_good"]
            and math.isfinite(top_positive_share)
            and top_positive_share >= float(args.top_share_threshold)
        )
        row["flag_suspicious"] = bool(
            row["flag_unstable_direction"]
            or row["flag_ci_crosses_zero"]
            or row["flag_median_nonpositive"]
            or row["flag_trimmed_below_threshold"]
            or row["flag_leave_top1_below_threshold"]
            or row["flag_top_batch_driven"]
        )
        stats.append(row)

    stats.sort(key=lambda row: (row["mean_margin"], row["positive_rate"]), reverse=True)
    return stats


def compact_pair_row(row, keep_margins=False):
    compact = {
        key: value
        for key, value in row.items()
        if key != "margins" or keep_margins
    }
    return compact


def write_csv(path, rows):
    scalar_keys = [
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
        "mean_minus_trimmed",
        "mean_without_top1_margin",
        "top10pct_positive_share",
        "min_margin",
        "max_margin",
        "num_valid_batches",
        "num_total_batches",
        "flag_mean_good",
        "flag_unstable_direction",
        "flag_ci_crosses_zero",
        "flag_median_nonpositive",
        "flag_trimmed_below_threshold",
        "flag_leave_top1_below_threshold",
        "flag_top_batch_driven",
        "flag_suspicious",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in scalar_keys:
                value = row.get(key)
                if isinstance(value, (list, tuple)):
                    value = json.dumps(value, ensure_ascii=False)
                out[key] = value
            writer.writerow(out)


def summarize(stats, args):
    mean_good = [row for row in stats if row["flag_mean_good"]]
    suspicious = [row for row in mean_good if row["flag_suspicious"]]
    unstable = [row for row in mean_good if row["flag_unstable_direction"]]
    top_driven = [row for row in mean_good if row["flag_top_batch_driven"]]
    ci_cross = [row for row in mean_good if row["flag_ci_crosses_zero"]]
    median_bad = [row for row in mean_good if row["flag_median_nonpositive"]]
    leave_top1_bad = [row for row in mean_good if row["flag_leave_top1_below_threshold"]]
    trimmed_bad = [row for row in mean_good if row["flag_trimmed_below_threshold"]]
    return {
        "num_scored_directed_pairs": int(len(stats)),
        "mean_margin_threshold": float(args.mean_margin_threshold),
        "positive_rate_threshold": float(args.positive_rate_threshold),
        "top_share_threshold": float(args.top_share_threshold),
        "trim_fraction": float(args.trim_fraction),
        "num_mean_good_pairs": int(len(mean_good)),
        "num_suspicious_mean_good_pairs": int(len(suspicious)),
        "fraction_suspicious_among_mean_good": (
            float(len(suspicious) / len(mean_good)) if mean_good else 0.0
        ),
        "num_unstable_direction": int(len(unstable)),
        "num_top_batch_driven": int(len(top_driven)),
        "num_ci_crosses_zero": int(len(ci_cross)),
        "num_median_nonpositive": int(len(median_bad)),
        "num_leave_top1_below_threshold": int(len(leave_top1_bad)),
        "num_trimmed_below_threshold": int(len(trimmed_bad)),
        "top_by_mean_margin": [compact_pair_row(row) for row in stats[: int(args.export_top_k)]],
        "top_suspicious_mean_good": [
            compact_pair_row(row)
            for row in sorted(
                suspicious,
                key=lambda row: (
                    row["mean_margin"],
                    row["top10pct_positive_share"]
                    if math.isfinite(float(row["top10pct_positive_share"]))
                    else -1.0,
                ),
                reverse=True,
            )[: int(args.export_top_k)]
        ],
        "top_unstable_direction": [
            compact_pair_row(row)
            for row in sorted(
                unstable,
                key=lambda row: (row["mean_margin"], -row["positive_rate"]),
                reverse=True,
            )[: int(args.export_top_k)]
        ],
        "top_batch_driven": [
            compact_pair_row(row)
            for row in sorted(
                top_driven,
                key=lambda row: (
                    row["top10pct_positive_share"]
                    if math.isfinite(float(row["top10pct_positive_share"]))
                    else -1.0,
                    row["mean_margin"],
                ),
                reverse=True,
            )[: int(args.export_top_k)]
        ],
    }


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
    token_perm = None if permutation_state is None else permutation_state["token_perm"]

    if args.initial_units_json is not None:
        units = load_initial_units(args.initial_units_json, model.num_blocks)
    else:
        units = build_initial_units(model.num_blocks)

    unit_pairs, attention_pruning = build_unit_pairs(
        args,
        model=model,
        units=units,
        tokens=tokens,
        token_perm=token_perm,
        autocast_context=autocast_context,
    )
    unit_pairs = limit_pair_list_symmetric(unit_pairs, args.max_directed_pairs)
    print(
        f"[probe] checkpoint={ckpt_path} num_units={len(units)} "
        f"directed_pairs={len(unit_pairs)} num_batches={int(args.num_batches)}"
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
    stats = compute_pair_stats(
        args,
        units=units,
        unit_pairs=unit_pairs,
        batch_score_matrices=batch_score_matrices,
        block_perm=block_perm,
    )
    summary = summarize(stats, args)
    payload = {
        "run_meta": {
            "ckpt_path": str(ckpt_path),
            "dataset": args.dataset or checkpoint.get("config", {}).get("dataset"),
            "split": str(args.split),
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
            "pair_mining_mode": str(args.pair_mining_mode),
            "attn_top_k": int(args.attn_top_k),
            "attn_num_batches": int(args.attn_num_batches),
            "attn_batch_size": int(args.attn_batch_size),
            "attn_mode": str(args.attn_mode),
            "attn_symmetrize": str(args.attn_symmetrize),
            "attn_export_type": str(args.attn_export_type),
            "max_directed_pairs": int(args.max_directed_pairs),
            "num_units": int(len(units)),
            "num_directed_pairs": int(len(unit_pairs)),
            "block_order_block_len": int(model.block_order_block_len),
            "order_impl": checkpoint.get("model_args", {}).get("order_impl", "block"),
            "permute_data": bool(checkpoint.get("config", {}).get("permute_data", False)),
            "permute_seed": checkpoint.get("config", {}).get("permute_seed", None),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "seed": int(args.seed),
            "note": (
                "Margins are score(first, second) - score(second, first), computed "
                "per sampled data batch using the same early-loss scoring path as "
                "hierarchical_structured_benchmark.py. Positive rate and robust "
                "statistics test whether a high mean is batch-stable."
            ),
        },
        "attention_pruning": (
            {
                "num_attention_samples": int(attention_pruning["num_samples"]),
                "num_undirected_edges": int(len(attention_pruning["undirected_edges"])),
            }
            if attention_pruning is not None
            else None
        ),
        "summary": summary,
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
    with (out_dir / "pair_stats.jsonl").open("w", encoding="utf-8") as handle:
        for row in stats:
            handle.write(json.dumps(compact_pair_row(row, keep_margins=True), ensure_ascii=False) + "\n")
    write_csv(out_dir / "pair_stats.csv", stats)

    print(f"[probe] saved summary to {out_dir / 'summary.json'}")
    print(f"[probe] saved pair stats to {out_dir / 'pair_stats.csv'}")
    print(
        "[probe] mean-good pairs="
        f"{summary['num_mean_good_pairs']} suspicious="
        f"{summary['num_suspicious_mean_good_pairs']} "
        f"fraction={summary['fraction_suspicious_among_mean_good']:.3f}"
    )


if __name__ == "__main__":
    main()
