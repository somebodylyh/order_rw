"""
Probe directional context gains inside original-frame local spans.

This is different from ranking whole-span mean prefix loss. It fixes a target
token and asks:

1. Pair context gain:
   Does target j get lower loss when source i is revealed immediately before j,
   compared with same-depth random control contexts that do not include i?

2. Prefix-order-to-target:
   For target k and the same set of previous original tokens, which ordering of
   that prefix gives the lowest target loss? For example, compare [10, 11, 12]
   against [11, 10, 12] by looking only at token 12's loss.

Default seq80 usage:

python scripts/analysis/original_span_conditional_gain_probe.py
"""

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AOGPT import AOGPT, AOGPTConfig  # noqa: E402
from order_utils import (  # noqa: E402
    block_permutation_to_token_permutation,
    build_fixed_block_permutation,
    evaluate_block_order_quality,
    invert_permutation,
)


DEFAULT_CKPT = (
    "out/base/permute/seq80/block1/"
    "out-wikitext103-seq80-random-b1-permute/ckpt.pt"
)
DEFAULT_OUT_DIR = "Report/analysis/original_span_conditional_gain_probe/seq80_b1_permute"
DEFAULT_SPANS = "0:3,10:3,20:4,35:4,50:5,65:5"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure fixed-target conditional gains for original-frame local spans."
    )
    parser.add_argument("--ckpt_path", type=Path, default=Path(DEFAULT_CKPT))
    parser.add_argument("--out_dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--spans", type=str, default=DEFAULT_SPANS)
    parser.add_argument(
        "--adjacent_pair_mode",
        action="store_true",
        help=(
            "Ignore --spans for pair gain and evaluate many original-frame adjacent "
            "pairs i<->i+1 for statistical +1 vs -1 comparison."
        ),
    )
    parser.add_argument(
        "--adjacent_pair_start",
        type=int,
        default=0,
        help="First original i to include for adjacent pair i,i+1.",
    )
    parser.add_argument(
        "--adjacent_pair_end",
        type=int,
        default=0,
        help=(
            "Exclusive upper bound for original i in adjacent pair i,i+1. "
            "0 means num_blocks-1."
        ),
    )
    parser.add_argument(
        "--adjacent_pair_stride",
        type=int,
        default=1,
        help="Stride over original adjacent pair starts.",
    )
    parser.add_argument(
        "--adjacent_pair_sample",
        type=int,
        default=0,
        help="If >0, sample this many adjacent pair starts after range/stride filtering.",
    )
    parser.add_argument("--eval_batches", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument(
        "--random_control_contexts",
        type=int,
        default=16,
        help="Number of same-depth random single-token contexts for pair gain baselines.",
    )
    parser.add_argument(
        "--random_suffixes",
        type=int,
        default=1,
        help=(
            "Random suffixes after the fixed prefix. Target loss is causal and should "
            "not depend on the suffix; this is kept as a sanity knob."
        ),
    )
    parser.add_argument("--forward_eval_batch_size", type=int, default=128)
    parser.add_argument("--max_span_len", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=(
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float32"
        ),
        choices=["float32", "float16", "bfloat16"],
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def get_autocast_context(device: str, dtype: str):
    if "cuda" not in device or dtype == "float32":
        return nullcontext()
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)


def load_checkpoint(ckpt_path: Path, device: str):
    return torch.load(ckpt_path, map_location=device)


def build_model(checkpoint, device: str):
    model = AOGPT(AOGPTConfig(**checkpoint["model_args"]))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def resolve_data_dir(args, checkpoint):
    if args.data_dir is not None:
        return args.data_dir
    dataset = args.dataset or checkpoint.get("config", {}).get("dataset")
    if dataset is None:
        raise ValueError("Could not infer dataset. Pass --dataset or --data_dir.")
    return REPO_ROOT / "data" / dataset


def load_tokens(data_dir: Path, split: str):
    split_path = data_dir / f"{split}.bin"
    if not split_path.exists():
        raise FileNotFoundError(f"Could not find split file: {split_path}")
    return np.memmap(split_path, dtype=np.uint16, mode="r")


def sample_batch(tokens, batch_size: int, block_size: int, rng, device: str, token_perm=None):
    max_start = len(tokens) - block_size
    if max_start <= 0:
        raise ValueError("Dataset split is shorter than block_size.")
    starts = rng.integers(0, max_start, size=int(batch_size))
    batch = torch.stack(
        [torch.from_numpy(tokens[start : start + block_size].astype(np.int64)) for start in starts]
    ).to(device)
    if token_perm is not None:
        batch = batch[:, token_perm.to(device)]
    return batch


def load_block_permutation_from_checkpoint(checkpoint, num_blocks, block_len):
    config = checkpoint.get("config", {})
    if not bool(config.get("permute_data", False)):
        return None
    perm_state = checkpoint.get("data_permutation")
    if perm_state is not None and perm_state.get("block_perm") is not None:
        block_perm = torch.tensor(perm_state["block_perm"], dtype=torch.long, device="cpu")
    else:
        block_perm = build_fixed_block_permutation(num_blocks, int(config.get("permute_seed", 42)))
    inverse_block_perm = invert_permutation(block_perm)
    token_perm = block_permutation_to_token_permutation(block_perm, block_len=block_len)
    return {
        "block_perm": block_perm,
        "inverse_block_perm": inverse_block_perm,
        "token_perm": token_perm,
    }


def parse_spans(raw_spans: str, block_size: int, max_span_len: int):
    spans = []
    for item in str(raw_spans).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Span must use START:LENGTH format, got {item!r}")
        raw_start, raw_len = item.split(":", 1)
        start = int(raw_start)
        length = int(raw_len)
        if length < 2:
            raise ValueError(f"Span length must be >=2, got {item!r}")
        if length > int(max_span_len):
            raise ValueError(f"Span length {length} exceeds --max_span_len={int(max_span_len)}.")
        if start < 0 or start + length > int(block_size):
            raise ValueError(f"Span {item!r} is outside block_size={int(block_size)}.")
        spans.append({"start": start, "length": length})
    if not spans:
        raise ValueError("No spans were provided.")
    return spans


def summarize(values):
    values = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def summarize_with_quantiles(values):
    values = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        **summarize(values),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def normalized_kendall(order_original):
    order_original = [int(value) for value in order_original]
    denom = len(order_original) * (len(order_original) - 1) / 2.0
    if denom <= 0:
        return 0.0
    inversions = 0
    for idx, left in enumerate(order_original):
        for right in order_original[idx + 1:]:
            if left > right:
                inversions += 1
    return float(inversions / denom)


def map_original_to_current(original_ids, permutation_state):
    if permutation_state is None:
        return [int(value) for value in original_ids]
    inverse = permutation_state["inverse_block_perm"].to(dtype=torch.long, device="cpu")
    return [int(inverse[int(value)].item()) for value in original_ids]


def map_current_to_original(current_ids, permutation_state):
    if permutation_state is None:
        return [int(value) for value in current_ids]
    block_perm = permutation_state["block_perm"].to(dtype=torch.long, device="cpu")
    return [int(block_perm[int(value)].item()) for value in current_ids]


def make_order(prefix_current, num_blocks, rng):
    prefix_current = [int(value) for value in prefix_current]
    prefix_set = set(prefix_current)
    remaining = [idx for idx in range(int(num_blocks)) if idx not in prefix_set]
    rng.shuffle(remaining)
    return prefix_current + remaining


@torch.no_grad()
def evaluate_prefix_rows(model, tokens, permutation_state, rows, args, autocast_context):
    if not rows:
        return {}
    device = str(args.device)
    eval_batch_size = max(1, int(args.eval_batch_size))
    random_suffixes = max(1, int(args.random_suffixes))
    forward_eval_batch_size = max(1, int(args.forward_eval_batch_size))
    token_perm = None if permutation_state is None else permutation_state["token_perm"]
    rng = np.random.default_rng(int(args.seed) + 98765)
    losses_by_id = {row["row_id"]: [] for row in rows}

    for batch_idx in range(max(1, int(args.eval_batches))):
        idx = sample_batch(
            tokens,
            eval_batch_size,
            model.config.block_size,
            rng,
            device,
            token_perm=token_perm,
        )
        for suffix_idx in range(random_suffixes):
            orders = torch.tensor(
                [make_order(row["prefix_current"], model.num_blocks, rng) for row in rows],
                dtype=torch.long,
                device=device,
            )
            target_positions = torch.tensor(
                [int(row["target_position"]) for row in rows],
                dtype=torch.long,
                device=device,
            )
            flat_orders = orders.unsqueeze(1).expand(len(rows), idx.size(0), model.num_blocks).reshape(
                len(rows) * idx.size(0),
                model.num_blocks,
            )
            flat_idx = idx.unsqueeze(0).expand(len(rows), idx.size(0), idx.size(1)).reshape(
                len(rows) * idx.size(0),
                idx.size(1),
            )
            flat_target_positions = target_positions.unsqueeze(1).expand(len(rows), idx.size(0)).reshape(-1)

            flat_losses = []
            for start in range(0, flat_orders.size(0), forward_eval_batch_size):
                end = min(start + forward_eval_batch_size, flat_orders.size(0))
                metrics = evaluate_block_order_quality(
                    model,
                    flat_idx[start:end],
                    flat_orders[start:end],
                    prefix_k=int(flat_target_positions[start:end].max().item()) + 1,
                    block_len=model.block_order_block_len,
                    autocast_context=autocast_context,
                )
                block_losses = metrics["block_losses"].float()
                local_positions = flat_target_positions[start:end].to(device=block_losses.device)
                local_losses = block_losses[
                    torch.arange(block_losses.size(0), device=block_losses.device),
                    local_positions,
                ]
                flat_losses.append(local_losses.detach().cpu())

            losses = torch.cat(flat_losses, dim=0).view(len(rows), idx.size(0))
            for row_idx, row in enumerate(rows):
                losses_by_id[row["row_id"]].extend(float(v) for v in losses[row_idx].tolist())

    result = {}
    for row in rows:
        stats = summarize(losses_by_id[row["row_id"]])
        result[row["row_id"]] = {
            **stats,
            "loss_samples": losses_by_id[row["row_id"]],
        }
    return result


def choose_control_contexts(num_blocks, forbidden_current, count, rng):
    candidates = [idx for idx in range(int(num_blocks)) if idx not in forbidden_current]
    if not candidates:
        return []
    count = max(1, int(count))
    if count <= len(candidates):
        return [int(v) for v in rng.choice(candidates, size=count, replace=False).tolist()]
    return [int(v) for v in rng.choice(candidates, size=count, replace=True).tolist()]


def build_pair_context_rows(spans, permutation_state, model, args):
    rows = []
    comparison_meta = []
    rng = np.random.default_rng(int(args.seed) + 111)
    for span in spans:
        original_tokens = list(range(int(span["start"]), int(span["start"]) + int(span["length"])))
        span_label = f"{original_tokens[0]}_{original_tokens[-1]}_len{len(original_tokens)}"
        current_tokens = map_original_to_current(original_tokens, permutation_state)
        original_to_current = dict(zip(original_tokens, current_tokens))
        for left, right in zip(original_tokens[:-1], original_tokens[1:]):
            for source_original, target_original in ((left, right), (right, left)):
                source_current = int(original_to_current[source_original])
                target_current = int(original_to_current[target_original])
                comp_id = f"pair_{span_label}_{source_original}_to_{target_original}"
                rows.append(
                    {
                        "row_id": f"{comp_id}_source",
                        "comparison_id": comp_id,
                        "kind": "source_context",
                        "prefix_current": [source_current, target_current],
                        "target_position": 1,
                    }
                )
                rows.append(
                    {
                        "row_id": f"{comp_id}_target_first",
                        "comparison_id": comp_id,
                        "kind": "target_first",
                        "prefix_current": [target_current],
                        "target_position": 0,
                    }
                )
                controls = choose_control_contexts(
                    model.num_blocks,
                    forbidden_current={source_current, target_current},
                    count=int(args.random_control_contexts),
                    rng=rng,
                )
                for control_idx, control_current in enumerate(controls, start=1):
                    rows.append(
                        {
                            "row_id": f"{comp_id}_control_{control_idx}",
                            "comparison_id": comp_id,
                            "kind": "random_control_context",
                            "control_current": int(control_current),
                            "control_original": map_current_to_original([control_current], permutation_state)[0],
                            "prefix_current": [int(control_current), target_current],
                            "target_position": 1,
                        }
                    )
                comparison_meta.append(
                    {
                        "comparison_id": comp_id,
                        "span_label": span_label,
                        "source_original": int(source_original),
                        "target_original": int(target_original),
                        "source_current": int(source_current),
                        "target_current": int(target_current),
                        "original_gap": int(target_original - source_original),
                    }
                )
    return rows, comparison_meta


def build_adjacent_pair_context_rows(num_blocks, permutation_state, model, args):
    start = max(0, int(args.adjacent_pair_start))
    end = int(args.adjacent_pair_end)
    if end <= 0:
        end = int(num_blocks) - 1
    end = max(start, min(end, int(num_blocks) - 1))
    stride = max(1, int(args.adjacent_pair_stride))
    pair_starts = list(range(start, end, stride))
    if int(args.adjacent_pair_sample) > 0 and int(args.adjacent_pair_sample) < len(pair_starts):
        rng = np.random.default_rng(int(args.seed) + 222)
        pair_starts = sorted(
            int(value)
            for value in rng.choice(pair_starts, size=int(args.adjacent_pair_sample), replace=False).tolist()
        )

    spans = [{"start": int(pair_start), "length": 2} for pair_start in pair_starts]
    rows, comparison_meta = build_pair_context_rows(spans, permutation_state, model, args)
    for row in comparison_meta:
        row["adjacent_pair_start"] = int(min(row["source_original"], row["target_original"]))
    return rows, comparison_meta, pair_starts


def summarize_pair_context(rows, comparison_meta, losses):
    by_comp = defaultdict(list)
    for row in rows:
        by_comp[row["comparison_id"]].append(row)

    output = []
    meta_by_id = {row["comparison_id"]: row for row in comparison_meta}
    for comp_id, comp_rows in sorted(by_comp.items()):
        meta = dict(meta_by_id[comp_id])
        source_row = next(row for row in comp_rows if row["kind"] == "source_context")
        target_first_row = next(row for row in comp_rows if row["kind"] == "target_first")
        control_rows = [row for row in comp_rows if row["kind"] == "random_control_context"]
        source_loss = float(losses[source_row["row_id"]]["mean"])
        target_first_loss = float(losses[target_first_row["row_id"]]["mean"])
        control_means = [float(losses[row["row_id"]]["mean"]) for row in control_rows]
        control_summary = summarize(control_means)
        output.append(
            {
                **meta,
                "source_context_loss": source_loss,
                "target_first_loss": target_first_loss,
                "random_control_context_mean_loss": float(control_summary["mean"]),
                "random_control_context_std_loss": float(control_summary["std"]),
                "num_random_control_contexts": int(control_summary["n"]),
                "gain_vs_target_first": float(target_first_loss - source_loss),
                "gain_vs_random_control": float(control_summary["mean"] - source_loss),
                "num_loss_samples_per_context": int(losses[source_row["row_id"]]["n"]),
            }
        )
    return output


def summarize_adjacent_pair_directions(pair_output):
    forward_rows = [row for row in pair_output if int(row["original_gap"]) == 1]
    reverse_rows = [row for row in pair_output if int(row["original_gap"]) == -1]
    by_start = {}
    for row in pair_output:
        source = int(row["source_original"])
        target = int(row["target_original"])
        start = min(source, target)
        entry = by_start.setdefault(start, {})
        if target - source == 1:
            entry["forward"] = row
        elif target - source == -1:
            entry["reverse"] = row

    comparisons = []
    for start, entry in sorted(by_start.items()):
        forward = entry.get("forward")
        reverse = entry.get("reverse")
        if forward is None or reverse is None:
            continue
        forward_gain = float(forward["gain_vs_random_control"])
        reverse_gain = float(reverse["gain_vs_random_control"])
        forward_loss = float(forward["source_context_loss"])
        reverse_loss = float(reverse["source_context_loss"])
        comparisons.append(
            {
                "pair_start_original": int(start),
                "forward_source_original": int(forward["source_original"]),
                "forward_target_original": int(forward["target_original"]),
                "reverse_source_original": int(reverse["source_original"]),
                "reverse_target_original": int(reverse["target_original"]),
                "forward_gain_vs_random_control": forward_gain,
                "reverse_gain_vs_random_control": reverse_gain,
                "gain_diff_forward_minus_reverse": float(forward_gain - reverse_gain),
                "forward_source_context_loss": forward_loss,
                "reverse_source_context_loss": reverse_loss,
                "loss_diff_reverse_minus_forward": float(reverse_loss - forward_loss),
                "forward_wins_by_gain": bool(forward_gain > reverse_gain),
                "forward_wins_by_loss": bool(forward_loss < reverse_loss),
            }
        )

    gain_diffs = [row["gain_diff_forward_minus_reverse"] for row in comparisons]
    loss_diffs = [row["loss_diff_reverse_minus_forward"] for row in comparisons]
    return {
        "num_adjacent_pairs": int(len(comparisons)),
        "forward_gap_plus1_gain": summarize_with_quantiles(
            [row["gain_vs_random_control"] for row in forward_rows]
        ),
        "reverse_gap_minus1_gain": summarize_with_quantiles(
            [row["gain_vs_random_control"] for row in reverse_rows]
        ),
        "gain_diff_forward_minus_reverse": summarize_with_quantiles(gain_diffs),
        "loss_diff_reverse_minus_forward": summarize_with_quantiles(loss_diffs),
        "forward_win_rate_by_gain": (
            float(np.mean([row["forward_wins_by_gain"] for row in comparisons]))
            if comparisons
            else float("nan")
        ),
        "forward_win_rate_by_loss": (
            float(np.mean([row["forward_wins_by_loss"] for row in comparisons]))
            if comparisons
            else float("nan")
        ),
        "comparisons": comparisons,
    }


def build_prefix_order_rows(spans, permutation_state):
    rows = []
    groups = []
    for span in spans:
        original_tokens = list(range(int(span["start"]), int(span["start"]) + int(span["length"])))
        span_label = f"{original_tokens[0]}_{original_tokens[-1]}_len{len(original_tokens)}"
        for target_offset in range(2, len(original_tokens)):
            prefix_original = original_tokens[:target_offset]
            target_original = original_tokens[target_offset]
            group_id = f"prefix_{span_label}_target_{target_original}"
            current_map = dict(
                zip(original_tokens, map_original_to_current(original_tokens, permutation_state))
            )
            for perm_idx, perm_original in enumerate(itertools.permutations(prefix_original), start=1):
                order_original = [int(v) for v in perm_original] + [int(target_original)]
                prefix_current = [int(current_map[v]) for v in order_original]
                rows.append(
                    {
                        "row_id": f"{group_id}_perm_{perm_idx}",
                        "group_id": group_id,
                        "span_label": span_label,
                        "target_original": int(target_original),
                        "target_current": int(current_map[target_original]),
                        "prefix_original": [int(v) for v in perm_original],
                        "order_original": order_original,
                        "prefix_current": prefix_current,
                        "target_position": len(order_original) - 1,
                        "is_l2r_prefix": list(perm_original) == sorted(perm_original),
                        "is_reverse_prefix": list(perm_original) == sorted(perm_original, reverse=True),
                        "prefix_kendall_vs_l2r": normalized_kendall(perm_original),
                    }
                )
            groups.append(
                {
                    "group_id": group_id,
                    "span_label": span_label,
                    "target_original": int(target_original),
                    "prefix_original_l2r": [int(v) for v in prefix_original],
                    "num_prefix_permutations": math.factorial(len(prefix_original)),
                }
            )
    return rows, groups


def summarize_prefix_order(rows, groups, losses):
    by_group = defaultdict(list)
    for row in rows:
        row_loss = float(losses[row["row_id"]]["mean"])
        by_group[row["group_id"]].append({**row, "target_loss": row_loss})

    output_rows = []
    summary_rows = []
    group_meta = {row["group_id"]: row for row in groups}
    for group_id, group_rows in sorted(by_group.items()):
        group_rows.sort(
            key=lambda row: (
                row["target_loss"],
                row["prefix_kendall_vs_l2r"],
                tuple(row["prefix_original"]),
            )
        )
        best_loss = float(group_rows[0]["target_loss"])
        mean_loss = float(np.mean([row["target_loss"] for row in group_rows]))
        for rank, row in enumerate(group_rows, start=1):
            output_rows.append(
                {
                    **row,
                    "rank": int(rank),
                    "delta_vs_best": float(row["target_loss"] - best_loss),
                    "delta_vs_uniform_prefix_mean": float(row["target_loss"] - mean_loss),
                    "num_loss_samples": int(losses[row["row_id"]]["n"]),
                }
            )
        l2r_row = next(row for row in output_rows if row["group_id"] == group_id and row["is_l2r_prefix"])
        reverse_row = next(row for row in output_rows if row["group_id"] == group_id and row["is_reverse_prefix"])
        summary_rows.append(
            {
                **group_meta[group_id],
                "best_prefix_original": group_rows[0]["prefix_original"],
                "best_order_original": group_rows[0]["order_original"],
                "best_target_loss": best_loss,
                "uniform_prefix_mean_loss": mean_loss,
                "best_vs_uniform_prefix_mean": float(mean_loss - best_loss),
                "l2r_prefix_rank": int(l2r_row["rank"]),
                "l2r_prefix_target_loss": float(l2r_row["target_loss"]),
                "l2r_delta_vs_best": float(l2r_row["delta_vs_best"]),
                "reverse_prefix_rank": int(reverse_row["rank"]),
                "reverse_prefix_target_loss": float(reverse_row["target_loss"]),
                "reverse_delta_vs_best": float(reverse_row["delta_vs_best"]),
            }
        )
    return output_rows, summary_rows


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (list, tuple)):
                    value = json.dumps([int(v) for v in value])
                else:
                    value = value
                out[key] = value
            writer.writerow(out)


def main():
    args = parse_args()
    ckpt_path = resolve_path(args.ckpt_path)
    out_dir = resolve_path(args.out_dir)
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
    spans = parse_spans(args.spans, block_size=model.num_blocks, max_span_len=args.max_span_len)

    if model.block_order_block_len != 1:
        print(
            f"[warn] model.block_order_block_len={model.block_order_block_len}; "
            "this script is intended for token/block1 diagnostics."
        )

    adjacent_pair_starts = []
    if bool(args.adjacent_pair_mode):
        pair_rows, pair_meta, adjacent_pair_starts = build_adjacent_pair_context_rows(
            model.num_blocks,
            permutation_state,
            model,
            args,
        )
        prefix_rows, prefix_groups = [], []
    else:
        pair_rows, pair_meta = build_pair_context_rows(spans, permutation_state, model, args)
        prefix_rows, prefix_groups = build_prefix_order_rows(spans, permutation_state)
    all_rows = pair_rows + prefix_rows
    print(
        f"[probe] rows={len(all_rows)} pair_rows={len(pair_rows)} "
        f"prefix_order_rows={len(prefix_rows)} eval_batches={int(args.eval_batches)}"
    )
    losses = evaluate_prefix_rows(
        model=model,
        tokens=tokens,
        permutation_state=permutation_state,
        rows=all_rows,
        args=args,
        autocast_context=autocast_context,
    )

    pair_output = summarize_pair_context(pair_rows, pair_meta, losses)
    prefix_output, prefix_summary = summarize_prefix_order(prefix_rows, prefix_groups, losses)
    adjacent_pair_summary = (
        summarize_adjacent_pair_directions(pair_output)
        if bool(args.adjacent_pair_mode)
        else None
    )

    pair_csv = out_dir / "pair_context_gain.csv"
    write_csv(
        pair_csv,
        pair_output,
        [
            "span_label",
            "source_original",
            "target_original",
            "original_gap",
            "source_current",
            "target_current",
            "source_context_loss",
            "target_first_loss",
            "random_control_context_mean_loss",
            "random_control_context_std_loss",
            "num_random_control_contexts",
            "gain_vs_target_first",
            "gain_vs_random_control",
            "num_loss_samples_per_context",
        ],
    )

    prefix_csv = out_dir / "prefix_order_target_loss.csv"
    write_csv(
        prefix_csv,
        prefix_output,
        [
            "group_id",
            "span_label",
            "target_original",
            "target_current",
            "rank",
            "prefix_original",
            "order_original",
            "target_loss",
            "delta_vs_best",
            "delta_vs_uniform_prefix_mean",
            "is_l2r_prefix",
            "is_reverse_prefix",
            "prefix_kendall_vs_l2r",
            "num_loss_samples",
        ],
    )

    if adjacent_pair_summary is not None:
        adjacent_comparison_csv = out_dir / "adjacent_pair_direction_comparison.csv"
        write_csv(
            adjacent_comparison_csv,
            adjacent_pair_summary["comparisons"],
            [
                "pair_start_original",
                "forward_source_original",
                "forward_target_original",
                "reverse_source_original",
                "reverse_target_original",
                "forward_gain_vs_random_control",
                "reverse_gain_vs_random_control",
                "gain_diff_forward_minus_reverse",
                "forward_source_context_loss",
                "reverse_source_context_loss",
                "loss_diff_reverse_minus_forward",
                "forward_wins_by_gain",
                "forward_wins_by_loss",
            ],
        )
        adjacent_summary_path = out_dir / "adjacent_pair_gain_summary.json"
        adjacent_summary_path.write_text(
            json.dumps(adjacent_pair_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "run_meta": {
            "ckpt_path": str(ckpt_path),
            "dataset": args.dataset or checkpoint.get("config", {}).get("dataset"),
            "split": str(args.split),
            "spans": str(args.spans),
            "adjacent_pair_mode": bool(args.adjacent_pair_mode),
            "adjacent_pair_start": int(args.adjacent_pair_start),
            "adjacent_pair_end": int(args.adjacent_pair_end),
            "adjacent_pair_stride": int(args.adjacent_pair_stride),
            "adjacent_pair_sample": int(args.adjacent_pair_sample),
            "num_adjacent_pair_starts": int(len(adjacent_pair_starts)),
            "adjacent_pair_starts": [int(value) for value in adjacent_pair_starts],
            "eval_batches": int(args.eval_batches),
            "eval_batch_size": int(args.eval_batch_size),
            "random_control_contexts": int(args.random_control_contexts),
            "random_suffixes": int(args.random_suffixes),
            "forward_eval_batch_size": int(args.forward_eval_batch_size),
            "seed": int(args.seed),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "num_blocks": int(model.num_blocks),
            "block_order_block_len": int(model.block_order_block_len),
            "permute_data": bool(checkpoint.get("config", {}).get("permute_data", False)),
        },
        "pair_context_gain": pair_output,
        "prefix_order_summary": prefix_summary,
        "adjacent_pair_summary": adjacent_pair_summary,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] wrote {pair_csv}")
    print(f"[done] wrote {prefix_csv}")
    if adjacent_pair_summary is not None:
        print(f"[done] wrote {adjacent_comparison_csv}")
        print(f"[done] wrote {adjacent_summary_path}")
        print(
            "[adjacent summary] "
            f"n={adjacent_pair_summary['num_adjacent_pairs']} "
            f"+1_mean={adjacent_pair_summary['forward_gap_plus1_gain']['mean']:.6f} "
            f"-1_mean={adjacent_pair_summary['reverse_gap_minus1_gain']['mean']:.6f} "
            f"+1_win_rate={adjacent_pair_summary['forward_win_rate_by_gain']:.3f}"
        )
    print(f"[done] wrote {summary_path}")
    print("[preview] pair gains:")
    for row in pair_output[:12]:
        print(
            f"  {row['source_original']} -> {row['target_original']}: "
            f"gain_control={row['gain_vs_random_control']:.6f} "
            f"source_loss={row['source_context_loss']:.6f} "
            f"control={row['random_control_context_mean_loss']:.6f}"
        )
    print("[preview] prefix order summaries:")
    for row in prefix_summary[:12]:
        print(
            f"  target={row['target_original']} prefix={row['prefix_original_l2r']}: "
            f"l2r_rank={row['l2r_prefix_rank']} reverse_rank={row['reverse_prefix_rank']} "
            f"best={row['best_prefix_original']} best_vs_uniform={row['best_vs_uniform_prefix_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
