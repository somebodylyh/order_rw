"""
Evaluate all internal orders for original-frame consecutive token spans.

This is a frozen-checkpoint diagnostic for permuted token/block1 AO-GPT runs.
For each original-frame span such as 10:3 = [10, 11, 12], the script maps those
tokens to current-frame ids, enumerates every internal permutation, places that
permutation at the beginning of the reveal order, appends shared random suffixes,
and reports the mean early prefix loss.

Default seq80 usage:

python scripts/analysis/original_span_permutation_loss_probe.py

Manual spans:

python scripts/analysis/original_span_permutation_loss_probe.py \
  --spans 10:3,20:4,35:5
"""

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from order_utils import evaluate_block_order_quality  # noqa: E402
from scripts.benchmark.hierarchical_structured_benchmark import (  # noqa: E402
    build_model,
    get_autocast_context,
    load_block_permutation_from_checkpoint,
    load_checkpoint,
    load_tokens,
    resolve_data_dir,
    sample_batch,
)


DEFAULT_CKPT = (
    "out/base/permute/seq80/block1/"
    "out-wikitext103-seq80-random-b1-permute/ckpt.pt"
)
DEFAULT_OUT_DIR = "Report/analysis/original_span_permutation_loss_probe/seq80_b1_permute"
DEFAULT_SPANS = "0:3,10:3,20:4,35:4,50:5,65:5"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate internal reveal orders for original-frame consecutive spans "
            "and compare their prefix losses on a frozen AO-GPT checkpoint."
        )
    )
    parser.add_argument("--ckpt_path", type=Path, default=Path(DEFAULT_CKPT))
    parser.add_argument("--out_dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument(
        "--spans",
        type=str,
        default=DEFAULT_SPANS,
        help=(
            "Comma-separated original-frame spans START:LENGTH, for example "
            "'10:3,20:4,35:5'. Lengths 3-5 are recommended."
        ),
    )
    parser.add_argument("--eval_batches", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument(
        "--random_suffixes",
        type=int,
        default=2,
        help="Shared random suffixes per sampled data batch.",
    )
    parser.add_argument(
        "--forward_eval_batch_size",
        type=int,
        default=64,
        help="Flattened model eval batch size for permutation batches.",
    )
    parser.add_argument(
        "--max_span_len",
        type=int,
        default=5,
        help="Safety cap. 5 means at most 120 permutations per span.",
    )
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
    parser.add_argument("--print_top_k", type=int, default=8)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_spans(raw_spans: str, block_size: int, max_span_len: int):
    spans = []
    for raw_item in str(raw_spans).split(","):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        if ":" not in raw_item:
            raise ValueError(f"Span must use START:LENGTH format, got {raw_item!r}")
        raw_start, raw_len = raw_item.split(":", 1)
        start = int(raw_start)
        length = int(raw_len)
        if length < 2:
            raise ValueError(f"Span length must be >=2, got {raw_item!r}")
        if length > int(max_span_len):
            raise ValueError(
                f"Span {raw_item!r} length exceeds --max_span_len={int(max_span_len)}."
            )
        if start < 0 or start + length > int(block_size):
            raise ValueError(
                f"Span {raw_item!r} is outside original-frame block_size={int(block_size)}."
            )
        spans.append({"start": start, "length": length})
    if not spans:
        raise ValueError("No spans were provided.")
    return spans


def normalized_kendall_vs_l2r(order_original):
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


def summarize(values):
    values = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def map_original_span_to_current(original_tokens, permutation_state):
    if permutation_state is None:
        return [int(value) for value in original_tokens]
    inverse = permutation_state["inverse_block_perm"].to(dtype=torch.long, device="cpu")
    return [int(inverse[int(original_idx)].item()) for original_idx in original_tokens]


def map_current_order_to_original(current_order, permutation_state):
    if permutation_state is None:
        return [int(value) for value in current_order]
    block_perm = permutation_state["block_perm"].to(dtype=torch.long, device="cpu")
    return [int(block_perm[int(current_idx)].item()) for current_idx in current_order]


@torch.no_grad()
def evaluate_span_permutations(
    model,
    tokens,
    permutation_state,
    original_tokens,
    current_tokens,
    args,
    autocast_context,
):
    device = str(args.device)
    rng = np.random.default_rng(int(args.seed) + 1000 * int(original_tokens[0]) + len(original_tokens))
    all_current = list(range(int(model.num_blocks)))
    component_set = set(int(value) for value in current_tokens)
    remaining = [idx for idx in all_current if idx not in component_set]
    permutations = [tuple(int(v) for v in order) for order in itertools.permutations(current_tokens)]
    num_perms = len(permutations)
    span_len = len(current_tokens)
    eval_batch_size = max(1, int(args.eval_batch_size))
    random_suffixes = max(1, int(args.random_suffixes))
    forward_eval_batch_size = max(1, int(args.forward_eval_batch_size))
    token_perm = None if permutation_state is None else permutation_state["token_perm"]

    loss_parts = [[] for _ in range(num_perms)]

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
            suffix = list(remaining)
            rng.shuffle(suffix)
            orders = torch.tensor(
                [list(order) + suffix for order in permutations],
                dtype=torch.long,
                device=device,
            )
            flat_orders = orders.unsqueeze(1).expand(num_perms, idx.size(0), model.num_blocks).reshape(
                num_perms * idx.size(0),
                model.num_blocks,
            )
            flat_idx = idx.unsqueeze(0).expand(num_perms, idx.size(0), idx.size(1)).reshape(
                num_perms * idx.size(0),
                idx.size(1),
            )

            prefix_losses = []
            for start in range(0, flat_orders.size(0), forward_eval_batch_size):
                end = min(start + forward_eval_batch_size, flat_orders.size(0))
                metrics = evaluate_block_order_quality(
                    model,
                    flat_idx[start:end],
                    flat_orders[start:end],
                    prefix_k=span_len,
                    block_len=model.block_order_block_len,
                    autocast_context=autocast_context,
                )
                loss = metrics["block_losses"].float()[:, :span_len].mean(dim=1)
                prefix_losses.append(loss.detach().cpu())
            prefix_loss = torch.cat(prefix_losses, dim=0).view(num_perms, idx.size(0))
            for perm_idx in range(num_perms):
                loss_parts[perm_idx].extend(float(v) for v in prefix_loss[perm_idx].tolist())

    rows = []
    for perm_idx, current_order in enumerate(permutations):
        order_original = map_current_order_to_original(current_order, permutation_state)
        losses = loss_parts[perm_idx]
        loss_summary = summarize(losses)
        rows.append(
            {
                "order_current": [int(value) for value in current_order],
                "order_original": order_original,
                "mean_prefix_loss": float(loss_summary["mean"]),
                "std_prefix_loss": float(loss_summary["std"]),
                "num_loss_samples": int(loss_summary["n"]),
                "kendall_vs_original_l2r": normalized_kendall_vs_l2r(order_original),
                "is_original_l2r": order_original == sorted(order_original),
                "is_reverse_l2r": order_original == sorted(order_original, reverse=True),
            }
        )

    rows.sort(
        key=lambda row: (
            row["mean_prefix_loss"],
            row["kendall_vs_original_l2r"],
            tuple(row["order_original"]),
        )
    )
    random_uniform_mean = float(np.mean([row["mean_prefix_loss"] for row in rows]))
    best_loss = float(rows[0]["mean_prefix_loss"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)
        row["delta_vs_best"] = float(row["mean_prefix_loss"] - best_loss)
        row["delta_vs_uniform_permutation_mean"] = float(
            row["mean_prefix_loss"] - random_uniform_mean
        )
    return rows, {
        "original_tokens": [int(value) for value in original_tokens],
        "current_tokens": [int(value) for value in current_tokens],
        "num_permutations": int(num_perms),
        "random_uniform_permutation_mean_loss": random_uniform_mean,
        "best_order_original": rows[0]["order_original"],
        "best_order_current": rows[0]["order_current"],
        "best_loss": best_loss,
        "best_vs_uniform_permutation_mean": float(random_uniform_mean - best_loss),
        "original_l2r_row": next((row for row in rows if row["is_original_l2r"]), None),
        "reverse_l2r_row": next((row for row in rows if row["is_reverse_l2r"]), None),
        "loss_distribution": summarize([row["mean_prefix_loss"] for row in rows]),
    }


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

    if model.block_order_block_len != 1:
        print(
            f"[warn] model.block_order_block_len={model.block_order_block_len}; "
            "this script is intended for token/block1 diagnostics."
        )

    spans = parse_spans(args.spans, block_size=model.num_blocks, max_span_len=args.max_span_len)
    all_rows = []
    summaries = []

    for span_idx, span in enumerate(spans, start=1):
        original_tokens = list(range(int(span["start"]), int(span["start"]) + int(span["length"])))
        current_tokens = map_original_span_to_current(original_tokens, permutation_state)
        print(
            f"[span {span_idx}/{len(spans)}] original={original_tokens} "
            f"current={current_tokens} permutations={math.factorial(len(original_tokens))}"
        )
        rows, summary = evaluate_span_permutations(
            model=model,
            tokens=tokens,
            permutation_state=permutation_state,
            original_tokens=original_tokens,
            current_tokens=current_tokens,
            args=args,
            autocast_context=autocast_context,
        )
        span_label = f"{original_tokens[0]}_{original_tokens[-1]}_len{len(original_tokens)}"
        for row in rows:
            all_rows.append(
                {
                    "span_label": span_label,
                    "span_start_original": int(original_tokens[0]),
                    "span_len": int(len(original_tokens)),
                    **row,
                    "original_tokens": original_tokens,
                    "current_tokens": current_tokens,
                }
            )
        summaries.append({"span_label": span_label, **summary})

        print(
            f"  best original order={summary['best_order_original']} "
            f"loss={summary['best_loss']:.6f} "
            f"best_vs_uniform={summary['best_vs_uniform_permutation_mean']:.6f}"
        )
        original_row = summary["original_l2r_row"]
        reverse_row = summary["reverse_l2r_row"]
        if original_row is not None:
            print(
                f"  original L2R rank={original_row['rank']} "
                f"loss={original_row['mean_prefix_loss']:.6f} "
                f"delta_best={original_row['delta_vs_best']:.6f}"
            )
        if reverse_row is not None:
            print(
                f"  reverse L2R rank={reverse_row['rank']} "
                f"loss={reverse_row['mean_prefix_loss']:.6f} "
                f"delta_best={reverse_row['delta_vs_best']:.6f}"
            )
        print(f"  top {int(args.print_top_k)}:")
        for row in rows[: max(1, int(args.print_top_k))]:
            print(
                f"    #{row['rank']:03d} order={row['order_original']} "
                f"loss={row['mean_prefix_loss']:.6f} "
                f"kendall={row['kendall_vs_original_l2r']:.3f}"
            )

    csv_path = out_dir / "permutation_losses.csv"
    fieldnames = [
        "span_label",
        "span_start_original",
        "span_len",
        "rank",
        "order_original",
        "order_current",
        "original_tokens",
        "current_tokens",
        "mean_prefix_loss",
        "std_prefix_loss",
        "num_loss_samples",
        "delta_vs_best",
        "delta_vs_uniform_permutation_mean",
        "kendall_vs_original_l2r",
        "is_original_l2r",
        "is_reverse_l2r",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            out = dict(row)
            for key in ("order_original", "order_current", "original_tokens", "current_tokens"):
                out[key] = json.dumps(out[key])
            writer.writerow({key: out.get(key) for key in fieldnames})

    summary_path = out_dir / "summary.json"
    payload = {
        "run_meta": {
            "ckpt_path": str(ckpt_path),
            "dataset": args.dataset or checkpoint.get("config", {}).get("dataset"),
            "split": str(args.split),
            "spans": str(args.spans),
            "eval_batches": int(args.eval_batches),
            "eval_batch_size": int(args.eval_batch_size),
            "random_suffixes": int(args.random_suffixes),
            "forward_eval_batch_size": int(args.forward_eval_batch_size),
            "seed": int(args.seed),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "block_order_block_len": int(model.block_order_block_len),
            "num_blocks": int(model.num_blocks),
            "permute_data": bool(checkpoint.get("config", {}).get("permute_data", False)),
        },
        "permute_map": (
            {
                "block_perm_current_to_original": [
                    int(v) for v in permutation_state["block_perm"].tolist()
                ],
                "inverse_block_perm_original_to_current": [
                    int(v) for v in permutation_state["inverse_block_perm"].tolist()
                ],
            }
            if permutation_state is not None
            else None
        ),
        "span_summaries": summaries,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
