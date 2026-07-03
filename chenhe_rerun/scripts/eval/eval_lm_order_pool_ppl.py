"""
Evaluate AO-GPT language PPL using a fixed pool of block orders.

This is useful for comparing a no-prior pool of learned/random orders against
single-order modes such as OriginalL2R. The order pool is assigned to validation
samples in a balanced cycle by default, so K orders and N samples produce one
aggregate PPL over N (sample, order) pairs plus rough per-order stats.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_lm_original_order_ppl import (  # noqa: E402
    apply_sequential_tail_coverage,
    expand_orders,
    get_autocast,
    infer_data_record_mode,
    iter_starts,
    load_checkpoint,
    load_model,
    load_tokens,
    maybe_to_original_frame,
    resolve_data_dir,
    resolve_permutation_state,
    unshuffle_to_current,
)
from order_utils import token_losses_to_block_losses  # noqa: E402


def _nested_get(row: dict[str, Any], field: str) -> Any:
    cur: Any = row
    for part in str(field).split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Could not find field {field!r} in history row.")
        cur = cur[part]
    return cur


def _coerce_order(raw: Any) -> list[int]:
    if isinstance(raw, dict):
        for key in ("order_current", "cached_order_current", "order", "blocks", "block_order"):
            if key in raw:
                return [int(v) for v in raw[key]]
    if isinstance(raw, list):
        return [int(v) for v in raw]
    raise ValueError(f"Cannot coerce order from object of type {type(raw).__name__}.")


def _validate_order(order: list[int], num_blocks: int, *, source: str):
    if len(order) != int(num_blocks) or sorted(order) != list(range(int(num_blocks))):
        raise ValueError(
            f"Invalid order from {source}: len={len(order)}, unique={len(set(order))}, "
            f"expected permutation of 0..{int(num_blocks) - 1}."
        )


def _history_row_orders(row: dict[str, Any], field: str) -> list[tuple[str, list[int]]]:
    if str(field) == "top_candidates.order_current":
        out = []
        for idx, candidate in enumerate(row.get("top_candidates") or []):
            out.append((f"top_candidates[{idx}].order_current", _coerce_order(candidate.get("order_current"))))
        return out
    return [(str(field), _coerce_order(_nested_get(row, field)))]


def _select_records(records: list[dict[str, Any]], selection: str, num_orders: int) -> list[dict[str, Any]]:
    if int(num_orders) <= 0:
        raise ValueError("--num_orders must be positive.")
    if not records:
        raise ValueError("No orders were loaded.")
    if selection == "all":
        return records[: int(num_orders)]
    if selection == "first":
        return records[: int(num_orders)]
    if selection == "last":
        return records[-int(num_orders) :]
    if selection == "uniform":
        if len(records) <= int(num_orders):
            return records
        indices = np.linspace(0, len(records) - 1, int(num_orders), dtype=int).tolist()
        return [records[idx] for idx in indices]
    raise ValueError(f"Unsupported history_selection={selection!r}.")


def load_history_order_pool(path: Path, num_blocks: int, field: str, selection: str, num_orders: int):
    records = []
    for line_idx, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        for field_source, order in _history_row_orders(row, field):
            _validate_order(order, num_blocks, source=f"{path}:{line_idx + 1}:{field_source}")
            records.append(
                {
                    "order": order,
                    "iter": int(row.get("iter", -1)),
                    "line": int(line_idx + 1),
                    "field": field_source,
                    "name": (row.get("top_candidate") or {}).get("name", ""),
                }
            )
    selected = _select_records(records, selection, num_orders)
    if len(selected) < int(num_orders):
        raise ValueError(f"Requested {num_orders} orders but only selected {len(selected)} from {path}.")
    orders = [row["order"] for row in selected]
    return orders, {
        "source": "history_jsonl",
        "path": str(path),
        "field": str(field),
        "selection": str(selection),
        "num_records_loaded": int(len(records)),
        "selected_first_iter": int(selected[0].get("iter", -1)),
        "selected_last_iter": int(selected[-1].get("iter", -1)),
        "selected_first_line": int(selected[0].get("line", -1)),
        "selected_last_line": int(selected[-1].get("line", -1)),
        "selected_records": [
            {
                "idx": int(idx),
                "iter": int(row.get("iter", -1)),
                "line": int(row.get("line", -1)),
                "field": str(row.get("field", "")),
                "name": str(row.get("name", "")),
            }
            for idx, row in enumerate(selected)
        ],
    }


def make_random_order_pool(num_orders: int, num_blocks: int, seed: int):
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    orders = [
        [int(v) for v in torch.randperm(int(num_blocks), generator=rng, device="cpu").tolist()]
        for _ in range(int(num_orders))
    ]
    return orders, {
        "source": "random",
        "seed": int(seed),
    }


def order_pool_stats(orders: list[list[int]]) -> dict[str, Any]:
    unique_count = len({tuple(order) for order in orders})
    ar = list(range(len(orders[0]))) if orders else []
    exact_ar_count = sum(1 for order in orders if order == ar)
    mean_abs_pos_delta = []
    for order in orders:
        pos = [0] * len(order)
        for reveal_idx, block_idx in enumerate(order):
            pos[int(block_idx)] = int(reveal_idx)
        mean_abs_pos_delta.append(float(np.mean([abs(pos[i] - i) for i in range(len(order))])))
    return {
        "num_orders": int(len(orders)),
        "unique_orders": int(unique_count),
        "exact_ar_orders": int(exact_ar_count),
        "mean_abs_position_delta_to_ar_mean": float(np.mean(mean_abs_pos_delta)) if mean_abs_pos_delta else None,
        "mean_abs_position_delta_to_ar_min": float(np.min(mean_abs_pos_delta)) if mean_abs_pos_delta else None,
        "mean_abs_position_delta_to_ar_max": float(np.max(mean_abs_pos_delta)) if mean_abs_pos_delta else None,
    }


def per_order_summary(loss_sum: torch.Tensor, token_count: torch.Tensor, sample_count: torch.Tensor):
    mean_nll = loss_sum / token_count.clamp_min(1.0)
    ppl = torch.exp(mean_nll)
    valid = (sample_count > 0) & torch.isfinite(ppl)
    finite_ppl = ppl[valid]
    if finite_ppl.numel() == 0:
        stats = {"min": None, "median": None, "mean": None, "max": None}
    else:
        stats = {
            "min": float(finite_ppl.min().item()),
            "median": float(finite_ppl.median().item()),
            "mean": float(finite_ppl.mean().item()),
            "max": float(finite_ppl.max().item()),
        }
    rows = []
    for idx in range(int(loss_sum.numel())):
        rows.append(
            {
                "order_idx": int(idx),
                "num_samples": int(sample_count[idx].item()),
                "num_scored_tokens": int(token_count[idx].item()),
                "mean_nll_original_frame": None if sample_count[idx].item() <= 0 else float(mean_nll[idx].item()),
                "ppl_original_frame": None if sample_count[idx].item() <= 0 else float(ppl[idx].item()),
            }
        )
    return stats, rows


@torch.no_grad()
def evaluate_order_pool(args):
    ckpt_path = Path(args.ckpt_path)
    checkpoint = load_checkpoint(ckpt_path)
    model = load_model(checkpoint, args.device)
    block_size = int(args.block_size or checkpoint["model_args"]["block_size"])
    data_dir = resolve_data_dir(args, checkpoint)
    data_record_mode = infer_data_record_mode(data_dir, checkpoint)
    tokens = load_tokens(data_dir, args.split)
    permutation_state = resolve_permutation_state(checkpoint, model, block_size)
    token_perm = None if permutation_state is None else permutation_state["token_perm"]
    token_perm_device = None if token_perm is None else token_perm.to(args.device, dtype=torch.long)

    if str(args.order_source) == "history_jsonl":
        if args.order_history_jsonl is None:
            raise ValueError("--order_source history_jsonl requires --order_history_jsonl.")
        orders, order_source_info = load_history_order_pool(
            Path(args.order_history_jsonl),
            int(model.num_blocks),
            str(args.history_field),
            str(args.history_selection),
            int(args.num_orders),
        )
    elif str(args.order_source) == "random":
        orders, order_source_info = make_random_order_pool(
            int(args.num_orders),
            int(model.num_blocks),
            int(args.order_seed),
        )
    else:
        raise ValueError(f"Unsupported order_source={args.order_source!r}.")

    for idx, order in enumerate(orders):
        _validate_order(order, int(model.num_blocks), source=f"order_pool[{idx}]")
    order_pool = torch.tensor(orders, device=args.device, dtype=torch.long)

    if int(args.num_batches) < 0:
        raise ValueError("--num_batches must be >= 0; use 0 with --sample_mode sequential to evaluate all chunks.")
    if args.sample_mode == "random" and int(args.num_batches) == 0:
        raise ValueError("--sample_mode random requires --num_batches > 0")
    num_samples = int(args.num_batches) * int(args.batch_size)
    starts = iter_starts(
        len(tokens),
        block_size,
        data_record_mode,
        args.sample_mode,
        num_samples,
        int(args.seed),
    )
    starts, score_masks, coverage_info = apply_sequential_tail_coverage(
        starts,
        len(tokens),
        block_size,
        data_record_mode,
        args.sample_mode,
        int(args.num_batches),
        bool(getattr(args, "sequential_cover_tail", False)),
    )
    if not starts:
        raise ValueError("No evaluation samples were selected.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = get_autocast(args.device, args.dtype)
    assign_rng = torch.Generator(device="cuda" if "cuda" in str(args.device) else "cpu")
    assign_rng.manual_seed(int(args.order_assignment_seed))

    total_nll = 0.0
    total_tokens = 0
    total_reveal_nll = 0.0
    total_samples = 0
    token_loss_sum = torch.zeros(block_size, dtype=torch.float64)
    token_loss_count = torch.zeros(block_size, dtype=torch.float64)
    per_order_loss_sum = torch.zeros(int(args.num_orders), dtype=torch.float64)
    per_order_token_count = torch.zeros(int(args.num_orders), dtype=torch.float64)
    per_order_sample_count = torch.zeros(int(args.num_orders), dtype=torch.float64)
    first_order_ids = None

    for batch_start in range(0, len(starts), int(args.batch_size)):
        start_batch = starts[batch_start : batch_start + int(args.batch_size)]
        mask_batch = None
        if score_masks is not None:
            mask_batch = score_masks[batch_start : batch_start + int(args.batch_size)]
        batch_np = np.stack([np.asarray(tokens[s : s + block_size], dtype=np.int64) for s in start_batch], axis=0)
        batch_original = torch.from_numpy(batch_np).to(args.device, non_blocking=True)
        if token_perm_device is not None:
            batch_current = batch_original.index_select(1, token_perm_device)
        else:
            batch_current = batch_original

        batch_size_actual = int(batch_current.size(0))
        if str(args.order_assignment) == "cycle":
            order_ids = (
                torch.arange(total_samples, total_samples + batch_size_actual, device=args.device, dtype=torch.long)
                % int(args.num_orders)
            )
        elif str(args.order_assignment) == "random":
            order_ids = torch.randint(
                low=0,
                high=int(args.num_orders),
                size=(batch_size_actual,),
                generator=assign_rng,
                device=args.device,
                dtype=torch.long,
            )
        else:
            raise ValueError(f"Unsupported order_assignment={args.order_assignment!r}.")
        block_orders = order_pool.index_select(0, order_ids)
        if first_order_ids is None:
            first_order_ids = [int(v) for v in order_ids.detach().cpu().tolist()]

        token_orders = expand_orders(model, block_orders)
        with ctx:
            _, _loss, token_losses_reveal = model(
                batch_current,
                mode=None,
                orders=token_orders,
                return_token_loss=True,
                return_logits=False,
            )
        token_losses_current = unshuffle_to_current(token_losses_reveal, token_orders)
        token_losses_original = maybe_to_original_frame(token_losses_current, permutation_state)
        mask_original = None
        mask_reveal = None
        if mask_batch is not None:
            mask_original = torch.stack(mask_batch, dim=0).to(args.device)
            if token_perm_device is not None:
                mask_current = mask_original.index_select(1, token_perm_device)
            else:
                mask_current = mask_original
            mask_reveal = torch.gather(mask_current, 1, token_orders.long())
        if bool(args.ignore_first_token):
            token_losses_original = token_losses_original[:, 1:]
            token_losses_reveal_for_mean = token_losses_reveal[:, 1:]
            if mask_original is not None:
                mask_original = mask_original[:, 1:]
                mask_reveal = mask_reveal[:, 1:]
            position_offset = 1
        else:
            token_losses_reveal_for_mean = token_losses_reveal
            position_offset = 0

        if mask_original is None:
            token_losses_original_for_mean = token_losses_original
            token_losses_reveal_selected = token_losses_reveal_for_mean
            per_pos_sum = token_losses_original.double().sum(dim=0).detach().cpu()
            per_pos_count = torch.full_like(per_pos_sum, fill_value=float(batch_size_actual), dtype=torch.float64)
            sample_loss_sum = token_losses_original.double().sum(dim=1).detach().cpu()
            sample_token_count = torch.full((batch_size_actual,), float(token_losses_original.size(1)), dtype=torch.float64)
        else:
            token_losses_original_for_mean = token_losses_original[mask_original]
            token_losses_reveal_selected = token_losses_reveal_for_mean[mask_reveal]
            per_pos_sum = (token_losses_original.double() * mask_original.double()).sum(dim=0).detach().cpu()
            per_pos_count = mask_original.double().sum(dim=0).detach().cpu()
            sample_loss_sum = (token_losses_original.double() * mask_original.double()).sum(dim=1).detach().cpu()
            sample_token_count = mask_original.double().sum(dim=1).detach().cpu()

        order_ids_cpu = order_ids.detach().cpu()
        for local_idx, order_idx in enumerate(order_ids_cpu.tolist()):
            per_order_loss_sum[int(order_idx)] += sample_loss_sum[local_idx]
            per_order_token_count[int(order_idx)] += sample_token_count[local_idx]
            per_order_sample_count[int(order_idx)] += 1.0

        total_nll += float(token_losses_original_for_mean.double().sum().item())
        total_reveal_nll += float(token_losses_reveal_selected.double().sum().item())
        total_tokens += int(token_losses_original_for_mean.numel())
        total_samples += batch_size_actual
        token_loss_sum[position_offset : position_offset + per_pos_sum.numel()] += per_pos_sum
        token_loss_count[position_offset : position_offset + per_pos_count.numel()] += per_pos_count

    mean_nll = total_nll / max(1, total_tokens)
    reveal_mean_nll = total_reveal_nll / max(1, total_tokens)
    token_loss_mean = token_loss_sum / token_loss_count.clamp_min(1.0)
    if bool(args.ignore_first_token):
        token_loss_mean[0] = float("nan")
    block_loss_mean = token_losses_to_block_losses(
        token_loss_mean.view(1, -1).float(),
        block_len=int(model.block_order_block_len),
    ).view(-1)
    per_order_stats, per_order_rows = per_order_summary(
        per_order_loss_sum,
        per_order_token_count,
        per_order_sample_count,
    )

    orders_path = out_dir / "orders_used.json"
    orders_path.write_text(json.dumps({"orders": orders, "order_source_info": order_source_info}, indent=2))

    summary = {
        "ckpt_path": str(ckpt_path),
        "checkpoint_iter": int(checkpoint.get("iter_num", -1)),
        "checkpoint_best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "dataset": str(args.dataset or checkpoint.get("config", {}).get("dataset")),
        "data_dir": str(data_dir),
        "split": str(args.split),
        "data_record_mode": str(data_record_mode),
        "sample_mode": str(args.sample_mode),
        "num_batches_arg": int(args.num_batches),
        "batch_size": int(args.batch_size),
        "num_samples": int(total_samples),
        "num_scored_tokens": int(total_tokens),
        "block_size": int(block_size),
        "block_order_block_len": int(model.block_order_block_len),
        "num_blocks": int(model.num_blocks),
        "eval_mode": "OrderPool",
        "order_source_info": order_source_info,
        "order_pool_stats": order_pool_stats(orders),
        "order_assignment": str(args.order_assignment),
        "order_assignment_seed": int(args.order_assignment_seed),
        "first_batch_order_ids": first_order_ids,
        "orders_used_path": str(orders_path),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "ignore_first_token": bool(args.ignore_first_token),
        "coverage_info": coverage_info,
        "mean_nll_original_frame": float(mean_nll),
        "ppl_original_frame": float(math.exp(mean_nll)),
        "mean_nll_reveal_frame": float(reveal_mean_nll),
        "ppl_reveal_frame": float(math.exp(reveal_mean_nll)),
        "per_order_ppl_stats": per_order_stats,
        "per_order_results": per_order_rows,
        "frame_mapping": {
            "input_batch_frame": "current_permuted" if permutation_state is not None else "original",
            "loss_report_frame": "true_original_0_to_T_minus_1",
            "unshuffled_from_reveal_order": True,
            "data_permutation_applied": bool(permutation_state is not None),
        },
        "position_loss_original_frame": [None if not math.isfinite(float(v)) else float(v) for v in token_loss_mean.tolist()],
        "block_loss_original_frame": [float(v) for v in block_loss_mean.tolist()],
    }
    if permutation_state is not None:
        summary["data_permutation"] = {
            "block_perm_first16": [int(v) for v in permutation_state["block_perm"][:16].tolist()],
            "inverse_block_perm_first16": [int(v) for v in permutation_state["inverse_block_perm"][:16].tolist()],
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                "mean_nll_original_frame": summary["mean_nll_original_frame"],
                "ppl_original_frame": summary["ppl_original_frame"],
                "mean_nll_reveal_frame": summary["mean_nll_reveal_frame"],
                "ppl_reveal_frame": summary["ppl_reveal_frame"],
                "per_order_ppl_stats": summary["per_order_ppl_stats"],
                "num_samples": summary["num_samples"],
                "num_scored_tokens": summary["num_scored_tokens"],
                "order_pool_stats": summary["order_pool_stats"],
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LM PPL using a fixed pool of block orders.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--sample_mode", choices=("random", "sequential"), default="random")
    parser.add_argument("--sequential_cover_tail", action="store_true")
    parser.add_argument("--order_source", choices=("history_jsonl", "random"), required=True)
    parser.add_argument("--order_history_jsonl", type=Path, default=None)
    parser.add_argument("--history_field", type=str, default="cached_order_current")
    parser.add_argument("--history_selection", choices=("first", "last", "uniform", "all"), default="last")
    parser.add_argument("--num_orders", type=int, default=100)
    parser.add_argument("--order_seed", type=int, default=12345)
    parser.add_argument("--order_assignment", choices=("cycle", "random"), default="cycle")
    parser.add_argument("--order_assignment_seed", type=int, default=23456)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
    )
    parser.add_argument("--ignore_first_token", action="store_true")
    return parser.parse_args()


def main():
    evaluate_order_pool(parse_args())


if __name__ == "__main__":
    main()
