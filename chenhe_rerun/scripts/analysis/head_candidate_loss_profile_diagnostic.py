#!/usr/bin/env python3
"""No-prior candidate reranking from current loss profiles.

Attention supplies a top-M set of unsigned/signed candidate orders. This script
does not use original order, tau, PPL, target index, or history to select among
them. It evaluates candidate orders with the current language-model objective
and uses reveal-step loss profiles to choose an orientation/order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ANALYSIS_DIR = REPO_ROOT / "scripts" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from head_angle_tau_diagnostic import (  # noqa: E402
    build_model,
    kendall_between_orders,
    load_checkpoint,
    load_permutation_state,
    load_tokens,
    order_to_original,
    parse_heads,
    resolve_data_dir,
    sample_batch,
    tau_to_l2r,
)
from order_utils import expand_block_orders_to_token_orders, token_losses_to_block_losses  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _parse_order(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    return [int(v) for v in json.loads(str(value))]


def _sign(value: float, eps: float = 1e-9) -> int:
    value = float(value)
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _precedence_matrix(order: list[int]) -> np.ndarray:
    order = [int(v) for v in order]
    rank = {value: idx for idx, value in enumerate(order)}
    matrix = np.zeros((len(order), len(order)), dtype=np.float64)
    for i in range(len(order)):
        for j in range(len(order)):
            if i == j:
                continue
            matrix[i, j] = 1.0 if rank[i] < rank[j] else -1.0
    return matrix


def _pairwise_abs_tau(orders: list[list[int]]) -> dict[str, float | int]:
    taus = []
    for i in range(len(orders)):
        for j in range(i + 1, len(orders)):
            taus.append(float(kendall_between_orders(orders[i], orders[j])))
    arr = np.asarray([v for v in taus if math.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {
            "pairwise_order_tau_mean": float("nan"),
            "pairwise_abs_order_tau_mean": float("nan"),
            "num_order_pairs": 0,
        }
    return {
        "pairwise_order_tau_mean": float(arr.mean()),
        "pairwise_abs_order_tau_mean": float(np.abs(arr).mean()),
        "num_order_pairs": int(arr.size),
    }


def _get_autocast_context(device: str, dtype: str):
    if "cuda" not in str(device) or str(dtype) == "float32":
        return nullcontext()
    return torch.amp.autocast(
        device_type="cuda",
        dtype={"float16": torch.float16, "bfloat16": torch.bfloat16}[str(dtype)],
    )


def _expand_orders(model, block_orders: torch.Tensor) -> torch.Tensor:
    return expand_block_orders_to_token_orders(
        block_orders,
        block_len=int(model.block_order_block_len),
        block_order_layout=str(getattr(model, "block_order_layout", "contiguous")),
        image_size=int(getattr(model, "image_size", 0)),
        image_block_size=int(getattr(model, "image_block_size", 0)),
        image_block_height=int(getattr(model, "image_block_height", 0)),
        image_block_width=int(getattr(model, "image_block_width", 0)),
    )


def _loss_profile_scores(profile: np.ndarray, prefix_k: int, prefix_weight: float, full_weight: float, exp_tau: float) -> dict[str, float]:
    values = np.asarray(profile, dtype=np.float64)
    prefix = values[: max(1, min(int(prefix_k), values.size))]
    full = values
    linear_weights = np.linspace(1.0, 0.1, values.size, dtype=np.float64)
    linear_weights /= linear_weights.sum()
    exp_weights = np.exp(-np.arange(values.size, dtype=np.float64) / max(float(exp_tau), 1e-6))
    exp_weights /= exp_weights.sum()
    return {
        "prefix": float(prefix.mean()),
        "full": float(full.mean()),
        "prefix_full": float(float(prefix_weight) * prefix.mean() + float(full_weight) * full.mean()),
        "linear_profile": float((linear_weights * values).sum()),
        "exp_profile": float((exp_weights * values).sum()),
    }


@torch.no_grad()
def _evaluate_loss_profiles(
    *,
    model,
    tokens,
    orders: list[list[int]],
    sample_count: int,
    batch_size: int,
    candidate_batch_size: int,
    seed: int,
    token_perm,
    device: str,
    dtype: str,
) -> dict[tuple[int, ...], dict[str, Any]]:
    unique_orders: list[list[int]] = []
    seen = set()
    for order in orders:
        key = tuple(int(v) for v in order)
        if key not in seen:
            seen.add(key)
            unique_orders.append(list(key))
    if not unique_orders:
        return {}

    num_blocks = len(unique_orders[0])
    sums = {
        tuple(order): {
            "profile_sum": np.zeros(num_blocks, dtype=np.float64),
            "count": 0,
        }
        for order in unique_orders
    }
    order_tensor = torch.tensor(unique_orders, dtype=torch.long, device="cpu")
    rng = np.random.default_rng(int(seed))
    ctx = _get_autocast_context(device, dtype)
    total = 0
    model.eval()
    while total < int(sample_count):
        local_batch = min(int(batch_size), int(sample_count) - int(total))
        idx = sample_batch(
            tokens,
            local_batch,
            int(model.config.block_size),
            rng,
            device,
            token_perm=token_perm,
        )
        for start in range(0, len(unique_orders), int(candidate_batch_size)):
            end = min(len(unique_orders), start + int(candidate_batch_size))
            chunk = order_tensor[start:end].to(device=device)
            chunk_size = int(chunk.size(0))
            idx_flat = idx.unsqueeze(1).expand(-1, chunk_size, -1).reshape(local_batch * chunk_size, -1)
            block_orders = (
                chunk.unsqueeze(0)
                .expand(local_batch, -1, -1)
                .reshape(local_batch * chunk_size, -1)
            )
            token_orders = _expand_orders(model, block_orders)
            with ctx:
                outputs = model(
                    idx_flat,
                    mode=None,
                    orders=token_orders,
                    return_token_loss=True,
                    return_logits=False,
                )
            token_losses = outputs[2]
            block_losses = token_losses_to_block_losses(
                token_losses,
                block_len=int(model.block_order_block_len),
            ).float().view(local_batch, chunk_size, num_blocks)
            profile = block_losses.double().sum(dim=0).detach().cpu().numpy()
            for local_idx, order_idx in enumerate(range(start, end)):
                key = tuple(unique_orders[order_idx])
                sums[key]["profile_sum"] += profile[local_idx]
                sums[key]["count"] += int(local_batch)
        total += int(local_batch)

    out = {}
    for key, item in sums.items():
        count = max(1, int(item["count"]))
        profile = np.asarray(item["profile_sum"], dtype=np.float64) / float(count)
        out[key] = {
            "profile": profile,
            "count": int(item["count"]),
        }
    return out


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), int(row["layer"]), int(row["head"]))].append(row)
    out = []
    for (method, layer, head), items in sorted(groups.items()):
        items = sorted(items, key=lambda r: int(r.get("repeat_idx", 0)))
        taus = [float(row["tau_origin_l2r_diagnostic"]) for row in items]
        signs = [_sign(v) for v in taus]
        nonzero = [s for s in signs if s != 0]
        flips = sum(1 for a, b in zip(nonzero, nonzero[1:]) if a * b < 0)
        orders = [_parse_order(row["order_current"]) for row in items]
        out.append(
            {
                "method": method,
                "layer": int(layer),
                "head": int(head),
                "head_id": f"L{layer}H{head}",
                "num_points": int(len(items)),
                "tau_mean": float(np.mean(taus)),
                "tau_std": float(np.std(taus)),
                "abs_tau_mean": float(np.mean(np.abs(taus))),
                "positive_rate": float(sum(1 for s in signs if s > 0) / max(1, len(signs))),
                "negative_rate": float(sum(1 for s in signs if s < 0) / max(1, len(signs))),
                "sign_flips": int(flips),
                "stable_sign": int(flips == 0 and bool(nonzero)),
                **_pairwise_abs_tau(orders),
            }
        )
    return out


def _candidate_rows_for_group(rows: list[dict[str, Any]], include_reverse: bool) -> dict[tuple[int, int], list[list[int]]]:
    out: dict[tuple[int, int], list[list[int]]] = defaultdict(list)
    seen: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    for row in rows:
        key = (int(row["layer"]), int(row["head"]))
        order = _parse_order(row["candidate_order_current"])
        for candidate in ([order, list(reversed(order))] if include_reverse else [order]):
            ckey = tuple(int(v) for v in candidate)
            if ckey in seen[key]:
                continue
            seen[key].add(ckey)
            out[key].append(list(ckey))
    return out


def _select_best(candidates: list[list[int]], loss_by_order: dict[tuple[int, ...], dict[str, Any]], score_name: str) -> tuple[list[int], float, float]:
    scored = []
    for order in candidates:
        key = tuple(int(v) for v in order)
        item = loss_by_order[key]
        scored.append((float(item["scores"][score_name]), order))
    scored.sort(key=lambda x: (x[0], json.dumps(x[1], separators=(",", ":"))))
    best_score, best_order = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else best_score
    return list(best_order), float(best_score), float(second_score - best_score)


def run(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(Path(args.ckpt_path))
    model = build_model(checkpoint, str(args.device))
    data_dir = resolve_data_dir(args, checkpoint)
    tokens = load_tokens(data_dir, str(args.split), checkpoint)
    permutation_state = load_permutation_state(checkpoint, model)
    token_perm = None if permutation_state is None else permutation_state["token_perm"]
    block_perm = None if permutation_state is None else permutation_state["block_perm"]
    selected_heads = set(parse_heads(str(args.heads), int(model.config.n_layer), int(model.config.n_head), 0))

    candidate_rows_all = [
        row
        for row in _read_csv(Path(args.candidate_csv))
        if (int(row["layer"]), int(row["head"])) in selected_heads
        and int(row["candidate_rank"]) <= int(args.max_candidate_rank)
    ]
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows_all:
        groups[int(row["repeat_idx"])].append(row)

    config = {
        "ckpt_path": str(args.ckpt_path),
        "candidate_csv": str(args.candidate_csv),
        "out_dir": str(out_dir),
        "sample_count": int(args.sample_count),
        "batch_size": int(args.batch_size),
        "candidate_batch_size": int(args.candidate_batch_size),
        "prefix_k": int(args.prefix_k),
        "prefix_weight": float(args.prefix_weight),
        "full_weight": float(args.full_weight),
        "exp_tau": float(args.exp_tau),
        "include_reverse": bool(args.include_reverse),
        "max_candidate_rank": int(args.max_candidate_rank),
        "diagnostic_note": "Original/tau fields are reported after selection only.",
    }
    (out_dir / "config.json").write_text(json.dumps(_json_safe(config), indent=2), encoding="utf-8")

    per_selection_rows: list[dict[str, Any]] = []
    per_candidate_loss_rows: list[dict[str, Any]] = []
    score_names = ["prefix", "prefix_full", "linear_profile", "exp_profile"]

    for repeat_idx in sorted(groups):
        group_rows = groups[repeat_idx]
        candidates_by_head = _candidate_rows_for_group(group_rows, bool(args.include_reverse))
        all_orders = []
        for candidates in candidates_by_head.values():
            all_orders.extend(candidates)
        seed = int(args.seed) + int(args.seed_stride) * int(repeat_idx)
        print(
            f"[candidate-loss-profile] repeat={repeat_idx} heads={len(candidates_by_head)} "
            f"orders={len({tuple(o) for o in all_orders})} samples={int(args.sample_count)}",
            flush=True,
        )
        loss_by_order = _evaluate_loss_profiles(
            model=model,
            tokens=tokens,
            orders=all_orders,
            sample_count=int(args.sample_count),
            batch_size=int(args.batch_size),
            candidate_batch_size=int(args.candidate_batch_size),
            seed=seed,
            token_perm=token_perm,
            device=str(args.device),
            dtype=str(args.dtype),
        )
        for key, item in loss_by_order.items():
            scores = _loss_profile_scores(
                item["profile"],
                prefix_k=int(args.prefix_k),
                prefix_weight=float(args.prefix_weight),
                full_weight=float(args.full_weight),
                exp_tau=float(args.exp_tau),
            )
            item["scores"] = scores
            per_candidate_loss_rows.append(
                {
                    "repeat_idx": int(repeat_idx),
                    "order_current": json.dumps(list(key), separators=(",", ":")),
                    "loss_eval_count": int(item["count"]),
                    "profile_json": json.dumps([float(v) for v in item["profile"].tolist()], separators=(",", ":")),
                    **{f"score_{name}": float(value) for name, value in scores.items()},
                }
            )

        # Direct loss-profile best candidate per head.
        direct_best: dict[str, dict[tuple[int, int], tuple[list[int], float, float]]] = {}
        for score_name in score_names:
            direct_best[score_name] = {}
            for (layer, head), candidates in candidates_by_head.items():
                direct_best[score_name][(layer, head)] = _select_best(candidates, loss_by_order, score_name)

        # Leave-one-out consensus built from each score's loss-profile winners.
        consensus_q: dict[str, dict[tuple[int, int], np.ndarray]] = {name: {} for name in score_names}
        for score_name in score_names:
            items = []
            for key, (order, _score, gap) in direct_best[score_name].items():
                items.append((key, _precedence_matrix(order), max(float(gap), 0.0) + 1e-4))
            for key, _matrix, _weight in items:
                denom = sum(weight for other_key, _m, weight in items if other_key != key)
                if denom <= 0.0:
                    denom = sum(weight for _other_key, _m, weight in items)
                q_matrix = sum(
                    weight * matrix
                    for other_key, matrix, weight in items
                    if other_key != key
                ) / float(max(denom, 1e-8))
                consensus_q[score_name][key] = q_matrix

        for (layer, head), candidates in sorted(candidates_by_head.items()):
            for score_name in score_names:
                best_order, best_score, gap = direct_best[score_name][(layer, head)]
                for method, selected_order, selected_score, selected_gap, align in [
                    (f"{score_name}_best", best_order, best_score, gap, float("nan")),
                ]:
                    original = order_to_original(selected_order, block_perm)
                    tau = tau_to_l2r(original)
                    per_selection_rows.append(
                        {
                            "method": method,
                            "repeat_idx": int(repeat_idx),
                            "layer": int(layer),
                            "head": int(head),
                            "score_name": str(score_name),
                            "selected_score": float(selected_score),
                            "score_gap": float(selected_gap),
                            "consensus_alignment": float(align),
                            "tau_origin_l2r_diagnostic": float(tau),
                            "abs_tau_origin_l2r_diagnostic": abs(float(tau)),
                            "sign_diagnostic": int(_sign(tau)),
                            "order_current": json.dumps(selected_order, separators=(",", ":")),
                            "order_original": json.dumps(original, separators=(",", ":")),
                        }
                    )
                q_matrix = consensus_q[score_name][(layer, head)]
                aligned = []
                for order in candidates:
                    matrix = _precedence_matrix(order)
                    align_score = float((matrix * q_matrix).sum())
                    loss_score = float(loss_by_order[tuple(order)]["scores"][score_name])
                    aligned.append((align_score, -loss_score, order, loss_score))
                aligned.sort(key=lambda item: (-item[0], -item[1], json.dumps(item[2], separators=(",", ":"))))
                align_score, _neg_loss, selected_order, loss_score = aligned[0]
                original = order_to_original(selected_order, block_perm)
                tau = tau_to_l2r(original)
                per_selection_rows.append(
                    {
                        "method": f"{score_name}_consensus_candidate",
                        "repeat_idx": int(repeat_idx),
                        "layer": int(layer),
                        "head": int(head),
                        "score_name": str(score_name),
                        "selected_score": float(loss_score),
                        "score_gap": float(gap),
                        "consensus_alignment": float(align_score),
                        "tau_origin_l2r_diagnostic": float(tau),
                        "abs_tau_origin_l2r_diagnostic": abs(float(tau)),
                        "sign_diagnostic": int(_sign(tau)),
                        "order_current": json.dumps(selected_order, separators=(",", ":")),
                        "order_original": json.dumps(original, separators=(",", ":")),
                    }
                )

        _write_csv(out_dir / "per_selection_results.csv", per_selection_rows)
        _write_csv(out_dir / "per_candidate_loss_profiles.csv", per_candidate_loss_rows)

    head_summary = _summarize(per_selection_rows)
    _write_csv(out_dir / "per_selection_results.csv", per_selection_rows)
    _write_csv(out_dir / "per_candidate_loss_profiles.csv", per_candidate_loss_rows)
    _write_csv(out_dir / "head_summary.csv", head_summary)

    global_summary = {}
    for method in sorted(set(row["method"] for row in head_summary)):
        rows = [row for row in head_summary if row["method"] == method]
        global_summary[method] = {
            "heads": int(len(rows)),
            "stable_sign_heads": int(sum(int(row["stable_sign"]) for row in rows)),
            "total_sign_flips": int(sum(int(row["sign_flips"]) for row in rows)),
            "mean_abs_tau": float(np.mean([float(row["abs_tau_mean"]) for row in rows])) if rows else float("nan"),
            "heads_abs_tau_ge_0.4": int(sum(float(row["abs_tau_mean"]) >= 0.4 for row in rows)),
            "heads_abs_tau_ge_0.5": int(sum(float(row["abs_tau_mean"]) >= 0.5 for row in rows)),
            "mean_pairwise_abs_order_tau": (
                float(np.mean([float(row["pairwise_abs_order_tau_mean"]) for row in rows])) if rows else float("nan")
            ),
        }
    (out_dir / "summary.json").write_text(
        json.dumps(_json_safe({"global_summary": global_summary}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(global_summary), indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--candidate_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--heads", type=str, default="all")
    parser.add_argument("--sample_count", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--candidate_batch_size", type=int, default=8)
    parser.add_argument("--prefix_k", type=int, default=16)
    parser.add_argument("--prefix_weight", type=float, default=0.7)
    parser.add_argument("--full_weight", type=float, default=0.3)
    parser.add_argument("--exp_tau", type=float, default=16.0)
    parser.add_argument("--include_reverse", action="store_true")
    parser.add_argument("--max_candidate_rank", type=int, default=6)
    parser.add_argument("--seed", type=int, default=991337)
    parser.add_argument("--seed_stride", type=int, default=1009)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
        choices=("float32", "float16", "bfloat16"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
