#!/usr/bin/env python3
"""Summarize fixed or online head-signal orientation stability.

This script is diagnostic-only. It can:

1. Read fixed-checkpoint per-probe CSV rows from
   ``head_orientation_loss_flow_diagnostic.py`` and add the same-step
   loss-anchored cross-head consensus orientation.
2. Read online training JSONL rows emitted by ``train.py``.

It reports tau/sign stability against original L2R only as diagnostics; no
original-order field is used to choose orientation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("raw", "loss_oriented", "consensus", "position_anchor", "position_consensus")


def _to_float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _sign(value: float, eps: float = 1e-9) -> int:
    value = float(value)
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _parse_list(value: Any) -> list[int] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, tuple):
        return [int(v) for v in value]
    if isinstance(value, np.ndarray):
        return [int(v) for v in value.tolist()]
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(parsed, (list, tuple)):
        return None
    return [int(v) for v in parsed]


def _tau_to_l2r(order: list[int] | None) -> float:
    if not order:
        return float("nan")
    values = [int(v) for v in order]
    n = len(values)
    total = n * (n - 1) / 2.0
    if total <= 0:
        return 1.0
    inv = 0
    for i in range(n):
        left = values[i]
        for j in range(i + 1, n):
            if left > values[j]:
                inv += 1
    return float(1.0 - 2.0 * float(inv) / float(total))


def _kendall_between_orders(a: list[int], b: list[int]) -> float:
    if len(a) != len(b) or len(a) <= 1:
        return float("nan")
    rank_a = {int(v): i for i, v in enumerate(a)}
    rank_b = {int(v): i for i, v in enumerate(b)}
    values = [int(v) for v in a if int(v) in rank_b]
    n = len(values)
    total = n * (n - 1) / 2.0
    if total <= 0:
        return float("nan")
    concordant = 0
    discordant = 0
    for i in range(n):
        vi = values[i]
        for j in range(i + 1, n):
            vj = values[j]
            same = (rank_a[vi] - rank_a[vj]) * (rank_b[vi] - rank_b[vj])
            if same > 0:
                concordant += 1
            elif same < 0:
                discordant += 1
    return float((concordant - discordant) / total)


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


def _mean_std(values: list[float]) -> tuple[float, float, float, float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        nan = float("nan")
        return nan, nan, nan, nan
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())


def _pairwise_order_summary(orders: list[list[int]]) -> dict[str, float | int]:
    taus = []
    for i in range(len(orders)):
        for j in range(i + 1, len(orders)):
            tau = _kendall_between_orders(orders[i], orders[j])
            if math.isfinite(tau):
                taus.append(float(tau))
    mean, std, min_v, max_v = _mean_std(taus)
    abs_mean, abs_std, abs_min, abs_max = _mean_std([abs(v) for v in taus])
    return {
        "pairwise_order_tau_mean": mean,
        "pairwise_order_tau_std": std,
        "pairwise_order_tau_min": min_v,
        "pairwise_order_tau_max": max_v,
        "pairwise_abs_order_tau_mean": abs_mean,
        "pairwise_abs_order_tau_std": abs_std,
        "pairwise_abs_order_tau_min": abs_min,
        "pairwise_abs_order_tau_max": abs_max,
        "num_order_pairs": int(len(taus)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
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


def _read_fixed_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _add_fixed_consensus(rows: list[dict[str, Any]], leave_one_out: bool) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("ckpt_label", ""),
            row.get("ckpt_iter", ""),
            row.get("split", ""),
            row.get("repeat_idx", ""),
        )
        groups[key].append(row)

    for group_rows in groups.values():
        matrices = []
        weights = []
        for row in group_rows:
            raw_order = _parse_list(row.get("raw_order_current"))
            if not raw_order:
                continue
            margin = _to_float(row.get("loss_selection_margin_reverse_minus_forward"))
            sign = 1.0 if margin >= 0.0 else -1.0
            row["_consensus_index"] = int(len(matrices))
            matrices.append(_precedence_matrix(raw_order) * sign)
            weights.append(abs(float(margin)) + 1e-4)
        if not matrices:
            continue
        weighted_sum = sum(float(w) * m for w, m in zip(weights, matrices))
        total_weight = max(float(sum(weights)), 1e-8)
        full_q = weighted_sum / total_weight
        for row in group_rows:
            idx = row.get("_consensus_index")
            raw_order = _parse_list(row.get("raw_order_current"))
            raw_original = _parse_list(row.get("raw_order_original"))
            if idx is None or not raw_order or not raw_original:
                continue
            idx = int(idx)
            q_matrix = full_q
            if leave_one_out and len(matrices) > 1:
                denom = float(sum(w for j, w in enumerate(weights) if j != idx))
                if denom > 0.0:
                    q_matrix = (
                        sum(float(weights[j]) * matrices[j] for j in range(len(matrices)) if j != idx)
                        / denom
                    )
            raw_matrix = _precedence_matrix(raw_order)
            align = float((raw_matrix * q_matrix).sum())
            reverse = align < 0.0
            consensus_order_current = list(reversed(raw_order)) if reverse else raw_order
            consensus_order_original = list(reversed(raw_original)) if reverse else raw_original
            consensus_tau = _tau_to_l2r(consensus_order_original)
            row["consensus_alignment_margin"] = align
            row["consensus_selected_reverse"] = bool(reverse)
            row["consensus_tau_diagnostic"] = consensus_tau
            row["consensus_abs_tau_diagnostic"] = abs(float(consensus_tau))
            row["consensus_sign_diagnostic"] = _sign(consensus_tau)
            row["consensus_order_current"] = consensus_order_current
            row["consensus_order_original_diagnostic"] = consensus_order_original


def _fixed_timeseries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        layer = _to_int(row.get("layer"))
        head = _to_int(row.get("head"))
        base = {
            "source": "fixed",
            "ckpt_label": row.get("ckpt_label", ""),
            "ckpt_iter": _to_int(row.get("ckpt_iter")),
            "split": row.get("split", ""),
            "repeat_idx": _to_int(row.get("repeat_idx")),
            "iter": _to_int(row.get("ckpt_iter")),
            "time_index": _to_int(row.get("repeat_idx")),
            "layer": layer,
            "head": head,
            "head_id": f"L{layer}H{head}",
            "loss_selection_margin_reverse_minus_forward": _to_float(
                row.get("loss_selection_margin_reverse_minus_forward")
            ),
            "consensus_alignment_margin": _to_float(row.get("consensus_alignment_margin")),
        }
        raw_order_original = _parse_list(row.get("raw_order_original"))
        loss_order_original = _parse_list(row.get("loss_oriented_order_original"))
        consensus_order_original = _parse_list(row.get("consensus_order_original_diagnostic"))
        position_anchor_order_original = _parse_list(
            row.get("position_anchor_order_original_diagnostic")
            or row.get("position_anchor_order_original")
        )
        position_consensus_order_original = _parse_list(
            row.get("position_consensus_order_original_diagnostic")
            or row.get("position_consensus_order_original")
        )
        values = dict(base)
        raw_tau = _to_float(row.get("raw_tau_origin_l2r_diagnostic"))
        loss_tau = _to_float(row.get("loss_oriented_tau_origin_l2r_diagnostic"))
        consensus_tau = _to_float(row.get("consensus_tau_diagnostic"))
        position_anchor_tau = _to_float(
            row.get("position_anchor_tau_origin_l2r_diagnostic")
            if "position_anchor_tau_origin_l2r_diagnostic" in row
            else row.get("position_anchor_tau_diagnostic")
        )
        position_consensus_tau = _to_float(
            row.get("position_consensus_tau_origin_l2r_diagnostic")
            if "position_consensus_tau_origin_l2r_diagnostic" in row
            else row.get("position_consensus_tau_diagnostic")
        )
        values.update(
            {
                "raw_tau": raw_tau,
                "raw_abs_tau": abs(raw_tau) if math.isfinite(raw_tau) else float("nan"),
                "raw_sign": _sign(raw_tau),
                "loss_oriented_tau": loss_tau,
                "loss_oriented_abs_tau": abs(loss_tau) if math.isfinite(loss_tau) else float("nan"),
                "loss_oriented_sign": _sign(loss_tau),
                "consensus_tau": consensus_tau,
                "consensus_abs_tau": abs(consensus_tau) if math.isfinite(consensus_tau) else float("nan"),
                "consensus_sign": _sign(consensus_tau),
                "position_anchor_tau": position_anchor_tau,
                "position_anchor_abs_tau": (
                    abs(position_anchor_tau) if math.isfinite(position_anchor_tau) else float("nan")
                ),
                "position_anchor_sign": _sign(position_anchor_tau),
                "position_consensus_tau": position_consensus_tau,
                "position_consensus_abs_tau": (
                    abs(position_consensus_tau) if math.isfinite(position_consensus_tau) else float("nan")
                ),
                "position_consensus_sign": _sign(position_consensus_tau),
                "raw_order_original": raw_order_original,
                "loss_oriented_order_original": loss_order_original,
                "consensus_order_original": consensus_order_original,
                "position_anchor_order_original": position_anchor_order_original,
                "position_consensus_order_original": position_consensus_order_original,
            }
        )
        out.append(values)
    return out


def _read_history_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _history_timeseries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        layer = _to_int(row.get("layer"))
        head = _to_int(row.get("head"))
        iter_num = _to_int(row.get("iter"))
        raw_tau = _to_float(row.get("raw_tau_diagnostic"))
        loss_tau = _to_float(row.get("loss_oriented_tau_diagnostic"))
        consensus_tau = _to_float(row.get("consensus_tau_diagnostic"))
        position_anchor_tau = _to_float(row.get("position_anchor_tau_diagnostic"))
        position_consensus_tau = _to_float(row.get("position_consensus_tau_diagnostic"))
        out.append(
            {
                "source": "online_training",
                "iter": iter_num,
                "time_index": iter_num,
                "layer": layer,
                "head": head,
                "head_id": f"L{layer}H{head}",
                "samples": _to_int(row.get("samples")),
                "probe_batches": _to_int(row.get("probe_batches")),
                "loss_batches": _to_int(row.get("loss_batches")),
                "loss_eval_count": _to_int(row.get("loss_eval_count")),
                "loss_selection_margin_reverse_minus_forward": _to_float(
                    row.get("loss_selection_margin_reverse_minus_forward")
                ),
                "consensus_alignment_margin": _to_float(row.get("consensus_alignment_margin")),
                "raw_tau": raw_tau,
                "raw_abs_tau": abs(raw_tau) if math.isfinite(raw_tau) else float("nan"),
                "raw_sign": _sign(raw_tau),
                "loss_oriented_tau": loss_tau,
                "loss_oriented_abs_tau": abs(loss_tau) if math.isfinite(loss_tau) else float("nan"),
                "loss_oriented_sign": _sign(loss_tau),
                "consensus_tau": consensus_tau,
                "consensus_abs_tau": abs(consensus_tau) if math.isfinite(consensus_tau) else float("nan"),
                "consensus_sign": _sign(consensus_tau),
                "position_anchor_tau": position_anchor_tau,
                "position_anchor_abs_tau": (
                    abs(position_anchor_tau) if math.isfinite(position_anchor_tau) else float("nan")
                ),
                "position_anchor_sign": _sign(position_anchor_tau),
                "position_consensus_tau": position_consensus_tau,
                "position_consensus_abs_tau": (
                    abs(position_consensus_tau) if math.isfinite(position_consensus_tau) else float("nan")
                ),
                "position_consensus_sign": _sign(position_consensus_tau),
                "raw_order_original": _parse_list(row.get("raw_order_original_diagnostic")),
                "loss_oriented_order_original": _parse_list(row.get("loss_oriented_order_original_diagnostic")),
                "consensus_order_original": _parse_list(row.get("consensus_order_original_diagnostic")),
                "position_anchor_order_original": _parse_list(row.get("position_anchor_order_original_diagnostic")),
                "position_consensus_order_original": _parse_list(row.get("position_consensus_order_original_diagnostic")),
                "error": row.get("error", ""),
            }
        )
    return out


def _summarize_timeseries(timeseries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_head_method: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in timeseries:
        for method in METHODS:
            tau = _to_float(row.get(f"{method}_tau"))
            if not math.isfinite(tau):
                continue
            by_head_method[(_to_int(row.get("layer")), _to_int(row.get("head")), method)].append(row)

    summaries = []
    for (layer, head, method), rows in sorted(by_head_method.items()):
        rows = sorted(rows, key=lambda item: (_to_int(item.get("time_index")), _to_int(item.get("iter"))))
        taus = [_to_float(row.get(f"{method}_tau")) for row in rows]
        abs_taus = [abs(v) for v in taus if math.isfinite(v)]
        signs = [_to_int(row.get(f"{method}_sign")) for row in rows]
        nonzero_signs = [s for s in signs if s != 0]
        sign_flips = 0
        for prev, cur in zip(nonzero_signs, nonzero_signs[1:]):
            if prev * cur < 0:
                sign_flips += 1
        tau_mean, tau_std, tau_min, tau_max = _mean_std(taus)
        abs_mean, abs_std, abs_min, abs_max = _mean_std(abs_taus)
        orders = [
            row.get(f"{method}_order_original")
            for row in rows
            if isinstance(row.get(f"{method}_order_original"), list)
        ]
        pairwise = _pairwise_order_summary(orders)
        num_points = len(rows)
        pos = sum(1 for s in signs if s > 0)
        neg = sum(1 for s in signs if s < 0)
        zero = sum(1 for s in signs if s == 0)
        summaries.append(
            {
                "method": method,
                "layer": int(layer),
                "head": int(head),
                "head_id": f"L{layer}H{head}",
                "num_points": int(num_points),
                "iter_first": _to_int(rows[0].get("iter")) if rows else 0,
                "iter_last": _to_int(rows[-1].get("iter")) if rows else 0,
                "tau_mean": tau_mean,
                "tau_std": tau_std,
                "tau_min": tau_min,
                "tau_max": tau_max,
                "abs_tau_mean": abs_mean,
                "abs_tau_std": abs_std,
                "abs_tau_min": abs_min,
                "abs_tau_max": abs_max,
                "positive_rate": float(pos / num_points) if num_points else float("nan"),
                "negative_rate": float(neg / num_points) if num_points else float("nan"),
                "zero_rate": float(zero / num_points) if num_points else float("nan"),
                "first_nonzero_sign": int(nonzero_signs[0]) if nonzero_signs else 0,
                "last_nonzero_sign": int(nonzero_signs[-1]) if nonzero_signs else 0,
                "sign_flips": int(sign_flips),
                "stable_sign": int(sign_flips == 0 and len(nonzero_signs) > 0),
                **pairwise,
            }
        )
    return summaries


def _global_summary(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in METHODS:
        rows = [row for row in summary_rows if row.get("method") == method]
        if not rows:
            continue
        out[method] = {
            "heads": int(len(rows)),
            "stable_sign_heads": int(sum(_to_int(row.get("stable_sign")) for row in rows)),
            "total_sign_flips": int(sum(_to_int(row.get("sign_flips")) for row in rows)),
            "mean_abs_tau": float(np.nanmean([_to_float(row.get("abs_tau_mean")) for row in rows])),
            "mean_tau_std": float(np.nanmean([_to_float(row.get("tau_std")) for row in rows])),
            "mean_pairwise_abs_order_tau": float(
                np.nanmean([_to_float(row.get("pairwise_abs_order_tau_mean")) for row in rows])
            ),
            "heads_abs_tau_ge_0.4": int(sum(_to_float(row.get("abs_tau_mean")) >= 0.4 for row in rows)),
            "heads_abs_tau_ge_0.5": int(sum(_to_float(row.get("abs_tau_mean")) >= 0.5 for row in rows)),
        }
    return out


def _plot_timeseries(timeseries: list[dict[str, Any]], out_dir: Path, prefix: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        (out_dir / "plots").mkdir(parents=True, exist_ok=True)
        (out_dir / "plots" / f"{prefix}_plot_error.txt").write_text(str(exc), encoding="utf-8")
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    by_head: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in timeseries:
        by_head[str(row.get("head_id"))].append(row)
    for method in METHODS:
        fig, ax = plt.subplots(figsize=(11, 5))
        for head_id, rows in sorted(by_head.items()):
            rows = sorted(rows, key=lambda item: (_to_int(item.get("time_index")), _to_int(item.get("iter"))))
            xs = [_to_int(row.get("time_index")) for row in rows]
            ys = [_to_float(row.get(f"{method}_tau")) for row in rows]
            if not any(math.isfinite(y) for y in ys):
                continue
            ax.plot(xs, ys, marker="o", linewidth=1.2, markersize=2.5, label=head_id)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        ax.set_xlabel("probe repeat / train iter")
        ax.set_ylabel("diagnostic tau vs original L2R")
        ax.set_title(f"{prefix} {method} tau")
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{prefix}_{method}_tau.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 4))
        for head_id, rows in sorted(by_head.items()):
            rows = sorted(rows, key=lambda item: (_to_int(item.get("time_index")), _to_int(item.get("iter"))))
            xs = [_to_int(row.get("time_index")) for row in rows]
            ys = [_to_int(row.get(f"{method}_sign")) for row in rows]
            if not ys:
                continue
            ax.step(xs, ys, where="post", linewidth=1.1, label=head_id)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        ax.set_yticks([-1, 0, 1])
        ax.set_xlabel("probe repeat / train iter")
        ax.set_ylabel("diagnostic tau sign")
        ax.set_title(f"{prefix} {method} sign")
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{prefix}_{method}_sign.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed_csv", type=Path, default=None)
    parser.add_argument("--history_jsonl", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--summary_csv", default="")
    parser.add_argument("--timeseries_csv", default="")
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--plot_prefix", default="")
    parser.add_argument("--leave_one_out", action="store_true")
    args = parser.parse_args()

    if (args.fixed_csv is None) == (args.history_jsonl is None):
        raise SystemExit("Pass exactly one of --fixed_csv or --history_jsonl.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.fixed_csv is not None:
        rows = _read_fixed_csv(args.fixed_csv)
        _add_fixed_consensus(rows, leave_one_out=bool(args.leave_one_out))
        timeseries = _fixed_timeseries(rows)
        default_summary_csv = "head_summary.csv"
        default_timeseries_csv = "fixed_timeseries.csv"
        default_summary_json = "fixed_summary.json"
        default_plot_prefix = "fixed"
    else:
        rows = _read_history_jsonl(args.history_jsonl)
        timeseries = _history_timeseries(rows)
        default_summary_csv = "training_probe_summary.csv"
        default_timeseries_csv = "training_probe_timeseries.csv"
        default_summary_json = "training_probe_summary.json"
        default_plot_prefix = "training"

    summary_rows = _summarize_timeseries(timeseries)
    global_summary = _global_summary(summary_rows)
    summary_csv = args.summary_csv or default_summary_csv
    timeseries_csv = args.timeseries_csv or default_timeseries_csv
    summary_json = args.summary_json or default_summary_json
    plot_prefix = args.plot_prefix or default_plot_prefix
    _write_csv(args.out_dir / timeseries_csv, timeseries)
    _write_csv(args.out_dir / summary_csv, summary_rows)
    (args.out_dir / summary_json).write_text(
        json.dumps(
            _json_safe(
                {
                    "source": "fixed_csv" if args.fixed_csv is not None else "history_jsonl",
                    "input": str(args.fixed_csv or args.history_jsonl),
                    "num_timeseries_rows": int(len(timeseries)),
                    "num_summary_rows": int(len(summary_rows)),
                    "global_summary": global_summary,
                    "diagnostic_note": (
                        "Tau/original fields are diagnostic-only. Fixed consensus uses current "
                        "loss margins and same-repeat cross-head precedence agreement; online "
                        "consensus is emitted during training from current model/sample signals."
                    ),
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _plot_timeseries(timeseries, args.out_dir, plot_prefix)
    print(json.dumps(_json_safe(global_summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
