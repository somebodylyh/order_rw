#!/usr/bin/env python3
"""Evaluate Attn-MLP distillation students and baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attn_mlp_order_policy import FlatAttentionOrderMLP  # noqa: E402
from scripts.analysis.directed_head_order_probe import (  # noqa: E402
    infer_record_mode,
    load_checkpoint,
    load_model,
    load_tokens,
    permutation_state,
)
from scripts.analysis.test_10k_direct_asym_eig_head_method import evaluate_loss_profiles  # noqa: E402
from scripts.train.train_attn_mlp_distillation import (  # noqa: E402
    DistillTensorDataset,
    constant_priority_order,
    evaluate_constant,
    evaluate_model,
    parse_int_list,
    transform_attention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Attn-MLP distillation checkpoints.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, default=Path("Report/MLP_distillation/try_1"))
    parser.add_argument("--ckpt_path", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--seeds", type=str, default="1337,2026,2027")
    parser.add_argument("--modes", type=str, default="normal,zero,shuffled")
    parser.add_argument("--compute_model_loss", action="store_true")
    parser.add_argument("--model_loss_splits", type=str, default="test_probe")
    parser.add_argument("--model_loss_max_orders_per_key", type=int, default=64)
    parser.add_argument("--model_loss_samples", type=int, default=64)
    parser.add_argument("--model_loss_batch_size", type=int, default=16)
    parser.add_argument("--model_loss_candidate_batch_size", type=int, default=4)
    parser.add_argument("--model_loss_seed", type=int, default=515151)
    parser.add_argument("--prefix_k", type=int, default=16)
    parser.add_argument("--exp_tau", type=float, default=16.0)
    parser.add_argument("--latency_repeats", type=int, default=200)
    return parser.parse_args()


def autocast_context(args: argparse.Namespace):
    if "cuda" not in str(args.device) or str(args.dtype) == "float32":
        return nullcontext()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[str(args.dtype)]
    return torch.amp.autocast("cuda", dtype=dtype)


def write_command(report_dir: Path, filename: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(part) for part in sys.argv)
    (report_dir / filename).write_text(command + "\n", encoding="utf-8")


def load_dataset_manifest(dataset_dir: Path, report_dir: Path) -> Dict:
    for path in (dataset_dir / "manifest.json", report_dir / "dataset_manifest.json"):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def teacher_label_from_manifest(manifest: Dict) -> str:
    teacher = manifest.get("teacher", {}) if isinstance(manifest, dict) else {}
    if teacher.get("label"):
        return str(teacher["label"])
    if "layer" in teacher and "head" in teacher:
        return f"L{int(teacher['layer'])}H{int(teacher['head'])}"
    return str(teacher.get("head", "FixedHead"))


def try_label(report_dir: Path) -> str:
    for part in report_dir.parts:
        if part.startswith("try_"):
            suffix = part.split("_", 1)[1]
            if suffix:
                return f"Try {suffix}"
    return "Distillation"


def pair_indices(n: int):
    return torch.triu_indices(int(n), int(n), offset=1)


def teacher_upper_metrics(dataset: DistillTensorDataset):
    n = int(dataset.teacher_rank.size(1))
    pair_i, pair_j = pair_indices(n)
    priority = 1.0 - dataset.teacher_rank.float() / float(n - 1)
    pred = priority[:, pair_i] - priority[:, pair_j]
    target = (dataset.teacher_rank[:, pair_i] < dataset.teacher_rank[:, pair_j]).float()
    bce = F.binary_cross_entropy_with_logits(pred, target)
    return {
        "pair_bce": float(bce.item()),
        "pair_accuracy": 1.0,
        "kendall_tau_mean": 1.0,
        "kendall_tau_std": 0.0,
        "kendall_distance_mean": 0.0,
        "exact_match_rate": 1.0,
        "orientation_agreement": 1.0,
        "num_samples": int(len(dataset)),
    }


@torch.no_grad()
def predict_orders(model, dataset: DistillTensorDataset, mode: str, device: torch.device, batch_size: int, seed: int):
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    orders: List[Tuple[int, ...]] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 123)
    model.eval()
    for batch in loader:
        attn = transform_attention(batch["attention"].to(device), mode, gen)
        logits = model(attn)
        pred = torch.argsort(logits.detach().float(), dim=1, descending=True).cpu()
        orders.extend([tuple(int(v) for v in row.tolist()) for row in pred])
    return orders


def constant_orders(order: torch.Tensor, count: int):
    key = tuple(int(v) for v in order.tolist())
    return [key for _ in range(int(count))]


def teacher_orders(dataset: DistillTensorDataset):
    return [tuple(int(v) for v in row.tolist()) for row in dataset.teacher_order]


def parse_str_list(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def deterministic_order_subset(key: str, orders: List[Tuple[int, ...]], max_orders: int, seed: int):
    if int(max_orders) <= 0 or len(orders) <= int(max_orders):
        return orders
    digest = hashlib.sha256(f"{key}:{int(seed)}".encode("utf-8")).digest()
    rng_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(rng_seed)
    indices = sorted(int(v) for v in rng.choice(len(orders), size=int(max_orders), replace=False))
    return [orders[idx] for idx in indices]


def mean_profile_loss_for_orders(args, order_lists: Dict[str, List[Tuple[int, ...]]], model, tokens, record_mode, perm_state, ctx):
    class LossArgs:
        pass

    loss_args = LossArgs()
    loss_args.device = args.device
    loss_args.loss_candidate_batch_size = int(args.model_loss_candidate_batch_size)
    loss_args.prefix_k = int(args.prefix_k)
    loss_args.exp_tau = float(args.exp_tau)
    sampled_order_lists = {
        key: deterministic_order_subset(key, orders, int(args.model_loss_max_orders_per_key), int(args.model_loss_seed))
        for key, orders in order_lists.items()
    }
    unique = []
    seen = set()
    for orders in sampled_order_lists.values():
        for order in orders:
            if order not in seen:
                seen.add(order)
                unique.append(list(order))
    if not unique:
        return {key: float("nan") for key in order_lists}
    loss_by_order = evaluate_loss_profiles(
        loss_args,
        model,
        tokens,
        record_mode,
        perm_state,
        unique,
        split_name="eval_model_loss",
        num_samples=int(args.model_loss_samples),
        batch_size=int(args.model_loss_batch_size),
        seed=int(args.model_loss_seed),
        ctx=ctx,
    )
    out = {}
    for key, orders in sampled_order_lists.items():
        vals = [float(loss_by_order[tuple(order)]["linear_profile_loss"]) for order in orders if tuple(order) in loss_by_order]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def load_student(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device)
    model = FlatAttentionOrderMLP(**payload["config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def measure_student_latency(model, dataset: DistillTensorDataset, mode: str, device: torch.device, repeats: int):
    count = max(1, min(int(repeats), len(dataset)))
    attn = dataset.attention[:count].to(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(12345)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for idx in range(count):
            row = transform_attention(attn[idx : idx + 1], mode, gen)
            logits = model(row)
            _ = torch.argsort(logits[0], descending=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    dt = time.perf_counter() - t0
    return {"student_order_ms_mean": float(dt * 1000.0 / count), "student_peak_memory_bytes": int(peak)}


def load_teacher_latency(dataset_dir: Path):
    summary_path = dataset_dir / "collection_summary.json"
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    # Per-sample latency is stored in shard metadata, so summarize from shards.
    vals = []
    for split in ("train", "val_probe", "test_probe"):
        for shard in sorted((dataset_dir / split).glob("shard_*.pt")):
            item = torch.load(shard, map_location="cpu")
            for meta in item.get("metadata", []):
                latency = meta.get("latency_ms", {}) if isinstance(meta, dict) else {}
                if "teacher_total" in latency:
                    vals.append(float(latency["teacher_total"]))
    if not vals:
        return {}
    return {
        "teacher_order_ms_mean": float(np.mean(vals)),
        "teacher_order_ms_median": float(np.median(vals)),
    }


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def row_summary(rows: Sequence[Dict], mode: str, split: str):
    vals = [row for row in rows if row.get("mode") == mode and row.get("split") == split and row.get("seed", -1) >= 0]
    out = {"mode": mode, "split": split, "runs": len(vals)}
    for key in (
        "pair_bce",
        "pair_accuracy",
        "kendall_tau_mean",
        "kendall_distance_mean",
        "exact_match_rate",
        "orientation_agreement",
        "linear_profile_loss_mean",
        "linear_profile_loss_minus_teacher",
    ):
        arr = np.asarray([float(row[key]) for row in vals], dtype=np.float64)
        if arr.size:
            out[f"{key}_mean"] = float(arr.mean())
            out[f"{key}_std"] = float(arr.std())
    return out


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_command(args.report_dir, "eval_command.sh")
    device = torch.device(args.device)
    dataset_manifest = load_dataset_manifest(args.dataset_dir, args.report_dir)
    teacher_label = teacher_label_from_manifest(dataset_manifest)
    train_data = DistillTensorDataset(args.dataset_dir, "train")
    val_data = DistillTensorDataset(args.dataset_dir, "val_probe")
    test_data = DistillTensorDataset(args.dataset_dir, "test_probe")
    loaders = {
        "val_probe": DataLoader(val_data, batch_size=int(args.batch_size), shuffle=False),
        "test_probe": DataLoader(test_data, batch_size=int(args.batch_size), shuffle=False),
    }
    datasets = {"val_probe": val_data, "test_probe": test_data}
    rows: List[Dict] = []
    order_lists_for_loss: Dict[str, List[Tuple[int, ...]]] = {}
    model_loss_splits = set(parse_str_list(args.model_loss_splits))

    const_order = constant_priority_order(train_data)
    for split, dataset in datasets.items():
        const_metrics = evaluate_constant(const_order, dataset, args.tau)
        rows.append({"method": "baseline", "mode": "constant", "seed": -1, "split": split, **const_metrics})
        rows.append({"method": "upper_bound", "mode": "teacher", "seed": -1, "split": split, **teacher_upper_metrics(dataset)})
        if split in model_loss_splits:
            order_lists_for_loss[f"constant::{split}"] = constant_orders(const_order, len(dataset))
            order_lists_for_loss[f"teacher::{split}"] = teacher_orders(dataset)

    latency_rows = []
    for mode in [v.strip() for v in str(args.modes).split(",") if v.strip()]:
        for seed in parse_int_list(args.seeds):
            ckpt_path = args.checkpoint_dir / f"{mode}_seed{seed}" / "best_by_val_pair_loss.pt"
            if not ckpt_path.exists():
                continue
            model, payload = load_student(ckpt_path, device)
            for split, loader in loaders.items():
                metrics = evaluate_model(model, loader, mode, device, args.tau, int(seed) + 42)
                row = {
                    "method": "student",
                    "mode": mode,
                    "seed": int(seed),
                    "split": split,
                    "checkpoint": str(ckpt_path),
                    "best_epoch": int(payload.get("training_meta", {}).get("best_epoch", -1)),
                    **metrics,
                }
                rows.append(row)
                if split in model_loss_splits:
                    order_lists_for_loss[f"{mode}:{seed}:{split}"] = predict_orders(
                        model, datasets[split], mode, device, args.batch_size, int(seed) + 42
                    )
            latency = measure_student_latency(model, test_data, mode, device, int(args.latency_repeats))
            latency_rows.append({"mode": mode, "seed": int(seed), **latency})

    if bool(args.compute_model_loss):
        if args.ckpt_path is None:
            raise ValueError("--compute_model_loss requires --ckpt_path")
        ckpt = load_checkpoint(args.ckpt_path)
        frozen_model = load_model(ckpt, args.device)
        frozen_model.eval()
        tokens, data_dir = load_tokens(ckpt, "train", args.data_dir)
        record_mode = infer_record_mode(ckpt, data_dir)
        perm_state = permutation_state(ckpt, frozen_model, int(frozen_model.config.block_size))
        ctx = autocast_context(args)
        loss_values = mean_profile_loss_for_orders(
            args, order_lists_for_loss, frozen_model, tokens, record_mode, perm_state, ctx
        )
    else:
        loss_values = {}

    for row in rows:
        if row["mode"] == "constant":
            key = f"constant::{row['split']}"
        elif row["mode"] == "teacher":
            key = f"teacher::{row['split']}"
        else:
            key = f"{row['mode']}:{row['seed']}:{row['split']}"
        row["linear_profile_loss_mean"] = float(loss_values.get(key, float("nan")))
        teacher_key = f"teacher::{row['split']}"
        teacher_loss = float(loss_values.get(teacher_key, float("nan")))
        row["linear_profile_loss_minus_teacher"] = (
            float(row["linear_profile_loss_mean"] - teacher_loss)
            if math.isfinite(float(row["linear_profile_loss_mean"])) and math.isfinite(teacher_loss)
            else float("nan")
        )

    teacher_latency = load_teacher_latency(args.dataset_dir)
    latency_summary = {
        "teacher": teacher_latency,
        "students": latency_rows,
    }
    if teacher_latency and latency_rows:
        teacher_ms = float(teacher_latency["teacher_order_ms_mean"])
        for item in latency_rows:
            item["speedup_vs_teacher"] = float(teacher_ms / max(float(item["student_order_ms_mean"]), 1e-12))

    write_csv(args.report_dir / "metrics.csv", rows)
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "rows": rows,
        "aggregates": [
            row_summary(rows, mode, "test_probe")
            for mode in ("normal", "zero", "shuffled")
        ],
        "latency": latency_summary,
        "model_loss_eval": {
            "enabled": bool(args.compute_model_loss),
            "splits": sorted(model_loss_splits),
            "max_orders_per_key": int(args.model_loss_max_orders_per_key),
            "samples": int(args.model_loss_samples),
            "batch_size": int(args.model_loss_batch_size),
            "candidate_batch_size": int(args.model_loss_candidate_batch_size),
            "seed": int(args.model_loss_seed),
        },
        "dataset_manifest": dataset_manifest,
    }
    (args.report_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    test_rows = [row for row in rows if row.get("split") == "test_probe"]
    normal = row_summary(rows, "normal", "test_probe")
    constant = next(row for row in test_rows if row["mode"] == "constant")
    zero = row_summary(rows, "zero", "test_probe")
    shuffled = row_summary(rows, "shuffled", "test_probe")
    teacher = next(row for row in test_rows if row["mode"] == "teacher")
    lines = [
        f"# {try_label(args.report_dir)} Result: {teacher_label} Direct-Asym-Eig Teacher Distillation",
        "",
        f"This experiment tests teacher compression only: can a simple Attn-MLP reproduce the fixed current-frame {teacher_label} teacher orders?",
        "",
        "## Test-Probe Summary",
        "",
        "| method | runs | pair BCE | pair acc | tau | distance | exact | orientation | linear loss minus teacher |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| teacher upper | 1 | {teacher['pair_bce']:.6f} | {teacher['pair_accuracy']:.4f} | "
            f"{teacher['kendall_tau_mean']:.4f} | {teacher['kendall_distance_mean']:.4f} | "
            f"{teacher['exact_match_rate']:.4f} | {teacher['orientation_agreement']:.4f} | "
            f"{teacher['linear_profile_loss_minus_teacher']:.6f} |"
        ),
        (
            f"| constant | 1 | {constant['pair_bce']:.6f} | {constant['pair_accuracy']:.4f} | "
            f"{constant['kendall_tau_mean']:.4f} | {constant['kendall_distance_mean']:.4f} | "
            f"{constant['exact_match_rate']:.4f} | {constant['orientation_agreement']:.4f} | "
            f"{constant['linear_profile_loss_minus_teacher']:.6f} |"
        ),
    ]
    for label, item in (("normal", normal), ("zero", zero), ("shuffled", shuffled)):
        lines.append(
            f"| {label} | {item.get('runs', 0)} | {item.get('pair_bce_mean', float('nan')):.6f} | "
            f"{item.get('pair_accuracy_mean', float('nan')):.4f} | "
            f"{item.get('kendall_tau_mean_mean', float('nan')):.4f} | "
            f"{item.get('kendall_distance_mean_mean', float('nan')):.4f} | "
            f"{item.get('exact_match_rate_mean', float('nan')):.4f} | "
            f"{item.get('orientation_agreement_mean', float('nan')):.4f} | "
            f"{item.get('linear_profile_loss_minus_teacher_mean', float('nan')):.6f} |"
        )
    lines.extend(["", "## Latency", ""])
    if latency_rows and teacher_latency:
        lines.extend(["| mode | seed | student ms/order | speedup | peak memory MB |", "| --- | ---: | ---: | ---: | ---: |"])
        for item in latency_rows:
            lines.append(
                f"| {item['mode']} | {item['seed']} | {item['student_order_ms_mean']:.4f} | "
                f"{item.get('speedup_vs_teacher', float('nan')):.2f} | "
                f"{item['student_peak_memory_bytes'] / (1024**2):.1f} |"
            )
        lines.append("")
        lines.append(f"Teacher pipeline mean latency: `{teacher_latency['teacher_order_ms_mean']:.4f}` ms/order.")
    else:
        lines.append("Latency summary unavailable.")
    normal_acc = float(normal.get("pair_accuracy_mean", float("nan")))
    constant_acc = float(constant["pair_accuracy"])
    zero_acc = float(zero.get("pair_accuracy_mean", float("nan")))
    shuffled_acc = float(shuffled.get("pair_accuracy_mean", float("nan")))
    strong = normal_acc > max(constant_acc, zero_acc, shuffled_acc) + 0.02
    normal_linear = float(normal.get("linear_profile_loss_minus_teacher_mean", float("nan")))
    constant_linear = float(constant.get("linear_profile_loss_minus_teacher", float("nan")))
    zero_linear = float(zero.get("linear_profile_loss_minus_teacher_mean", float("nan")))
    shuffled_linear = float(shuffled.get("linear_profile_loss_minus_teacher_mean", float("nan")))
    baseline_linear_vals = [v for v in (constant_linear, zero_linear, shuffled_linear) if math.isfinite(v)]
    best_baseline_linear = min(baseline_linear_vals) if baseline_linear_vals else float("nan")
    linear_caveat = math.isfinite(normal_linear) and math.isfinite(best_baseline_linear) and normal_linear > best_baseline_linear
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                "The attention-conditioned MLP beats constant/zero/shuffle baselines on held-out test probes, "
                "supporting an attention-conditioned teacher mapping."
                if strong
                else "The attention-conditioned MLP does not clearly beat constant/zero/shuffle baselines; current evidence mainly supports a stable or nearly constant teacher order rather than a proven attention-conditioned mapping."
            ),
            (
                "Caveat: the capped current-model linear_profile_loss diagnostic favors a constant/zero/shuffle baseline over the normal student, so this run is evidence for teacher compression, not yet evidence that the student order is better for frozen-model loss."
                if linear_caveat
                else "The capped current-model linear_profile_loss diagnostic does not contradict the pairwise-imitation result."
            ),
            "",
            "Original-frame tau/distance and validation PPL are not used for checkpoint selection or early stopping.",
        ]
    )
    (args.report_dir / "experiment_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    failure_lines = [
        "# Failure Analysis",
        "",
        "This file is written for both success and failure cases.",
        "",
        f"- normal test pair accuracy mean: `{normal_acc:.6f}`",
        f"- constant test pair accuracy: `{constant_acc:.6f}`",
        f"- zero test pair accuracy mean: `{zero_acc:.6f}`",
        f"- shuffled test pair accuracy mean: `{shuffled_acc:.6f}`",
        f"- normal linear_profile_loss minus teacher mean: `{normal_linear:.6f}`",
        f"- best baseline linear_profile_loss minus teacher: `{best_baseline_linear:.6f}`",
        "",
    ]
    if strong:
        if linear_caveat:
            failure_lines.append(
                "Residual caveat: normal MLP clears the pairwise baseline margin, but the current-model loss diagnostic still prefers a simpler baseline."
            )
        else:
            failure_lines.append("No primary failure: normal MLP clears the configured baseline margin.")
    else:
        failure_lines.append(
            "Primary risk: the teacher may be too stable across probe batches, so imitation can be solved by a constant-order baseline."
        )
    (args.report_dir / "failure_analysis.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
    print(json.dumps({"report_dir": str(args.report_dir), "rows": len(rows), "strong_success": bool(strong)}, indent=2))


if __name__ == "__main__":
    main()
