#!/usr/bin/env python3
"""Build fixed-head direct-asym-eig teacher datasets for Attn-MLP distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.directed_head_order_probe import (  # noqa: E402
    attention_block_orders,
    batched,
    current_batch,
    expand_block_orders,
    extract_attentions,
    infer_record_mode,
    iter_starts,
    load_checkpoint,
    load_model,
    load_tokens,
    permutation_state,
)
from scripts.analysis.test_10k_direct_asym_eig_head_method import (  # noqa: E402
    aggregate_layerhead_attention,
    direct_asym_eig_order,
    evaluate_loss_profiles,
    loss_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed-head direct-asym-eig Attn-MLP distillation datasets.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, default=Path("Report/MLP_distillation/try_1"))
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train_target", type=int, default=5000)
    parser.add_argument("--val_target", type=int, default=1000)
    parser.add_argument("--test_target", type=int, default=1000)
    parser.add_argument("--smoke_train_target", type=int, default=256)
    parser.add_argument("--smoke_val_target", type=int, default=64)
    parser.add_argument("--smoke_test_target", type=int, default=64)
    parser.add_argument("--max_attempt_factor", type=float, default=20.0)
    parser.add_argument("--shard_size", type=int, default=256)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--head", type=int, default=2)
    parser.add_argument("--export_type", choices=("with_none", "without_none"), default="with_none")
    parser.add_argument("--attention_order_mode", choices=("random", "current_ar", "original_l2r"), default="random")
    parser.add_argument("--attention_batch_size", type=int, default=16)
    parser.add_argument("--loss_samples", type=int, default=16)
    parser.add_argument("--loss_batch_size", type=int, default=16)
    parser.add_argument("--loss_candidate_batch_size", type=int, default=2)
    parser.add_argument("--prefix_k", type=int, default=16)
    parser.add_argument("--loss_score", choices=("linear_profile", "exp_profile", "prefix", "full"), default="linear_profile")
    parser.add_argument("--exp_tau", type=float, default=16.0)
    parser.add_argument("--direct_asym_eig_mode", type=str, default="raw_right_largest_real_real")
    parser.add_argument("--gap_threshold", type=float, default=0.01)
    parser.add_argument("--train_seed", type=int, default=110000)
    parser.add_argument("--val_seed", type=int, default=220000)
    parser.add_argument("--test_seed", type=int, default=330000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def autocast_context(args: argparse.Namespace):
    if "cuda" not in str(args.device) or str(args.dtype) == "float32":
        return nullcontext()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[str(args.dtype)]
    return torch.amp.autocast("cuda", dtype=dtype)


def git_value(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def valid_perm(values: Sequence[int], n: int) -> bool:
    return len(values) == int(n) and sorted(int(v) for v in values) == list(range(int(n)))


def rank_from_order(order: Sequence[int], n: int) -> List[int]:
    rank = [0 for _ in range(int(n))]
    for idx, value in enumerate(order):
        rank[int(value)] = int(idx)
    return rank


@torch.no_grad()
def collect_single_head_attention(
    args: argparse.Namespace,
    model,
    tokens,
    record_mode: str,
    perm_state,
    ctx,
    seed: int,
):
    block_size = int(model.config.block_size)
    num_blocks = int(model.num_blocks)
    starts = iter_starts(
        len(tokens),
        block_size,
        record_mode,
        int(args.attention_batch_size),
        int(seed),
    )
    rng_device = "cuda" if "cuda" in str(args.device) else "cpu"
    rng = torch.Generator(device=rng_device)
    rng.manual_seed(int(seed) + 17)
    batch = current_batch(tokens, starts, block_size, args.device, perm_state)
    block_orders = attention_block_orders(
        str(args.attention_order_mode),
        int(batch.size(0)),
        num_blocks,
        args.device,
        rng,
        perm_state,
    )
    token_orders = expand_block_orders(model, block_orders)
    with ctx:
        outputs = model(
            batch,
            mode=None,
            orders=token_orders,
            return_attentions=True,
            return_logits=False,
        )
    attentions = extract_attentions(outputs)
    if not attentions:
        raise RuntimeError("Model did not return attentions.")
    layer_heads = aggregate_layerhead_attention(
        attentions[int(args.layer)].detach(),
        block_orders,
        int(model.block_order_block_len),
        str(args.export_type),
    )
    matrix = layer_heads[int(args.head)].detach().cpu().to(dtype=torch.float64).numpy()
    np.fill_diagonal(matrix, 0.0)
    return matrix, [int(v) for v in starts]


def choose_teacher_order(
    args: argparse.Namespace,
    model,
    tokens,
    record_mode: str,
    perm_state,
    ctx,
    matrix: np.ndarray,
    loss_seed: int,
):
    raw_order, eig_meta = direct_asym_eig_order(matrix, str(args.direct_asym_eig_mode))
    reverse_order = list(reversed(raw_order))
    if not valid_perm(raw_order, int(model.num_blocks)):
        raise ValueError("direct_asym_eig raw order is not a valid permutation")
    loss_by_order = evaluate_loss_profiles(
        args,
        model,
        tokens,
        record_mode,
        perm_state,
        [raw_order, reverse_order],
        split_name="train_orientation",
        num_samples=int(args.loss_samples),
        batch_size=int(args.loss_batch_size),
        seed=int(loss_seed),
        ctx=ctx,
    )
    raw_loss_item = loss_by_order[tuple(raw_order)]
    reverse_loss_item = loss_by_order[tuple(reverse_order)]
    raw_loss = float(loss_score(raw_loss_item, str(args.loss_score)))
    reverse_loss = float(loss_score(reverse_loss_item, str(args.loss_score)))
    selected_reverse = bool(reverse_loss < raw_loss)
    teacher_order = reverse_order if selected_reverse else raw_order
    chosen_loss = reverse_loss if selected_reverse else raw_loss
    return {
        "teacher_order": [int(v) for v in teacher_order],
        "teacher_rank": rank_from_order(teacher_order, int(model.num_blocks)),
        "raw_order": [int(v) for v in raw_order],
        "reverse_order": [int(v) for v in reverse_order],
        "raw_loss": float(raw_loss),
        "reverse_loss": float(reverse_loss),
        "chosen_loss": float(chosen_loss),
        "teacher_gap": float(abs(raw_loss - reverse_loss)),
        "selected_reverse": bool(selected_reverse),
        "eig_meta": eig_meta,
    }


def shard_path(out_dir: Path, split: str, shard_idx: int) -> Path:
    return out_dir / split / f"shard_{int(shard_idx):03d}.pt"


def save_shard(path: Path, samples: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "attention": torch.stack([sample["attention"] for sample in samples], dim=0).to(torch.float16),
        "teacher_rank": torch.stack([sample["teacher_rank"] for sample in samples], dim=0).to(torch.uint8),
        "teacher_order": torch.stack([sample["teacher_order"] for sample in samples], dim=0).to(torch.uint8),
        "teacher_gap": torch.tensor([sample["teacher_gap"] for sample in samples], dtype=torch.float32),
        "raw_loss": torch.tensor([sample["raw_loss"] for sample in samples], dtype=torch.float32),
        "reverse_loss": torch.tensor([sample["reverse_loss"] for sample in samples], dtype=torch.float32),
        "chosen_loss": torch.tensor([sample["chosen_loss"] for sample in samples], dtype=torch.float32),
        "probe_seed": torch.tensor([sample["probe_seed"] for sample in samples], dtype=torch.int64),
        "loss_seed": torch.tensor([sample["loss_seed"] for sample in samples], dtype=torch.int64),
        "sample_id": torch.tensor([sample["sample_id"] for sample in samples], dtype=torch.int64),
        "selected_reverse": torch.tensor([sample["selected_reverse"] for sample in samples], dtype=torch.bool),
        "metadata": [sample["metadata"] for sample in samples],
    }
    torch.save(payload, path)


def existing_count(out_dir: Path, split: str) -> int:
    total = 0
    for path in sorted((out_dir / split).glob("shard_*.pt")):
        payload = torch.load(path, map_location="cpu")
        total += int(payload["attention"].shape[0])
    return int(total)


def split_targets(args: argparse.Namespace) -> Dict[str, int]:
    if bool(args.smoke):
        return {
            "train": int(args.smoke_train_target),
            "val_probe": int(args.smoke_val_target),
            "test_probe": int(args.smoke_test_target),
        }
    return {
        "train": int(args.train_target),
        "val_probe": int(args.val_target),
        "test_probe": int(args.test_target),
    }


def seed_for_split(args: argparse.Namespace, split: str) -> int:
    return {
        "train": int(args.train_seed),
        "val_probe": int(args.val_seed),
        "test_probe": int(args.test_seed),
    }[split]


def histogram(values: Sequence[float]) -> Dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"bins": [], "counts": []}
    counts, bins = np.histogram(arr, bins=10)
    return {"bins": [float(v) for v in bins.tolist()], "counts": [int(v) for v in counts.tolist()]}


def summarize_split(split: str, target: int, accepted: int, rejected: int, gaps: Sequence[float], shard_paths: Sequence[Path]) -> Dict:
    arr = np.asarray(list(gaps), dtype=np.float64)
    return {
        "split": split,
        "target_valid_samples": int(target),
        "accepted_samples": int(accepted),
        "rejected_samples": int(rejected),
        "attempted_samples": int(accepted + rejected),
        "accept_rate": float(accepted / max(1, accepted + rejected)),
        "gap_mean": float(arr.mean()) if arr.size else None,
        "gap_median": float(np.median(arr)) if arr.size else None,
        "gap_quantiles": (
            {str(q): float(np.quantile(arr, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)}
            if arr.size
            else {}
        ),
        "gap_histogram": histogram(gaps),
        "shards": [str(path) for path in shard_paths],
    }


def collect_split(
    args: argparse.Namespace,
    split: str,
    target: int,
    model,
    tokens,
    record_mode: str,
    perm_state,
    ctx,
):
    out_dir = Path(args.out_dir)
    accepted_existing = existing_count(out_dir, split) if bool(args.resume) else 0
    accepted = int(accepted_existing)
    rejected = 0
    gaps: List[float] = []
    shard_paths = sorted((out_dir / split).glob("shard_*.pt")) if accepted_existing else []
    pending: List[Dict] = []
    shard_idx = len(shard_paths)
    attempts = 0
    max_attempts = int(max(target * float(args.max_attempt_factor), target + 1))
    base_seed = seed_for_split(args, split)
    while accepted < int(target) and attempts < max_attempts:
        sample_id = accepted + rejected
        probe_seed = int(base_seed + attempts * 1009)
        loss_seed = int(base_seed + attempts * 1009 + 503)
        attempts += 1
        t0 = time.perf_counter()
        try:
            t_attn0 = time.perf_counter()
            matrix, attention_starts = collect_single_head_attention(
                args, model, tokens, record_mode, perm_state, ctx, probe_seed
            )
            t_attn1 = time.perf_counter()
            teacher = choose_teacher_order(
                args, model, tokens, record_mode, perm_state, ctx, matrix, loss_seed
            )
            t1 = time.perf_counter()
        except Exception as exc:
            rejected += 1
            print(f"[distill-dataset] split={split} attempt={attempts} error={exc}", flush=True)
            continue
        gap = float(teacher["teacher_gap"])
        gaps.append(gap)
        if gap < float(args.gap_threshold):
            rejected += 1
            continue
        metadata = {
            "checkpoint": str(args.ckpt_path),
            "teacher_head": f"L{int(args.layer)}H{int(args.head)}",
            "export_type": str(args.export_type),
            "teacher_method": "direct_asym_eig",
            "eig_mode": str(args.direct_asym_eig_mode),
            "orientation_score": f"{args.loss_score}_loss",
            "frame": "current",
            "split": str(split),
            "attention_order_mode": str(args.attention_order_mode),
            "attention_starts": attention_starts,
            "eig_meta": teacher["eig_meta"],
            "latency_ms": {
                "attention": float((t_attn1 - t_attn0) * 1000.0),
                "teacher_total": float((t1 - t0) * 1000.0),
            },
        }
        pending.append(
            {
                "attention": torch.from_numpy(matrix.astype(np.float32)).to(dtype=torch.float16),
                "teacher_rank": torch.tensor(teacher["teacher_rank"], dtype=torch.uint8),
                "teacher_order": torch.tensor(teacher["teacher_order"], dtype=torch.uint8),
                "teacher_gap": float(gap),
                "raw_loss": float(teacher["raw_loss"]),
                "reverse_loss": float(teacher["reverse_loss"]),
                "chosen_loss": float(teacher["chosen_loss"]),
                "probe_seed": int(probe_seed),
                "loss_seed": int(loss_seed),
                "sample_id": int(sample_id),
                "selected_reverse": bool(teacher["selected_reverse"]),
                "metadata": metadata,
            }
        )
        accepted += 1
        if len(pending) >= int(args.shard_size):
            path = shard_path(out_dir, split, shard_idx)
            save_shard(path, pending)
            shard_paths.append(path)
            shard_idx += 1
            pending = []
        if accepted == 1 or accepted % max(1, min(100, int(args.shard_size))) == 0 or accepted >= int(target):
            print(
                f"[distill-dataset] split={split} accepted={accepted}/{target} "
                f"rejected={rejected} last_gap={gap:.6f}",
                flush=True,
            )
    if pending:
        path = shard_path(out_dir, split, shard_idx)
        save_shard(path, pending)
        shard_paths.append(path)
    if accepted < int(target):
        raise RuntimeError(f"split={split} accepted {accepted}/{target} before max_attempts={max_attempts}")
    return summarize_split(split, target, accepted, rejected, gaps, shard_paths)


def write_command(report_dir: Path, filename: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(part) for part in sys.argv)
    (report_dir / filename).write_text(command + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_command(args.report_dir, "dataset_command.sh")

    ckpt = load_checkpoint(args.ckpt_path)
    checkpoint_iter = int(ckpt.get("iter_num", -1))
    if checkpoint_iter != 10000:
        raise ValueError(f"Expected clean Random-10k checkpoint, got iter_num={checkpoint_iter}")
    if ckpt.get("attn_mlp_policy_state") is not None:
        raise ValueError("Checkpoint contains attn_mlp_policy_state; refusing to build teacher dataset.")
    if not bool(ckpt.get("config", {}).get("permute_data", False)):
        raise ValueError("Checkpoint config does not indicate permute_data=True.")
    if not ckpt.get("data_permutation"):
        raise ValueError("Checkpoint is missing data_permutation metadata.")

    model = load_model(ckpt, args.device)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    tokens, data_dir = load_tokens(ckpt, "train", args.data_dir)
    record_mode = infer_record_mode(ckpt, data_dir)
    perm_state = permutation_state(ckpt, model, int(model.config.block_size))
    ctx = autocast_context(args)

    manifest = {
        "git_branch": git_value(["branch", "--show-current"]),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "checkpoint": {
            "path": str(args.ckpt_path),
            "sha256": sha256_file(args.ckpt_path),
            "iter_num": checkpoint_iter,
            "model_args": ckpt.get("model_args", {}),
            "config_subset": {
                "dataset": ckpt.get("config", {}).get("dataset"),
                "permute_data": ckpt.get("config", {}).get("permute_data"),
                "permute_seed": ckpt.get("config", {}).get("permute_seed"),
                "data_record_mode": ckpt.get("config", {}).get("data_record_mode"),
            },
        },
        "teacher": {
            "label": f"L{int(args.layer)}H{int(args.head)}",
            "layer": int(args.layer),
            "head": int(args.head),
            "export_type": str(args.export_type),
            "method": "direct_asym_eig",
            "eig_mode": str(args.direct_asym_eig_mode),
            "orientation_score": str(args.loss_score),
            "gap_threshold": float(args.gap_threshold),
            "frame": "current",
        },
        "attention": {
            "order_mode": str(args.attention_order_mode),
            "batch_size": int(args.attention_batch_size),
            "dtype": "float16",
            "shape": [int(model.num_blocks), int(model.num_blocks)],
        },
        "loss_profile": {
            "split": "train",
            "samples": int(args.loss_samples),
            "batch_size": int(args.loss_batch_size),
            "candidate_batch_size": int(args.loss_candidate_batch_size),
            "prefix_k": int(args.prefix_k),
            "score": str(args.loss_score),
            "exp_tau": float(args.exp_tau),
        },
        "split_seed_groups": {
            "train": int(args.train_seed),
            "val_probe": int(args.val_seed),
            "test_probe": int(args.test_seed),
        },
        "no_prior_note": (
            "Teacher construction uses current-frame attention and current-model train linear_profile_loss only. "
            "Original-frame order/tau/distance and validation PPL are not used."
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    targets = split_targets(args)
    summaries = {}
    for split, target in targets.items():
        summaries[split] = collect_split(
            args, split, int(target), model, tokens, record_mode, perm_state, ctx
        )
    collection_summary = {
        "dataset_dir": str(args.out_dir),
        "smoke": bool(args.smoke),
        "targets": targets,
        "splits": summaries,
    }
    (args.out_dir / "collection_summary.json").write_text(
        json.dumps(collection_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.report_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.report_dir / "collection_summary.json").write_text(
        json.dumps(collection_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(collection_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
