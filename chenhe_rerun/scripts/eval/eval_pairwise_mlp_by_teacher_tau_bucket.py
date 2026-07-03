#!/usr/bin/env python3
"""Evaluate full teacher-order MLP by teacher OriginalL2R tau buckets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attn_mlp_order_policy import FlatAttentionOrderMLP  # noqa: E402
from order_utils import build_fixed_block_permutation, invert_permutation  # noqa: E402


BUCKETS = [
    ("neg_high", -1.0000001, -0.8),
    ("neg_mid", -0.8, -0.2),
    ("near_zero", -0.2, 0.2),
    ("pos_mid", 0.2, 0.8),
    ("pos_high", 0.8, 1.0000001),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--splits", type=str, default="train,val")
    parser.add_argument("--permute_seed", type=int, default=42)
    return parser.parse_args()


def pair_indices(num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    pair_i = []
    pair_j = []
    for i in range(int(num_blocks)):
        for j in range(i + 1, int(num_blocks)):
            pair_i.append(i)
            pair_j.append(j)
    return torch.tensor(pair_i, dtype=torch.long), torch.tensor(pair_j, dtype=torch.long)


def original_target_rank(num_blocks: int, permute_seed: int) -> torch.Tensor:
    block_perm = build_fixed_block_permutation(int(num_blocks), int(permute_seed)).long()
    original_l2r = invert_permutation(block_perm).long()
    rank = torch.empty(int(num_blocks), dtype=torch.long)
    rank[original_l2r] = torch.arange(int(num_blocks), dtype=torch.long)
    return rank


def tau_to_original(order: torch.Tensor, target_rank: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor) -> torch.Tensor:
    original_positions = target_rank[order.long()]
    concordant = (original_positions[:, pair_i] < original_positions[:, pair_j]).float().mean(dim=1)
    return 2.0 * concordant - 1.0


def bucket_name(value: float) -> str:
    for name, lo, hi in BUCKETS:
        if float(value) >= lo and float(value) < hi:
            return name
    return "unknown"


def summarize(values: List[float]) -> Dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def update_stats(stats: Dict, loss: float, acc: float, order_acc: float, teacher_tau: float) -> None:
    stats["losses"].append(float(loss))
    stats["pair_accs"].append(float(acc))
    stats["order_accs"].append(float(order_acc))
    stats["taus"].append(float(2.0 * order_acc - 1.0))
    stats["teacher_original_taus"].append(float(teacher_tau))


def row_from_stats(split: str, group_type: str, group: str, stats: Dict) -> Dict:
    return {
        "split": split,
        "group_type": group_type,
        "group": group,
        "num_samples": int(len(stats["taus"])),
        "teacher_original_tau_mean": float(np.mean(stats["teacher_original_taus"])) if stats["taus"] else float("nan"),
        "pair_bce": float(np.mean(stats["losses"])) if stats["taus"] else float("nan"),
        "pair_accuracy": float(np.mean(stats["pair_accs"])) if stats["taus"] else float("nan"),
        "order_pair_accuracy": float(np.mean(stats["order_accs"])) if stats["taus"] else float("nan"),
        "kendall_tau_mean": float(np.mean(stats["taus"])) if stats["taus"] else float("nan"),
        "kendall_tau_std": float(np.std(stats["taus"])) if stats["taus"] else float("nan"),
        "kendall_tau_min": float(np.min(stats["taus"])) if stats["taus"] else float("nan"),
        "kendall_tau_max": float(np.max(stats["taus"])) if stats["taus"] else float("nan"),
    }


def empty_stats() -> Dict:
    return {"losses": [], "pair_accs": [], "order_accs": [], "taus": [], "teacher_original_taus": []}


@torch.no_grad()
def eval_split(model: FlatAttentionOrderMLP, dataset_dir: Path, split: str, device: torch.device, batch_size: int, permute_seed: int) -> List[Dict]:
    paths = sorted((dataset_dir / split).glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt files under {dataset_dir / split}")
    first = torch.load(paths[0], map_location="cpu")
    num_blocks = int(first["attention"].shape[-1])
    pair_i = first["pair_i"].long()
    pair_j = first["pair_j"].long()
    if pair_i.numel() == 0:
        pair_i, pair_j = pair_indices(num_blocks)
    target_original_rank = original_target_rank(num_blocks, int(permute_seed))
    overall = empty_stats()
    bucket_stats = defaultdict(empty_stats)
    head_stats = defaultdict(empty_stats)
    model.eval()
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        attention = payload["attention"]
        rank = payload["teacher_rank"].long()
        order = payload["teacher_order"].long()
        teacher_tau = tau_to_original(order, target_original_rank, pair_i, pair_j)
        layer = payload["layer"].long()
        head = payload["head"].long()
        pair_i_dev = pair_i.to(device)
        pair_j_dev = pair_j.to(device)
        for start in range(0, int(attention.size(0)), int(batch_size)):
            end = min(start + int(batch_size), int(attention.size(0)))
            attn_batch = attention[start:end].to(device)
            rank_batch = rank[start:end].to(device)
            logits = model(attn_batch)
            pred = logits[:, pair_i_dev] - logits[:, pair_j_dev]
            target = rank_batch[:, pair_i_dev] < rank_batch[:, pair_j_dev]
            loss_per = F.binary_cross_entropy_with_logits(pred.float(), target.float(), reduction="none").mean(dim=1).cpu()
            pair_acc_per = ((pred.cpu() > 0.0) == target.cpu()).float().mean(dim=1)
            pred_order = torch.argsort(logits.detach().cpu().float(), dim=1, descending=True)
            for local_idx in range(int(end - start)):
                sample_idx = start + local_idx
                pred_rank = torch.empty_like(pred_order[local_idx])
                pred_rank[pred_order[local_idx]] = torch.arange(num_blocks, dtype=pred_rank.dtype)
                target_rank = rank[sample_idx]
                order_acc = (
                    (pred_rank[pair_i] < pred_rank[pair_j]) == (target_rank[pair_i] < target_rank[pair_j])
                ).float().mean()
                t_tau = float(teacher_tau[sample_idx].item())
                loss = float(loss_per[local_idx].item())
                pair_acc = float(pair_acc_per[local_idx].item())
                order_pair_acc = float(order_acc.item())
                bname = bucket_name(t_tau)
                hname = f"L{int(layer[sample_idx].item())}H{int(head[sample_idx].item())}"
                for stats in (overall, bucket_stats[bname], head_stats[hname]):
                    update_stats(stats, loss, pair_acc, order_pair_acc, t_tau)
    rows = [row_from_stats(split, "overall", "all", overall)]
    for name, _, _ in BUCKETS:
        rows.append(row_from_stats(split, "bucket", name, bucket_stats[name]))
    for name in sorted(head_stats):
        rows.append(row_from_stats(split, "head", name, head_stats[name]))
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    fields = [
        "split",
        "group_type",
        "group",
        "num_samples",
        "teacher_original_tau_mean",
        "pair_bce",
        "pair_accuracy",
        "order_pair_accuracy",
        "kendall_tau_mean",
        "kendall_tau_std",
        "kendall_tau_min",
        "kendall_tau_max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = FlatAttentionOrderMLP(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    rows = []
    for split in [item.strip() for item in str(args.splits).split(",") if item.strip()]:
        rows.extend(eval_split(model, args.dataset_dir, split, device, int(args.batch_size), int(args.permute_seed)))
    write_csv(args.out_dir / "teacher_tau_bucket_eval.csv", rows)
    write_json = {
        "dataset_dir": str(args.dataset_dir),
        "checkpoint": str(args.checkpoint),
        "rows": rows,
    }
    (args.out_dir / "teacher_tau_bucket_eval.json").write_text(json.dumps(write_json, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(args.out_dir / "teacher_tau_bucket_eval.csv")}, indent=2))


if __name__ == "__main__":
    main()
