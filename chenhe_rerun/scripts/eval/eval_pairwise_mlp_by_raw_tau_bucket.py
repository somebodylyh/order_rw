#!/usr/bin/env python3
"""Evaluate a pairwise Attn-MLP by raw OriginalL2R tau buckets."""

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
    return parser.parse_args()


def bucket_name(value: float) -> str:
    for name, lo, hi in BUCKETS:
        if float(value) >= lo and float(value) < hi:
            return name
    raise ValueError(f"tau={value} fell outside buckets")


def pair_indices_from_payload(payload: Dict) -> tuple[torch.Tensor, torch.Tensor]:
    return payload["pair_i"].long(), payload["pair_j"].long()


def metrics_from_stats(stats: Dict[str, List[float]]) -> Dict:
    arr_loss = np.asarray(stats["losses"], dtype=np.float64)
    arr_acc = np.asarray(stats["accs"], dtype=np.float64)
    arr_order = np.asarray(stats["order_accs"], dtype=np.float64)
    arr_tau = np.asarray(stats["taus"], dtype=np.float64)
    if arr_tau.size == 0:
        return {
            "num_samples": 0,
            "pair_bce": float("nan"),
            "pair_accuracy": float("nan"),
            "order_pair_accuracy": float("nan"),
            "kendall_tau_mean": float("nan"),
            "kendall_tau_std": float("nan"),
            "kendall_tau_min": float("nan"),
            "kendall_tau_max": float("nan"),
        }
    return {
        "num_samples": int(arr_tau.size),
        "pair_bce": float(arr_loss.mean()),
        "pair_accuracy": float(arr_acc.mean()),
        "order_pair_accuracy": float(arr_order.mean()),
        "kendall_tau_mean": float(arr_tau.mean()),
        "kendall_tau_std": float(arr_tau.std()),
        "kendall_tau_min": float(arr_tau.min()),
        "kendall_tau_max": float(arr_tau.max()),
    }


@torch.no_grad()
def eval_split(model: FlatAttentionOrderMLP, dataset_dir: Path, split: str, device: torch.device, batch_size: int) -> List[Dict]:
    bucket_stats = defaultdict(lambda: {"losses": [], "accs": [], "order_accs": [], "taus": [], "raw_taus": []})
    head_stats = defaultdict(lambda: {"losses": [], "accs": [], "order_accs": [], "taus": [], "raw_taus": []})
    overall = {"losses": [], "accs": [], "order_accs": [], "taus": [], "raw_taus": []}
    paths = sorted((dataset_dir / split).glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No shards for split={split} under {dataset_dir}")
    model.eval()
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        pair_i, pair_j = pair_indices_from_payload(payload)
        pair_i_dev = pair_i.to(device)
        pair_j_dev = pair_j.to(device)
        attention = payload["attention"]
        rank = payload["teacher_rank"].long()
        raw_tau = payload["raw_original_l2r_tau"].float()
        layer = payload["layer"].long()
        head = payload["head"].long()
        for start in range(0, int(attention.size(0)), int(batch_size)):
            end = min(start + int(batch_size), int(attention.size(0)))
            attn_batch = attention[start:end].to(device)
            rank_batch = rank[start:end].to(device)
            logits = model(attn_batch)
            pred = logits[:, pair_i_dev] - logits[:, pair_j_dev]
            target = (rank_batch[:, pair_i_dev] < rank_batch[:, pair_j_dev]).float()
            loss_per = F.binary_cross_entropy_with_logits(pred.float(), target, reduction="none").mean(dim=1).cpu()
            acc_per = ((pred.cpu() > 0.0) == (target.cpu() > 0.5)).float().mean(dim=1)
            pred_order = torch.argsort(logits.detach().cpu().float(), dim=1, descending=True)
            order_accs = []
            for row_idx in range(pred_order.size(0)):
                pred_rank = torch.empty_like(pred_order[row_idx])
                pred_rank[pred_order[row_idx]] = torch.arange(pred_order.size(1), dtype=pred_rank.dtype)
                target_rank = rank[start + row_idx]
                order_acc = ((pred_rank[pair_i] < pred_rank[pair_j]) == (target_rank[pair_i] < target_rank[pair_j])).float().mean()
                order_accs.append(float(order_acc.item()))
            tau_per = [2.0 * value - 1.0 for value in order_accs]
            for local_idx, (loss, acc, order_acc, tau_value) in enumerate(
                zip(loss_per.tolist(), acc_per.tolist(), order_accs, tau_per)
            ):
                sample_idx = start + local_idx
                bname = bucket_name(float(raw_tau[sample_idx].item()))
                hname = f"L{int(layer[sample_idx].item())}H{int(head[sample_idx].item())}"
                for stats in (overall, bucket_stats[bname], head_stats[hname]):
                    stats["losses"].append(float(loss))
                    stats["accs"].append(float(acc))
                    stats["order_accs"].append(float(order_acc))
                    stats["taus"].append(float(tau_value))
                    stats["raw_taus"].append(float(raw_tau[sample_idx].item()))
    rows = []
    overall_metrics = metrics_from_stats(overall)
    overall_metrics.update({"split": split, "group_type": "overall", "group": "all", "raw_tau_mean": float(np.mean(overall["raw_taus"]))})
    rows.append(overall_metrics)
    for name, _, _ in BUCKETS:
        stats = bucket_stats[name]
        metrics = metrics_from_stats(stats)
        metrics.update(
            {
                "split": split,
                "group_type": "bucket",
                "group": name,
                "raw_tau_mean": float(np.mean(stats["raw_taus"])) if stats["raw_taus"] else float("nan"),
            }
        )
        rows.append(metrics)
    for name in sorted(head_stats.keys()):
        stats = head_stats[name]
        metrics = metrics_from_stats(stats)
        metrics.update(
            {
                "split": split,
                "group_type": "head",
                "group": name,
                "raw_tau_mean": float(np.mean(stats["raw_taus"])) if stats["raw_taus"] else float("nan"),
            }
        )
        rows.append(metrics)
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    fields = [
        "split",
        "group_type",
        "group",
        "num_samples",
        "raw_tau_mean",
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
    payload = torch.load(args.checkpoint, map_location=device)
    model = FlatAttentionOrderMLP(**payload["config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    rows = []
    for split in [item.strip() for item in str(args.splits).split(",") if item.strip()]:
        rows.extend(eval_split(model, args.dataset_dir, split, device, int(args.batch_size)))
    write_csv(args.out_dir / "bucket_and_head_eval.csv", rows)
    summary = {"checkpoint": str(args.checkpoint), "dataset_dir": str(args.dataset_dir), "rows": rows}
    (args.out_dir / "bucket_and_head_eval.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(args.out_dir / "bucket_and_head_eval.csv")}, indent=2))


if __name__ == "__main__":
    main()
