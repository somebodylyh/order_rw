#!/usr/bin/env python3
"""Train an Attn-MLP to fit the raw/reverse eig axis, ignoring arbitrary sign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

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


class AxisPairwiseDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str):
        self.dataset_dir = Path(dataset_dir)
        self.split = str(split)
        paths = sorted((self.dataset_dir / self.split).glob("shard_*.pt"))
        if not paths:
            raise FileNotFoundError(f"No shard_*.pt files under {self.dataset_dir / self.split}")
        payloads = [torch.load(path, map_location="cpu") for path in paths]
        self.attention = torch.cat([p["attention"] for p in payloads], dim=0).contiguous()
        self.raw_rank = torch.cat([p["raw_rank"] for p in payloads], dim=0).long().contiguous()
        self.reverse_rank = torch.cat([p["reverse_rank"] for p in payloads], dim=0).long().contiguous()
        self.layer = torch.cat([p["layer"] for p in payloads], dim=0).long().contiguous()
        self.head = torch.cat([p["head"] for p in payloads], dim=0).long().contiguous()
        self.iter = torch.cat([p["iter"] for p in payloads], dim=0).long().contiguous()
        self.record_index = torch.cat([p["record_index"] for p in payloads], dim=0).long().contiguous()
        if "raw_original_l2r_tau" in payloads[0]:
            self.raw_original_l2r_tau = torch.cat([p["raw_original_l2r_tau"] for p in payloads], dim=0).float()
        else:
            self.raw_original_l2r_tau = torch.full((self.attention.size(0),), float("nan"))
        self.pair_i = payloads[0]["pair_i"].long().contiguous()
        self.pair_j = payloads[0]["pair_j"].long().contiguous()
        self.num_blocks = int(self.attention.shape[-1])

    def __len__(self) -> int:
        return int(self.attention.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "attention": self.attention[idx],
            "raw_rank": self.raw_rank[idx],
            "reverse_rank": self.reverse_rank[idx],
            "layer": self.layer[idx],
            "head": self.head[idx],
            "iter": self.iter[idx],
            "record_index": self.record_index[idx],
            "raw_original_l2r_tau": self.raw_original_l2r_tau[idx],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--num_blocks", type=int, default=64)
    parser.add_argument("--hidden_dims", type=str, default="8192,8192,4096")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--input_normalization", type=str, default="zscore")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--axis_tau_gate", type=float, default=0.98)
    return parser.parse_args()


def parse_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_command(report_dir: Path) -> None:
    command = " ".join(shlex.quote(part) for part in sys.argv)
    (report_dir / "train_command.sh").write_text(command + "\n", encoding="utf-8")


def append_csv(path: Path, row: Dict, fields: Sequence[str]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def bucket_name(value: float) -> str:
    for name, lo, hi in BUCKETS:
        if float(value) >= lo and float(value) < hi:
            return name
    return "unknown"


def per_sample_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(pred.float(), target.float(), reduction="none").mean(dim=1)


def axis_loss_and_metrics(
    logits: torch.Tensor,
    raw_rank: torch.Tensor,
    reverse_rank: torch.Tensor,
    tau: float,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pred = (logits[:, pair_i] - logits[:, pair_j]) / max(float(tau), 1e-6)
    raw_target = raw_rank[:, pair_i] < raw_rank[:, pair_j]
    reverse_target = reverse_rank[:, pair_i] < reverse_rank[:, pair_j]
    raw_loss = per_sample_bce(pred, raw_target)
    reverse_loss = per_sample_bce(pred, reverse_target)
    choose_reverse = reverse_loss < raw_loss
    loss = torch.minimum(raw_loss, reverse_loss).mean()
    raw_pair_acc = ((pred.detach() > 0.0) == raw_target).float().mean(dim=1)
    reverse_pair_acc = ((pred.detach() > 0.0) == reverse_target).float().mean(dim=1)
    axis_pair_acc = torch.maximum(raw_pair_acc, reverse_pair_acc)
    return loss, {
        "raw_pair_acc": raw_pair_acc,
        "reverse_pair_acc": reverse_pair_acc,
        "axis_pair_acc": axis_pair_acc,
        "choose_reverse": choose_reverse.float(),
        "raw_loss": raw_loss.detach(),
        "reverse_loss": reverse_loss.detach(),
        "axis_loss": torch.minimum(raw_loss, reverse_loss).detach(),
    }


def order_axis_metrics_from_logits(
    logits: torch.Tensor,
    raw_rank: torch.Tensor,
    reverse_rank: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> Dict[str, np.ndarray | float]:
    pred_order = torch.argsort(logits.detach().float(), dim=1, descending=True).cpu()
    raw_rank = raw_rank.cpu()
    reverse_rank = reverse_rank.cpu()
    pair_i = pair_i.cpu()
    pair_j = pair_j.cpu()
    raw_accs = []
    reverse_accs = []
    axis_accs = []
    chosen_reverse = []
    for idx in range(pred_order.size(0)):
        pred_rank = torch.empty_like(pred_order[idx])
        pred_rank[pred_order[idx]] = torch.arange(pred_order.size(1), dtype=pred_rank.dtype)
        raw_acc = ((pred_rank[pair_i] < pred_rank[pair_j]) == (raw_rank[idx][pair_i] < raw_rank[idx][pair_j])).float().mean()
        reverse_acc = (
            (pred_rank[pair_i] < pred_rank[pair_j]) == (reverse_rank[idx][pair_i] < reverse_rank[idx][pair_j])
        ).float().mean()
        raw_value = float(raw_acc.item())
        reverse_value = float(reverse_acc.item())
        raw_accs.append(raw_value)
        reverse_accs.append(reverse_value)
        axis_accs.append(max(raw_value, reverse_value))
        chosen_reverse.append(float(reverse_value > raw_value))
    raw_arr = np.asarray(raw_accs, dtype=np.float64)
    reverse_arr = np.asarray(reverse_accs, dtype=np.float64)
    axis_arr = np.asarray(axis_accs, dtype=np.float64)
    return {
        "raw_order_pair_accuracy": float(raw_arr.mean()),
        "raw_kendall_tau_mean": float((2.0 * raw_arr - 1.0).mean()),
        "reverse_order_pair_accuracy": float(reverse_arr.mean()),
        "reverse_kendall_tau_mean": float((2.0 * reverse_arr - 1.0).mean()),
        "axis_order_pair_accuracy": float(axis_arr.mean()),
        "axis_kendall_tau_mean": float((2.0 * axis_arr - 1.0).mean()),
        "axis_kendall_tau_std": float((2.0 * axis_arr - 1.0).std()),
        "axis_kendall_tau_min": float((2.0 * axis_arr - 1.0).min()),
        "axis_choose_reverse_rate": float(np.mean(chosen_reverse)),
        "axis_tau_per_sample": 2.0 * axis_arr - 1.0,
    }


@torch.no_grad()
def evaluate(
    model: FlatAttentionOrderMLP,
    dataset: Dataset,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    device: torch.device,
    batch_size: int,
    tau: float,
    num_workers: int,
    pin_memory: bool,
    use_amp: bool,
) -> Dict[str, float | int]:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers), pin_memory=pin_memory)
    model.eval()
    losses = []
    raw_accs = []
    reverse_accs = []
    axis_accs = []
    choose_reverse = []
    logits_all = []
    raw_rank_all = []
    reverse_rank_all = []
    total = 0
    pair_i_dev = pair_i.to(device)
    pair_j_dev = pair_j.to(device)
    for batch in loader:
        attn = batch["attention"].to(device, non_blocking=pin_memory)
        raw_rank = batch["raw_rank"].to(device, non_blocking=pin_memory)
        reverse_rank = batch["reverse_rank"].to(device, non_blocking=pin_memory)
        with torch.autocast(device_type=device.type, enabled=bool(use_amp and device.type == "cuda")):
            logits = model(attn)
            loss, metrics = axis_loss_and_metrics(logits, raw_rank, reverse_rank, tau, pair_i_dev, pair_j_dev)
        count = int(attn.size(0))
        losses.append(float(loss.item()) * count)
        raw_accs.append(metrics["raw_pair_acc"].detach().cpu())
        reverse_accs.append(metrics["reverse_pair_acc"].detach().cpu())
        axis_accs.append(metrics["axis_pair_acc"].detach().cpu())
        choose_reverse.append(metrics["choose_reverse"].detach().cpu())
        logits_all.append(logits.detach().cpu())
        raw_rank_all.append(raw_rank.detach().cpu())
        reverse_rank_all.append(reverse_rank.detach().cpu())
        total += count
    raw_pair = torch.cat(raw_accs).numpy()
    reverse_pair = torch.cat(reverse_accs).numpy()
    axis_pair = torch.cat(axis_accs).numpy()
    chosen = torch.cat(choose_reverse).numpy()
    order_metrics = order_axis_metrics_from_logits(
        torch.cat(logits_all), torch.cat(raw_rank_all), torch.cat(reverse_rank_all), pair_i, pair_j
    )
    return {
        "axis_bce": float(sum(losses) / max(1, total)),
        "raw_pair_accuracy": float(raw_pair.mean()),
        "raw_pair_tau": float((2.0 * raw_pair - 1.0).mean()),
        "reverse_pair_accuracy": float(reverse_pair.mean()),
        "reverse_pair_tau": float((2.0 * reverse_pair - 1.0).mean()),
        "axis_pair_accuracy": float(axis_pair.mean()),
        "axis_pair_tau": float((2.0 * axis_pair - 1.0).mean()),
        "loss_choose_reverse_rate": float(chosen.mean()),
        "num_samples": int(total),
        **{key: value for key, value in order_metrics.items() if key != "axis_tau_per_sample"},
    }


def subset_indices_by_head(dataset: AxisPairwiseDataset, layer: int, head: int) -> List[int]:
    mask = (dataset.layer == int(layer)) & (dataset.head == int(head))
    return torch.nonzero(mask, as_tuple=False).flatten().tolist()


def all_layer_heads(dataset: AxisPairwiseDataset) -> List[tuple[int, int]]:
    return sorted({(int(l), int(h)) for l, h in zip(dataset.layer.tolist(), dataset.head.tolist())})


def dataset_summary(dataset: AxisPairwiseDataset) -> Dict:
    out = {
        "split": dataset.split,
        "num_samples": int(len(dataset)),
        "num_blocks": int(dataset.num_blocks),
        "iter_min": int(dataset.iter.min().item()),
        "iter_max": int(dataset.iter.max().item()),
        "record_min": int(dataset.record_index.min().item()),
        "record_max": int(dataset.record_index.max().item()),
        "raw_original_l2r_tau_mean": float(dataset.raw_original_l2r_tau.mean().item()),
    }
    bucket_counts = defaultdict(int)
    for value in dataset.raw_original_l2r_tau.tolist():
        bucket_counts[bucket_name(float(value))] += 1
    out["bucket_counts"] = dict(bucket_counts)
    head_counts = {}
    for layer, head in all_layer_heads(dataset):
        head_counts[f"L{layer}H{head}"] = int(len(subset_indices_by_head(dataset, layer, head)))
    out["head_counts"] = head_counts
    return out


def maybe_plot(report_dir: Path) -> None:
    metrics_path = report_dir / "metrics.csv"
    if not metrics_path.exists():
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (report_dir / "plot_skipped.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return
    rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8")))
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if not train_rows or not val_rows:
        return
    plt.figure(figsize=(7, 4))
    plt.plot([int(r["epoch"]) for r in train_rows], [float(r["axis_bce"]) for r in train_rows], label="train axis BCE")
    plt.plot([int(r["epoch"]) for r in val_rows], [float(r["axis_bce"]) for r in val_rows], label="val axis BCE")
    plt.xlabel("epoch")
    plt.ylabel("axis BCE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "loss_curve.png", dpi=180)
    plt.close()
    plt.figure(figsize=(7, 4))
    plt.plot([int(r["epoch"]) for r in val_rows], [float(r["axis_kendall_tau_mean"]) for r in val_rows], label="val axis tau")
    plt.axhline(0.98, color="black", linestyle="--", linewidth=1, label="0.98 gate")
    plt.xlabel("epoch")
    plt.ylabel("Kendall tau to eig axis")
    plt.ylim(-0.05, 1.01)
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "axis_tau_curve.png", dpi=180)
    plt.close()


@torch.no_grad()
def grouped_eval(
    model: FlatAttentionOrderMLP,
    dataset: AxisPairwiseDataset,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    device: torch.device,
    batch_size: int,
    tau: float,
    num_workers: int,
    pin_memory: bool,
    use_amp: bool,
) -> List[Dict]:
    rows = []
    rows.append(
        {
            "split": dataset.split,
            "group_type": "overall",
            "group": "all",
            **evaluate(model, dataset, pair_i, pair_j, device, batch_size, tau, num_workers, pin_memory, use_amp),
        }
    )
    for bucket, _, _ in BUCKETS:
        indices = [
            idx
            for idx, value in enumerate(dataset.raw_original_l2r_tau.tolist())
            if bucket_name(float(value)) == bucket
        ]
        subset = Subset(dataset, indices)
        rows.append(
            {
                "split": dataset.split,
                "group_type": "bucket",
                "group": bucket,
                **evaluate(model, subset, pair_i, pair_j, device, batch_size, tau, num_workers, pin_memory, use_amp),
            }
        )
    for layer, head in all_layer_heads(dataset):
        subset = Subset(dataset, subset_indices_by_head(dataset, layer, head))
        rows.append(
            {
                "split": dataset.split,
                "group_type": "head",
                "group": f"L{layer}H{head}",
                **evaluate(model, subset, pair_i, pair_j, device, batch_size, tau, num_workers, pin_memory, use_amp),
            }
        )
    return rows


def write_grouped_csv(path: Path, rows: List[Dict]) -> None:
    fields = [
        "split",
        "group_type",
        "group",
        "num_samples",
        "axis_bce",
        "axis_order_pair_accuracy",
        "axis_kendall_tau_mean",
        "raw_order_pair_accuracy",
        "raw_kendall_tau_mean",
        "reverse_order_pair_accuracy",
        "reverse_kendall_tau_mean",
        "axis_choose_reverse_rate",
        "loss_choose_reverse_rate",
        "axis_kendall_tau_std",
        "axis_kendall_tau_min",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_command(args.report_dir)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    train_data = AxisPairwiseDataset(args.dataset_dir, "train")
    val_data = AxisPairwiseDataset(args.dataset_dir, "val")
    pair_i = train_data.pair_i
    pair_j = train_data.pair_j
    model_config = {
        "num_blocks": int(args.num_blocks),
        "hidden_dims": parse_ints(args.hidden_dims),
        "dropout": float(args.dropout),
        "activation": str(args.activation),
        "input_normalization": str(args.input_normalization),
    }
    model = FlatAttentionOrderMLP(**model_config).to(device)
    params = int(sum(p.numel() for p in model.parameters()))
    run_config = {
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        "report_dir": str(args.report_dir),
        "seed": int(args.seed),
        "model_config": model_config,
        "parameter_count": int(params),
        "loss": {"axis_pairwise_bce": "min(raw_bce, reverse_bce)", "pairwise_bce_tau": float(args.tau)},
        "optimizer": {
            "name": "AdamW",
            "lr": float(args.lr),
            "min_lr": float(args.min_lr),
            "weight_decay": float(args.weight_decay),
            "betas": [float(args.beta1), float(args.beta2)],
            "grad_clip": float(args.grad_clip),
        },
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "amp": bool(args.amp),
        "axis_tau_gate": float(args.axis_tau_gate),
        "train": dataset_summary(train_data),
        "val": dataset_summary(val_data),
    }
    write_json(args.report_dir / "config.json", run_config)
    write_json(args.out_dir / "config.json", run_config)

    train_loader = DataLoader(
        train_data,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(float(args.beta1), float(args.beta2)),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
        eta_min=float(args.min_lr),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    pair_i_dev = pair_i.to(device)
    pair_j_dev = pair_j.to(device)
    metric_fields = [
        "epoch",
        "split",
        "axis_bce",
        "raw_pair_tau",
        "reverse_pair_tau",
        "axis_pair_tau",
        "axis_kendall_tau_mean",
        "axis_order_pair_accuracy",
        "axis_choose_reverse_rate",
        "num_samples",
        "epoch_seconds",
        "lr",
    ]
    metrics_path = args.report_dir / "metrics.csv"
    best_val_axis_tau = -float("inf")
    best_val_bce = float("inf")
    best_epoch = -1
    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_axis_sum = 0.0
        train_total = 0
        for batch in train_loader:
            attn = batch["attention"].to(device, non_blocking=bool(args.pin_memory))
            raw_rank = batch["raw_rank"].to(device, non_blocking=bool(args.pin_memory))
            reverse_rank = batch["reverse_rank"].to(device, non_blocking=bool(args.pin_memory))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=bool(args.amp and device.type == "cuda")):
                logits = model(attn)
                loss, train_metrics = axis_loss_and_metrics(
                    logits, raw_rank, reverse_rank, float(args.tau), pair_i_dev, pair_j_dev
                )
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            count = int(attn.size(0))
            train_loss_sum += float(loss.item()) * count
            train_axis_sum += float(train_metrics["axis_pair_acc"].mean().item()) * count
            train_total += count
        scheduler.step()
        epoch_seconds = time.perf_counter() - t0
        train_row = {
            "epoch": int(epoch),
            "split": "train",
            "axis_bce": float(train_loss_sum / max(1, train_total)),
            "raw_pair_tau": float("nan"),
            "reverse_pair_tau": float("nan"),
            "axis_pair_tau": float(2.0 * train_axis_sum / max(1, train_total) - 1.0),
            "axis_kendall_tau_mean": float("nan"),
            "axis_order_pair_accuracy": float("nan"),
            "axis_choose_reverse_rate": float("nan"),
            "num_samples": int(train_total),
            "epoch_seconds": float(epoch_seconds),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        append_csv(metrics_path, train_row, metric_fields)
        should_eval = epoch == 1 or epoch == int(args.epochs) or epoch % max(1, int(args.eval_every)) == 0
        if should_eval:
            val_metrics = evaluate(
                model,
                val_data,
                pair_i,
                pair_j,
                device,
                int(args.batch_size),
                float(args.tau),
                int(args.num_workers),
                bool(args.pin_memory),
                bool(args.amp),
            )
            val_row = {
                "epoch": int(epoch),
                "split": "val",
                **{key: val_metrics.get(key, float("nan")) for key in metric_fields if key not in {"epoch", "split", "epoch_seconds", "lr"}},
                "epoch_seconds": float(epoch_seconds),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            append_csv(metrics_path, val_row, metric_fields)
            val_axis_tau = float(val_metrics["axis_kendall_tau_mean"])
            is_best = val_axis_tau > best_val_axis_tau or (
                math.isclose(val_axis_tau, best_val_axis_tau) and float(val_metrics["axis_bce"]) < best_val_bce
            )
            if is_best:
                best_val_axis_tau = val_axis_tau
                best_val_bce = float(val_metrics["axis_bce"])
                best_epoch = int(epoch)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": model_config,
                        "policy_type": "direct_asym_eig_axis_distilled_mlp",
                        "target_direction": "raw_reverse_axis_sign_invariant",
                        "training_meta": {
                            "best_epoch": int(best_epoch),
                            "best_val_axis_tau": float(best_val_axis_tau),
                            "best_val_axis_bce": float(best_val_bce),
                            "dataset_dir": str(args.dataset_dir),
                            "parameter_count": int(params),
                            "seed": int(args.seed),
                        },
                    },
                    args.out_dir / "best_by_val_axis_tau.pt",
                )
            print(
                f"[axis-mlp] epoch={epoch}/{args.epochs} "
                f"train_bce={train_row['axis_bce']:.6f} train_axis_tau={train_row['axis_pair_tau']:.4f} "
                f"val_bce={val_metrics['axis_bce']:.6f} val_axis_tau={val_axis_tau:.4f} "
                f"val_raw_tau={val_metrics['raw_kendall_tau_mean']:.4f} "
                f"val_rev_tau={val_metrics['reverse_kendall_tau_mean']:.4f}",
                flush=True,
            )
        maybe_plot(args.report_dir)

    best_payload = torch.load(args.out_dir / "best_by_val_axis_tau.pt", map_location=device)
    best_model = FlatAttentionOrderMLP(**best_payload["config"]).to(device)
    best_model.load_state_dict(best_payload["model_state_dict"])
    grouped_rows = []
    for split_data in [train_data, val_data]:
        grouped_rows.extend(
            grouped_eval(
                best_model,
                split_data,
                pair_i,
                pair_j,
                device,
                int(args.batch_size),
                float(args.tau),
                int(args.num_workers),
                bool(args.pin_memory),
                bool(args.amp),
            )
        )
    write_grouped_csv(args.report_dir / "bucket_and_head_axis_eval.csv", grouped_rows)
    write_json(args.report_dir / "bucket_and_head_axis_eval.json", {"rows": grouped_rows})
    passed_gate = bool(best_val_axis_tau >= float(args.axis_tau_gate))
    summary = {
        "config": run_config,
        "best_epoch": int(best_epoch),
        "best_val_axis_tau": float(best_val_axis_tau),
        "best_val_axis_bce": float(best_val_bce),
        "passed_axis_tau_gate": bool(passed_gate),
        "checkpoint": str(args.out_dir / "best_by_val_axis_tau.pt"),
        "grouped_eval": grouped_rows,
    }
    write_json(args.out_dir / "training_summary.json", summary)
    write_json(args.report_dir / "training_summary.json", summary)
    maybe_plot(args.report_dir)

    bucket_rows = [row for row in grouped_rows if row["split"] == "val" and row["group_type"] in {"overall", "bucket"}]
    lines = [
        "# try15: Sign-Invariant Eig-Axis MLP Distillation",
        "",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Checkpoint: `{args.out_dir / 'best_by_val_axis_tau.pt'}`",
        f"- Parameters: `{params}`",
        f"- Best epoch: `{best_epoch}`",
        f"- Best val axis tau: `{best_val_axis_tau:.6f}`",
        f"- Best val axis BCE: `{best_val_bce:.6f}`",
        f"- Axis tau gate `{float(args.axis_tau_gate):.3f}` passed: `{passed_gate}`",
        "",
        "## Val Axis Buckets",
        "",
        "| bucket | samples | axis tau | axis pair acc | raw tau | reverse tau | choose reverse | BCE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bucket_rows:
        lines.append(
            f"| {row['group']} | {int(row['num_samples'])} | {float(row['axis_kendall_tau_mean']):.4f} | "
            f"{float(row['axis_order_pair_accuracy']):.4f} | {float(row['raw_kendall_tau_mean']):.4f} | "
            f"{float(row['reverse_kendall_tau_mean']):.4f} | {float(row['axis_choose_reverse_rate']):.4f} | "
            f"{float(row['axis_bce']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics.csv`: train/val axis BCE and tau by epoch.",
            "- `loss_curve.png`: train/val axis BCE.",
            "- `axis_tau_curve.png`: val axis Kendall tau.",
            "- `bucket_and_head_axis_eval.csv`: bucket and per-head held-out axis metrics.",
        ]
    )
    (args.report_dir / "training_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "best_val_axis_tau": best_val_axis_tau, "passed_axis_tau_gate": passed_gate}, indent=2))


if __name__ == "__main__":
    main()
