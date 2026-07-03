#!/usr/bin/env python3
"""Train an Attn-MLP to imitate a pairwise teacher operator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attn_mlp_order_policy import FlatAttentionOrderMLP  # noqa: E402


class PairwiseOperatorDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        min_loss_score_gap: float = 0.0,
        include_layers: Sequence[int] | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = str(split)
        self.min_loss_score_gap = float(min_loss_score_gap)
        self.include_layers = None if include_layers is None else tuple(int(value) for value in include_layers)
        paths = sorted((self.dataset_dir / self.split).glob("shard_*.pt"))
        if not paths:
            raise FileNotFoundError(f"No shard_*.pt files for split={split} under {dataset_dir}.")
        payloads = [torch.load(path, map_location="cpu") for path in paths]
        self.attention = torch.cat([p["attention"] for p in payloads], dim=0).contiguous()
        self.teacher_rank = torch.cat([p["teacher_rank"] for p in payloads], dim=0).long().contiguous()
        self.teacher_order = torch.cat([p["teacher_order"] for p in payloads], dim=0).long().contiguous()
        self.layer = torch.cat([p["layer"] for p in payloads], dim=0).long().contiguous()
        self.head = torch.cat([p["head"] for p in payloads], dim=0).long().contiguous()
        self.iter = torch.cat([p["iter"] for p in payloads], dim=0).long().contiguous()
        self.record_index = torch.cat([p["record_index"] for p in payloads], dim=0).long().contiguous()
        self.selected_reverse = torch.cat([p["selected_reverse"] for p in payloads], dim=0).bool().contiguous()
        self.loss_score_gap = torch.cat([p["loss_score_gap"] for p in payloads], dim=0).float().contiguous()
        self.eigval_abs = torch.cat([p["eigval_abs"] for p in payloads], dim=0).float().contiguous()
        self.vector_std = torch.cat([p["vector_std"] for p in payloads], dim=0).float().contiguous()
        self.pair_i = payloads[0]["pair_i"].long().contiguous()
        self.pair_j = payloads[0]["pair_j"].long().contiguous()
        self.num_blocks = int(self.attention.shape[-1])
        mask = torch.ones((self.attention.size(0),), dtype=torch.bool)
        if self.min_loss_score_gap > 0.0:
            mask &= self.loss_score_gap >= float(self.min_loss_score_gap)
        if self.include_layers:
            layer_mask = torch.zeros_like(mask)
            for layer in self.include_layers:
                layer_mask |= self.layer == int(layer)
            mask &= layer_mask
        if not bool(mask.any()):
            raise ValueError(
                f"No samples left for split={split} after filters: "
                f"min_loss_score_gap={self.min_loss_score_gap}, include_layers={self.include_layers}."
            )
        if not bool(mask.all()):
            self.attention = self.attention[mask].contiguous()
            self.teacher_rank = self.teacher_rank[mask].contiguous()
            self.teacher_order = self.teacher_order[mask].contiguous()
            self.layer = self.layer[mask].contiguous()
            self.head = self.head[mask].contiguous()
            self.iter = self.iter[mask].contiguous()
            self.record_index = self.record_index[mask].contiguous()
            self.selected_reverse = self.selected_reverse[mask].contiguous()
            self.loss_score_gap = self.loss_score_gap[mask].contiguous()
            self.eigval_abs = self.eigval_abs[mask].contiguous()
            self.vector_std = self.vector_std[mask].contiguous()

    def __len__(self) -> int:
        return int(self.attention.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "attention": self.attention[idx],
            "teacher_rank": self.teacher_rank[idx],
            "teacher_order": self.teacher_order[idx],
            "layer": self.layer[idx],
            "head": self.head[idx],
            "iter": self.iter[idx],
            "record_index": self.record_index[idx],
            "selected_reverse": self.selected_reverse[idx],
            "loss_score_gap": self.loss_score_gap[idx],
            "eigval_abs": self.eigval_abs[idx],
            "vector_std": self.vector_std[idx],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a large pairwise Attn-MLP operator student.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260622)
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
    parser.add_argument("--lambda_rank_mse", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--tau_gate", type=float, default=0.98)
    parser.add_argument("--eval_input_modes", type=str, default="real,zero,gaussian_like,gaussian_std1,shuffled")
    parser.add_argument("--min_loss_score_gap", type=float, default=0.0)
    parser.add_argument("--val_min_loss_score_gap", type=float, default=None)
    parser.add_argument("--include_layers", type=str, default="")
    return parser.parse_args()


def parse_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_strings(raw: str) -> List[str]:
    return [str(item.strip()) for item in str(raw).split(",") if item.strip()]


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_command(report_dir: Path) -> None:
    command = " ".join(shlex.quote(part) for part in sys.argv)
    (report_dir / "train_command.sh").write_text(command + "\n", encoding="utf-8")


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def pair_loss_and_acc(
    logits: torch.Tensor,
    rank: torch.Tensor,
    tau: float,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = (logits[:, pair_i] - logits[:, pair_j]) / max(float(tau), 1e-6)
    target = (rank[:, pair_i] < rank[:, pair_j]).float()
    loss = F.binary_cross_entropy_with_logits(pred.float(), target)
    acc = ((pred.detach() > 0.0) == (target > 0.5)).float().mean()
    return loss, acc


def rank_mse_loss(logits: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
    n = int(logits.size(1))
    target_priority = 1.0 - rank.float() / float(max(1, n - 1))
    target_priority = target_priority - target_priority.mean(dim=1, keepdim=True)
    target_priority = target_priority / target_priority.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    pred = logits.float() - logits.float().mean(dim=1, keepdim=True)
    pred = pred / pred.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return F.mse_loss(pred, target_priority)


def transform_attention(
    attn: torch.Tensor,
    mode: str,
    generator: torch.Generator,
    data_mean: float,
    data_std: float,
) -> torch.Tensor:
    mode = str(mode)
    if mode == "real":
        return attn
    if mode == "zero":
        return torch.zeros_like(attn)
    if mode == "gaussian_like":
        return torch.randn(attn.shape, generator=generator, device=attn.device, dtype=attn.dtype) * float(data_std) + float(data_mean)
    if mode == "gaussian_std1":
        return torch.randn(attn.shape, generator=generator, device=attn.device, dtype=attn.dtype)
    if mode == "shuffled":
        rows = []
        n = int(attn.size(-1))
        for item in attn:
            perm = torch.randperm(n, generator=generator, device=attn.device)
            rows.append(item.index_select(0, perm).index_select(1, perm))
        return torch.stack(rows, dim=0)
    raise ValueError(f"Unsupported eval input mode={mode!r}.")


def order_metrics_from_logits(
    logits: torch.Tensor,
    teacher_rank: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> Dict[str, float]:
    pred_order = torch.argsort(logits.detach().float(), dim=1, descending=True).cpu()
    teacher_rank = teacher_rank.cpu()
    pair_i = pair_i.cpu()
    pair_j = pair_j.cpu()
    pair_accs = []
    exact = []
    for idx in range(pred_order.size(0)):
        pred_rank = torch.empty_like(pred_order[idx])
        pred_rank[pred_order[idx]] = torch.arange(pred_order.size(1), dtype=pred_rank.dtype)
        target = teacher_rank[idx]
        pair_acc = ((pred_rank[pair_i] < pred_rank[pair_j]) == (target[pair_i] < target[pair_j])).float().mean()
        pair_accs.append(float(pair_acc.item()))
        teacher_order = torch.argsort(target, descending=False)
        exact.append(float(torch.equal(pred_order[idx], teacher_order)))
    pair_acc_arr = np.asarray(pair_accs, dtype=np.float64)
    tau_arr = 2.0 * pair_acc_arr - 1.0
    return {
        "order_pair_accuracy": float(pair_acc_arr.mean()) if pair_acc_arr.size else float("nan"),
        "kendall_tau_mean": float(tau_arr.mean()) if tau_arr.size else float("nan"),
        "kendall_tau_std": float(tau_arr.std()) if tau_arr.size else float("nan"),
        "kendall_tau_min": float(tau_arr.min()) if tau_arr.size else float("nan"),
        "exact_match_rate": float(np.mean(exact)) if exact else float("nan"),
    }


def subset_indices_by_head(dataset: PairwiseOperatorDataset, layer: int, head: int) -> List[int]:
    mask = (dataset.layer == int(layer)) & (dataset.head == int(head))
    return torch.nonzero(mask, as_tuple=False).flatten().tolist()


def all_layer_heads(dataset: PairwiseOperatorDataset) -> List[tuple[int, int]]:
    pairs = sorted({(int(l), int(h)) for l, h in zip(dataset.layer.tolist(), dataset.head.tolist())})
    return pairs


def append_csv(path: Path, row: Dict, fields: Sequence[str]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def evaluate(
    model: FlatAttentionOrderMLP,
    dataset: Dataset,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    device: torch.device,
    batch_size: int,
    tau: float,
    input_mode: str,
    data_mean: float,
    data_std: float,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    use_amp: bool,
) -> Dict[str, float | int | str]:
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
    model.eval()
    losses = []
    accs = []
    logits_all = []
    rank_all = []
    total = 0
    generator_device = device.type if device.type == "cuda" else "cpu"
    gen = torch.Generator(device=generator_device)
    gen.manual_seed(int(seed))
    pair_i_dev = pair_i.to(device)
    pair_j_dev = pair_j.to(device)
    for batch in loader:
        attn = batch["attention"].to(device, non_blocking=bool(pin_memory))
        rank = batch["teacher_rank"].to(device, non_blocking=bool(pin_memory))
        attn = transform_attention(attn, input_mode, gen, data_mean, data_std)
        with torch.autocast(device_type=device.type, enabled=bool(use_amp and device.type == "cuda")):
            logits = model(attn)
            loss, acc = pair_loss_and_acc(logits, rank, tau, pair_i_dev, pair_j_dev)
        count = int(attn.size(0))
        losses.append(float(loss.item()) * count)
        accs.append(float(acc.item()) * count)
        logits_all.append(logits.detach().cpu())
        rank_all.append(rank.detach().cpu())
        total += count
    logits_cat = torch.cat(logits_all, dim=0)
    rank_cat = torch.cat(rank_all, dim=0)
    metrics = order_metrics_from_logits(logits_cat, rank_cat, pair_i, pair_j)
    metrics.update(
        {
            "pair_bce": float(sum(losses) / max(1, total)),
            "pair_accuracy": float(sum(accs) / max(1, total)),
            "num_samples": int(total),
            "input_mode": str(input_mode),
        }
    )
    return metrics


def constant_order_from_train(train_data: PairwiseOperatorDataset) -> torch.Tensor:
    priority = 1.0 - train_data.teacher_rank.float() / float(train_data.num_blocks - 1)
    mean_priority = priority.mean(dim=0)
    return torch.argsort(mean_priority, descending=True)


def evaluate_constant(order: torch.Tensor, dataset: Dataset, pair_i: torch.Tensor, pair_j: torch.Tensor, tau: float) -> Dict:
    n = int(order.numel())
    rank_pred = torch.empty(n, dtype=torch.long)
    rank_pred[order.long()] = torch.arange(n, dtype=torch.long)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    losses = []
    accs = []
    logits_all = []
    rank_all = []
    total = 0
    pred = ((rank_pred[pair_i] < rank_pred[pair_j]).float() * 2.0 - 1.0) * 8.0
    for batch in loader:
        rank = batch["teacher_rank"].long()
        target = (rank[:, pair_i] < rank[:, pair_j]).float()
        pred_batch = pred.unsqueeze(0).expand(target.size(0), -1)
        loss = F.binary_cross_entropy_with_logits(pred_batch / max(float(tau), 1e-6), target)
        acc = ((pred_batch > 0.0) == (target > 0.5)).float().mean()
        count = int(rank.size(0))
        losses.append(float(loss.item()) * count)
        accs.append(float(acc.item()) * count)
        logits = torch.zeros((count, n), dtype=torch.float32)
        logits[:, order] = torch.linspace(float(n), 1.0, steps=n)
        logits_all.append(logits)
        rank_all.append(rank)
        total += count
    metrics = order_metrics_from_logits(torch.cat(logits_all, dim=0), torch.cat(rank_all, dim=0), pair_i, pair_j)
    metrics.update(
        {
            "pair_bce": float(sum(losses) / max(1, total)),
            "pair_accuracy": float(sum(accs) / max(1, total)),
            "num_samples": int(total),
            "input_mode": "constant_train_mean_order",
        }
    )
    return metrics


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
    rows = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    train_rows = [r for r in rows if r.get("split") == "train" and r.get("input_mode") == "real"]
    val_rows = [r for r in rows if r.get("split") == "val" and r.get("input_mode") == "real"]
    if not train_rows or not val_rows:
        return
    train_x = [int(r["epoch"]) for r in train_rows]
    val_x = [int(r["epoch"]) for r in val_rows]
    plt.figure(figsize=(7, 4))
    plt.plot(train_x, [float(r["pair_bce"]) for r in train_rows], label="train pair BCE")
    plt.plot(val_x, [float(r["pair_bce"]) for r in val_rows], label="val pair BCE")
    plt.xlabel("epoch")
    plt.ylabel("BCE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "loss_curve.png", dpi=180)
    plt.close()
    plt.figure(figsize=(7, 4))
    plt.plot(val_x, [float(r["kendall_tau_mean"]) for r in val_rows], label="val tau")
    plt.axhline(0.98, color="black", linestyle="--", linewidth=1, label="0.98 gate")
    plt.xlabel("epoch")
    plt.ylabel("Kendall tau to teacher")
    plt.ylim(-0.05, 1.01)
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "tau_curve.png", dpi=180)
    plt.close()


def dataset_summary(dataset: PairwiseOperatorDataset) -> Dict:
    out = {
        "split": dataset.split,
        "num_samples": int(len(dataset)),
        "num_blocks": int(dataset.num_blocks),
        "iter_min": int(dataset.iter.min().item()),
        "iter_max": int(dataset.iter.max().item()),
        "record_min": int(dataset.record_index.min().item()),
        "record_max": int(dataset.record_index.max().item()),
        "selected_reverse_rate": float(dataset.selected_reverse.float().mean().item()),
        "loss_score_gap_mean": float(dataset.loss_score_gap.mean().item()),
        "loss_score_gap_std": float(dataset.loss_score_gap.std(unbiased=False).item()),
    }
    layer_counts = {}
    for layer in sorted({int(value) for value in dataset.layer.tolist()}):
        layer_counts[f"L{layer}"] = int((dataset.layer == int(layer)).sum().item())
    out["layer_counts"] = layer_counts
    head_counts = {}
    for layer, head in all_layer_heads(dataset):
        indices = subset_indices_by_head(dataset, layer, head)
        head_counts[f"L{layer}H{head}"] = int(len(indices))
    out["head_counts"] = head_counts
    return out


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

    val_min_loss_score_gap = (
        float(args.min_loss_score_gap)
        if args.val_min_loss_score_gap is None
        else float(args.val_min_loss_score_gap)
    )
    include_layers = parse_ints(args.include_layers)
    train_data = PairwiseOperatorDataset(
        args.dataset_dir,
        "train",
        min_loss_score_gap=float(args.min_loss_score_gap),
        include_layers=include_layers,
    )
    val_data = PairwiseOperatorDataset(
        args.dataset_dir,
        "val",
        min_loss_score_gap=val_min_loss_score_gap,
        include_layers=include_layers,
    )
    pair_i = train_data.pair_i
    pair_j = train_data.pair_j
    data_mean = float(train_data.attention.float().mean().item())
    data_std = float(train_data.attention.float().std(unbiased=False).item())
    model_config = {
        "num_blocks": int(args.num_blocks),
        "hidden_dims": parse_ints(args.hidden_dims),
        "dropout": float(args.dropout),
        "activation": str(args.activation),
        "input_normalization": str(args.input_normalization),
    }
    model = FlatAttentionOrderMLP(**model_config).to(device)
    if bool(args.compile):
        model = torch.compile(model)
    params = parameter_count(model)
    run_config = {
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        "report_dir": str(args.report_dir),
        "seed": int(args.seed),
        "model_config": model_config,
        "parameter_count": int(params),
        "optimizer": {
            "name": "AdamW",
            "lr": float(args.lr),
            "min_lr": float(args.min_lr),
            "weight_decay": float(args.weight_decay),
            "betas": [float(args.beta1), float(args.beta2)],
            "grad_clip": float(args.grad_clip),
        },
        "loss": {
            "pairwise_bce_tau": float(args.tau),
            "lambda_rank_mse": float(args.lambda_rank_mse),
        },
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "eval_every": int(args.eval_every),
        "amp": bool(args.amp),
        "tau_gate": float(args.tau_gate),
        "filters": {
            "train_min_loss_score_gap": float(args.min_loss_score_gap),
            "val_min_loss_score_gap": float(val_min_loss_score_gap),
            "include_layers": include_layers,
        },
        "train": dataset_summary(train_data),
        "val": dataset_summary(val_data),
        "data_mean": float(data_mean),
        "data_std": float(data_std),
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
    best_val_tau = -float("inf")
    best_val_bce = float("inf")
    best_epoch = -1
    metric_fields = [
        "epoch",
        "split",
        "input_mode",
        "pair_bce",
        "pair_accuracy",
        "order_pair_accuracy",
        "kendall_tau_mean",
        "kendall_tau_std",
        "kendall_tau_min",
        "exact_match_rate",
        "num_samples",
        "epoch_seconds",
        "lr",
    ]
    metrics_path = args.report_dir / "metrics.csv"
    const_order = constant_order_from_train(train_data)
    const_val = evaluate_constant(const_order, val_data, pair_i, pair_j, float(args.tau))
    append_csv(metrics_path, {"epoch": 0, "split": "val", **const_val, "epoch_seconds": 0.0, "lr": float(args.lr)}, metric_fields)

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.perf_counter()
        model.train()
        train_losses = []
        train_accs = []
        train_total = 0
        for batch in train_loader:
            attn = batch["attention"].to(device, non_blocking=bool(args.pin_memory))
            rank = batch["teacher_rank"].to(device, non_blocking=bool(args.pin_memory))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=bool(args.amp and device.type == "cuda")):
                logits = model(attn)
                loss, acc = pair_loss_and_acc(logits, rank, float(args.tau), pair_i_dev, pair_j_dev)
                if float(args.lambda_rank_mse) > 0.0:
                    loss = loss + float(args.lambda_rank_mse) * rank_mse_loss(logits, rank)
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            count = int(attn.size(0))
            train_losses.append(float(loss.item()) * count)
            train_accs.append(float(acc.item()) * count)
            train_total += count
        scheduler.step()
        epoch_seconds = time.perf_counter() - t0
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_row = {
            "epoch": int(epoch),
            "split": "train",
            "input_mode": "real",
            "pair_bce": float(sum(train_losses) / max(1, train_total)),
            "pair_accuracy": float(sum(train_accs) / max(1, train_total)),
            "order_pair_accuracy": float("nan"),
            "kendall_tau_mean": float("nan"),
            "kendall_tau_std": float("nan"),
            "kendall_tau_min": float("nan"),
            "exact_match_rate": float("nan"),
            "num_samples": int(train_total),
            "epoch_seconds": float(epoch_seconds),
            "lr": float(current_lr),
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
                "real",
                data_mean,
                data_std,
                int(args.seed) + epoch,
                int(args.num_workers),
                bool(args.pin_memory),
                bool(args.amp),
            )
            val_row = {
                "epoch": int(epoch),
                "split": "val",
                **val_metrics,
                "epoch_seconds": float(epoch_seconds),
                "lr": float(current_lr),
            }
            append_csv(metrics_path, val_row, metric_fields)
            is_best = (
                float(val_metrics["kendall_tau_mean"]) > best_val_tau
                or (
                    math.isclose(float(val_metrics["kendall_tau_mean"]), best_val_tau)
                    and float(val_metrics["pair_bce"]) < best_val_bce
                )
            )
            if is_best:
                best_val_tau = float(val_metrics["kendall_tau_mean"])
                best_val_bce = float(val_metrics["pair_bce"])
                best_epoch = int(epoch)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": model_config,
                        "policy_type": "pairwise_operator_distilled_mlp",
                        "target_direction": "larger_logit_reveals_earlier",
                        "training_meta": {
                            "best_epoch": int(best_epoch),
                            "best_val_tau": float(best_val_tau),
                            "best_val_pair_bce": float(best_val_bce),
                            "dataset_dir": str(args.dataset_dir),
                            "parameter_count": int(params),
                            "seed": int(args.seed),
                        },
                    },
                    args.out_dir / "best_by_val_tau.pt",
                )
            print(
                f"[operator-mlp] epoch={epoch}/{args.epochs} "
                f"train_bce={train_row['pair_bce']:.6f} train_acc={train_row['pair_accuracy']:.4f} "
                f"val_bce={val_metrics['pair_bce']:.6f} val_tau={val_metrics['kendall_tau_mean']:.4f} "
                f"val_exact={val_metrics['exact_match_rate']:.4f}",
                flush=True,
            )
        else:
            print(
                f"[operator-mlp] epoch={epoch}/{args.epochs} "
                f"train_bce={train_row['pair_bce']:.6f} train_acc={train_row['pair_accuracy']:.4f}",
                flush=True,
            )
        maybe_plot(args.report_dir)

    best_payload = torch.load(args.out_dir / "best_by_val_tau.pt", map_location=device)
    best_model = FlatAttentionOrderMLP(**best_payload["config"]).to(device)
    best_model.load_state_dict(best_payload["model_state_dict"])
    final_eval = {}
    for mode in parse_strings(args.eval_input_modes):
        final_eval[mode] = evaluate(
            best_model,
            val_data,
            pair_i,
            pair_j,
            device,
            int(args.batch_size),
            float(args.tau),
            mode,
            data_mean,
            data_std,
            int(args.seed) + 98765,
            int(args.num_workers),
            bool(args.pin_memory),
            bool(args.amp),
        )
    per_head_rows = []
    per_head_fields = [
        "layer",
        "head",
        "label",
        "num_samples",
        "pair_bce",
        "pair_accuracy",
        "order_pair_accuracy",
        "kendall_tau_mean",
        "kendall_tau_std",
        "kendall_tau_min",
        "exact_match_rate",
    ]
    per_head_path = args.report_dir / "per_head_val_metrics.csv"
    for layer, head in all_layer_heads(val_data):
        subset = Subset(val_data, subset_indices_by_head(val_data, layer, head))
        metrics = evaluate(
            best_model,
            subset,
            pair_i,
            pair_j,
            device,
            int(args.batch_size),
            float(args.tau),
            "real",
            data_mean,
            data_std,
            int(args.seed) + 3333 + int(layer) * 100 + int(head),
            int(args.num_workers),
            bool(args.pin_memory),
            bool(args.amp),
        )
        row = {"layer": int(layer), "head": int(head), "label": f"L{layer}H{head}", **metrics}
        per_head_rows.append(row)
        append_csv(per_head_path, row, per_head_fields)
    per_head_rows.sort(key=lambda item: float(item["kendall_tau_mean"]))
    passed_gate = bool(float(best_val_tau) >= float(args.tau_gate))
    summary = {
        "config": run_config,
        "best_epoch": int(best_epoch),
        "best_val_tau": float(best_val_tau),
        "best_val_pair_bce": float(best_val_bce),
        "passed_tau_gate": bool(passed_gate),
        "constant_val": const_val,
        "final_eval": final_eval,
        "per_head_val_metrics": per_head_rows,
        "checkpoint": str(args.out_dir / "best_by_val_tau.pt"),
    }
    write_json(args.out_dir / "training_summary.json", summary)
    write_json(args.report_dir / "training_summary.json", summary)
    maybe_plot(args.report_dir)

    report_label = args.report_dir.name or "operator_mlp_distillation"
    lines = [
        f"# {report_label}: Large Operator MLP Distillation",
        "",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Checkpoint: `{args.out_dir / 'best_by_val_tau.pt'}`",
        f"- Parameters: `{params}`",
        f"- Best epoch: `{best_epoch}`",
        f"- Best val tau: `{best_val_tau:.6f}`",
        f"- Best val pair BCE: `{best_val_bce:.6f}`",
        f"- Tau gate `{float(args.tau_gate):.3f}` passed: `{passed_gate}`",
        "",
        "## Val Input Ablations",
        "",
        "| input | pair acc | tau | pair BCE | exact | samples |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| constant train mean order | {const_val['pair_accuracy']:.4f} | "
            f"{const_val['kendall_tau_mean']:.4f} | {const_val['pair_bce']:.6f} | "
            f"{const_val['exact_match_rate']:.4f} | {const_val['num_samples']} |"
        ),
    ]
    for mode in parse_strings(args.eval_input_modes):
        item = final_eval[mode]
        lines.append(
            f"| {mode} | {item['pair_accuracy']:.4f} | {item['kendall_tau_mean']:.4f} | "
            f"{item['pair_bce']:.6f} | {item['exact_match_rate']:.4f} | {item['num_samples']} |"
        )
    lines.extend(["", "## Worst Per-Head Val Tau", "", "| head | tau | pair acc | samples |", "|---|---:|---:|---:|"])
    for item in per_head_rows[:8]:
        lines.append(
            f"| {item['label']} | {float(item['kendall_tau_mean']):.4f} | "
            f"{float(item['pair_accuracy']):.4f} | {int(item['num_samples'])} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics.csv`: train/val BCE and tau by epoch.",
            "- `loss_curve.png`: train/val pairwise BCE.",
            "- `tau_curve.png`: val Kendall tau to teacher order.",
            "- `per_head_val_metrics.csv`: per-head held-out tau.",
        ]
    )
    (args.report_dir / "training_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "best_val_tau": best_val_tau, "passed_tau_gate": passed_gate}, indent=2))


if __name__ == "__main__":
    main()
