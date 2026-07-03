#!/usr/bin/env python3
"""Train a FlatAttentionOrderMLP on all-head pairwise direct-asym-eig targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
import time
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


class PairwiseAttentionDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str):
        paths = sorted((dataset_dir / split).glob("shard_*.pt"))
        if not paths:
            raise FileNotFoundError(f"No shards for split={split} under {dataset_dir}")
        payloads = [torch.load(path, map_location="cpu") for path in paths]
        self.attention = torch.cat([p["attention"] for p in payloads], dim=0).float()
        self.teacher_rank = torch.cat([p["teacher_rank"] for p in payloads], dim=0).long()
        self.teacher_order = torch.cat([p["teacher_order"] for p in payloads], dim=0).long()
        self.layer = torch.cat([p["layer"] for p in payloads], dim=0).long()
        self.head = torch.cat([p["head"] for p in payloads], dim=0).long()
        self.iter = torch.cat([p["iter"] for p in payloads], dim=0).long()
        self.record_index = torch.cat([p["record_index"] for p in payloads], dim=0).long()
        self.eigval_abs = torch.cat([p["eigval_abs"] for p in payloads], dim=0).float()
        self.vector_std = torch.cat([p["vector_std"] for p in payloads], dim=0).float()
        self.pair_i = payloads[0]["pair_i"].long()
        self.pair_j = payloads[0]["pair_j"].long()
        self.num_blocks = int(self.attention.shape[-1])

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
            "eigval_abs": self.eigval_abs[idx],
            "vector_std": self.vector_std[idx],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all-head pairwise Attn-MLP.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num_blocks", type=int, default=64)
    parser.add_argument("--hidden_dims", type=str, default="1024,1024")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--input_normalization", type=str, default="none")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--strong_top_k_heads", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def parse_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def write_command(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(part) for part in sys.argv)
    (report_dir / "train_pairwise_mlp_command.sh").write_text(command + "\n", encoding="utf-8")


def pair_loss_and_acc(
    logits: torch.Tensor,
    rank: torch.Tensor,
    tau: float,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = (logits[:, pair_i] - logits[:, pair_j]) / max(float(tau), 1e-6)
    target = (rank[:, pair_i] < rank[:, pair_j]).float()
    loss = F.binary_cross_entropy_with_logits(pred, target)
    acc = ((pred.detach() > 0.0) == (target > 0.5)).float().mean()
    return loss, acc


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
        return torch.randn(
            attn.shape,
            generator=generator,
            device=attn.device,
            dtype=attn.dtype,
        ) * float(data_std) + float(data_mean)
    if mode == "gaussian_std1":
        return torch.randn(attn.shape, generator=generator, device=attn.device, dtype=attn.dtype)
    if mode == "shuffled":
        rows = []
        n = int(attn.size(-1))
        for item in attn:
            perm = torch.randperm(n, generator=generator, device=attn.device)
            rows.append(item.index_select(0, perm).index_select(1, perm))
        return torch.stack(rows, dim=0)
    raise ValueError(f"Unsupported eval input mode={mode!r}")


def order_metrics_from_logits(logits: torch.Tensor, teacher_rank: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor) -> Dict:
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
        "exact_match_rate": float(np.mean(exact)) if exact else float("nan"),
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
    input_mode: str,
    data_mean: float,
    data_std: float,
    seed: int,
) -> Dict:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    model.eval()
    losses = []
    accs = []
    logits_all = []
    rank_all = []
    total = 0
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    pair_i_dev = pair_i.to(device)
    pair_j_dev = pair_j.to(device)
    for batch in loader:
        attn = batch["attention"].to(device)
        rank = batch["teacher_rank"].to(device)
        attn = transform_attention(attn, input_mode, gen, data_mean, data_std)
        logits = model(attn)
        loss, acc = pair_loss_and_acc(logits, rank, tau, pair_i_dev, pair_j_dev)
        count = int(attn.size(0))
        losses.append(float(loss.item()) * count)
        accs.append(float(acc.item()) * count)
        logits_all.append(logits.cpu())
        rank_all.append(rank.cpu())
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


def constant_order_from_train(train_data: PairwiseAttentionDataset) -> torch.Tensor:
    priority = 1.0 - train_data.teacher_rank.float() / float(train_data.num_blocks - 1)
    mean_priority = priority.mean(dim=0)
    return torch.argsort(mean_priority, descending=True)


def evaluate_constant(order: torch.Tensor, dataset: Dataset, pair_i: torch.Tensor, pair_j: torch.Tensor, tau: float) -> Dict:
    n = int(order.numel())
    rank_pred = torch.empty(n, dtype=torch.long)
    rank_pred[order.long()] = torch.arange(n, dtype=torch.long)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
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


def append_csv(path: Path, row: Dict, fields: Sequence[str]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def head_id(layer: int, head: int) -> str:
    return f"L{int(layer)}H{int(head)}"


def top_heads_by_train_signal(train_data: PairwiseAttentionDataset, signal: str, top_k: int) -> List[Dict]:
    values = train_data.vector_std if str(signal) == "vector_std" else train_data.eigval_abs
    rows = []
    for layer in sorted(set(int(v) for v in train_data.layer.tolist())):
        for head in sorted(set(int(v) for v in train_data.head.tolist())):
            mask = (train_data.layer == int(layer)) & (train_data.head == int(head))
            if bool(mask.any()):
                rows.append(
                    {
                        "layer": int(layer),
                        "head": int(head),
                        "label": head_id(layer, head),
                        "mean": float(values[mask].mean().item()),
                        "count": int(mask.sum().item()),
                    }
                )
    rows.sort(key=lambda item: item["mean"], reverse=True)
    return rows[: max(1, int(top_k))]


def subset_for_heads(dataset: PairwiseAttentionDataset, heads: Sequence[Dict]) -> Subset:
    selected = {(int(item["layer"]), int(item["head"])) for item in heads}
    indices = [
        idx
        for idx in range(len(dataset))
        if (int(dataset.layer[idx].item()), int(dataset.head[idx].item())) in selected
    ]
    return Subset(dataset, indices)


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

    train_data = PairwiseAttentionDataset(args.dataset_dir, "train")
    val_data = PairwiseAttentionDataset(args.dataset_dir, "val")
    pair_i = train_data.pair_i
    pair_j = train_data.pair_j
    data_mean = float(train_data.attention.mean().item())
    data_std = float(train_data.attention.std(unbiased=False).item())
    config = {
        "num_blocks": int(args.num_blocks),
        "hidden_dims": parse_ints(args.hidden_dims),
        "dropout": float(args.dropout),
        "activation": str(args.activation),
        "input_normalization": str(args.input_normalization),
    }
    run_config = {
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        "report_dir": str(args.report_dir),
        "seed": int(args.seed),
        "model_config": config,
        "optimizer": {
            "name": "AdamW",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "betas": [float(args.beta1), float(args.beta2)],
            "grad_clip": float(args.grad_clip),
        },
        "loss": {"name": "pairwise_bce", "tau": float(args.tau)},
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "train_samples": int(len(train_data)),
        "val_samples": int(len(val_data)),
        "data_mean": float(data_mean),
        "data_std": float(data_std),
    }
    (args.report_dir / "config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    train_loader = DataLoader(
        train_data,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    model = FlatAttentionOrderMLP(**config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(float(args.beta1), float(args.beta2)),
        weight_decay=float(args.weight_decay),
    )
    pair_i_dev = pair_i.to(device)
    pair_j_dev = pair_j.to(device)
    best_val_bce = float("inf")
    best_epoch = -1
    metric_rows: List[Dict] = []
    fields = [
        "epoch",
        "split",
        "input_mode",
        "pair_bce",
        "pair_accuracy",
        "order_pair_accuracy",
        "kendall_tau_mean",
        "kendall_tau_std",
        "exact_match_rate",
        "num_samples",
        "epoch_seconds",
    ]

    const_order = constant_order_from_train(train_data)
    const_val = evaluate_constant(const_order, val_data, pair_i, pair_j, float(args.tau))
    const_row = {"epoch": 0, "split": "val", **const_val, "epoch_seconds": 0.0}
    metric_rows.append(const_row)
    append_csv(args.report_dir / "metrics.csv", const_row, fields)

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.perf_counter()
        model.train()
        train_losses = []
        train_accs = []
        train_total = 0
        for batch in train_loader:
            attn = batch["attention"].to(device)
            rank = batch["teacher_rank"].to(device)
            logits = model(attn)
            loss, acc = pair_loss_and_acc(logits, rank, float(args.tau), pair_i_dev, pair_j_dev)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            count = int(attn.size(0))
            train_losses.append(float(loss.item()) * count)
            train_accs.append(float(acc.item()) * count)
            train_total += count
        epoch_seconds = time.perf_counter() - t0
        train_row = {
            "epoch": int(epoch),
            "split": "train",
            "input_mode": "real",
            "pair_bce": float(sum(train_losses) / max(1, train_total)),
            "pair_accuracy": float(sum(train_accs) / max(1, train_total)),
            "order_pair_accuracy": float("nan"),
            "kendall_tau_mean": float("nan"),
            "kendall_tau_std": float("nan"),
            "exact_match_rate": float("nan"),
            "num_samples": int(train_total),
            "epoch_seconds": float(epoch_seconds),
        }
        metric_rows.append(train_row)
        append_csv(args.report_dir / "metrics.csv", train_row, fields)
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
        )
        val_row = {"epoch": int(epoch), "split": "val", **val_metrics, "epoch_seconds": float(epoch_seconds)}
        metric_rows.append(val_row)
        append_csv(args.report_dir / "metrics.csv", val_row, fields)
        if float(val_metrics["pair_bce"]) < best_val_bce:
            best_val_bce = float(val_metrics["pair_bce"])
            best_epoch = int(epoch)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "training_meta": {
                        "best_epoch": int(best_epoch),
                        "best_val_pair_bce": float(best_val_bce),
                        "dataset_dir": str(args.dataset_dir),
                        "seed": int(args.seed),
                    },
                },
                args.out_dir / "best_by_val_pair_bce.pt",
            )
        print(
            f"[pairwise-mlp] epoch={epoch}/{args.epochs} "
            f"train_bce={train_row['pair_bce']:.6f} train_acc={train_row['pair_accuracy']:.4f} "
            f"val_bce={val_metrics['pair_bce']:.6f} val_acc={val_metrics['pair_accuracy']:.4f} "
            f"val_tau={val_metrics['kendall_tau_mean']:.4f}",
            flush=True,
        )

    best_payload = torch.load(args.out_dir / "best_by_val_pair_bce.pt", map_location=device)
    best_model = FlatAttentionOrderMLP(**best_payload["config"]).to(device)
    best_model.load_state_dict(best_payload["model_state_dict"])
    eval_modes = ["real", "zero", "gaussian_like", "gaussian_std1", "shuffled"]
    final_eval = {}
    for mode in eval_modes:
        final_eval[f"val_{mode}"] = evaluate(
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
        )
    strong_heads = {
        "top_vector_std": top_heads_by_train_signal(train_data, "vector_std", int(args.strong_top_k_heads)),
        "top_eigval_abs": top_heads_by_train_signal(train_data, "eigval_abs", int(args.strong_top_k_heads)),
    }
    strong_eval = {}
    for name, heads in strong_heads.items():
        subset = subset_for_heads(val_data, heads)
        strong_eval[name] = {
            "heads": heads,
            "val_real": evaluate(
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
                int(args.seed) + 2222,
            ),
        }

    summary = {
        "config": run_config,
        "best_epoch": int(best_epoch),
        "best_val_pair_bce": float(best_val_bce),
        "constant_val": const_val,
        "final_eval": final_eval,
        "strong_heads": strong_eval,
        "constant_order": [int(v) for v in const_order.tolist()],
        "metric_rows": metric_rows,
    }
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (args.report_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Try 4 Pairwise MLP Training Result",
        "",
        f"Best epoch: `{best_epoch}`",
        f"Best val pair BCE: `{best_val_bce:.6f}`",
        "",
        "## Val Input Ablations",
        "",
        "| input | pair acc | tau to eig order | pair BCE | samples |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.append(
        f"| constant train mean order | {const_val['pair_accuracy']:.4f} | {const_val['kendall_tau_mean']:.4f} | "
        f"{const_val['pair_bce']:.6f} | {const_val['num_samples']} |"
    )
    for mode in eval_modes:
        item = final_eval[f"val_{mode}"]
        lines.append(
            f"| {mode} | {item['pair_accuracy']:.4f} | {item['kendall_tau_mean']:.4f} | "
            f"{item['pair_bce']:.6f} | {item['num_samples']} |"
        )
    lines.extend(["", "## Strong-Signal Heads", ""])
    for name, item in strong_eval.items():
        head_labels = ", ".join(head["label"] for head in item["heads"])
        metrics = item["val_real"]
        lines.append(f"- `{name}` heads: {head_labels}")
        lines.append(
            f"  val real: pair acc `{metrics['pair_accuracy']:.4f}`, "
            f"tau `{metrics['kendall_tau_mean']:.4f}`, samples `{metrics['num_samples']}`"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "The teacher target is raw direct-asym-eig order from the try4 pairwise dataset. It is not raw/reverse oriented by current-model loss.",
        ]
    )
    (args.report_dir / "training_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "best_val_pair_bce": best_val_bce, "report": str(args.report_dir)}, indent=2))


if __name__ == "__main__":
    main()
