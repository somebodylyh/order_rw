#!/usr/bin/env python3
"""Train Attn-MLP students on fixed direct-asym-eig teacher ranks."""

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
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attn_mlp_order_policy import FlatAttentionOrderMLP  # noqa: E402


class DistillTensorDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str):
        payloads = [torch.load(path, map_location="cpu") for path in sorted((dataset_dir / split).glob("shard_*.pt"))]
        if not payloads:
            raise FileNotFoundError(f"No shards found for split={split} under {dataset_dir}")
        self.attention = torch.cat([payload["attention"] for payload in payloads], dim=0).float()
        self.teacher_rank = torch.cat([payload["teacher_rank"] for payload in payloads], dim=0).long()
        self.teacher_order = torch.cat([payload["teacher_order"] for payload in payloads], dim=0).long()
        self.teacher_gap = torch.cat([payload["teacher_gap"] for payload in payloads], dim=0).float()
        self.chosen_loss = torch.cat([payload["chosen_loss"] for payload in payloads], dim=0).float()
        self.raw_loss = torch.cat([payload["raw_loss"] for payload in payloads], dim=0).float()
        self.reverse_loss = torch.cat([payload["reverse_loss"] for payload in payloads], dim=0).float()
        self.selected_reverse = torch.cat([payload["selected_reverse"] for payload in payloads], dim=0).bool()

    def __len__(self) -> int:
        return int(self.attention.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "attention": self.attention[idx],
            "teacher_rank": self.teacher_rank[idx],
            "teacher_order": self.teacher_order[idx],
            "teacher_gap": self.teacher_gap[idx],
            "chosen_loss": self.chosen_loss[idx],
            "raw_loss": self.raw_loss[idx],
            "reverse_loss": self.reverse_loss[idx],
            "selected_reverse": self.selected_reverse[idx],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Attn-MLP distillation baselines.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, default=Path("Report/MLP_distillation/try_1"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--modes", type=str, default="normal,zero,shuffled")
    parser.add_argument("--seeds", type=str, default="1337,2026,2027")
    parser.add_argument("--num_blocks", type=int, default=64)
    parser.add_argument("--hidden_dims", type=str, default="1024,1024")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--input_normalization", type=str, default="none")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_str_list(raw: str) -> List[str]:
    return [str(item.strip()) for item in str(raw).split(",") if item.strip()]


def hidden_dims(raw: str) -> List[int]:
    return parse_int_list(raw)


def pair_indices(num_blocks: int, device: torch.device):
    return torch.triu_indices(int(num_blocks), int(num_blocks), offset=1, device=device)


def pair_loss_and_acc(logits: torch.Tensor, rank: torch.Tensor, tau: float, pair_i: torch.Tensor, pair_j: torch.Tensor):
    pred = (logits[:, pair_i] - logits[:, pair_j]) / max(float(tau), 1e-6)
    target = (rank[:, pair_i] < rank[:, pair_j]).float()
    loss = F.binary_cross_entropy_with_logits(pred, target)
    acc = ((pred.detach() > 0.0) == (target > 0.5)).float().mean()
    return loss, acc


def order_metrics(logits: torch.Tensor, teacher_order: torch.Tensor, teacher_rank: torch.Tensor):
    pred_orders = torch.argsort(logits.detach().float(), dim=1, descending=True).cpu()
    teacher_order = teacher_order.cpu()
    teacher_rank = teacher_rank.cpu()
    pair_i, pair_j = torch.triu_indices(logits.size(1), logits.size(1), offset=1)
    taus = []
    distances = []
    exact = []
    orient = []
    for idx in range(pred_orders.size(0)):
        pred_rank = torch.empty_like(pred_orders[idx])
        pred_rank[pred_orders[idx]] = torch.arange(pred_orders.size(1), dtype=pred_rank.dtype)
        target = teacher_rank[idx]
        pair_acc = ((pred_rank[pair_i] < pred_rank[pair_j]) == (target[pair_i] < target[pair_j])).float().mean().item()
        tau = 2.0 * float(pair_acc) - 1.0
        taus.append(tau)
        distances.append((1.0 - tau) / 2.0)
        exact.append(float(torch.equal(pred_orders[idx], teacher_order[idx])))
        reverse_order = torch.flip(teacher_order[idx], dims=[0])
        reverse_rank = torch.empty_like(reverse_order)
        reverse_rank[reverse_order] = torch.arange(reverse_order.numel(), dtype=reverse_rank.dtype)
        reverse_pair_acc = (
            (pred_rank[pair_i] < pred_rank[pair_j]) == (reverse_rank[pair_i] < reverse_rank[pair_j])
        ).float().mean().item()
        orient.append(float(pair_acc >= reverse_pair_acc))
    return {
        "kendall_tau_mean": float(np.mean(taus)),
        "kendall_tau_std": float(np.std(taus)),
        "kendall_distance_mean": float(np.mean(distances)),
        "exact_match_rate": float(np.mean(exact)),
        "orientation_agreement": float(np.mean(orient)),
    }


def transform_attention(attn: torch.Tensor, mode: str, generator: torch.Generator | None = None) -> torch.Tensor:
    mode = str(mode)
    if mode == "normal":
        return attn
    if mode == "zero":
        return torch.zeros_like(attn)
    if mode == "shuffled":
        rows = []
        n = int(attn.size(-1))
        for item in attn:
            perm = torch.randperm(n, generator=generator, device=attn.device)
            rows.append(item.index_select(0, perm).index_select(1, perm))
        return torch.stack(rows, dim=0)
    raise ValueError(f"Unsupported mode={mode!r}")


@torch.no_grad()
def evaluate_model(model, loader, mode: str, device: torch.device, tau: float, seed: int):
    model.eval()
    pair_i, pair_j = pair_indices(model.num_blocks, device)
    losses = []
    accs = []
    logits_all = []
    order_all = []
    rank_all = []
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 991)
    for batch in loader:
        attn = batch["attention"].to(device)
        rank = batch["teacher_rank"].to(device)
        attn = transform_attention(attn, mode, gen)
        logits = model(attn)
        loss, acc = pair_loss_and_acc(logits, rank, tau, pair_i, pair_j)
        losses.append(float(loss.item()) * int(attn.size(0)))
        accs.append(float(acc.item()) * int(attn.size(0)))
        logits_all.append(logits.detach().cpu())
        order_all.append(batch["teacher_order"].cpu())
        rank_all.append(batch["teacher_rank"].cpu())
    total = sum(int(x.size(0)) for x in logits_all)
    logits_cat = torch.cat(logits_all, dim=0)
    orders_cat = torch.cat(order_all, dim=0)
    ranks_cat = torch.cat(rank_all, dim=0)
    metrics = order_metrics(logits_cat, orders_cat, ranks_cat)
    metrics.update(
        {
            "pair_bce": float(sum(losses) / max(1, total)),
            "pair_accuracy": float(sum(accs) / max(1, total)),
            "num_samples": int(total),
        }
    )
    return metrics


def constant_priority_order(train_data: DistillTensorDataset) -> torch.Tensor:
    priority = 1.0 - train_data.teacher_rank.float() / float(train_data.teacher_rank.size(1) - 1)
    mean_priority = priority.mean(dim=0)
    return torch.argsort(mean_priority, descending=True)


def evaluate_constant(order: torch.Tensor, dataset: DistillTensorDataset, tau: float):
    n = int(dataset.teacher_rank.size(1))
    rank_pred = torch.empty(n, dtype=torch.long)
    rank_pred[order.long()] = torch.arange(n, dtype=torch.long)
    pair_i, pair_j = torch.triu_indices(n, n, offset=1)
    pred = ((rank_pred[pair_i] < rank_pred[pair_j]).float() * 2.0 - 1.0) * 8.0
    targets = (dataset.teacher_rank[:, pair_i] < dataset.teacher_rank[:, pair_j]).float()
    pred_batch = pred.unsqueeze(0).expand(targets.size(0), -1)
    loss = F.binary_cross_entropy_with_logits(pred_batch / max(float(tau), 1e-6), targets)
    acc = ((pred_batch > 0.0) == (targets > 0.5)).float().mean().item()
    logits = torch.zeros((len(dataset), n), dtype=torch.float32)
    logits[:, order] = torch.linspace(float(n), 1.0, steps=n)
    metrics = order_metrics(logits, dataset.teacher_order, dataset.teacher_rank)
    metrics.update({"pair_bce": float(loss.item()), "pair_accuracy": float(acc), "num_samples": len(dataset)})
    return metrics


def write_command(report_dir: Path, filename: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(part) for part in sys.argv)
    (report_dir / filename).write_text(command + "\n", encoding="utf-8")


def load_dataset_manifest(dataset_dir: Path) -> Dict:
    path = dataset_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def teacher_label_from_manifest(manifest: Dict) -> str:
    teacher = manifest.get("teacher", {}) if isinstance(manifest, dict) else {}
    if teacher.get("label"):
        return str(teacher["label"])
    if "layer" in teacher and "head" in teacher:
        return f"L{int(teacher['layer'])}H{int(teacher['head'])}"
    return str(teacher.get("head", "fixed_head"))


def teacher_policy_type(manifest: Dict) -> str:
    label = teacher_label_from_manifest(manifest).lower()
    label = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_") or "fixed_head"
    return f"{label}_direct_asym_eig_teacher_distillation"


def append_csv(path: Path, row: Dict, fieldnames: Sequence[str]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_command(args.report_dir, "train_command.sh")
    device = torch.device(args.device)
    dataset_manifest = load_dataset_manifest(args.dataset_dir)
    policy_type = teacher_policy_type(dataset_manifest)
    train_data = DistillTensorDataset(args.dataset_dir, "train")
    val_data = DistillTensorDataset(args.dataset_dir, "val_probe")
    test_data = DistillTensorDataset(args.dataset_dir, "test_probe")
    val_loader = DataLoader(val_data, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    test_loader = DataLoader(test_data, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    config = {
        "num_blocks": int(args.num_blocks),
        "hidden_dims": hidden_dims(args.hidden_dims),
        "dropout": float(args.dropout),
        "activation": str(args.activation),
        "input_normalization": str(args.input_normalization),
    }
    run_config = {
        "dataset_dir": str(args.dataset_dir),
        "out_dir": str(args.out_dir),
        "model_config": config,
        "optimizer": {
            "name": "AdamW",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "betas": [float(args.beta1), float(args.beta2)],
            "grad_clip": float(args.grad_clip),
        },
        "loss": {
            "name": "pairwise_bce",
            "tau": float(args.tau),
            "extra_research_losses": [],
        },
        "seeds": parse_int_list(args.seeds),
        "modes": parse_str_list(args.modes),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "dataset_manifest": dataset_manifest,
    }
    (args.report_dir / "config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_rows: List[Dict] = []
    fields = [
        "mode",
        "seed",
        "epoch",
        "split",
        "pair_bce",
        "pair_accuracy",
        "kendall_tau_mean",
        "kendall_tau_std",
        "kendall_distance_mean",
        "exact_match_rate",
        "orientation_agreement",
        "num_samples",
        "epoch_seconds",
    ]

    const_order = constant_priority_order(train_data)
    for split_name, dataset in (("val_probe", val_data), ("test_probe", test_data)):
        row = {"mode": "constant", "seed": -1, "epoch": 0, "split": split_name, **evaluate_constant(const_order, dataset, args.tau)}
        metric_rows.append(row)
        append_csv(args.report_dir / "metrics.csv", row, fields)

    train_loader_base = lambda seed: DataLoader(
        train_data,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        generator=torch.Generator().manual_seed(int(seed)),
    )
    pair_i, pair_j = pair_indices(int(args.num_blocks), device)
    for mode in parse_str_list(args.modes):
        for seed in parse_int_list(args.seeds):
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            model = FlatAttentionOrderMLP(**config).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(args.lr),
                betas=(float(args.beta1), float(args.beta2)),
                weight_decay=float(args.weight_decay),
            )
            best_loss = float("inf")
            best_epoch = -1
            run_dir = args.out_dir / f"{mode}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            for epoch in range(1, int(args.epochs) + 1):
                t0 = time.perf_counter()
                model.train()
                gen = torch.Generator(device=device)
                gen.manual_seed(int(seed) + epoch * 7919)
                for batch in train_loader_base(seed + epoch):
                    attn = batch["attention"].to(device)
                    rank = batch["teacher_rank"].to(device)
                    attn = transform_attention(attn, mode, gen)
                    logits = model(attn)
                    loss, _ = pair_loss_and_acc(logits, rank, args.tau, pair_i, pair_j)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if float(args.grad_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                    optimizer.step()
                epoch_seconds = time.perf_counter() - t0
                val_metrics = evaluate_model(model, val_loader, mode, device, args.tau, seed + epoch)
                row = {
                    "mode": mode,
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "split": "val_probe",
                    **val_metrics,
                    "epoch_seconds": float(epoch_seconds),
                }
                metric_rows.append(row)
                append_csv(args.report_dir / "metrics.csv", row, fields)
                if float(val_metrics["pair_bce"]) < best_loss:
                    best_loss = float(val_metrics["pair_bce"])
                    best_epoch = int(epoch)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "config": config,
                            "policy_type": policy_type,
                            "target_direction": "larger_logit_reveals_earlier",
                            "training_meta": {
                                "mode": mode,
                                "seed": int(seed),
                                "best_epoch": int(best_epoch),
                                "best_val_pair_loss": float(best_loss),
                                "dataset_dir": str(args.dataset_dir),
                                "dataset_manifest": dataset_manifest,
                            },
                        },
                        run_dir / "best_by_val_pair_loss.pt",
                    )
                print(
                    f"[distill-train] mode={mode} seed={seed} epoch={epoch}/{args.epochs} "
                    f"val_bce={val_metrics['pair_bce']:.6f} val_acc={val_metrics['pair_accuracy']:.4f}",
                    flush=True,
                )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "policy_type": policy_type,
                    "target_direction": "larger_logit_reveals_earlier",
                    "training_meta": {
                        "mode": mode,
                        "seed": int(seed),
                        "last_epoch": int(args.epochs),
                        "best_epoch": int(best_epoch),
                        "best_val_pair_loss": float(best_loss),
                        "dataset_dir": str(args.dataset_dir),
                        "dataset_manifest": dataset_manifest,
                    },
                },
                run_dir / "last.pt",
            )
            best_payload = torch.load(run_dir / "best_by_val_pair_loss.pt", map_location=device)
            best_model = FlatAttentionOrderMLP(**best_payload["config"]).to(device)
            best_model.load_state_dict(best_payload["model_state_dict"])
            for split_name, loader in (("val_probe", val_loader), ("test_probe", test_loader)):
                metrics = evaluate_model(best_model, loader, mode, device, args.tau, seed + 4242)
                row = {"mode": mode, "seed": int(seed), "epoch": int(best_epoch), "split": split_name, **metrics}
                metric_rows.append(row)
                append_csv(args.report_dir / "metrics.csv", row, fields)
    summary = {
        "config": run_config,
        "rows": metric_rows,
        "constant_order": [int(v) for v in const_order.tolist()],
    }
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "num_metric_rows": len(metric_rows)}, indent=2))


if __name__ == "__main__":
    main()
