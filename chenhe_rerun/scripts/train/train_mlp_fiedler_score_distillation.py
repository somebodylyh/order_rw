#!/usr/bin/env python3
"""Train an Attn-MLP to regress oriented Fiedler priority scores."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill the pairwise-max Laplacian Fiedler score operator into an Attn-MLP."
    )
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, required=True)
    parser.add_argument("--target_cache_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--num_blocks", type=int, default=64)
    parser.add_argument("--hidden_dims", type=str, default="2048,1024")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--input_normalization", type=str, default="zscore")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_corr_loss", type=float, default=0.0)
    parser.add_argument("--lambda_pairwise_loss", type=float, default=0.0)
    parser.add_argument("--pairwise_logit_scale", type=float, default=12.0)
    parser.add_argument(
        "--orientation",
        type=str,
        default="teacher_rank",
        choices=("teacher_rank", "selected_reverse", "raw", "reverse"),
        help=(
            "How to choose the arbitrary eigenvector sign. teacher_rank uses only the saved "
            "loss-selected teacher direction; the regression target remains the continuous score."
        ),
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--min_loss_score_gap", type=float, default=0.0)
    parser.add_argument("--val_min_loss_score_gap", type=float, default=None)
    parser.add_argument("--include_layers", type=str, default="")
    parser.add_argument("--score_mse_gate", type=float, default=0.01)
    parser.add_argument("--target_tau_gate", type=float, default=0.98)
    parser.add_argument(
        "--eval_input_modes",
        type=str,
        default="real,zero,gaussian_like,gaussian_std1,shuffled",
    )
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


def fiedler_vector_from_attention(matrix: np.ndarray) -> tuple[np.ndarray, Dict[str, float]]:
    values = np.asarray(matrix, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Expected square attention matrix, got shape={tuple(values.shape)}.")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(values, 0.0)
    affinity = np.maximum(values, values.T)
    affinity = np.nan_to_num(affinity, nan=0.0, posinf=0.0, neginf=0.0)
    affinity = np.maximum(affinity, 0.0)
    np.fill_diagonal(affinity, 0.0)
    degree = affinity.sum(axis=1)
    laplacian = np.diag(degree) - affinity
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    if eigvals.size <= 1:
        raise ValueError("Laplacian produced no nontrivial eigenvector.")
    vector = np.asarray(eigvecs[:, 1], dtype=np.float64)
    if (not np.isfinite(vector).all()) or float(np.std(vector)) <= 1e-12:
        raise ValueError("Laplacian Fiedler vector is degenerate or non-finite.")
    spectral_gap = float(eigvals[2] - eigvals[1]) if eigvals.size > 2 else float(abs(eigvals[1]))
    meta = {
        "laplacian_eigval0": float(eigvals[0]),
        "laplacian_fiedler_eigval": float(eigvals[1]),
        "laplacian_next_eigval": float(eigvals[2]) if eigvals.size > 2 else float("nan"),
        "spectral_gap": float(spectral_gap),
        "affinity_sum": float(affinity.sum()),
        "affinity_density": float(np.mean(affinity > 0.0)),
        "vector_std": float(np.std(vector)),
    }
    return vector, meta


def minmax(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    denom = max(hi - lo, float(eps))
    return (arr - lo) / denom


def priority_from_vector(
    vector: np.ndarray,
    orientation: str,
    teacher_rank: np.ndarray,
    selected_reverse: bool,
) -> tuple[np.ndarray, str]:
    # Larger priority means the block should appear earlier in argsort_desc.
    raw_priority = 1.0 - minmax(vector)
    reverse_priority = minmax(vector)
    orientation = str(orientation)
    if orientation == "raw":
        return raw_priority.astype(np.float32), "raw"
    if orientation == "reverse":
        return reverse_priority.astype(np.float32), "reverse"
    if orientation == "selected_reverse":
        chosen = reverse_priority if bool(selected_reverse) else raw_priority
        return chosen.astype(np.float32), "selected_reverse_reverse" if bool(selected_reverse) else "selected_reverse_raw"
    if orientation == "teacher_rank":
        teacher_rank = np.asarray(teacher_rank, dtype=np.int64)
        teacher_priority = 1.0 - teacher_rank.astype(np.float64) / float(max(1, teacher_rank.size - 1))
        raw_corr = pearson_np(raw_priority, teacher_priority)
        reverse_corr = pearson_np(reverse_priority, teacher_priority)
        if reverse_corr > raw_corr:
            return reverse_priority.astype(np.float32), "teacher_rank_reverse"
        return raw_priority.astype(np.float32), "teacher_rank_raw"
    raise ValueError(f"Unsupported orientation={orientation!r}.")


def pearson_np(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - float(left.mean())
    right = right - float(right.mean())
    denom = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    if denom <= eps:
        return 0.0
    return float(np.sum(left * right) / denom)


def pair_accuracy_to_rank(priority: torch.Tensor, rank: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor) -> torch.Tensor:
    pred = priority[:, pair_i] > priority[:, pair_j]
    target = rank[:, pair_i] < rank[:, pair_j]
    return (pred == target).float().mean(dim=1)


def pair_accuracy_to_priority(
    pred_priority: torch.Tensor,
    target_priority: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> torch.Tensor:
    pred = pred_priority[:, pair_i] > pred_priority[:, pair_j]
    target = target_priority[:, pair_i] > target_priority[:, pair_j]
    return (pred == target).float().mean(dim=1)


def batch_pearson(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.float() - pred.float().mean(dim=1, keepdim=True)
    target = target.float() - target.float().mean(dim=1, keepdim=True)
    denom = pred.norm(dim=1) * target.norm(dim=1)
    return (pred * target).sum(dim=1) / denom.clamp_min(float(eps))


def score_losses(
    logits: torch.Tensor,
    target_score: torch.Tensor,
    lambda_corr_loss: float,
    pair_i: torch.Tensor | None = None,
    pair_j: torch.Tensor | None = None,
    lambda_pairwise_loss: float = 0.0,
    pairwise_logit_scale: float = 12.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pred_score = torch.sigmoid(logits.float())
    mse = F.mse_loss(pred_score, target_score.float())
    mae = F.l1_loss(pred_score, target_score.float())
    pearson = batch_pearson(pred_score, target_score.float()).mean()
    loss = mse
    pairwise_bce = logits.new_tensor(float("nan"))
    if float(lambda_corr_loss) > 0.0:
        loss = loss + float(lambda_corr_loss) * (1.0 - pearson)
    if float(lambda_pairwise_loss) > 0.0:
        if pair_i is None or pair_j is None:
            raise ValueError("pair_i/pair_j are required when lambda_pairwise_loss > 0.")
        pred_diff = pred_score[:, pair_i] - pred_score[:, pair_j]
        target_pref = target_score[:, pair_i] > target_score[:, pair_j]
        pairwise_bce = F.binary_cross_entropy_with_logits(
            float(pairwise_logit_scale) * pred_diff.float(),
            target_pref.float(),
        )
        loss = loss + float(lambda_pairwise_loss) * pairwise_bce
    return loss, {
        "score_mse": mse.detach(),
        "score_mae": mae.detach(),
        "score_pearson": pearson.detach(),
        "pairwise_bce": pairwise_bce.detach(),
    }


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


def cache_name(split: str, orientation: str, num_samples: int, num_blocks: int, min_gap: float, include_layers: Sequence[int]) -> str:
    layers = "all" if not include_layers else "-".join(str(value) for value in include_layers)
    gap = str(float(min_gap)).replace(".", "p").replace("-", "m")
    return f"{split}_pairwise_max_fiedler_scores_{orientation}_n{num_samples}_b{num_blocks}_gap{gap}_layers{layers}.pt"


class FiedlerScoreDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        orientation: str,
        cache_dir: Path,
        min_loss_score_gap: float = 0.0,
        include_layers: Sequence[int] | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = str(split)
        self.orientation = str(orientation)
        self.cache_dir = Path(cache_dir)
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

        self.target_score, self.target_meta = self._load_or_build_targets()

    def _load_or_build_targets(self) -> tuple[torch.Tensor, Dict[str, float | int | str]]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / cache_name(
            self.split,
            self.orientation,
            int(self.attention.size(0)),
            int(self.num_blocks),
            float(self.min_loss_score_gap),
            self.include_layers or (),
        )
        fingerprint = {
            "split": self.split,
            "dataset_dir": str(self.dataset_dir),
            "orientation": self.orientation,
            "num_samples": int(self.attention.size(0)),
            "num_blocks": int(self.num_blocks),
            "record_index_first": int(self.record_index[0].item()),
            "record_index_last": int(self.record_index[-1].item()),
            "iter_first": int(self.iter[0].item()),
            "iter_last": int(self.iter[-1].item()),
        }
        if path.exists():
            payload = torch.load(path, map_location="cpu")
            if isinstance(payload, dict) and payload.get("fingerprint") == fingerprint:
                return payload["target_score"].float().contiguous(), dict(payload.get("meta", {}))

        scores = []
        orientation_choices: Dict[str, int] = {}
        eigval1 = []
        spectral_gaps = []
        affinity_sums = []
        attention_np = self.attention.float().numpy()
        rank_np = self.teacher_rank.numpy()
        selected_np = self.selected_reverse.numpy()
        for idx, matrix in enumerate(attention_np):
            vector, meta = fiedler_vector_from_attention(matrix)
            priority, choice = priority_from_vector(
                vector,
                self.orientation,
                rank_np[idx],
                bool(selected_np[idx]),
            )
            scores.append(torch.from_numpy(priority))
            orientation_choices[choice] = int(orientation_choices.get(choice, 0) + 1)
            eigval1.append(float(meta["laplacian_fiedler_eigval"]))
            spectral_gaps.append(float(meta["spectral_gap"]))
            affinity_sums.append(float(meta["affinity_sum"]))
            if (idx + 1) % 1000 == 0:
                print(f"[score-target] {self.split}: built {idx + 1}/{len(attention_np)} targets", flush=True)
        target_score = torch.stack(scores, dim=0).float().contiguous()
        pair_to_teacher = pair_accuracy_to_rank(target_score, self.teacher_rank, self.pair_i, self.pair_j)
        meta = {
            "cache_path": str(path),
            "orientation": self.orientation,
            "orientation_choices": orientation_choices,
            "target_to_teacher_pair_accuracy_mean": float(pair_to_teacher.mean().item()),
            "target_to_teacher_tau_mean": float((2.0 * pair_to_teacher - 1.0).mean().item()),
            "target_score_mean": float(target_score.mean().item()),
            "target_score_std": float(target_score.std(unbiased=False).item()),
            "laplacian_fiedler_eigval_mean": float(np.mean(eigval1)),
            "spectral_gap_mean": float(np.mean(spectral_gaps)),
            "affinity_sum_mean": float(np.mean(affinity_sums)),
        }
        torch.save({"fingerprint": fingerprint, "target_score": target_score, "meta": meta}, path)
        return target_score, meta

    def __len__(self) -> int:
        return int(self.attention.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "attention": self.attention[idx],
            "target_score": self.target_score[idx],
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


def all_layer_heads(dataset: FiedlerScoreDataset) -> List[tuple[int, int]]:
    pairs = sorted({(int(l), int(h)) for l, h in zip(dataset.layer.tolist(), dataset.head.tolist())})
    return pairs


def subset_indices_by_head(dataset: FiedlerScoreDataset, layer: int, head: int) -> List[int]:
    mask = (dataset.layer == int(layer)) & (dataset.head == int(head))
    return torch.nonzero(mask, as_tuple=False).flatten().tolist()


def dataset_summary(dataset: FiedlerScoreDataset) -> Dict:
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
        "target_meta": dataset.target_meta,
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
    maes = []
    pears = []
    target_accs = []
    teacher_accs = []
    pred_all = []
    target_all = []
    total = 0
    generator_device = device.type if device.type == "cuda" else "cpu"
    gen = torch.Generator(device=generator_device)
    gen.manual_seed(int(seed))
    pair_i_dev = pair_i.to(device)
    pair_j_dev = pair_j.to(device)
    for batch in loader:
        attn = batch["attention"].to(device, non_blocking=bool(pin_memory))
        target_score = batch["target_score"].to(device, non_blocking=bool(pin_memory))
        teacher_rank = batch["teacher_rank"].to(device, non_blocking=bool(pin_memory))
        attn = transform_attention(attn, input_mode, gen, data_mean, data_std)
        with torch.autocast(device_type=device.type, enabled=bool(use_amp and device.type == "cuda")):
            logits = model(attn)
            _, loss_items = score_losses(logits, target_score, lambda_corr_loss=0.0)
            pred_score = torch.sigmoid(logits.float())
        count = int(attn.size(0))
        losses.append(float(loss_items["score_mse"].item()) * count)
        maes.append(float(loss_items["score_mae"].item()) * count)
        pears.append(float(loss_items["score_pearson"].item()) * count)
        target_pair = pair_accuracy_to_priority(pred_score, target_score, pair_i_dev, pair_j_dev)
        teacher_pair = pair_accuracy_to_rank(pred_score, teacher_rank, pair_i_dev, pair_j_dev)
        target_accs.append(float(target_pair.mean().item()) * count)
        teacher_accs.append(float(teacher_pair.mean().item()) * count)
        pred_all.append(pred_score.detach().cpu())
        target_all.append(target_score.detach().cpu())
        total += count
    pred_cat = torch.cat(pred_all, dim=0)
    target_cat = torch.cat(target_all, dim=0)
    flat_pearson = pearson_np(pred_cat.reshape(-1).numpy(), target_cat.reshape(-1).numpy())
    target_pair_acc = float(sum(target_accs) / max(1, total))
    teacher_pair_acc = float(sum(teacher_accs) / max(1, total))
    return {
        "score_mse": float(sum(losses) / max(1, total)),
        "score_mae": float(sum(maes) / max(1, total)),
        "score_pearson": float(sum(pears) / max(1, total)),
        "score_flat_pearson": float(flat_pearson),
        "target_pair_accuracy": target_pair_acc,
        "target_kendall_tau_mean": float(2.0 * target_pair_acc - 1.0),
        "teacher_pair_accuracy": teacher_pair_acc,
        "teacher_kendall_tau_mean": float(2.0 * teacher_pair_acc - 1.0),
        "num_samples": int(total),
        "input_mode": str(input_mode),
    }


def _format_order(values: torch.Tensor, limit: int = 16) -> str:
    values = values.detach().cpu().reshape(-1).tolist()
    if int(limit) > 0:
        values = values[: int(limit)]
    return " ".join(str(int(value)) for value in values)


def _pair_accuracy_single_priority(
    pred_priority: torch.Tensor,
    target_priority: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> float:
    pred = pred_priority[pair_i] > pred_priority[pair_j]
    target = target_priority[pair_i] > target_priority[pair_j]
    return float((pred == target).float().mean().item())


def _pair_accuracy_single_rank(
    pred_priority: torch.Tensor,
    teacher_rank: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> float:
    pred = pred_priority[pair_i] > pred_priority[pair_j]
    target = teacher_rank[pair_i] < teacher_rank[pair_j]
    return float((pred == target).float().mean().item())


@torch.no_grad()
def write_val_order_examples(
    model: FlatAttentionOrderMLP,
    dataset: FiedlerScoreDataset,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    device: torch.device,
    report_dir: Path,
    num_examples: int = 16,
) -> Dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    example_count = min(int(num_examples), int(len(dataset)))
    if example_count <= 0:
        return {}
    indices = np.linspace(0, int(len(dataset)) - 1, num=example_count, dtype=int).tolist()
    indices = list(dict.fromkeys(int(index) for index in indices))
    fields = [
        "sample_id",
        "source_iter",
        "record_index",
        "layer",
        "head",
        "selected_reverse",
        "loss_score_gap",
        "pred_score_mean",
        "pred_score_std",
        "pred_score_min",
        "pred_score_max",
        "target_score_mean",
        "target_score_std",
        "target_pair_accuracy",
        "target_kendall_tau",
        "teacher_pair_accuracy",
        "teacher_kendall_tau",
        "pred_order_first16",
        "pred_order_last16",
        "target_order_first16",
        "target_order_last16",
        "teacher_order_first16",
        "teacher_order_last16",
    ]
    rows: List[Dict[str, str | int | float | bool]] = []
    pair_i = pair_i.long().cpu()
    pair_j = pair_j.long().cpu()
    model.eval()
    for sample_id in indices:
        item = dataset[int(sample_id)]
        attn = item["attention"].unsqueeze(0).to(device)
        pred_score = torch.sigmoid(model(attn).float()).squeeze(0).detach().cpu()
        target_score = item["target_score"].float().cpu()
        teacher_rank = item["teacher_rank"].long().cpu()
        teacher_order = item["teacher_order"].long().cpu()
        pred_order = torch.argsort(pred_score, descending=True)
        target_order = torch.argsort(target_score, descending=True)
        target_pair_acc = _pair_accuracy_single_priority(pred_score, target_score, pair_i, pair_j)
        teacher_pair_acc = _pair_accuracy_single_rank(pred_score, teacher_rank, pair_i, pair_j)
        rows.append(
            {
                "sample_id": int(sample_id),
                "source_iter": int(item["iter"].item()),
                "record_index": int(item["record_index"].item()),
                "layer": int(item["layer"].item()),
                "head": int(item["head"].item()),
                "selected_reverse": bool(item["selected_reverse"].item()),
                "loss_score_gap": float(item["loss_score_gap"].item()),
                "pred_score_mean": float(pred_score.mean().item()),
                "pred_score_std": float(pred_score.std(unbiased=False).item()),
                "pred_score_min": float(pred_score.min().item()),
                "pred_score_max": float(pred_score.max().item()),
                "target_score_mean": float(target_score.mean().item()),
                "target_score_std": float(target_score.std(unbiased=False).item()),
                "target_pair_accuracy": float(target_pair_acc),
                "target_kendall_tau": float(2.0 * target_pair_acc - 1.0),
                "teacher_pair_accuracy": float(teacher_pair_acc),
                "teacher_kendall_tau": float(2.0 * teacher_pair_acc - 1.0),
                "pred_order_first16": _format_order(pred_order, 16),
                "pred_order_last16": _format_order(torch.flip(pred_order, dims=(0,)), 16),
                "target_order_first16": _format_order(target_order, 16),
                "target_order_last16": _format_order(torch.flip(target_order, dims=(0,)), 16),
                "teacher_order_first16": _format_order(teacher_order, 16),
                "teacher_order_last16": _format_order(torch.flip(teacher_order, dims=(0,)), 16),
            }
        )
    csv_path = report_dir / "val_order_examples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = [
        "# Validation Order Examples",
        "",
        "All orders use descending score priority: larger score means earlier reveal.",
        "",
        "| sample | iter | L/H | target tau | teacher tau | pred first8 | target first8 | teacher first8 |",
        "|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['sample_id']} | {row['source_iter']} | L{row['layer']}H{row['head']} | "
            f"{float(row['target_kendall_tau']):.4f} | {float(row['teacher_kendall_tau']):.4f} | "
            f"{' '.join(str(row['pred_order_first16']).split()[:8])} | "
            f"{' '.join(str(row['target_order_first16']).split()[:8])} | "
            f"{' '.join(str(row['teacher_order_first16']).split()[:8])} |"
        )
    md_path = report_dir / "val_order_examples.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "md": str(md_path)}


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
    plt.plot(train_x, [float(r["score_mse"]) for r in train_rows], label="train score MSE")
    plt.plot(val_x, [float(r["score_mse"]) for r in val_rows], label="val score MSE")
    plt.xlabel("epoch")
    plt.ylabel("MSE on sigmoid(logits)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(val_x, [float(r["score_pearson"]) for r in val_rows], label="val score Pearson")
    plt.plot(val_x, [float(r["target_kendall_tau_mean"]) for r in val_rows], label="val order tau to score")
    plt.axhline(0.98, color="black", linestyle="--", linewidth=1, label="0.98 tau")
    plt.xlabel("epoch")
    plt.ylabel("correlation / tau")
    plt.ylim(-0.05, 1.01)
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "score_corr_tau_curve.png", dpi=180)
    plt.close()


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

    target_cache_dir = args.target_cache_dir or (args.report_dir / "target_cache")
    val_min_loss_score_gap = (
        float(args.min_loss_score_gap)
        if args.val_min_loss_score_gap is None
        else float(args.val_min_loss_score_gap)
    )
    include_layers = parse_ints(args.include_layers)
    train_data = FiedlerScoreDataset(
        args.dataset_dir,
        "train",
        orientation=str(args.orientation),
        cache_dir=target_cache_dir,
        min_loss_score_gap=float(args.min_loss_score_gap),
        include_layers=include_layers,
    )
    val_data = FiedlerScoreDataset(
        args.dataset_dir,
        "val",
        orientation=str(args.orientation),
        cache_dir=target_cache_dir,
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
        "target_cache_dir": str(target_cache_dir),
        "seed": int(args.seed),
        "model_config": model_config,
        "parameter_count": int(params),
        "target": {
            "source": "pairwise_max_fiedler",
            "affinity": "max(A,A.T)",
            "laplacian": "D-W",
            "coordinate": "second_smallest_laplacian_eigenvector",
            "score": "oriented_minmax_priority",
            "orientation": str(args.orientation),
            "larger_score_means": "earlier_in_order",
        },
        "optimizer": {
            "name": "AdamW",
            "lr": float(args.lr),
            "min_lr": float(args.min_lr),
            "weight_decay": float(args.weight_decay),
            "betas": [float(args.beta1), float(args.beta2)],
            "grad_clip": float(args.grad_clip),
        },
        "loss": {
            "name": "mse_sigmoid_logits_to_oriented_fiedler_priority",
            "lambda_corr_loss": float(args.lambda_corr_loss),
            "lambda_pairwise_loss": float(args.lambda_pairwise_loss),
            "pairwise_logit_scale": float(args.pairwise_logit_scale),
        },
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "eval_every": int(args.eval_every),
        "amp": bool(args.amp),
        "score_mse_gate": float(args.score_mse_gate),
        "target_tau_gate": float(args.target_tau_gate),
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
    best_val_mse = float("inf")
    best_val_tau = -float("inf")
    best_epoch = -1
    metric_fields = [
        "epoch",
        "split",
        "input_mode",
        "score_mse",
        "score_mae",
        "score_pearson",
        "score_flat_pearson",
        "target_pair_accuracy",
        "target_kendall_tau_mean",
        "teacher_pair_accuracy",
        "teacher_kendall_tau_mean",
        "pairwise_bce",
        "num_samples",
        "epoch_seconds",
        "lr",
    ]
    metrics_path = args.report_dir / "metrics.csv"
    pair_i_dev = pair_i.to(device)
    pair_j_dev = pair_j.to(device)

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss_total = 0.0
        train_mse_total = 0.0
        train_mae_total = 0.0
        train_pear_total = 0.0
        train_pair_bce_total = 0.0
        train_total = 0
        for batch in train_loader:
            attn = batch["attention"].to(device, non_blocking=bool(args.pin_memory))
            target_score = batch["target_score"].to(device, non_blocking=bool(args.pin_memory))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=bool(args.amp and device.type == "cuda")):
                logits = model(attn)
                loss, loss_items = score_losses(
                    logits,
                    target_score,
                    float(args.lambda_corr_loss),
                    pair_i=pair_i_dev,
                    pair_j=pair_j_dev,
                    lambda_pairwise_loss=float(args.lambda_pairwise_loss),
                    pairwise_logit_scale=float(args.pairwise_logit_scale),
                )
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            count = int(attn.size(0))
            train_loss_total += float(loss.item()) * count
            train_mse_total += float(loss_items["score_mse"].item()) * count
            train_mae_total += float(loss_items["score_mae"].item()) * count
            train_pear_total += float(loss_items["score_pearson"].item()) * count
            if torch.isfinite(loss_items["pairwise_bce"]):
                train_pair_bce_total += float(loss_items["pairwise_bce"].item()) * count
            train_total += count
        scheduler.step()
        epoch_seconds = time.perf_counter() - t0
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_row = {
            "epoch": int(epoch),
            "split": "train",
            "input_mode": "real",
            "score_mse": float(train_mse_total / max(1, train_total)),
            "score_mae": float(train_mae_total / max(1, train_total)),
            "score_pearson": float(train_pear_total / max(1, train_total)),
            "score_flat_pearson": float("nan"),
            "target_pair_accuracy": float("nan"),
            "target_kendall_tau_mean": float("nan"),
            "teacher_pair_accuracy": float("nan"),
            "teacher_kendall_tau_mean": float("nan"),
            "pairwise_bce": float(train_pair_bce_total / max(1, train_total))
            if float(args.lambda_pairwise_loss) > 0.0
            else float("nan"),
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
                "pairwise_bce": float("nan"),
                "epoch_seconds": float(epoch_seconds),
                "lr": float(current_lr),
            }
            append_csv(metrics_path, val_row, metric_fields)
            is_best = (
                float(val_metrics["score_mse"]) < best_val_mse
                or (
                    math.isclose(float(val_metrics["score_mse"]), best_val_mse)
                    and float(val_metrics["target_kendall_tau_mean"]) > best_val_tau
                )
            )
            if is_best:
                best_val_mse = float(val_metrics["score_mse"])
                best_val_tau = float(val_metrics["target_kendall_tau_mean"])
                best_epoch = int(epoch)
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "config": model_config,
                    "policy_type": "fiedler_score_distilled_mlp",
                    "target_direction": "larger_logit_reveals_larger_oriented_fiedler_priority",
                    "training_meta": {
                        "best_epoch": int(best_epoch),
                        "best_val_score_mse": float(best_val_mse),
                        "best_val_target_tau": float(best_val_tau),
                        "dataset_dir": str(args.dataset_dir),
                        "target_cache_dir": str(target_cache_dir),
                        "parameter_count": int(params),
                        "seed": int(args.seed),
                    },
                }
                torch.save(checkpoint, args.out_dir / "best_by_val_score.pt")
                torch.save(checkpoint, args.out_dir / "best_by_val_tau.pt")
            print(
                f"[score-mlp] epoch={epoch}/{args.epochs} "
                f"train_mse={train_row['score_mse']:.6f} train_pearson={train_row['score_pearson']:.4f} "
                f"val_mse={val_metrics['score_mse']:.6f} val_pearson={val_metrics['score_pearson']:.4f} "
                f"val_target_tau={val_metrics['target_kendall_tau_mean']:.4f}",
                flush=True,
            )
        else:
            print(
                f"[score-mlp] epoch={epoch}/{args.epochs} "
                f"train_mse={train_row['score_mse']:.6f} train_pearson={train_row['score_pearson']:.4f}",
                flush=True,
            )
        maybe_plot(args.report_dir)

    best_payload = torch.load(args.out_dir / "best_by_val_score.pt", map_location=device)
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
        "score_mse",
        "score_mae",
        "score_pearson",
        "target_pair_accuracy",
        "target_kendall_tau_mean",
        "teacher_pair_accuracy",
        "teacher_kendall_tau_mean",
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
    per_head_rows.sort(key=lambda item: float(item["target_kendall_tau_mean"]))
    example_paths = write_val_order_examples(
        best_model,
        val_data,
        pair_i,
        pair_j,
        device,
        args.report_dir,
        num_examples=16,
    )
    passed_gate = bool(float(best_val_mse) <= float(args.score_mse_gate) and float(best_val_tau) >= float(args.target_tau_gate))
    summary = {
        "config": run_config,
        "best_epoch": int(best_epoch),
        "best_val_score_mse": float(best_val_mse),
        "best_val_target_tau": float(best_val_tau),
        "passed_gate": bool(passed_gate),
        "final_eval": final_eval,
        "per_head_val_metrics": per_head_rows,
        "val_order_examples": example_paths,
        "checkpoint": str(args.out_dir / "best_by_val_score.pt"),
        "checkpoint_compat_tau_name": str(args.out_dir / "best_by_val_tau.pt"),
    }
    write_json(args.out_dir / "training_summary.json", summary)
    write_json(args.report_dir / "training_summary.json", summary)
    maybe_plot(args.report_dir)

    report_label = args.report_dir.name or "fiedler_score_mlp_distillation"
    lines = [
        f"# {report_label}: Fiedler Score MLP Distillation",
        "",
        f"- Dataset: `{args.dataset_dir}`",
        f"- Checkpoint: `{args.out_dir / 'best_by_val_score.pt'}`",
        f"- Compatibility checkpoint name: `{args.out_dir / 'best_by_val_tau.pt'}`",
        f"- Parameters: `{params}`",
        f"- Best epoch: `{best_epoch}`",
        f"- Best val score MSE: `{best_val_mse:.6f}`",
        f"- Best val target tau: `{best_val_tau:.6f}`",
        f"- Gate passed: `{passed_gate}`",
        "",
        "## Target",
        "",
        "- Attention target source: `pairwise_max_fiedler`.",
        "- Affinity: `max(A,A.T)`; Laplacian: `D-W`; coordinate: second-smallest eigenvector.",
        "- Target score: oriented min-max priority in `[0, 1]`; larger means earlier under `argsort_desc`.",
        f"- Orientation mode: `{args.orientation}`.",
        "",
        "## Val Input Ablations",
        "",
        "| input | score MSE | score Pearson | target tau | teacher tau | samples |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in parse_strings(args.eval_input_modes):
        item = final_eval[mode]
        lines.append(
            f"| {mode} | {item['score_mse']:.6f} | {item['score_pearson']:.4f} | "
            f"{item['target_kendall_tau_mean']:.4f} | {item['teacher_kendall_tau_mean']:.4f} | "
            f"{item['num_samples']} |"
        )
    lines.extend(["", "## Worst Per-Head Val Target Tau", "", "| head | target tau | teacher tau | score MSE | samples |", "|---|---:|---:|---:|---:|"])
    for item in per_head_rows[:8]:
        lines.append(
            f"| {item['label']} | {float(item['target_kendall_tau_mean']):.4f} | "
            f"{float(item['teacher_kendall_tau_mean']):.4f} | {float(item['score_mse']):.6f} | "
            f"{int(item['num_samples'])} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics.csv`: train/val score regression metrics by epoch.",
            "- `loss_curve.png`: train/val continuous-score MSE.",
            "- `score_corr_tau_curve.png`: val score Pearson and order tau to target score.",
            "- `per_head_val_metrics.csv`: per-head held-out score/tau metrics.",
            "- `val_order_examples.csv` and `val_order_examples.md`: concrete MLP-vs-target-vs-teacher order examples.",
            "- `target_cache/`: regenerated continuous Fiedler score labels.",
        ]
    )
    (args.report_dir / "training_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_score_mse": best_val_mse,
                "best_val_target_tau": best_val_tau,
                "passed_gate": passed_gate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
