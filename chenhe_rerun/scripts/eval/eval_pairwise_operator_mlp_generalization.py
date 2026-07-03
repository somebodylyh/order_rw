#!/usr/bin/env python3
"""Evaluate a pairwise-operator MLP on datasets and head-information matrices."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attn_mlp_order_policy import FlatAttentionOrderMLP  # noqa: E402
from scripts.train.train_pairwise_mlp_operator_distillation import PairwiseOperatorDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate distilled pairwise operator MLP.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dataset_eval", action="append", default=[], help="name:path for a pairwise dataset root")
    parser.add_argument("--head_info_root", type=Path, default=None)
    parser.add_argument("--base_ckpt", type=Path, required=True, help="Checkpoint containing data_permutation metadata.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--num_blocks", type=int, default=64)
    parser.add_argument("--head_info_key_filter", type=str, default="post_softmax_attention_without_none")
    parser.add_argument("--head_info_max_files", type=int, default=0)
    parser.add_argument("--max_dataset_samples", type=int, default=0)
    parser.add_argument("--zero_diagonal", action="store_true", default=True)
    return parser.parse_args()


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_model(path: Path, device: torch.device) -> FlatAttentionOrderMLP:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config") or checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError(f"Could not find model config in {path}")
    model = FlatAttentionOrderMLP(**config).to(device)
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"Could not find model state in {path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_inverse_perm(base_ckpt: Path, num_blocks: int) -> torch.Tensor:
    checkpoint = torch.load(base_ckpt, map_location="cpu")
    perm = checkpoint.get("data_permutation") or {}
    if perm.get("inverse_block_perm") is not None:
        return torch.tensor(perm["inverse_block_perm"], dtype=torch.long)
    if perm.get("block_perm") is not None:
        block_perm = torch.tensor(perm["block_perm"], dtype=torch.long)
        inverse = torch.empty_like(block_perm)
        inverse[block_perm] = torch.arange(block_perm.numel(), dtype=torch.long)
        return inverse
    return torch.arange(int(num_blocks), dtype=torch.long)


def pair_indices(num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.triu_indices(int(num_blocks), int(num_blocks), offset=1)


def ranks_from_order(order: torch.Tensor) -> torch.Tensor:
    rank = torch.empty_like(order)
    rank[order.long()] = torch.arange(order.numel(), dtype=rank.dtype)
    return rank


def kendall_tau_between_orders(order_a: torch.Tensor, order_b: torch.Tensor) -> float:
    a = order_a.detach().cpu().long()
    b = order_b.detach().cpu().long()
    n = int(a.numel())
    if n < 2:
        return 1.0
    pos_a = torch.empty(n, dtype=torch.long)
    pos_b = torch.empty(n, dtype=torch.long)
    pos_a[a] = torch.arange(n, dtype=torch.long)
    pos_b[b] = torch.arange(n, dtype=torch.long)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (pos_a[i] - pos_a[j]).item() * (pos_b[i] - pos_b[j]).item() < 0:
                inversions += 1
    total = n * (n - 1) / 2.0
    return float(1.0 - 2.0 * inversions / total)


def pair_acc_from_ranks(pred_rank: torch.Tensor, target_rank: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor) -> float:
    pred = pred_rank[pair_i] < pred_rank[pair_j]
    target = target_rank[pair_i] < target_rank[pair_j]
    return float((pred == target).float().mean().item())


def batched_ranks_from_orders(orders: torch.Tensor) -> torch.Tensor:
    ranks = torch.empty_like(orders)
    values = torch.arange(orders.size(1), dtype=orders.dtype, device=orders.device)
    ranks.scatter_(1, orders.long(), values.unsqueeze(0).expand_as(orders))
    return ranks


@torch.no_grad()
def eval_pairwise_dataset(
    *,
    name: str,
    dataset_dir: Path,
    split: str,
    model: FlatAttentionOrderMLP,
    device: torch.device,
    inverse_perm: torch.Tensor,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    max_samples: int,
) -> Dict:
    dataset = PairwiseOperatorDataset(dataset_dir, split)
    if int(max_samples) > 0 and len(dataset) > int(max_samples):
        from torch.utils.data import Subset

        dataset = Subset(dataset, list(range(int(max_samples))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
    pair_i = base_dataset.pair_i.cpu()
    pair_j = base_dataset.pair_j.cpu()
    original_rank = ranks_from_order(inverse_perm.cpu())
    original_pairs = original_rank[pair_i] < original_rank[pair_j]
    rows = []
    for batch in loader:
        attn = batch["attention"].to(device, non_blocking=pin_memory)
        logits = model(attn).detach().cpu()
        teacher_rank = batch["teacher_rank"].cpu().long()
        pred_order = torch.argsort(logits.float(), dim=1, descending=True)
        pred_rank = batched_ranks_from_orders(pred_order)
        pred_pairs = pred_rank[:, pair_i] < pred_rank[:, pair_j]
        teacher_pairs = teacher_rank[:, pair_i] < teacher_rank[:, pair_j]
        teacher_pair_acc = (pred_pairs == teacher_pairs).float().mean(dim=1)
        mlp_original_tau = 2.0 * (pred_pairs == original_pairs.unsqueeze(0)).float().mean(dim=1) - 1.0
        teacher_original_tau = 2.0 * (teacher_pairs == original_pairs.unsqueeze(0)).float().mean(dim=1) - 1.0
        for idx in range(pred_order.size(0)):
            rows.append({
                "eval_name": name,
                "split": split,
                "record_index": int(batch["record_index"][idx].item()),
                "iter": int(batch["iter"][idx].item()),
                "layer": int(batch["layer"][idx].item()),
                "head": int(batch["head"][idx].item()),
                "mlp_teacher_pair_acc": float(teacher_pair_acc[idx].item()),
                "mlp_teacher_tau": float(2.0 * teacher_pair_acc[idx].item() - 1.0),
                "mlp_original_tau": float(mlp_original_tau[idx].item()),
                "teacher_original_tau": float(teacher_original_tau[idx].item()),
                "loss_score_gap": float(batch["loss_score_gap"][idx].item()),
                "selected_reverse": bool(batch["selected_reverse"][idx].item()),
            })
    numeric_keys = [
        "mlp_teacher_pair_acc",
        "mlp_teacher_tau",
        "mlp_original_tau",
        "teacher_original_tau",
        "loss_score_gap",
    ]
    summary = {
        "eval_name": name,
        "split": split,
        "dataset_dir": str(dataset_dir),
        "num_samples": int(len(rows)),
    }
    for key in numeric_keys:
        arr = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(arr.mean()) if arr.size else float("nan")
        summary[f"{key}_std"] = float(arr.std()) if arr.size else float("nan")
        summary[f"{key}_min"] = float(arr.min()) if arr.size else float("nan")
        summary[f"{key}_max"] = float(arr.max()) if arr.size else float("nan")
    return {"summary": summary, "rows": rows}


def parse_dataset_eval(raw: str) -> tuple[str, Path]:
    if ":" not in raw:
        path = Path(raw)
        return path.name, path
    name, path = raw.split(":", 1)
    return name, Path(path)


def iter_head_info_npz(root: Path, max_files: int) -> Iterable[Path]:
    paths = sorted(root.rglob("*.npz"))
    if int(max_files) > 0:
        paths = paths[: int(max_files)]
    return paths


def extract_l0_matrices(array: np.ndarray, key: str, num_blocks: int) -> List[tuple[str, torch.Tensor]]:
    arr = np.asarray(array)
    if arr.shape == (num_blocks, num_blocks):
        return [(key, torch.from_numpy(arr).float())]
    if arr.ndim >= 4 and tuple(arr.shape[-4:]) == (4, 8, num_blocks, num_blocks):
        tensor = torch.from_numpy(arr).float()
        l0_mean = tensor[..., 0, :, :, :].mean(dim=-3)
        return [(f"{key}::L0mean", l0_mean.reshape(-1, num_blocks, num_blocks)[0])]
    if arr.ndim == 4 and tuple(arr.shape) == (4, 8, num_blocks, num_blocks):
        tensor = torch.from_numpy(arr).float()
        return [(f"{key}::L0mean", tensor[0].mean(dim=0))]
    return []


def infer_matrix_frame(key: str) -> str:
    key = str(key)
    current_markers = (
        "_current_raw_average",
        "_current_exposure_corrected",
        "_current_raw_",
        "_current_exposure_",
    )
    original_markers = (
        "_original_raw_average",
        "_original_exposure_corrected",
        "_original_raw_",
        "_original_exposure_",
    )
    if any(marker in key for marker in current_markers):
        return "current"
    if any(marker in key for marker in original_markers):
        return "original"
    return "current"


@torch.no_grad()
def eval_head_information(
    *,
    root: Path,
    model: FlatAttentionOrderMLP,
    device: torch.device,
    inverse_perm: torch.Tensor,
    key_filter: str,
    max_files: int,
    num_blocks: int,
    zero_diagonal: bool,
) -> List[Dict]:
    rows = []
    l2r = torch.arange(int(num_blocks), dtype=torch.long)
    for path in iter_head_info_npz(root, max_files):
        try:
            archive = np.load(path)
        except Exception as exc:
            rows.append({"path": str(path), "error": str(exc)})
            continue
        for key in archive.files:
            if key_filter and key_filter not in key:
                continue
            for matrix_label, matrix in extract_l0_matrices(archive[key], key, num_blocks):
                matrix = matrix.clone().float()
                if zero_diagonal:
                    matrix.fill_diagonal_(0.0)
                logits = model(matrix.to(device)).detach().cpu()
                order = torch.argsort(logits.float(), descending=True)
                frame = infer_matrix_frame(key)
                reference = l2r if frame == "original" else inverse_perm
                rows.append({
                    "path": str(path),
                    "key": key,
                    "matrix_label": matrix_label,
                    "frame": frame,
                    "mlp_original_tau": kendall_tau_between_orders(order, reference),
                    "mlp_current_l2r_tau": kendall_tau_between_orders(order, l2r),
                    "logits_mean": float(logits.float().mean().item()),
                    "logits_std": float(logits.float().std(unbiased=False).item()),
                    "logits_min": float(logits.float().min().item()),
                    "logits_max": float(logits.float().max().item()),
                    "matrix_mean": float(matrix.mean().item()),
                    "matrix_std": float(matrix.std(unbiased=False).item()),
                    "order_first16": " ".join(str(int(v)) for v in order.tolist()[:16]),
                })
    return rows


def maybe_plot(out_dir: Path, dataset_summaries: List[Dict], head_rows: List[Dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_dir / "plot_skipped.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return

    if dataset_summaries:
        labels = [f"{row['eval_name']}:{row['split']}" for row in dataset_summaries]
        teacher_tau = [float(row["mlp_teacher_tau_mean"]) for row in dataset_summaries]
        original_tau = [float(row["mlp_original_tau_mean"]) for row in dataset_summaries]
        x = np.arange(len(labels))
        plt.figure(figsize=(max(7, 0.55 * len(labels)), 4))
        plt.bar(x - 0.18, teacher_tau, width=0.36, label="MLP vs teacher tau")
        plt.bar(x + 0.18, original_tau, width=0.36, label="MLP original tau")
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel("tau")
        plt.ylim(-1.0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "dataset_eval_tau_summary.png", dpi=180)
        plt.close()

    clean_head_rows = [row for row in head_rows if "mlp_original_tau" in row]
    if clean_head_rows:
        values = [float(row["mlp_original_tau"]) for row in clean_head_rows]
        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=30)
        plt.xlabel("MLP original tau")
        plt.ylabel("matrix count")
        plt.tight_layout()
        plt.savefig(out_dir / "head_information_original_tau_hist.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    inverse_perm = load_inverse_perm(args.base_ckpt, int(args.num_blocks))

    dataset_summaries = []
    dataset_rows = []
    for item in args.dataset_eval:
        name, dataset_dir = parse_dataset_eval(item)
        for split in ("train", "val"):
            split_dir = dataset_dir / split
            if not split_dir.exists() or not list(split_dir.glob("shard_*.pt")):
                continue
            result = eval_pairwise_dataset(
                name=name,
                dataset_dir=dataset_dir,
                split=split,
                model=model,
                device=device,
                inverse_perm=inverse_perm,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                pin_memory=bool(args.pin_memory),
                max_samples=int(args.max_dataset_samples),
            )
            dataset_summaries.append(result["summary"])
            dataset_rows.extend(result["rows"])

    head_rows = []
    if args.head_info_root is not None:
        head_rows = eval_head_information(
            root=args.head_info_root,
            model=model,
            device=device,
            inverse_perm=inverse_perm,
            key_filter=str(args.head_info_key_filter),
            max_files=int(args.head_info_max_files),
            num_blocks=int(args.num_blocks),
            zero_diagonal=bool(args.zero_diagonal),
        )

    write_json(args.out_dir / "eval_summary.json", {
        "checkpoint": str(args.checkpoint),
        "base_ckpt": str(args.base_ckpt),
        "dataset_summaries": dataset_summaries,
        "head_information_rows": len(head_rows),
    })
    write_csv(args.out_dir / "dataset_eval_rows.csv", dataset_rows)
    write_csv(args.out_dir / "dataset_eval_summary.csv", dataset_summaries)
    write_csv(args.out_dir / "head_information_mlp_eval.csv", head_rows)
    maybe_plot(args.out_dir, dataset_summaries, head_rows)

    lines = [
        "# Pairwise Operator MLP Generalization Eval",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Base ckpt for original tau: `{args.base_ckpt}`",
        "",
        "## Dataset Eval",
        "",
        "| dataset | split | samples | MLP-teacher tau | MLP original tau | teacher original tau |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_summaries:
        lines.append(
            f"| {row['eval_name']} | {row['split']} | {row['num_samples']} | "
            f"{row['mlp_teacher_tau_mean']:.4f} | {row['mlp_original_tau_mean']:.4f} | "
            f"{row['teacher_original_tau_mean']:.4f} |"
        )
    lines.extend([
        "",
        "## Head Information Eval",
        "",
        f"- Rows: `{len(head_rows)}`",
        f"- Key filter: `{args.head_info_key_filter}`",
        "- See `head_information_mlp_eval.csv` for per-matrix order/logit details.",
        "",
        "## Figures",
        "",
        "- `dataset_eval_tau_summary.png`",
        "- `head_information_original_tau_hist.png`",
    ])
    (args.out_dir / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"dataset_summaries": dataset_summaries, "head_information_rows": len(head_rows)}, indent=2))


if __name__ == "__main__":
    main()
