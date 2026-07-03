#!/usr/bin/env python3
"""Probe a trained score Attn-MLP on upper/lower triangular attention inputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attn_mlp_order_policy import load_frozen_attn_mlp_policy, logits_to_order  # noqa: E402
from order_utils import build_fixed_block_permutation, kendall_tau_to_l2r_per_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints/attn_mlp_distillation/"
            "try33_joint_try20_try24_l0_layermean_fiedler_score_mlp_h2048_1024/"
            "best_by_val_tau.pt"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "Report/language/wikitext103/mlp/distillation/try_25/"
            "combined_try20_try24_l0_layermean_fiedler_dataset"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Report/language/wikitext103/mlp/distillation/try_33/triangular_probe_20260627"),
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--subset-size", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--order-mode", type=str, default="argsort_desc")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--band-scale", type=float, default=6.0)
    parser.add_argument("--permute-seed", type=int, default=42)
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def pearson_np(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - float(left.mean())
    right = right - float(right.mean())
    denom = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    if denom <= eps:
        return 0.0
    return float(np.sum(left * right) / denom)


def vector_stats(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().float().cpu().view(-1)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
    }


def order_tau(order: torch.Tensor) -> float:
    order = order.detach().cpu().long().view(1, -1)
    return float(kendall_tau_to_l2r_per_sample(order).float()[0].item())


def summarize_orders(orders: torch.Tensor, fixed_block_perm: torch.Tensor | None = None) -> Dict[str, float]:
    orders = orders.detach().cpu().long()
    taus = kendall_tau_to_l2r_per_sample(orders).float().cpu()
    out = {
        "n": int(orders.size(0)),
        "current_tau_mean": float(taus.mean().item()),
        "current_tau_std": float(taus.std(unbiased=False).item()),
        "current_tau_min": float(taus.min().item()),
        "current_tau_max": float(taus.max().item()),
        "current_tau_negative_count": int((taus < 0.0).sum().item()),
        "current_tau_positive_count": int((taus > 0.0).sum().item()),
    }
    if fixed_block_perm is not None:
        mapper = fixed_block_perm.detach().cpu().long()
        original_orders = mapper[orders.long()]
        original_taus = kendall_tau_to_l2r_per_sample(original_orders).float().cpu()
        out.update(
            {
                "original_tau_mean": float(original_taus.mean().item()),
                "original_tau_std": float(original_taus.std(unbiased=False).item()),
                "original_tau_min": float(original_taus.min().item()),
                "original_tau_max": float(original_taus.max().item()),
                "original_tau_negative_count": int((original_taus < 0.0).sum().item()),
                "original_tau_positive_count": int((original_taus > 0.0).sum().item()),
                "original_order_first16": [int(v) for v in original_orders[0, :16].tolist()],
                "original_order_last16": [int(v) for v in original_orders[0, -16:].tolist()],
            }
        )
    return out


def score_direction_stats(scores: torch.Tensor) -> Dict[str, float]:
    scores = scores.detach().float().cpu()
    n = int(scores.size(-1))
    l2r_priority = torch.linspace(1.0, 0.0, steps=n)
    reverse_priority = torch.linspace(0.0, 1.0, steps=n)
    l2r_corrs = []
    reverse_corrs = []
    margins = []
    for row in scores.view(-1, n):
        arr = row.numpy()
        l2r_corrs.append(pearson_np(arr, l2r_priority.numpy()))
        reverse_corrs.append(pearson_np(arr, reverse_priority.numpy()))
        margins.append(float(row[0].item() - row[-1].item()))
    return {
        "score_l2r_pearson_mean": float(np.mean(l2r_corrs)),
        "score_reverse_pearson_mean": float(np.mean(reverse_corrs)),
        "score_first_minus_last_mean": float(np.mean(margins)),
    }


def predict(
    model,
    matrix: torch.Tensor,
    device: torch.device,
    order_mode: str,
    fixed_block_perm: torch.Tensor | None = None,
) -> Dict:
    matrix = matrix.detach().float().to(device)
    if matrix.ndim == 2:
        matrix_b = matrix.unsqueeze(0)
    else:
        matrix_b = matrix
    with torch.no_grad():
        logits = model(matrix_b)
        scores = torch.sigmoid(logits.float())
        orders = torch.stack([logits_to_order(row, mode=order_mode).detach().cpu() for row in logits], dim=0)
    order_summary = summarize_orders(orders, fixed_block_perm=fixed_block_perm)
    score_stats = vector_stats(scores)
    out = {
        **order_summary,
        **{f"score_{k}": v for k, v in score_stats.items()},
        **score_direction_stats(scores),
        "order_first16": [int(v) for v in orders[0, :16].tolist()],
        "order_last16": [int(v) for v in orders[0, -16:].tolist()],
        "score_first16": [float(v) for v in scores[0, :16].detach().cpu().tolist()],
        "score_last16": [float(v) for v in scores[0, -16:].detach().cpu().tolist()],
    }
    return out


def row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    denom = matrix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return matrix / denom


def synthetic_matrices(num_blocks: int, band_scale: float) -> Dict[str, torch.Tensor]:
    n = int(num_blocks)
    idx = torch.arange(n)
    rows = idx[:, None]
    cols = idx[None, :]
    dist = (rows - cols).abs().float()
    lower = (rows > cols).float()
    upper = (rows < cols).float()
    diag = torch.eye(n)
    decay = torch.exp(-dist / float(band_scale))
    mats = {
        "synthetic_zero": torch.zeros(n, n),
        "synthetic_identity": diag,
        "synthetic_lower_strict_ones": lower,
        "synthetic_upper_strict_ones": upper,
        "synthetic_lower_decay": lower * decay,
        "synthetic_upper_decay": upper * decay,
        "synthetic_lower_decay_row_norm": row_normalize(lower * decay),
        "synthetic_upper_decay_row_norm": row_normalize(upper * decay),
        "synthetic_symmetric_decay": (lower + upper) * decay,
        "synthetic_causal_with_diag_decay": (lower + diag) * decay,
        "synthetic_anti_causal_with_diag_decay": (upper + diag) * decay,
    }
    return mats


def original_frame_synthetic_matrices(
    num_blocks: int,
    band_scale: float,
    fixed_block_perm: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    current_to_original = fixed_block_perm.detach().cpu().long()
    originals = synthetic_matrices(num_blocks, band_scale)
    keep = {
        "synthetic_lower_strict_ones",
        "synthetic_upper_strict_ones",
        "synthetic_lower_decay",
        "synthetic_upper_decay",
        "synthetic_lower_decay_row_norm",
        "synthetic_upper_decay_row_norm",
        "synthetic_symmetric_decay",
    }
    out = {}
    for name, original_matrix in originals.items():
        if name not in keep:
            continue
        current_matrix = original_matrix[current_to_original][:, current_to_original].contiguous()
        out[f"original_l2r_view_{name}"] = current_matrix
    return out


def load_attention_subset(dataset_dir: Path, split: str, subset_size: int) -> torch.Tensor:
    paths = sorted((dataset_dir / split).glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.pt under {dataset_dir / split}")
    chunks = []
    total = 0
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        attention = payload["attention"].float().contiguous()
        chunks.append(attention)
        total += int(attention.size(0))
        if total >= int(subset_size):
            break
    return torch.cat(chunks, dim=0)[: int(subset_size)].contiguous()


def real_variants(attention: torch.Tensor) -> Dict[str, torch.Tensor]:
    n = int(attention.size(-1))
    lower_strict = torch.tril(torch.ones(n, n), diagonal=-1).bool()
    upper_strict = torch.triu(torch.ones(n, n), diagonal=1).bool()
    lower_diag = torch.tril(torch.ones(n, n), diagonal=0).bool()
    upper_diag = torch.triu(torch.ones(n, n), diagonal=0).bool()
    diag = torch.eye(n).bool()
    transpose = attention.transpose(-1, -2).contiguous()
    return {
        "real_original": attention,
        "real_transpose": transpose,
        "real_lower_strict_only": attention.masked_fill(~lower_strict, 0.0),
        "real_upper_strict_only": attention.masked_fill(~upper_strict, 0.0),
        "real_lower_with_diag": attention.masked_fill(~lower_diag, 0.0),
        "real_upper_with_diag": attention.masked_fill(~upper_diag, 0.0),
        "real_diag_only": attention.masked_fill(~diag, 0.0),
        "real_symmetric_max": torch.maximum(attention, transpose),
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys() if not isinstance(row.get(key), (list, dict))})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def write_markdown(path: Path, rows: List[Dict], metadata: Dict) -> None:
    interesting = [
        "name",
        "family",
        "n",
        "current_tau_mean",
        "original_tau_mean",
        "original_tau_negative_count",
        "score_l2r_pearson_mean",
        "score_first_minus_last_mean",
        "order_first16",
    ]
    lines = [
        "# Attn-MLP triangular input probe",
        "",
        "This probe feeds the trained score-MSE Attn-MLP with synthetic triangular attention patterns and triangularized real validation attention subsets.",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(json_safe(metadata), indent=2, sort_keys=True),
        "```",
        "",
        "## Summary",
        "",
        "| name | family | n | current_tau | original_tau | orig_neg | score_l2r_corr | first-last | first16 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {family} | {n} | {current_tau_mean:.4f} | {original_tau_mean:.4f} | "
            "{original_tau_negative_count} | {score_l2r_pearson_mean:.4f} | {score_first_minus_last_mean:.4f} | `{order}` |".format(
                name=row["name"],
                family=row["family"],
                n=int(row["n"]),
                current_tau_mean=float(row["current_tau_mean"]),
                original_tau_mean=float(row.get("original_tau_mean", float("nan"))),
                original_tau_negative_count=int(row.get("original_tau_negative_count", 0)),
                score_l2r_pearson_mean=float(row["score_l2r_pearson_mean"]),
                score_first_minus_last_mean=float(row["score_first_minus_last_mean"]),
                order=" ".join(str(v) for v in row["order_first16"]),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation guide: `current_tau` is computed in current block ids. `original_tau` first maps current ids through `fixed_block_perm` and is the column to compare with original-L2R plots.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    fixed_block_perm = build_fixed_block_permutation(
        int(args.num_blocks),
        int(args.permute_seed),
    ).detach().cpu().long()
    model, config = load_frozen_attn_mlp_policy(
        str(args.checkpoint),
        num_blocks=int(args.num_blocks),
        device=device,
        input_normalization="zscore",
    )

    rows: List[Dict] = []
    synthetic = synthetic_matrices(int(args.num_blocks), float(args.band_scale))
    for name, matrix in synthetic.items():
        result = predict(
            model,
            matrix,
            device=device,
            order_mode=str(args.order_mode),
            fixed_block_perm=fixed_block_perm,
        )
        rows.append({"name": name, "family": "synthetic_current_frame", **result})

    original_synthetic = original_frame_synthetic_matrices(
        int(args.num_blocks),
        float(args.band_scale),
        fixed_block_perm,
    )
    for name, matrix in original_synthetic.items():
        result = predict(
            model,
            matrix,
            device=device,
            order_mode=str(args.order_mode),
            fixed_block_perm=fixed_block_perm,
        )
        rows.append({"name": name, "family": "synthetic_original_l2r_view", **result})

    attention = load_attention_subset(args.dataset_dir, args.split, int(args.subset_size))
    for name, matrix in real_variants(attention).items():
        result = predict(
            model,
            matrix,
            device=device,
            order_mode=str(args.order_mode),
            fixed_block_perm=fixed_block_perm,
        )
        rows.append({"name": name, "family": "real_subset", **result})

    metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_config": config,
        "dataset_dir": str(args.dataset_dir),
        "split": str(args.split),
        "subset_size": int(attention.size(0)),
        "attention_mean": float(attention.mean().item()),
        "attention_std": float(attention.std(unbiased=False).item()),
        "attention_min": float(attention.min().item()),
        "attention_max": float(attention.max().item()),
        "order_mode": str(args.order_mode),
        "device": str(device),
        "band_scale": float(args.band_scale),
        "permute_seed": int(args.permute_seed),
        "fixed_block_perm_current_to_original": [int(v) for v in fixed_block_perm.tolist()],
    }
    payload = {"metadata": metadata, "rows": rows}
    (args.out_dir / "triangular_probe_results.json").write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(args.out_dir / "triangular_probe_summary.csv", rows)
    write_markdown(args.out_dir / "triangular_probe_report.md", rows, metadata)
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
