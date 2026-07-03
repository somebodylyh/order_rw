#!/usr/bin/env python3
"""Export per-layer/per-head block attention maps from a checkpoint.

This is a narrow diagnostic helper for seq256/block64 AO-GPT language runs. It
aggregates every attention head separately, then writes both current-frame L2R
and true-original L2R coordinates for permuted-data checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.export_block_attention_heatmap import (  # noqa: E402
    build_model,
    build_orders,
    extract_logits_loss_attentions,
    get_autocast_context,
    infer_data_record_mode,
    load_checkpoint,
    load_tokens,
    maybe_apply_data_permutation,
    resolve_data_dir,
    resolve_data_permutation,
    sample_batch,
)


EXPORT_TYPES = ("with_none", "without_none")
FRAMES = ("current_l2r", "original_l2r")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all 32 head block-attention maps in current and original L2R frames."
    )
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--mode", type=str, default="Random", choices=["AR", "Random"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--force_manual_attention", action="store_true")
    parser.add_argument("--vmax_percentile", type=float, default=99.0)
    return parser.parse_args()


def shift_attention(layer_attn: torch.Tensor, export_type: str) -> torch.Tensor:
    if export_type == "with_none":
        return layer_attn[:, :, :-1, :-1]
    if export_type == "without_none":
        return layer_attn[:, :, 1:, 1:]
    raise ValueError(f"Unsupported export_type={export_type!r}")


def block_attention(layer_attn: torch.Tensor, block_len: int, export_type: str) -> torch.Tensor:
    shifted = shift_attention(layer_attn, export_type)
    batch, heads, seq_a, seq_b = shifted.shape
    if seq_a != seq_b or seq_a % block_len != 0:
        raise ValueError(f"attention shape {tuple(shifted.shape)} is incompatible with block_len={block_len}")
    num_blocks = seq_a // block_len
    return shifted.float().view(batch, heads, num_blocks, block_len, num_blocks, block_len).mean(dim=(3, 5))


def invert_batch_permutation(orders: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(orders)
    inv.scatter_(1, orders, torch.arange(orders.size(1), device=orders.device).expand_as(orders))
    return inv


def align_batch_to_current(block: torch.Tensor, block_orders: torch.Tensor) -> torch.Tensor:
    batch, heads, num_blocks, _ = block.shape
    inv = invert_batch_permutation(block_orders.detach()).to(device=block.device)
    gather_rows = inv[:, None, :, None].expand(batch, heads, num_blocks, num_blocks)
    gather_cols = inv[:, None, None, :].expand(batch, heads, num_blocks, num_blocks)
    return block.gather(2, gather_rows).gather(3, gather_cols)


def align_current_to_original(current: torch.Tensor, data_permutation) -> torch.Tensor:
    if data_permutation is None:
        return current
    original_to_current = data_permutation["inverse_block_perm"].to(device=current.device)
    return current[:, :, original_to_current, :][:, :, :, original_to_current]


def positive_vmax(matrix: np.ndarray, percentile: float) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    finite = values[np.isfinite(values)]
    finite = finite[finite > 0.0]
    if finite.size == 0:
        return 1.0
    if 0.0 < percentile < 100.0:
        vmax = float(np.percentile(finite, percentile))
    else:
        vmax = float(np.max(finite))
    return max(vmax, 1e-12)


def matrix_metrics(matrix: np.ndarray) -> dict:
    values = np.asarray(matrix, dtype=np.float64)
    n = values.shape[0]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)
    upper_sum = float(values[upper].sum())
    lower_sum = float(values[lower].sum())
    total = max(upper_sum + lower_sum, 1e-12)
    direction_score = float((upper_sum - lower_sum) / max(abs(upper_sum) + abs(lower_sum), 1e-12))
    row = values / np.clip(values.sum(axis=-1, keepdims=True), 1e-12, None)
    row_upper = float(row[upper].mean())
    row_lower = float(row[lower].mean())
    row_triangle = float((row_lower - row_upper) / max(row_lower + row_upper, 1e-12))
    best_offset = 0
    best_sum = -float("inf")
    for offset in range(-(n - 1), n):
        if offset == 0:
            continue
        band_sum = float(np.diagonal(values, offset=offset).sum())
        if band_sum > best_sum:
            best_sum = band_sum
            best_offset = int(offset)
    return {
        "upper_sum": upper_sum,
        "lower_sum": lower_sum,
        "upper_fraction": float(upper_sum / total),
        "lower_fraction": float(lower_sum / total),
        "direction_score_upper_minus_lower": direction_score,
        "row_normalized_lower_minus_upper": row_triangle,
        "attention_sum": float(values.sum()),
        "attention_max": float(values.max()),
        "top_band_offset": int(best_offset),
        "top_band_sum": float(best_sum if np.isfinite(best_sum) else 0.0),
    }


def save_grid(matrices: np.ndarray, metrics: list[dict], out_path: Path, title: str, vmax: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers, heads = matrices.shape[:2]
    fig, axes = plt.subplots(layers, heads, figsize=(2.05 * heads, 1.95 * layers), constrained_layout=True)
    if layers == 1:
        axes = np.expand_dims(axes, axis=0)
    if heads == 1:
        axes = np.expand_dims(axes, axis=1)
    last_im = None
    by_head = {(row["layer"], row["head"]): row for row in metrics}
    for layer_idx in range(layers):
        for head_idx in range(heads):
            ax = axes[layer_idx, head_idx]
            last_im = ax.imshow(matrices[layer_idx, head_idx], cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest")
            item = by_head[(layer_idx, head_idx)]
            ax.set_title(
                f"L{layer_idx}H{head_idx} u={item['upper_fraction']:.2f} d={item['direction_score_upper_minus_lower']:+.2f}",
                fontsize=7,
            )
            ax.set_xticks([])
            ax.set_yticks([])
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.72, pad=0.01)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_individual(matrices: np.ndarray, metrics: list[dict], out_root: Path, title_prefix: str, vmax: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_head = {(row["layer"], row["head"]): row for row in metrics}
    layers, heads = matrices.shape[:2]
    for layer_idx in range(layers):
        for head_idx in range(heads):
            item = by_head[(layer_idx, head_idx)]
            fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
            im = ax.imshow(matrices[layer_idx, head_idx], cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest")
            ax.set_title(
                f"{title_prefix} L{layer_idx}H{head_idx}\n"
                f"upper_frac={item['upper_fraction']:.3f}, dir={item['direction_score_upper_minus_lower']:+.3f}, "
                f"band={item['top_band_offset']:+d}",
                fontsize=10,
            )
            ax.set_xlabel("key block")
            ax.set_ylabel("query block")
            fig.colorbar(im, ax=ax, shrink=0.8)
            target = out_root / f"L{layer_idx}H{head_idx}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=180, bbox_inches="tight")
            plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.ckpt_path, args.device)
    model = build_model(checkpoint, args.device, force_manual_attention=args.force_manual_attention)
    data_dir = resolve_data_dir(args, checkpoint)
    data_record_mode = str(checkpoint.get("config", {}).get("data_record_mode", infer_data_record_mode(data_dir)))
    tokens = load_tokens(data_dir, args.split, data_record_mode=data_record_mode)
    rng = np.random.default_rng(args.seed)
    ctx = get_autocast_context(args.device, args.dtype)
    data_permutation = resolve_data_permutation(
        checkpoint,
        num_blocks=model.num_blocks,
        block_len=model.block_order_block_len,
        model=model,
    )

    n_layer = int(model.config.n_layer)
    n_head = int(model.config.n_head)
    num_blocks = int(model.num_blocks)
    sums = {
        export_type: {frame: torch.zeros((n_layer, n_head, num_blocks, num_blocks), dtype=torch.float64, device="cpu") for frame in FRAMES}
        for export_type in EXPORT_TYPES
    }
    total_samples = 0
    loss_sum = 0.0
    batch_start_offsets = []

    for batch_idx in range(int(args.num_batches)):
        idx, starts = sample_batch(tokens, args.batch_size, model.config.block_size, rng, args.device)
        idx = maybe_apply_data_permutation(idx, data_permutation)
        token_orders, block_orders = build_orders(model, idx, args.mode)
        with torch.no_grad():
            with ctx:
                outputs = model.forward_fn(idx, token_orders, return_attentions=True)
        _, loss, attentions = extract_logits_loss_attentions(outputs)
        batch_start_offsets.extend(int(v) for v in starts)
        for layer_idx, layer_attn in enumerate(attentions):
            for export_type in EXPORT_TYPES:
                block = block_attention(layer_attn.detach(), model.block_order_block_len, export_type)
                current = align_batch_to_current(block, block_orders)
                original = align_current_to_original(current, data_permutation)
                sums[export_type]["current_l2r"][layer_idx] += current.double().sum(dim=0).cpu()
                sums[export_type]["original_l2r"][layer_idx] += original.double().sum(dim=0).cpu()
        batch_samples = int(idx.size(0))
        total_samples += batch_samples
        loss_sum += float(loss.detach().item()) * batch_samples
        if (batch_idx + 1) % 5 == 0 or batch_idx == 0 or batch_idx + 1 == int(args.num_batches):
            print(f"processed {batch_idx + 1}/{args.num_batches} batches")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    metric_rows = []
    for export_type in EXPORT_TYPES:
        for frame in FRAMES:
            avg = (sums[export_type][frame] / float(max(1, total_samples))).numpy().astype(np.float32)
            arrays[f"{export_type}_{frame}"] = avg
            rows_for_grid = []
            for layer_idx in range(n_layer):
                for head_idx in range(n_head):
                    row = {
                        "export_type": export_type,
                        "frame": frame,
                        "layer": layer_idx,
                        "head": head_idx,
                        **matrix_metrics(avg[layer_idx, head_idx]),
                    }
                    metric_rows.append(row)
                    rows_for_grid.append(row)
            vmax = positive_vmax(avg, args.vmax_percentile)
            frame_dir = args.out_dir / export_type / frame
            frame_dir.mkdir(parents=True, exist_ok=True)
            np.save(frame_dir / "all_head_matrices.npy", avg)
            save_grid(
                avg,
                rows_for_grid,
                frame_dir / "all_heads_grid.png",
                title=f"{export_type} {frame} | {args.mode} | n={total_samples}",
                vmax=vmax,
            )
            save_individual(
                avg,
                rows_for_grid,
                frame_dir / "heads",
                title_prefix=f"{export_type} {frame}",
                vmax=vmax,
            )

    np.savez_compressed(args.out_dir / "all_head_attention_matrices.npz", **arrays)
    write_csv(args.out_dir / "head_attention_metrics.csv", metric_rows)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "ckpt_path": str(args.ckpt_path),
                "iter_num": int(checkpoint.get("iter_num", -1)),
                "best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
                "dataset": str(args.dataset or checkpoint.get("config", {}).get("dataset")),
                "split": args.split,
                "mode": args.mode,
                "batch_size": int(args.batch_size),
                "num_batches": int(args.num_batches),
                "num_samples_aggregated": int(total_samples),
                "mean_loss": float(loss_sum / max(1, total_samples)),
                "dtype": args.dtype,
                "data_permutation": None
                if data_permutation is None
                else {
                    "permute_mode": data_permutation["permute_mode"],
                    "permute_seed": data_permutation["permute_seed"],
                    "block_perm": data_permutation["block_perm"].tolist(),
                    "inverse_block_perm": data_permutation["inverse_block_perm"].tolist(),
                },
                "frame_definitions": {
                    "current_l2r": "block attention after undoing the sampled reveal order, in the permuted/current input frame",
                    "original_l2r": "current_l2r further reordered by inverse fixed block permutation, i.e. true unpermuted text block order",
                },
                "export_type_definitions": {
                    "with_none": "predictor-aligned attn[:-1, :-1], includes [None] in block 0",
                    "without_none": "real-token-only attn[1:, 1:]",
                },
                "outputs": {
                    "npz": "all_head_attention_matrices.npz",
                    "metrics_csv": "head_attention_metrics.csv",
                    "grid_png": "<export_type>/<frame>/all_heads_grid.png",
                    "individual_pngs": "<export_type>/<frame>/heads/LxHy.png",
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"saved all-head attention maps to {args.out_dir}")


if __name__ == "__main__":
    main()
