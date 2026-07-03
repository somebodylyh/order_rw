import argparse
import json
from contextlib import nullcontext
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
    infer_data_record_mode,
    load_checkpoint,
    load_tokens,
    maybe_apply_data_permutation,
    resolve_data_dir,
    resolve_data_permutation,
    sample_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare L0 layer-mean without_none block attention averages from "
            "the same random checkpoint using one 64-sample batch vs eight "
            "64-sample batches."
        )
    )
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--mode", type=str, default="Random", choices=["Random", "AR"])
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--force_manual_attention", action="store_true")
    return parser.parse_args()


def get_autocast(device: str, dtype: str):
    if "cuda" not in str(device) or str(dtype) == "float32":
        return nullcontext()
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[str(dtype)]
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)


def invert_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), dtype=perm.dtype, device=perm.device)
    return inv


def aggregate_l0_without_none_current(layer_attn, block_orders, block_len: int):
    shifted = layer_attn[:, :, 1:, 1:].detach().float()
    batch, heads, seq_a, seq_b = shifted.shape
    if seq_a != seq_b or seq_a % int(block_len) != 0:
        raise ValueError(f"attention shape {tuple(shifted.shape)} is incompatible with block_len={block_len}")
    num_blocks = seq_a // int(block_len)
    block_batch = shifted.view(batch, heads, num_blocks, int(block_len), num_blocks, int(block_len)).mean(dim=(3, 5))
    total = torch.zeros((num_blocks, num_blocks), dtype=torch.float64)
    for sample_idx in range(batch):
        inverse = invert_permutation(block_orders[sample_idx].detach().cpu().long())
        sample = block_batch[sample_idx].detach().cpu()[:, inverse, :][:, :, inverse].mean(dim=0)
        total += sample.double()
    return total / float(max(1, batch))


def remap_current_to_true_original(matrix, data_permutation):
    if data_permutation is None:
        return None
    block_perm = data_permutation["block_perm"].detach().cpu().long()
    original_to_current = invert_permutation(block_perm)
    return matrix.index_select(0, original_to_current).index_select(1, original_to_current)


def matrix_stats(matrix):
    values = matrix.detach().float().cpu().numpy()
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "abs_mean": float(np.abs(values).mean()),
        "abs_max": float(np.abs(values).max()),
        "diag_mean": float(np.diag(values).mean()),
        "offdiag_mean": float(values[~np.eye(values.shape[0], dtype=bool)].mean()),
    }


def save_triptych(path, first, eighth, title_prefix):
    import matplotlib.pyplot as plt

    first_np = first.detach().float().cpu().numpy()
    eighth_np = eighth.detach().float().cpu().numpy()
    diff_np = eighth_np - first_np
    pos = np.concatenate([first_np[first_np > 0], eighth_np[eighth_np > 0]])
    vmax = float(np.percentile(pos, 99.0)) if pos.size else None
    diff_vmax = float(np.percentile(np.abs(diff_np), 99.0))
    diff_vmax = max(diff_vmax, 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    im0 = axes[0].imshow(first_np, interpolation="nearest", cmap="viridis", vmax=vmax)
    axes[0].set_title("64 samples")
    im1 = axes[1].imshow(eighth_np, interpolation="nearest", cmap="viridis", vmax=vmax)
    axes[1].set_title("512 samples")
    im2 = axes[2].imshow(diff_np, interpolation="nearest", cmap="coolwarm", vmin=-diff_vmax, vmax=diff_vmax)
    axes[2].set_title("512 - 64")
    for ax in axes:
        ax.set_xlabel("key block")
        ax.set_ylabel("query block")
    fig.suptitle(title_prefix)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(args.ckpt_path, args.device)
    model = build_model(checkpoint, args.device, force_manual_attention=bool(args.force_manual_attention))
    data_dir = resolve_data_dir(args, checkpoint)
    data_record_mode = str(checkpoint.get("config", {}).get("data_record_mode", infer_data_record_mode(data_dir)))
    tokens = load_tokens(data_dir, args.split, data_record_mode=data_record_mode)
    data_permutation = resolve_data_permutation(
        checkpoint,
        num_blocks=int(model.num_blocks),
        block_len=int(model.block_order_block_len),
        model=model,
    )
    rng = np.random.default_rng(int(args.seed))
    autocast = get_autocast(args.device, args.dtype)

    current_sum_64 = None
    current_sum_512 = None
    true_sum_64 = None
    true_sum_512 = None
    total_seen = 0
    loss_sum_64 = 0.0
    loss_sum_512 = 0.0
    starts_by_batch = []

    for batch_idx in range(int(args.num_batches)):
        idx, starts = sample_batch(tokens, int(args.batch_size), int(model.config.block_size), rng, args.device)
        starts_by_batch.append([int(v) for v in starts])
        idx = maybe_apply_data_permutation(idx, data_permutation)
        token_orders, block_orders = build_orders(model, idx, str(args.mode))
        with torch.no_grad():
            with autocast:
                outputs = model.forward_fn(idx, token_orders, return_attentions=True)
        _logits, loss, attentions = extract_logits_loss_attentions(outputs)
        layer_attn = attentions[int(args.layer)]
        current = aggregate_l0_without_none_current(layer_attn, block_orders, int(model.block_order_block_len))
        true_original = remap_current_to_true_original(current, data_permutation)

        batch_n = int(idx.size(0))
        if batch_idx == 0:
            current_sum_64 = current * float(batch_n)
            if true_original is not None:
                true_sum_64 = true_original * float(batch_n)
            loss_sum_64 += float(loss.item()) * float(batch_n)

        if current_sum_512 is None:
            current_sum_512 = current * float(batch_n)
            if true_original is not None:
                true_sum_512 = true_original * float(batch_n)
        else:
            current_sum_512 += current * float(batch_n)
            if true_original is not None:
                true_sum_512 += true_original * float(batch_n)
        loss_sum_512 += float(loss.item()) * float(batch_n)
        total_seen += batch_n
        print(f"processed batch {batch_idx + 1}/{args.num_batches}", flush=True)

    if total_seen != int(args.batch_size) * int(args.num_batches):
        raise RuntimeError(f"unexpected sample count total_seen={total_seen}")

    current_64 = current_sum_64 / float(args.batch_size)
    current_512 = current_sum_512 / float(total_seen)
    np.save(args.out_dir / "current_l2r_64.npy", current_64.detach().float().cpu().numpy())
    np.save(args.out_dir / "current_l2r_512.npy", current_512.detach().float().cpu().numpy())
    np.save(args.out_dir / "current_l2r_diff_512_minus_64.npy", (current_512 - current_64).detach().float().cpu().numpy())
    save_triptych(
        args.out_dir / "random10k_l0_layermean_withoutnone_current_l2r_64_vs_512.png",
        current_64,
        current_512,
        "10k random ckpt | L0 layer-mean without_none | current L2R",
    )

    payload = {
        "ckpt_path": str(args.ckpt_path),
        "ckpt_iter_num": int(checkpoint.get("iter_num", -1)),
        "out_dir": str(args.out_dir),
        "dataset": str(args.dataset or checkpoint.get("config", {}).get("dataset")),
        "data_dir": str(data_dir),
        "split": str(args.split),
        "mode": str(args.mode),
        "layer": int(args.layer),
        "head_reduce": "mean",
        "export_type": "without_none",
        "batch_size": int(args.batch_size),
        "num_batches": int(args.num_batches),
        "samples_64": int(args.batch_size),
        "samples_512": int(total_seen),
        "seed": int(args.seed),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "mean_loss_64": float(loss_sum_64 / float(args.batch_size)),
        "mean_loss_512": float(loss_sum_512 / float(total_seen)),
        "current_l2r_64_stats": matrix_stats(current_64),
        "current_l2r_512_stats": matrix_stats(current_512),
        "current_l2r_diff_stats": matrix_stats(current_512 - current_64),
        "batch_start_offsets_preview": starts_by_batch[:2],
    }

    if true_sum_64 is not None and true_sum_512 is not None:
        true_64 = true_sum_64 / float(args.batch_size)
        true_512 = true_sum_512 / float(total_seen)
        np.save(args.out_dir / "true_original_l2r_64.npy", true_64.detach().float().cpu().numpy())
        np.save(args.out_dir / "true_original_l2r_512.npy", true_512.detach().float().cpu().numpy())
        np.save(
            args.out_dir / "true_original_l2r_diff_512_minus_64.npy",
            (true_512 - true_64).detach().float().cpu().numpy(),
        )
        save_triptych(
            args.out_dir / "random10k_l0_layermean_withoutnone_true_original_l2r_64_vs_512.png",
            true_64,
            true_512,
            "10k random ckpt | L0 layer-mean without_none | true original L2R",
        )
        payload.update(
            {
                "true_original_l2r_64_stats": matrix_stats(true_64),
                "true_original_l2r_512_stats": matrix_stats(true_512),
                "true_original_l2r_diff_stats": matrix_stats(true_512 - true_64),
            }
        )

    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
