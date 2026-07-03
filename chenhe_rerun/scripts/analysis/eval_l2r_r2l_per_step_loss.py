"""
Evaluate a trained language checkpoint under true-original L2R and R2L orders.

This is a narrow diagnostic for permuted-data AO-GPT checkpoints. For a
permuted checkpoint, OriginalL2R means revealing original blocks 0..N-1 after
mapping them into the checkpoint's current/permuted frame; OriginalR2L is the
reverse of that same mapped order.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "scripts" / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from eval_lm_original_order_ppl import (  # noqa: E402
    batched,
    expand_orders,
    get_autocast,
    infer_data_record_mode,
    iter_starts,
    load_checkpoint,
    load_model,
    load_tokens,
    resolve_data_dir,
    resolve_permutation_state,
)


ORDER_NAMES = ("OriginalL2R", "OriginalR2L")


def build_original_orders(model, permutation_state, batch_size: int):
    device = next(model.parameters()).device
    if permutation_state is None:
        l2r = torch.arange(model.num_blocks, device=device, dtype=torch.long)
        source = "current_frame_l2r_no_data_permutation"
    else:
        l2r = permutation_state["inverse_block_perm"].to(device=device, dtype=torch.long)
        source = "original_l2r_mapped_to_current_frame"
    r2l = torch.flip(l2r, dims=[0])
    return {
        "OriginalL2R": l2r.unsqueeze(0).expand(batch_size, -1),
        "OriginalR2L": r2l.unsqueeze(0).expand(batch_size, -1),
    }, {
        "OriginalL2R": source,
        "OriginalR2L": source + "_reversed",
    }


def as_float_list(values):
    out = []
    for value in values:
        value = float(value)
        out.append(value if math.isfinite(value) else None)
    return out


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_plot(block_csv: Path, token_csv: Path, block_png: Path, token_png: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local env
        return {"enabled": False, "error": repr(exc)}

    def load_csv(path):
        with path.open() as handle:
            return list(csv.DictReader(handle))

    def plot_rows(rows, step_key, png_path, title, xlabel):
        steps = [int(row[step_key]) for row in rows]
        l2r = [float(row["OriginalL2R_nll"]) for row in rows]
        r2l = [float(row["OriginalR2L_nll"]) for row in rows]
        delta = [float(row["delta_r2l_minus_l2r"]) for row in rows]

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[3, 1])
        axes[0].plot(steps, l2r, label="OriginalL2R", linewidth=1.8)
        axes[0].plot(steps, r2l, label="OriginalR2L", linewidth=1.8)
        axes[0].set_ylabel("Mean NLL")
        axes[0].set_title(title)
        axes[0].grid(True, alpha=0.25)
        axes[0].legend()

        axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axes[1].plot(steps, delta, color="tab:red", linewidth=1.4)
        axes[1].set_ylabel("R2L - L2R")
        axes[1].set_xlabel(xlabel)
        axes[1].grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(png_path, dpi=180)
        plt.close(fig)

    plot_rows(
        load_csv(block_csv),
        "reveal_block_step",
        block_png,
        "WikiText103 random checkpoint: per-block reveal loss",
        "Reveal block step",
    )
    plot_rows(
        load_csv(token_csv),
        "reveal_token_step",
        token_png,
        "WikiText103 random checkpoint: per-token reveal loss",
        "Reveal token step",
    )
    return {"enabled": True}


@torch.no_grad()
def evaluate(args):
    ckpt_path = Path(args.ckpt_path)
    checkpoint = load_checkpoint(ckpt_path)
    model = load_model(checkpoint, args.device)
    block_size = int(args.block_size or checkpoint["model_args"]["block_size"])
    data_dir = resolve_data_dir(args, checkpoint)
    data_record_mode = infer_data_record_mode(data_dir, checkpoint)
    tokens = load_tokens(data_dir, args.split)
    permutation_state = resolve_permutation_state(checkpoint, model, block_size)
    token_perm = None if permutation_state is None else permutation_state["token_perm"]
    token_perm_device = None if token_perm is None else token_perm.to(args.device, dtype=torch.long)

    starts = iter_starts(
        len(tokens),
        block_size,
        data_record_mode,
        args.sample_mode,
        int(args.num_samples),
        int(args.seed),
    )
    if len(starts) != int(args.num_samples):
        raise ValueError(f"Expected {args.num_samples} samples, got {len(starts)}")

    block_len = int(model.block_order_block_len)
    num_blocks = int(model.num_blocks)
    if block_size != num_blocks * block_len:
        raise ValueError(f"block_size={block_size} does not equal num_blocks*block_len={num_blocks * block_len}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = get_autocast(args.device, args.dtype)

    stats = {}
    first_orders = {}
    order_sources = {}
    for name in ORDER_NAMES:
        stats[name] = {
            "total_nll": 0.0,
            "total_tokens": 0,
            "token_sum": torch.zeros(block_size, dtype=torch.float64),
            "token_count": torch.zeros(block_size, dtype=torch.float64),
            "block_sum": torch.zeros(num_blocks, dtype=torch.float64),
            "block_count": torch.zeros(num_blocks, dtype=torch.float64),
        }

    total_samples = 0
    progress_every = max(1, int(args.progress_every))
    for batch_index, start_batch in enumerate(batched(starts, int(args.batch_size)), start=1):
        batch_np = np.stack([np.asarray(tokens[s : s + block_size], dtype=np.int64) for s in start_batch], axis=0)
        batch_original = torch.from_numpy(batch_np).to(args.device, non_blocking=True)
        if token_perm_device is not None:
            batch_current = batch_original.index_select(1, token_perm_device)
        else:
            batch_current = batch_original

        block_orders_by_name, source_by_name = build_original_orders(model, permutation_state, batch_current.size(0))
        for name, block_orders in block_orders_by_name.items():
            if name not in first_orders:
                first_orders[name] = [int(v) for v in block_orders[0].detach().cpu().tolist()]
                order_sources[name] = source_by_name[name]
            token_orders = expand_orders(model, block_orders)
            with ctx:
                _, _, token_losses_reveal = model(
                    batch_current,
                    mode=None,
                    orders=token_orders,
                    return_token_loss=True,
                    return_logits=False,
                )

            token_losses = token_losses_reveal.detach().double()
            batch_size = int(token_losses.size(0))
            stats[name]["total_nll"] += float(token_losses.sum().item())
            stats[name]["total_tokens"] += int(token_losses.numel())
            stats[name]["token_sum"] += token_losses.sum(dim=0).detach().cpu()
            stats[name]["token_count"] += torch.full((block_size,), float(batch_size), dtype=torch.float64)
            block_losses = token_losses.view(batch_size, num_blocks, block_len)
            stats[name]["block_sum"] += block_losses.sum(dim=(0, 2)).detach().cpu()
            stats[name]["block_count"] += torch.full((num_blocks,), float(batch_size * block_len), dtype=torch.float64)

        total_samples += int(batch_current.size(0))
        if args.verbose and (batch_index % progress_every == 0 or total_samples == len(starts)):
            print(f"processed {total_samples}/{len(starts)} samples", flush=True)

    results = {}
    for name in ORDER_NAMES:
        mean_nll = stats[name]["total_nll"] / max(1, stats[name]["total_tokens"])
        results[name] = {
            "mean_nll": float(mean_nll),
            "ppl": float(math.exp(mean_nll)),
            "num_scored_tokens": int(stats[name]["total_tokens"]),
            "block_loss_reveal_mean": as_float_list(
                (stats[name]["block_sum"] / stats[name]["block_count"].clamp_min(1.0)).tolist()
            ),
            "token_loss_reveal_mean": as_float_list(
                (stats[name]["token_sum"] / stats[name]["token_count"].clamp_min(1.0)).tolist()
            ),
        }

    block_rows = []
    for step in range(num_blocks):
        l2r = float(results["OriginalL2R"]["block_loss_reveal_mean"][step])
        r2l = float(results["OriginalR2L"]["block_loss_reveal_mean"][step])
        block_rows.append(
            {
                "reveal_block_step": step,
                "OriginalL2R_nll": l2r,
                "OriginalR2L_nll": r2l,
                "OriginalL2R_ppl": math.exp(l2r),
                "OriginalR2L_ppl": math.exp(r2l),
                "delta_r2l_minus_l2r": r2l - l2r,
            }
        )

    token_rows = []
    for step in range(block_size):
        l2r = float(results["OriginalL2R"]["token_loss_reveal_mean"][step])
        r2l = float(results["OriginalR2L"]["token_loss_reveal_mean"][step])
        token_rows.append(
            {
                "reveal_token_step": step,
                "OriginalL2R_nll": l2r,
                "OriginalR2L_nll": r2l,
                "OriginalL2R_ppl": math.exp(l2r),
                "OriginalR2L_ppl": math.exp(r2l),
                "delta_r2l_minus_l2r": r2l - l2r,
            }
        )

    block_csv = out_dir / "per_block_step_loss.csv"
    token_csv = out_dir / "per_token_step_loss.csv"
    write_csv(
        block_csv,
        block_rows,
        [
            "reveal_block_step",
            "OriginalL2R_nll",
            "OriginalR2L_nll",
            "OriginalL2R_ppl",
            "OriginalR2L_ppl",
            "delta_r2l_minus_l2r",
        ],
    )
    write_csv(
        token_csv,
        token_rows,
        [
            "reveal_token_step",
            "OriginalL2R_nll",
            "OriginalR2L_nll",
            "OriginalL2R_ppl",
            "OriginalR2L_ppl",
            "delta_r2l_minus_l2r",
        ],
    )

    block_png = out_dir / "per_block_step_loss.png"
    token_png = out_dir / "per_token_step_loss.png"
    plot_info = maybe_plot(block_csv, token_csv, block_png, token_png)

    summary = {
        "ckpt_path": str(ckpt_path),
        "checkpoint_iter": int(checkpoint.get("iter_num", -1)),
        "checkpoint_best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "dataset": str(args.dataset or checkpoint.get("config", {}).get("dataset")),
        "data_dir": str(data_dir),
        "split": str(args.split),
        "data_record_mode": str(data_record_mode),
        "sample_mode": str(args.sample_mode),
        "num_samples": int(total_samples),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "block_size": int(block_size),
        "block_order_block_len": int(block_len),
        "num_blocks": int(num_blocks),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "frame_mapping": {
            "input_batch_frame": "current_permuted" if permutation_state is not None else "original",
            "OriginalL2R": "true original block order 0..N-1 mapped to current frame",
            "OriginalR2L": "true original block order N-1..0 mapped to current frame",
            "data_permutation_applied": bool(permutation_state is not None),
        },
        "order_sources": order_sources,
        "first_block_orders_current_frame": first_orders,
        "results": results,
        "delta": {
            "mean_nll_r2l_minus_l2r": float(results["OriginalR2L"]["mean_nll"] - results["OriginalL2R"]["mean_nll"]),
            "ppl_r2l_minus_l2r": float(results["OriginalR2L"]["ppl"] - results["OriginalL2R"]["ppl"]),
        },
        "files": {
            "per_block_step_loss_csv": str(block_csv),
            "per_token_step_loss_csv": str(token_csv),
            "per_block_step_loss_png": str(block_png) if plot_info.get("enabled") else None,
            "per_token_step_loss_png": str(token_png) if plot_info.get("enabled") else None,
        },
        "plot": plot_info,
    }
    if permutation_state is not None:
        summary["data_permutation"] = {
            "block_perm_first16": [int(v) for v in permutation_state["block_perm"][:16].tolist()],
            "inverse_block_perm_first16": [int(v) for v in permutation_state["inverse_block_perm"][:16].tolist()],
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    readme = f"""# Original L2R vs R2L per-step loss

Checkpoint: `{ckpt_path}`
Split: `{args.split}`
Samples: `{total_samples}`
Scored tokens per order: `{results["OriginalL2R"]["num_scored_tokens"]}`

| order | mean NLL | PPL |
| --- | ---: | ---: |
| OriginalL2R | {results["OriginalL2R"]["mean_nll"]:.6f} | {results["OriginalL2R"]["ppl"]:.6f} |
| OriginalR2L | {results["OriginalR2L"]["mean_nll"]:.6f} | {results["OriginalR2L"]["ppl"]:.6f} |

Delta R2L - L2R mean NLL: `{summary["delta"]["mean_nll_r2l_minus_l2r"]:.6f}`

Files:
- `summary.json`
- `per_block_step_loss.csv`
- `per_block_step_loss.png`
- `per_token_step_loss.csv`
- `per_token_step_loss.png`

This is a diagnostic run only. The orders are not used as optimization targets.
"""
    (out_dir / "README.md").write_text(readme)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "num_samples": int(total_samples),
                "OriginalL2R": {
                    "mean_nll": results["OriginalL2R"]["mean_nll"],
                    "ppl": results["OriginalL2R"]["ppl"],
                },
                "OriginalR2L": {
                    "mean_nll": results["OriginalR2L"]["mean_nll"],
                    "ppl": results["OriginalR2L"]["ppl"],
                },
                "delta_mean_nll_r2l_minus_l2r": summary["delta"]["mean_nll_r2l_minus_l2r"],
            },
            indent=2,
        )
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Compare original L2R and R2L PPL plus per-step loss.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=2048)
    parser.add_argument("--sample_mode", choices=("random", "sequential"), default="random")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
    )
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
