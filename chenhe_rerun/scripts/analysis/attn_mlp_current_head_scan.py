from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.attn_mlp_implicit_axis_common import (  # noqa: E402
    aggregate_layerhead_attention_to_current_blocks,
    append_jsonl,
    autocast_context,
    expand_orders_for_model,
    infer_data_dir,
    infer_data_record_mode,
    load_frozen_aogpt,
    load_permutation_state,
    load_tokens,
    random_block_orders,
    robust_z_offdiag,
    sample_batch,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--pair_dataset", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="wikitext103")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--num_train_samples", type=int, default=2000)
    parser.add_argument("--num_val_samples", type=int, default=512)
    parser.add_argument("--probe_batch_size", type=int, default=64)
    parser.add_argument("--target_sign", type=float, default=-1.0)
    parser.add_argument("--export_type", type=str, default="with_none")
    parser.add_argument("--min_pair_conf", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3456)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--progress_interval", type=int, default=100)
    return parser.parse_args()


def weighted_corr(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
    mask = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(w) & (w > 0)
    if not bool(mask.any()):
        return float("nan")
    x = x[mask].float()
    y = y[mask].float()
    w = w[mask].float()
    w = w / w.sum().clamp_min(1e-8)
    xm = (w * x).sum()
    ym = (w * y).sum()
    xv = x - xm
    yv = y - ym
    cov = (w * xv * yv).sum()
    den = torch.sqrt((w * xv.square()).sum().clamp_min(1e-12) * (w * yv.square()).sum().clamp_min(1e-12))
    return float((cov / den).item())


def weighted_agreement(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
    mask = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(w) & (w > 0) & (y != 0)
    if not bool(mask.any()):
        return float("nan")
    x = x[mask].float()
    y = y[mask].float()
    w = w[mask].float()
    agree = (torch.sign(x) == torch.sign(y)).float()
    return float((agree * w).sum().div(w.sum().clamp_min(1e-8)).item())


def score_layer_heads(layer_heads: torch.Tensor, pref: torch.Tensor, weights: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor):
    rows = []
    layer_heads = layer_heads.detach().float().cpu()
    for head_idx in range(int(layer_heads.size(0))):
        z = robust_z_offdiag(layer_heads[head_idx])
        anti = z - z.t()
        values = anti[ii, jj]
        rows.append(
            {
                "corr": weighted_corr(values, pref, weights),
                "agreement": weighted_agreement(values, pref, weights),
                "attn_abs_mean": float(values.abs().mean().item()),
            }
        )
    return rows


@torch.no_grad()
def scan_split(
    *,
    split_name: str,
    tokens,
    num_samples: int,
    model,
    pair_delta: torch.Tensor,
    pair_conf: torch.Tensor,
    target_sign: float,
    rng: np.random.Generator,
    device: torch.device,
    dtype: str,
    token_perm: torch.Tensor | None,
    data_record_mode: str,
    probe_batch_size: int,
    export_type: str,
    min_pair_conf: float,
    progress_interval: int,
    log_path: Path,
):
    n = int(pair_delta.size(0))
    ii, jj = torch.triu_indices(n, n, offset=1)
    pref = (-float(target_sign) * pair_delta.float())[ii, jj]
    weights = pair_conf.float()[ii, jj]
    weights = torch.where(weights >= float(min_pair_conf), weights, torch.zeros_like(weights))
    accum: dict[tuple[int, int], dict[str, float]] = {}
    counts: dict[tuple[int, int], int] = {}
    first_orders = []
    for sample_idx in range(int(num_samples)):
        x = sample_batch(
            tokens,
            batch_size=int(probe_batch_size),
            block_size=int(model.config.block_size),
            rng=rng,
            device=device,
            token_perm=token_perm,
            data_record_mode=data_record_mode,
        )
        block_orders = random_block_orders(int(probe_batch_size), int(model.num_blocks), device)
        token_orders = expand_orders_for_model(model, block_orders)
        with autocast_context(device, dtype):
            outputs = model(
                x,
                mode=None,
                orders=token_orders,
                return_attentions=True,
                return_logits=False,
            )
        attentions = outputs[-1]
        for layer_idx, layer_attn in enumerate(attentions):
            layer_heads = aggregate_layerhead_attention_to_current_blocks(
                layer_attn.detach(),
                block_orders,
                block_len=int(model.block_order_block_len),
                export_type=export_type,
            )
            head_rows = score_layer_heads(layer_heads, pref, weights, ii, jj)
            for head_idx, row in enumerate(head_rows):
                key = (int(layer_idx), int(head_idx))
                if key not in accum:
                    accum[key] = {"corr": 0.0, "abs_corr": 0.0, "agreement": 0.0, "attn_abs_mean": 0.0}
                    counts[key] = 0
                corr = float(row["corr"])
                accum[key]["corr"] += corr
                accum[key]["abs_corr"] += abs(corr)
                accum[key]["agreement"] += float(row["agreement"])
                accum[key]["attn_abs_mean"] += float(row["attn_abs_mean"])
                counts[key] += 1
        if len(first_orders) < 4:
            first_orders.append(block_orders[0].detach().cpu().tolist())
        if (sample_idx + 1) % max(1, int(progress_interval)) == 0 or sample_idx + 1 == int(num_samples):
            message = f"{split_name}_head_scan_samples={sample_idx + 1}/{num_samples}"
            print(message, flush=True)
            append_jsonl(log_path, {"event": message})
    rows = []
    for key, values in sorted(accum.items()):
        count = max(1, counts[key])
        layer_idx, head_idx = key
        rows.append(
            {
                "split": split_name,
                "layer": layer_idx,
                "head": head_idx,
                "samples": int(num_samples),
                "probe_batch_size": int(probe_batch_size),
                "target_sign": float(target_sign),
                "corr_mean": values["corr"] / count,
                "abs_corr_mean": values["abs_corr"] / count,
                "agreement_mean": values["agreement"] / count,
                "attn_abs_mean": values["attn_abs_mean"] / count,
            }
        )
    return rows, first_orders


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "progress.jsonl"
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    device = torch.device(args.device)
    model, checkpoint = load_frozen_aogpt(args.ckpt_path, device)
    data_dir = infer_data_dir(args.dataset, args.data_dir)
    train_tokens = load_tokens(data_dir, args.train_split)
    val_tokens = load_tokens(data_dir, args.val_split)
    data_record_mode = infer_data_record_mode(checkpoint)
    permutation_state = load_permutation_state(
        checkpoint,
        num_blocks=int(model.num_blocks),
        block_len=int(model.block_order_block_len),
    )
    token_perm = None if permutation_state is None else permutation_state["token_perm"]
    pair_payload = torch.load(args.pair_dataset, map_location="cpu")
    pair_delta = pair_payload["pair_delta"].float()
    pair_conf = pair_payload["pair_conf"].float()
    config = vars(args).copy()
    config.update(
        {
            "frame": "current",
            "selection_signal": "train split weighted abs corr between current-frame attention antisymmetry and current-frame pair loss-drop preference",
            "forbidden_signals_not_used": [
                "original_l2r",
                "original_tau",
                "OriginalL2R_ppl",
                "validation_ppl_for_policy_selection",
                "generated_teacher_order_supervision",
            ],
        }
    )
    write_json(args.out_dir / "config.json", config)
    (args.out_dir / "commands.sh").write_text("CUDA_VISIBLE_DEVICES=1 python " + " ".join(sys.argv) + "\n", encoding="utf-8")
    train_rows, train_orders = scan_split(
        split_name="train",
        tokens=train_tokens,
        num_samples=int(args.num_train_samples),
        model=model,
        pair_delta=pair_delta,
        pair_conf=pair_conf,
        target_sign=float(args.target_sign),
        rng=rng,
        device=device,
        dtype=args.dtype,
        token_perm=token_perm,
        data_record_mode=data_record_mode,
        probe_batch_size=int(args.probe_batch_size),
        export_type=args.export_type,
        min_pair_conf=float(args.min_pair_conf),
        progress_interval=int(args.progress_interval),
        log_path=log_path,
    )
    val_rows, val_orders = scan_split(
        split_name="val",
        tokens=val_tokens,
        num_samples=int(args.num_val_samples),
        model=model,
        pair_delta=pair_delta,
        pair_conf=pair_conf,
        target_sign=float(args.target_sign),
        rng=rng,
        device=device,
        dtype=args.dtype,
        token_perm=token_perm,
        data_record_mode=data_record_mode,
        probe_batch_size=int(args.probe_batch_size),
        export_type=args.export_type,
        min_pair_conf=float(args.min_pair_conf),
        progress_interval=int(args.progress_interval),
        log_path=log_path,
    )
    rows = train_rows + val_rows
    write_csv(args.out_dir / "head_scan_summary.csv", rows)
    train_best = max(train_rows, key=lambda row: (float(row["abs_corr_mean"]), float(row["agreement_mean"])))
    selected = {
        "selected_layer": int(train_best["layer"]),
        "selected_head": int(train_best["head"]),
        "selection_split": "train",
        "selection_metric": "abs_corr_mean",
        "selected_train_row": train_best,
        "matching_val_row": next(
            (
                row
                for row in val_rows
                if int(row["layer"]) == int(train_best["layer"]) and int(row["head"]) == int(train_best["head"])
            ),
            None,
        ),
        "train_random_probe_orders_first": train_orders,
        "val_random_probe_orders_first": val_orders,
    }
    write_json(args.out_dir / "selection.json", selected)
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()
