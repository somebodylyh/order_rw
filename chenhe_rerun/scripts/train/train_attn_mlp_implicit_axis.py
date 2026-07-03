#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from attn_mlp_order_policy import FlatAttentionOrderMLP  # noqa: E402
from attn_mlp_implicit_axis_common import (  # noqa: E402
    FORBIDDEN_SIGNALS,
    append_jsonl,
    build_coefficients_for_matrices,
    collect_attention_matrices,
    evaluate_order_source_nll,
    infer_data_dir,
    infer_data_record_mode,
    implicit_axis_batch_loss,
    load_frozen_aogpt,
    load_permutation_state,
    load_tokens,
    mean_dicts,
    order_from_logits_desc,
    order_stability_summary,
    parse_int_list,
    tensor_metrics_to_floats,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Attn-MLP with implicit attention-axis graph loss.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default="wikitext103")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--num_train_attention_samples", type=int, default=2000)
    parser.add_argument("--num_val_attention_samples", type=int, default=512)
    parser.add_argument("--probe_batch_size", type=int, default=64)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=7)
    parser.add_argument("--export_type", choices=("with_none", "without_none"), default="with_none")
    parser.add_argument("--sym_mode", choices=("max", "mean"), default="max")
    parser.add_argument("--dir_mode", choices=("query_key", "key_query"), default="query_key")
    parser.add_argument("--threshold_percentile", type=float, default=60.0)
    parser.add_argument("--lambda_dir", type=float, default=0.3)
    parser.add_argument("--lambda_var", type=float, default=0.01)
    parser.add_argument("--min_std", type=float, default=0.1)
    parser.add_argument("--tau_d", type=float, default=0.5)
    parser.add_argument("--tau_s", type=float, default=1.0)
    parser.add_argument("--margin_d", type=float, default=0.5)
    parser.add_argument("--min_conf", type=float, default=0.2)
    parser.add_argument("--hidden_dims", type=str, default="1024,1024")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--input_normalization", type=str, default="robust_zscore")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_eval_attention_samples", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--random_eval_orders", type=int, default=8)
    parser.add_argument("--skip_frozen_nll_eval", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--progress_interval", type=int, default=50)
    return parser.parse_args()


def print_progress(text: str):
    print(text, flush=True)


def prefixed(prefix: str, metrics: dict):
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


@torch.no_grad()
def evaluate_objective(policy, matrices, coeffs, args, device):
    policy.eval()
    rows = []
    orders = []
    loss_kwargs = make_loss_kwargs(args)
    for start in range(0, int(matrices.size(0)), int(args.batch_size)):
        end = min(int(matrices.size(0)), start + int(args.batch_size))
        batch = matrices[start:end].to(device=device, dtype=torch.float32)
        logits = policy(batch)
        metrics = implicit_axis_batch_loss(logits, coeffs[start:end], **loss_kwargs)
        rows.append(tensor_metrics_to_floats(metrics))
        orders.extend([order_from_logits_desc(row).detach().cpu() for row in logits])
    summary = mean_dicts(rows)
    summary.update(order_stability_summary(orders))
    return summary


def make_loss_kwargs(args):
    return {
        "lambda_dir": float(args.lambda_dir),
        "lambda_var": float(args.lambda_var),
        "min_std": float(args.min_std),
        "tau_d": float(args.tau_d),
        "tau_s": float(args.tau_s),
        "margin_d": float(args.margin_d),
        "min_conf": float(args.min_conf),
    }


def train_one_epoch(policy, optimizer, matrices, coeffs, args, device):
    policy.train()
    loss_kwargs = make_loss_kwargs(args)
    order = torch.randperm(int(matrices.size(0)))
    rows = []
    for start in range(0, int(order.numel()), int(args.batch_size)):
        batch_idx = order[start : start + int(args.batch_size)]
        batch = matrices.index_select(0, batch_idx).to(device=device, dtype=torch.float32)
        coeff_batch = [coeffs[int(i)] for i in batch_idx.tolist()]
        optimizer.zero_grad(set_to_none=True)
        logits = policy(batch)
        metrics = implicit_axis_batch_loss(logits, coeff_batch, **loss_kwargs)
        metrics["loss_total"].backward()
        if float(args.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(args.grad_clip))
        optimizer.step()
        rows.append(tensor_metrics_to_floats(metrics))
    return mean_dicts(rows)


def make_report_readme(args, summary):
    return f"""# Attn-MLP Implicit Attention Axis

This report is for the frozen random-checkpoint implicit attention-axis Attn-MLP experiment.

- checkpoint: `{args.ckpt_path}`
- policy: `{summary.get("policy_path")}`
- frame: `current`
- loss: `L_smooth + lambda_dir * L_dir + lambda_var * L_var`
- sym_mode: `{args.sym_mode}`
- dir_mode: `{args.dir_mode}`
- train attention samples: `{args.num_train_attention_samples}`
- val attention samples: `{args.num_val_attention_samples}`
- tau/original diagnostics: logged only, not used for training, model selection, direction selection, or early stop

Key files:

- `metrics.csv`
- `train_log.jsonl`
- `eval_summary.json`
- `README.md`
"""


def main():
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np_rng = np.random.default_rng(int(args.seed))
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = args.report_dir
    if report_dir is None:
        report_dir = REPO_ROOT / "Report" / "attn_mlp_implicit_axis" / "seq256_perm_b64_l0h7_withnone"
    report_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_frozen_aogpt(args.ckpt_path, device)
    data_dir = infer_data_dir(args.dataset, args.data_dir)
    train_tokens = load_tokens(data_dir, args.split)
    val_tokens = load_tokens(data_dir, args.val_split)
    data_record_mode = infer_data_record_mode(checkpoint)
    permutation_state = load_permutation_state(
        checkpoint,
        num_blocks=int(model.num_blocks),
        block_len=int(model.block_order_block_len),
    )
    token_perm = None if permutation_state is None else permutation_state["token_perm"]

    print_progress(f"loaded_ckpt={args.ckpt_path}")
    print_progress(f"collecting_train_attention_samples={args.num_train_attention_samples}")
    train_matrices, train_probe_orders = collect_attention_matrices(
        model=model,
        tokens=train_tokens,
        num_samples=int(args.num_train_attention_samples),
        probe_batch_size=int(args.probe_batch_size),
        layer=int(args.layer),
        head=int(args.head),
        export_type=args.export_type,
        rng=np_rng,
        device=device,
        dtype=args.dtype,
        token_perm=token_perm,
        data_record_mode=data_record_mode,
        progress=print_progress,
        progress_interval=max(1, int(args.progress_interval)),
    )
    print_progress(f"collecting_val_attention_samples={args.num_val_attention_samples}")
    val_matrices, val_probe_orders = collect_attention_matrices(
        model=model,
        tokens=val_tokens,
        num_samples=int(args.num_val_attention_samples),
        probe_batch_size=int(args.probe_batch_size),
        layer=int(args.layer),
        head=int(args.head),
        export_type=args.export_type,
        rng=np_rng,
        device=device,
        dtype=args.dtype,
        token_perm=token_perm,
        data_record_mode=data_record_mode,
        progress=print_progress,
        progress_interval=max(1, int(args.progress_interval)),
    )
    train_coeffs = build_coefficients_for_matrices(
        train_matrices,
        sym_mode=args.sym_mode,
        dir_mode=args.dir_mode,
        threshold_percentile=float(args.threshold_percentile),
    )
    val_coeffs = build_coefficients_for_matrices(
        val_matrices,
        sym_mode=args.sym_mode,
        dir_mode=args.dir_mode,
        threshold_percentile=float(args.threshold_percentile),
    )
    torch.save(
        {
            "train_matrices": train_matrices,
            "val_matrices": val_matrices,
            "meta": {
                "ckpt_path": str(args.ckpt_path),
                "layer": int(args.layer),
                "head": int(args.head),
                "export_type": args.export_type,
                "sym_mode": args.sym_mode,
                "dir_mode": args.dir_mode,
                "threshold_percentile": float(args.threshold_percentile),
            },
        },
        args.out_dir / "attention_dataset.pt",
    )

    hidden_dims = parse_int_list(args.hidden_dims, default=[1024, 1024])
    policy_config = {
        "num_blocks": int(model.num_blocks),
        "hidden_dims": hidden_dims,
        "dropout": float(args.dropout),
        "activation": args.activation,
        "input_normalization": args.input_normalization,
    }
    policy = FlatAttentionOrderMLP(**policy_config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    train_log_path = args.out_dir / "train_log.jsonl"
    report_train_log_path = report_dir / "train_log.jsonl"
    for path in (train_log_path, report_train_log_path):
        if path.exists():
            path.unlink()

    metrics_rows = []
    init_train = evaluate_objective(policy, train_matrices, train_coeffs, args, device)
    init_val = evaluate_objective(policy, val_matrices, val_coeffs, args, device)
    init_row = {"epoch": 0, "phase": "initial"}
    init_row.update(prefixed("train", init_train))
    init_row.update(prefixed("val", init_val))
    metrics_rows.append(init_row)
    append_jsonl(train_log_path, init_row)
    append_jsonl(report_train_log_path, init_row)
    print_progress(json.dumps(init_row))

    best_val_objective = float(init_val.get("loss_total", float("inf")))
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(policy, optimizer, train_matrices, train_coeffs, args, device)
        val_metrics = evaluate_objective(policy, val_matrices, val_coeffs, args, device)
        row = {"epoch": epoch, "phase": "train"}
        row.update(prefixed("train", train_metrics))
        row.update(prefixed("val", val_metrics))
        metrics_rows.append(row)
        append_jsonl(train_log_path, row)
        append_jsonl(report_train_log_path, row)
        print_progress(json.dumps(row))
        val_objective = float(val_metrics.get("loss_total", float("inf")))
        if val_objective < best_val_objective:
            best_val_objective = val_objective
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}

    policy.load_state_dict(best_state, strict=True)
    final_train = evaluate_objective(policy, train_matrices, train_coeffs, args, device)
    final_val = evaluate_objective(policy, val_matrices, val_coeffs, args, device)
    eval_summary = {}
    eval_rows = []
    if not args.skip_frozen_nll_eval and int(args.num_eval_attention_samples) > 0:
        n_eval = min(int(args.num_eval_attention_samples), int(val_matrices.size(0)))

        def order_source(matrix: torch.Tensor, _coeff: dict):
            with torch.no_grad():
                logits = policy(matrix.unsqueeze(0).to(device=device, dtype=torch.float32)).squeeze(0).detach().cpu()
            return logits, order_from_logits_desc(logits)

        print_progress(f"running_frozen_nll_eval_samples={n_eval}")
        eval_rows, eval_summary = evaluate_order_source_nll(
            model=model,
            tokens=val_tokens,
            matrices=val_matrices[:n_eval],
            coeffs=val_coeffs[:n_eval],
            order_source=order_source,
            rng=np_rng,
            device=device,
            dtype=args.dtype,
            token_perm=token_perm,
            data_record_mode=data_record_mode,
            eval_batch_size=int(args.eval_batch_size),
            random_eval_orders=int(args.random_eval_orders),
            permutation_state=permutation_state,
            loss_kwargs=make_loss_kwargs(args),
            progress=print_progress,
            progress_interval=max(1, int(args.progress_interval)),
        )
        write_csv(args.out_dir / "eval_rows.csv", eval_rows)
        write_csv(report_dir / "eval_rows.csv", eval_rows)

    policy_path = args.out_dir / "policy.pt"
    summary = {
        "command": " ".join(sys.argv),
        "script": "scripts/train/train_attn_mlp_implicit_axis.py",
        "policy_path": str(policy_path),
        "report_dir": str(report_dir),
        "run_meta": {
            "ckpt_path": str(args.ckpt_path),
            "checkpoint_iter": checkpoint.get("iter_num"),
            "checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
            "dataset": args.dataset,
            "data_dir": str(data_dir),
            "train_split": args.split,
            "val_split": args.val_split,
            "permute_data": permutation_state is not None,
            "block_size": int(model.config.block_size),
            "block_order_block_len": int(model.block_order_block_len),
            "num_blocks": int(model.num_blocks),
            "frame": "current",
            "forbidden_signals": FORBIDDEN_SIGNALS,
            "block_perm_first16_diagnostic_only": None
            if permutation_state is None
            else [int(v) for v in permutation_state["block_perm"].tolist()[:16]],
        },
        "config": policy_config,
        "policy_type": "implicit_attention_axis_mlp",
        "target_direction": "larger_logit_reveals_earlier",
        "training_meta": {
            "ckpt_path": str(args.ckpt_path),
            "frame": "current",
            "layer": int(args.layer),
            "head": int(args.head),
            "export_type": args.export_type,
            "loss": "L_smooth + lambda_dir * L_dir + lambda_var * L_var",
            "sym_mode": args.sym_mode,
            "dir_mode": args.dir_mode,
            "threshold_percentile": float(args.threshold_percentile),
            "lambda_dir": float(args.lambda_dir),
            "lambda_var": float(args.lambda_var),
            "min_std": float(args.min_std),
            "tau_d": float(args.tau_d),
            "tau_s": float(args.tau_s),
            "margin_d": float(args.margin_d),
            "min_conf": float(args.min_conf),
            "forbidden_signals": FORBIDDEN_SIGNALS,
            "tau_is_diagnostic_only": True,
            "original_l2r_is_diagnostic_only": True,
        },
        "samples": {
            "train_attention_samples": int(args.num_train_attention_samples),
            "val_attention_samples": int(args.num_val_attention_samples),
            "probe_batch_size": int(args.probe_batch_size),
            "train_random_probe_block_orders_first": train_probe_orders,
            "val_random_probe_block_orders_first": val_probe_orders,
        },
        "optimization": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "best_epoch_by_val_implicit_objective": int(best_epoch),
            "initial_train": init_train,
            "initial_val": init_val,
            "final_train": final_train,
            "final_val": final_val,
            "val_objective_drop": float(init_val.get("loss_total", float("nan")))
            - float(final_val.get("loss_total", float("nan"))),
        },
        "metrics": {
            "train": final_train,
            "val": final_val,
            "eval": eval_summary,
        },
    }
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "config": policy_config,
            "policy_type": "implicit_attention_axis_mlp",
            "target_direction": "larger_logit_reveals_earlier",
            "training_meta": summary["training_meta"],
            "metrics": summary["metrics"],
        },
        policy_path,
    )
    write_csv(args.out_dir / "metrics.csv", metrics_rows)
    write_csv(report_dir / "metrics.csv", metrics_rows)
    write_json(args.out_dir / "summary.json", summary)
    write_json(report_dir / "eval_summary.json", summary)
    (report_dir / "README.md").write_text(make_report_readme(args, summary), encoding="utf-8")
    if train_log_path.resolve() != report_train_log_path.resolve():
        shutil.copyfile(train_log_path, report_train_log_path)
    print_progress(f"policy_path={policy_path}")
    print_progress(f"report_dir={report_dir}")
    print_progress(json.dumps({"final_val": final_val, "eval": eval_summary}, indent=2))


if __name__ == "__main__":
    main()
