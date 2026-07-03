#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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

from attn_mlp_implicit_axis_common import (  # noqa: E402
    FORBIDDEN_SIGNALS,
    append_jsonl,
    build_coefficients_for_matrices,
    collect_attention_matrices,
    evaluate_order_source_nll,
    infer_data_dir,
    infer_data_record_mode,
    implicit_axis_loss,
    load_frozen_aogpt,
    load_permutation_state,
    load_tokens,
    mean_dicts,
    order_from_logits_desc,
    tensor_metrics_to_floats,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Direct free-vector sanity check for implicit attention-axis loss.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="wikitext103")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_attention_samples", type=int, default=32)
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
    parser.add_argument("--opt_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--random_eval_orders", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--progress_interval", type=int, default=10)
    return parser.parse_args()


def print_progress(text: str):
    print(text, flush=True)


def optimize_one(coeffs: dict, sample_id: int, args, device: torch.device):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + int(sample_id) * 1009)
    num_blocks = int(coeffs["W"].size(0))
    logits = torch.randn(num_blocks, generator=generator, device=device, dtype=torch.float32) * 0.01
    logits.requires_grad_(True)
    optimizer = torch.optim.AdamW([logits], lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_kwargs = {
        "lambda_dir": float(args.lambda_dir),
        "lambda_var": float(args.lambda_var),
        "min_std": float(args.min_std),
        "tau_d": float(args.tau_d),
        "tau_s": float(args.tau_s),
        "margin_d": float(args.margin_d),
        "min_conf": float(args.min_conf),
    }
    initial = tensor_metrics_to_floats(implicit_axis_loss(logits, coeffs, **loss_kwargs))
    last = initial
    for step in range(int(args.opt_steps)):
        optimizer.zero_grad(set_to_none=True)
        metrics = implicit_axis_loss(logits, coeffs, **loss_kwargs)
        metrics["loss_total"].backward()
        if float(args.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_([logits], float(args.grad_clip))
        optimizer.step()
        last = tensor_metrics_to_floats(metrics)
    final = tensor_metrics_to_floats(implicit_axis_loss(logits, coeffs, **loss_kwargs))
    order = order_from_logits_desc(logits)
    row = {
        "sample_id": int(sample_id),
        "initial_loss_total": initial["loss_total"],
        "final_loss_total": final["loss_total"],
        "loss_drop": initial["loss_total"] - final["loss_total"],
        "initial_loss_smooth": initial["loss_smooth"],
        "final_loss_smooth": final["loss_smooth"],
        "initial_loss_dir": initial["loss_dir"],
        "final_loss_dir": final["loss_dir"],
        "initial_logit_std": initial["logit_std"],
        "final_logit_std": final["logit_std"],
        "final_logit_entropy": final["logit_entropy"],
        "final_pairwise_direction_agreement": final["pairwise_direction_agreement"],
        "order_current_first16": json.dumps([int(v) for v in order.detach().cpu().tolist()[:16]]),
        "last_step_loss_total": last["loss_total"],
    }
    return logits.detach().cpu(), row


def main():
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np_rng = np.random.default_rng(int(args.seed))
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_frozen_aogpt(args.ckpt_path, device)
    data_dir = infer_data_dir(args.dataset, args.data_dir)
    tokens = load_tokens(data_dir, args.split)
    data_record_mode = infer_data_record_mode(checkpoint)
    permutation_state = load_permutation_state(
        checkpoint,
        num_blocks=int(model.num_blocks),
        block_len=int(model.block_order_block_len),
    )
    token_perm = None if permutation_state is None else permutation_state["token_perm"]

    print_progress(f"loaded_ckpt={args.ckpt_path}")
    print_progress(f"collecting_attention_samples={args.num_attention_samples}")
    matrices, block_orders_first = collect_attention_matrices(
        model=model,
        tokens=tokens,
        num_samples=int(args.num_attention_samples),
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
    coeffs = build_coefficients_for_matrices(
        matrices,
        sym_mode=args.sym_mode,
        dir_mode=args.dir_mode,
        threshold_percentile=float(args.threshold_percentile),
    )

    loss_kwargs = {
        "lambda_dir": float(args.lambda_dir),
        "lambda_var": float(args.lambda_var),
        "min_std": float(args.min_std),
        "tau_d": float(args.tau_d),
        "tau_s": float(args.tau_s),
        "margin_d": float(args.margin_d),
        "min_conf": float(args.min_conf),
    }
    free_logits = []
    opt_rows = []
    train_log_path = args.out_dir / "free_vector_train_log.jsonl"
    if train_log_path.exists():
        train_log_path.unlink()
    for sample_id, coeff in enumerate(coeffs):
        logits, row = optimize_one(coeff, sample_id, args, device)
        free_logits.append(logits)
        opt_rows.append(row)
        append_jsonl(train_log_path, row)
        if (sample_id + 1) % max(1, int(args.progress_interval)) == 0:
            print_progress(f"optimized_free_vectors={sample_id + 1}/{len(coeffs)}")

    state = {"idx": 0}

    def order_source(_matrix: torch.Tensor, _coeff: dict):
        idx = state["idx"]
        state["idx"] += 1
        logits = free_logits[idx]
        return logits, order_from_logits_desc(logits)

    print_progress("evaluating_frozen_nll")
    nll_rows, nll_summary = evaluate_order_source_nll(
        model=model,
        tokens=tokens,
        matrices=matrices,
        coeffs=coeffs,
        order_source=order_source,
        rng=np_rng,
        device=device,
        dtype=args.dtype,
        token_perm=token_perm,
        data_record_mode=data_record_mode,
        eval_batch_size=int(args.eval_batch_size),
        random_eval_orders=int(args.random_eval_orders),
        permutation_state=permutation_state,
        loss_kwargs=loss_kwargs,
        progress=print_progress,
        progress_interval=max(1, int(args.progress_interval)),
    )

    merged_rows = []
    nll_by_id = {int(row["sample_id"]): row for row in nll_rows}
    for row in opt_rows:
        merged = dict(row)
        merged.update(nll_by_id.get(int(row["sample_id"]), {}))
        merged_rows.append(merged)
    write_csv(args.out_dir / "samples.csv", merged_rows)
    write_json(args.out_dir / "free_vector_logits.json", {"logits": [row.tolist() for row in free_logits]})
    graph_stats = mean_dicts([coeff["stats"] for coeff in coeffs])
    opt_summary = mean_dicts(opt_rows)
    summary = {
        "script": "scripts/analysis/attn_axis_free_vector_sanity.py",
        "ckpt_path": str(args.ckpt_path),
        "checkpoint_iter": checkpoint.get("iter_num"),
        "checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
        "dataset": args.dataset,
        "split": args.split,
        "data_dir": str(data_dir),
        "frame": "current",
        "num_attention_samples": int(args.num_attention_samples),
        "probe_batch_size": int(args.probe_batch_size),
        "layer": int(args.layer),
        "head": int(args.head),
        "export_type": args.export_type,
        "sym_mode": args.sym_mode,
        "dir_mode": args.dir_mode,
        "threshold_percentile": float(args.threshold_percentile),
        "loss": "L_smooth + lambda_dir * L_dir + lambda_var * L_var",
        "lambda_dir": float(args.lambda_dir),
        "lambda_var": float(args.lambda_var),
        "min_std": float(args.min_std),
        "tau_d": float(args.tau_d),
        "tau_s": float(args.tau_s),
        "margin_d": float(args.margin_d),
        "min_conf": float(args.min_conf),
        "opt_steps": int(args.opt_steps),
        "lr": float(args.lr),
        "forbidden_signals": FORBIDDEN_SIGNALS,
        "tau_is_diagnostic_only": True,
        "original_l2r_is_diagnostic_only": True,
        "data_permutation_applied": permutation_state is not None,
        "block_perm_first16_diagnostic_only": None
        if permutation_state is None
        else [int(v) for v in permutation_state["block_perm"].tolist()[:16]],
        "random_probe_block_orders_first": block_orders_first,
        "optimization_summary": opt_summary,
        "graph_stats": graph_stats,
        "nll_summary": nll_summary,
        "objective_decreased": bool(
            math.isfinite(float(opt_summary.get("loss_drop", float("nan"))))
            and float(opt_summary.get("loss_drop", 0.0)) > 0.0
        ),
        "logits_noncollapsed": bool(float(opt_summary.get("final_logit_std", 0.0)) >= float(args.min_std)),
    }
    write_json(args.out_dir / "summary.json", summary)
    readme = f"""# Free-Vector Implicit Attention-Axis Sanity

This is a direct free-vector check for the implicit attention-axis loss. It freezes the AO-GPT
checkpoint and optimizes one free logit vector per current-frame attention matrix.

- checkpoint: `{args.ckpt_path}`
- samples: `{args.num_attention_samples}`
- loss: `L_smooth + lambda_dir * L_dir + lambda_var * L_var`
- sym_mode: `{args.sym_mode}`
- dir_mode: `{args.dir_mode}`
- tau/original diagnostics: logged only, not used for optimization or selection

Key files:

- `summary.json`
- `samples.csv`
- `free_vector_train_log.jsonl`
- `free_vector_logits.json`
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print_progress(f"summary_path={args.out_dir / 'summary.json'}")
    print_progress(json.dumps({"optimization_summary": opt_summary, "nll_summary": nll_summary}, indent=2))


if __name__ == "__main__":
    main()
