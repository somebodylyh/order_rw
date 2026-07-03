#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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

from attn_mlp_order_policy import load_frozen_attn_mlp_policy  # noqa: E402
from attn_mlp_implicit_axis_common import (  # noqa: E402
    FORBIDDEN_SIGNALS,
    build_coefficients_for_matrices,
    collect_attention_matrices,
    evaluate_order_source_nll,
    infer_data_dir,
    infer_data_record_mode,
    load_frozen_aogpt,
    load_permutation_state,
    load_tokens,
    order_from_logits_desc,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate implicit-axis Attn-MLP hard orders with frozen AO-GPT NLL.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--policy_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="wikitext103")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num_attention_samples", type=int, default=64)
    parser.add_argument("--probe_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--random_eval_orders", type=int, default=8)
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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--progress_interval", type=int, default=10)
    return parser.parse_args()


def print_progress(text: str):
    print(text, flush=True)


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


def main():
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np_rng = np.random.default_rng(int(args.seed))
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_frozen_aogpt(args.ckpt_path, device)
    policy, policy_config = load_frozen_attn_mlp_policy(
        args.policy_path,
        num_blocks=int(model.num_blocks),
        device=device,
    )
    data_dir = infer_data_dir(args.dataset, args.data_dir)
    tokens = load_tokens(data_dir, args.split)
    data_record_mode = infer_data_record_mode(checkpoint)
    permutation_state = load_permutation_state(
        checkpoint,
        num_blocks=int(model.num_blocks),
        block_len=int(model.block_order_block_len),
    )
    token_perm = None if permutation_state is None else permutation_state["token_perm"]

    matrices, probe_orders = collect_attention_matrices(
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

    def order_source(matrix: torch.Tensor, _coeff: dict):
        with torch.no_grad():
            logits = policy(matrix.unsqueeze(0).to(device=device, dtype=torch.float32)).squeeze(0).detach().cpu()
        return logits, order_from_logits_desc(logits)

    rows, nll_summary = evaluate_order_source_nll(
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
        loss_kwargs=make_loss_kwargs(args),
        progress=print_progress,
        progress_interval=max(1, int(args.progress_interval)),
    )
    write_csv(args.out_dir / "eval_rows.csv", rows)
    summary = {
        "command": " ".join(sys.argv),
        "script": "scripts/eval/eval_attn_mlp_implicit_axis_order_nll.py",
        "ckpt_path": str(args.ckpt_path),
        "policy_path": str(args.policy_path),
        "policy_config": policy_config,
        "dataset": args.dataset,
        "split": args.split,
        "data_dir": str(data_dir),
        "frame": "current",
        "layer": int(args.layer),
        "head": int(args.head),
        "export_type": args.export_type,
        "sym_mode": args.sym_mode,
        "dir_mode": args.dir_mode,
        "threshold_percentile": float(args.threshold_percentile),
        "loss": "L_smooth + lambda_dir * L_dir + lambda_var * L_var",
        "forbidden_signals": FORBIDDEN_SIGNALS,
        "tau_is_diagnostic_only": True,
        "original_l2r_is_diagnostic_only": True,
        "num_attention_samples": int(args.num_attention_samples),
        "probe_batch_size": int(args.probe_batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "random_eval_orders": int(args.random_eval_orders),
        "random_probe_block_orders_first": probe_orders,
        "data_permutation_applied": permutation_state is not None,
        "block_perm_first16_diagnostic_only": None
        if permutation_state is None
        else [int(v) for v in permutation_state["block_perm"].tolist()[:16]],
        "nll_summary": nll_summary,
    }
    write_json(args.out_dir / "eval_summary.json", summary)
    print_progress(f"eval_summary_path={args.out_dir / 'eval_summary.json'}")
    print_progress(json.dumps(nll_summary, indent=2))


if __name__ == "__main__":
    main()
