#!/usr/bin/env python3
"""Unified fixed-order AO-GPT eval protocol.

This intentionally does not reproduce the original MDM val loss.  It evaluates
checkpoints on the same train-chunk heldout split used by the Graph-RW/order
experiments, with explicit block-order modes and token-averaged CE.
"""

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from directed_graph_policy import build_directed_graph, sample_order
from train_aogpt_graph_rw import (
    AO_GPT_CKPT,
    AOGPT,
    AOGPTConfig,
    A_PATH_DEFAULT,
    BLOCK_LEN,
    N,
    SEQ_LEN,
    load_train_chunks,
    phys_to_model_idx,
)


DEFAULT_OUT_DIR = Path(A_PATH_DEFAULT).parent / "eval_protocol_check"


@dataclass
class EvalBlockOrder:
    physical: torch.Tensor
    model: torch.Tensor


def _clean_state_dict_keys(state_dict):
    state_dict = dict(state_dict)
    for key in list(state_dict.keys()):
        clean = key.replace("_orig_mod.", "")
        if clean != key:
            state_dict[clean] = state_dict.pop(key)
    return state_dict


def load_aogpt_any(ckpt_path, permutation_ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    perm_ckpt = ckpt
    if "data_permutation" not in perm_ckpt:
        perm_ckpt = torch.load(permutation_ckpt_path, map_location="cpu", weights_only=False)

    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    model_args = ckpt.get("model_args") or ckpt.get("config")
    if model_args is None and permutation_ckpt_path:
        model_args = perm_ckpt.get("model_args") or perm_ckpt.get("config")
    if model_args is None:
        raise KeyError("checkpoint does not contain model_args/config")

    valid = {k: v for k, v in dict(model_args).items() if k in sig}
    model = AOGPT(AOGPTConfig(**valid))
    state_dict = ckpt.get("model") or ckpt.get("model_state_dict")
    if state_dict is None:
        raise KeyError("checkpoint does not contain model/model_state_dict")
    model.load_state_dict(_clean_state_dict_keys(state_dict))
    model.crop_block_size(SEQ_LEN)
    model.to(device)
    model.eval()

    data_perm = perm_ckpt["data_permutation"]
    block_perm = torch.tensor(data_perm["block_perm"], dtype=torch.long)
    inv_perm = torch.tensor(data_perm["inverse_block_perm"], dtype=torch.long)
    return model, block_perm, inv_perm


def block_orders_to_token_orders(model_block_orders, block_len=BLOCK_LEN):
    batch_size, num_blocks = model_block_orders.shape
    offsets = torch.arange(block_len, device=model_block_orders.device).view(1, 1, block_len)
    token_orders = model_block_orders.unsqueeze(-1) * block_len + offsets
    return token_orders.reshape(batch_size, num_blocks * block_len).long()


def original_l2r_block_order(inv_perm):
    physical = torch.arange(inv_perm.numel(), dtype=torch.long)
    model = inv_perm[physical]
    return EvalBlockOrder(physical=physical, model=model)


def model_coordinate_block_order(block_perm):
    model = torch.arange(block_perm.numel(), dtype=torch.long)
    physical = block_perm[model]
    return EvalBlockOrder(physical=physical, model=model)


def describe_block_order(model_order, block_perm, block_len=BLOCK_LEN):
    model_order = model_order.detach().cpu().long()
    block_perm = block_perm.detach().cpu().long()
    physical_order = block_perm[model_order]
    ranges = []
    for physical_block in physical_order[:8].tolist():
        start = physical_block * block_len
        ranges.append(f"phys[{start}:{start + block_len}]")
    return {
        "model_order_first16": model_order[:16].tolist(),
        "physical_order_first16": physical_order[:16].tolist(),
        "token_ranges_first8": ranges,
    }


def physical_orders_to_model_orders(physical_orders, inv_perm):
    inv_perm = inv_perm.to(device=physical_orders.device)
    return inv_perm[physical_orders.long()]


def _sum_ce_for_model_orders(model, idx_batch, model_block_orders, device):
    token_orders = block_orders_to_token_orders(model_block_orders, BLOCK_LEN).to(device)
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logits, model_loss = model.forward_fn(idx_batch, token_orders)
        targets = idx_batch.gather(1, token_orders)
        shift_logits = logits[:, :-1, :].contiguous()
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view(idx_batch.size(0), idx_batch.size(1))

    loss_sum = float(token_losses.float().sum().item())
    token_count = int(token_losses.numel())
    block_losses = token_losses.float().view(idx_batch.size(0), N, BLOCK_LEN).mean(dim=-1)
    step_loss_sum = float(block_losses.sum().item())
    step_count = int(block_losses.numel())
    return loss_sum, token_count, step_loss_sum, step_count, float(model_loss.item())


def evaluate_order_matrices(model, idx_eval, order_matrices, device, batch_size):
    total_ce = 0.0
    total_tokens = 0
    total_step_ce = 0.0
    total_steps = 0
    model_loss_weighted = 0.0
    model_loss_batches = 0

    for model_orders_cpu in order_matrices:
        if model_orders_cpu.dim() == 1:
            model_orders_cpu = model_orders_cpu.unsqueeze(0).expand(idx_eval.size(0), -1)
        for start in range(0, idx_eval.size(0), batch_size):
            stop = min(start + batch_size, idx_eval.size(0))
            idx_batch = idx_eval[start:stop].to(device)
            model_orders = model_orders_cpu[start:stop].to(device)
            ce, n_tok, step_ce, n_step, model_loss = _sum_ce_for_model_orders(
                model, idx_batch, model_orders, device
            )
            total_ce += ce
            total_tokens += n_tok
            total_step_ce += step_ce
            total_steps += n_step
            model_loss_weighted += model_loss * (stop - start)
            model_loss_batches += stop - start

    return {
        "loss_token_avg": total_ce / total_tokens,
        "loss_step_avg": total_step_ce / total_steps,
        "num_predicted_tokens": total_tokens,
        "num_order_steps": total_steps,
        "model_forward_loss_avg": model_loss_weighted / max(model_loss_batches, 1),
    }


def build_stochastic_physical_orders(num_eval, num_blocks, eval_order_seeds, sampler):
    matrices = []
    for seed in eval_order_seeds:
        rows = []
        for seq_idx in range(num_eval):
            rows.append(sampler(seed, seq_idx))
        matrices.append(torch.tensor(np.stack(rows), dtype=torch.long))
    return matrices


def sha256_int_array(values):
    arr = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def make_eval_indices(total_chunks, seed, val_fraction, max_eval_seqs):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total_chunks)
    val_size = max(1, int(round(total_chunks * val_fraction)))
    val_indices = np.array(sorted(int(x) for x in perm[:val_size]), dtype=np.int64)
    return val_indices[: min(max_eval_seqs, len(val_indices))], val_size


def main():
    parser = argparse.ArgumentParser(description="Run unified fixed-order eval on an AO-GPT checkpoint.")
    parser.add_argument("--ckpt", default=AO_GPT_CKPT)
    parser.add_argument("--permutation-ckpt", default=AO_GPT_CKPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--label", default="random_perm_50k")
    parser.add_argument("--a-path", default=A_PATH_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--max-eval-seqs", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-order-seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--tau-start", type=float, default=0.1)
    parser.add_argument("--tau-step", type=float, default=0.1)
    parser.add_argument("--rw-top-k", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, block_perm, inv_perm = load_aogpt_any(args.ckpt, args.permutation_ckpt, device)

    if not torch.equal(inv_perm[block_perm], torch.arange(block_perm.numel())):
        raise RuntimeError("invalid checkpoint permutation: inv_perm[block_perm] != arange")
    if not torch.equal(block_perm[inv_perm], torch.arange(block_perm.numel())):
        raise RuntimeError("invalid checkpoint permutation: block_perm[inv_perm] != arange")

    print("Loading eval chunks using Graph-RW/order split protocol...", flush=True)
    idx_phys = load_train_chunks(n_chunks=None)
    eval_indices, full_val_size = make_eval_indices(
        idx_phys.size(0), args.seed, args.val_fraction, args.max_eval_seqs
    )
    idx_model = phys_to_model_idx(idx_phys[eval_indices], inv_perm).to(device)

    print(f"Eval chunks: {len(eval_indices)} / full val split {full_val_size}", flush=True)

    A_all = np.load(args.a_path)
    A_global = A_all.mean(axis=0).astype(np.float32)
    np.fill_diagonal(A_global, 0.0)
    graph = build_directed_graph(A_global)
    rw_params = {
        "tau_start": args.tau_start,
        "tau_step": args.tau_step,
        "alpha_dep": 0.5,
        "alpha_pr": 0.85,
        "beta_sup": 1.0,
        "beta_fut": 0.5,
        "beta_src": 0.2,
        "beta_loc": 0.5,
    }
    if args.rw_top_k > 0:
        rw_params["top_k"] = int(args.rw_top_k)

    ori = original_l2r_block_order(inv_perm)
    model_order = model_coordinate_block_order(block_perm)

    def random_sampler(seed, seq_idx):
        return np.random.default_rng(seed * 10000 + seq_idx).permutation(N)

    def rw_sampler(seed, seq_idx):
        order, _ = sample_order(
            graph,
            "progressive_rw",
            rw_params,
            seed=seed * 10000 + seq_idx,
        )
        return order

    unstructured_phys = build_stochastic_physical_orders(
        len(eval_indices), N, args.eval_order_seeds, random_sampler
    )
    rw_phys = build_stochastic_physical_orders(
        len(eval_indices), N, args.eval_order_seeds, rw_sampler
    )

    modes = {
        "val_ori_l2r_block": {
            "orders": [ori.model],
            "desc_model_order": ori.model,
            "eval_seeds": None,
            "notes": "physical block L2R [0..63], mapped to model positions with inv_perm[physical]",
        },
        "val_ar_l2r": {
            "orders": [ori.model],
            "desc_model_order": ori.model,
            "eval_seeds": None,
            "notes": "true original token L2R; identical to val_ori_l2r_block for block_len=4",
        },
        "val_model_order": {
            "orders": [model_order.model],
            "desc_model_order": model_order.model,
            "eval_seeds": None,
            "notes": "model-coordinate ascending order; diagnostic, not original L2R",
        },
        "val_unstructured_order": {
            "orders": [physical_orders_to_model_orders(order, inv_perm) for order in unstructured_phys],
            "desc_model_order": physical_orders_to_model_orders(unstructured_phys[0][0], inv_perm),
            "eval_seeds": args.eval_order_seeds,
            "notes": "random physical block permutations mapped to model positions",
        },
        "val_rw_order": {
            "orders": [physical_orders_to_model_orders(order, inv_perm) for order in rw_phys],
            "desc_model_order": physical_orders_to_model_orders(rw_phys[0][0], inv_perm),
            "eval_seeds": args.eval_order_seeds,
            "notes": "Graph-RW physical block orders mapped to model positions",
        },
    }

    results = {}
    for mode_name, spec in modes.items():
        print(f"Evaluating {mode_name}...", flush=True)
        metrics = evaluate_order_matrices(
            model, idx_model, spec["orders"], device, args.eval_batch_size
        )
        desc = describe_block_order(spec["desc_model_order"], block_perm, BLOCK_LEN)
        results[mode_name] = {
            **metrics,
            **desc,
            "eval_seeds": spec["eval_seeds"],
            "notes": spec["notes"],
        }

    metadata = {
        "label": args.label,
        "ckpt": os.path.abspath(os.path.expanduser(args.ckpt)),
        "permutation_ckpt": os.path.abspath(os.path.expanduser(args.permutation_ckpt)),
        "data_protocol": "Graph-RW/order train-arrow chunks, deterministic heldout split",
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "full_val_split_count": int(full_val_size),
        "eval_chunk_count": int(len(eval_indices)),
        "max_eval_seqs": args.max_eval_seqs,
        "eval_indices_sha256": sha256_int_array(eval_indices),
        "eval_indices_first16": eval_indices[:16].tolist(),
        "loss_convention": "token-averaged cross entropy over predicted reveal tokens; nats/token",
        "coordinate_convention": {
            "block_perm": "block_perm[model_block] = physical_block",
            "inv_perm": "inv_perm[physical_block] = model_block",
            "physical_to_model": "inv_perm[physical_order]",
            "model_to_physical": "block_perm[model_order]",
        },
        "block_len": BLOCK_LEN,
        "num_blocks": N,
        "rw_policy": "progressive_rw",
        "rw_params": rw_params,
    }

    payload = {"metadata": metadata, "results": results}
    json_path = output_dir / f"{args.label}_eval.json"
    tsv_path = output_dir / f"{args.label}_eval.tsv"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    columns = [
        "mode",
        "loss_token_avg",
        "loss_step_avg",
        "num_predicted_tokens",
        "num_order_steps",
        "model_order_first16",
        "physical_order_first16",
        "token_ranges_first8",
        "eval_seeds",
    ]
    with tsv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for mode_name, row in results.items():
            writer.writerow({
                "mode": mode_name,
                "loss_token_avg": f"{row['loss_token_avg']:.6f}",
                "loss_step_avg": f"{row['loss_step_avg']:.6f}",
                "num_predicted_tokens": row["num_predicted_tokens"],
                "num_order_steps": row["num_order_steps"],
                "model_order_first16": json.dumps(row["model_order_first16"]),
                "physical_order_first16": json.dumps(row["physical_order_first16"]),
                "token_ranges_first8": json.dumps(row["token_ranges_first8"]),
                "eval_seeds": json.dumps(row["eval_seeds"]),
            })

    print(f"Saved JSON: {json_path}", flush=True)
    print(f"Saved TSV: {tsv_path}", flush=True)


if __name__ == "__main__":
    main()
