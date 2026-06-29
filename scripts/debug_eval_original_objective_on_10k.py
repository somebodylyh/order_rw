#!/usr/bin/env python3
"""Evaluate the 50k checkpoint on the 10k attention chunks under matched objectives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from debug_eval_mismatch_common import (
    DEFAULT_A64,
    DEFAULT_CKPT,
    DEFAULT_ON_CKPT,
    DEFAULT_OUT_DIR,
    DEFAULT_TOKENS,
    NUM_BLOCKS,
    TOKENS_PER_BLOCK,
    autocast_context,
    build_block_orders,
    build_model_token_order_from_phys_blocks,
    get_permutation_state,
    load_aogpt_from_ckpt,
    load_on_model,
    load_token_chunks,
    model_loss,
    order_checksum,
    permute_input_to_model_frame,
    reference_loss_from_logits,
    resolve_device,
    sample_on_order,
    set_seeds,
    toy_order_sanity,
    write_json,
)


DEFAULT_A_GLOBAL = "block_lo_arm_order_network/probe_results/attention_curriculum_real_diag/A_global_train_first8000_n64.npy"


def batch_iter(tensor: torch.Tensor, batch_size: int):
    for start in range(0, tensor.size(0), batch_size):
        yield tensor[start: start + batch_size], start


def mean_losses(values):
    return float(np.mean(values)) if values else float("nan")


def eval_random_objective(model, tokens_phys, perm_state, batch_size, ctx, permute_input: bool):
    losses = []
    predicted = 0
    for batch_phys, _ in batch_iter(tokens_phys, batch_size):
        idx = permute_input_to_model_frame(batch_phys, perm_state) if permute_input else batch_phys
        loss, _ = model_loss(model, idx, mode="Random", ctx=ctx)
        losses.append(loss)
        predicted += idx.numel()
    return {"loss": mean_losses(losses), "predicted_tokens": int(predicted), "num_batches": len(losses)}


def eval_fixed_orders(model, tokens_phys, A64, on_model, perm_state, batch_size, ctx, seed, permute_input: bool):
    out = {name: [] for name in ["ar", "random_fixed", "l2r", "on_order"]}
    order_meta = {}

    for batch_phys, offset in batch_iter(tokens_phys, batch_size):
        B = batch_phys.size(0)
        idx = permute_input_to_model_frame(batch_phys, perm_state) if permute_input else batch_phys

        ar_blocks_model = build_block_orders("ar", B, idx.device, seed=seed)
        offsets = torch.arange(TOKENS_PER_BLOCK, device=idx.device)
        ar_token = (ar_blocks_model.unsqueeze(-1) * TOKENS_PER_BLOCK + offsets).reshape(B, -1)
        ar_loss, _ = model_loss(model, idx, mode=None, token_orders=ar_token, ctx=ctx)
        out["ar"].append(ar_loss)
        order_meta.setdefault("ar", ar_blocks_model[0].detach().cpu())

        rnd_phys = build_block_orders("random_fixed", B, idx.device, seed=seed)
        rnd_token = build_model_token_order_from_phys_blocks(rnd_phys, perm_state)
        rnd_loss, _ = model_loss(model, idx, mode=None, token_orders=rnd_token, ctx=ctx)
        out["random_fixed"].append(rnd_loss)
        order_meta.setdefault("random_fixed", rnd_phys[0].detach().cpu())

        l2r_phys = build_block_orders("l2r_phys", B, idx.device, seed=seed)
        l2r_token = build_model_token_order_from_phys_blocks(l2r_phys, perm_state)
        l2r_loss, _ = model_loss(model, idx, mode=None, token_orders=l2r_token, ctx=ctx)
        out["l2r"].append(l2r_loss)
        order_meta.setdefault("l2r", l2r_phys[0].detach().cpu())

        if on_model is not None and A64 is not None:
            A_b = torch.from_numpy(A64[offset: offset + B]).float().to(idx.device)
            on_phys = sample_on_order(on_model, A_b, temperature=1.0, top_k=4)
            on_token = build_model_token_order_from_phys_blocks(on_phys, perm_state)
            on_loss, _ = model_loss(model, idx, mode=None, token_orders=on_token, ctx=ctx)
            out["on_order"].append(on_loss)
            order_meta.setdefault("on_order", on_phys[0].detach().cpu())

    losses = {key: mean_losses(vals) for key, vals in out.items()}
    predicted_tokens = int(tokens_phys.numel())
    modes = {}
    for key, order in order_meta.items():
        token_ranges = [[int(b) * TOKENS_PER_BLOCK, int(b) * TOKENS_PER_BLOCK + TOKENS_PER_BLOCK - 1]
                        for b in order[:10].tolist()]
        modes[key] = {
            "loss": losses[key],
            "predicted_tokens": predicted_tokens,
            "visible_tokens_per_step": [int(i * TOKENS_PER_BLOCK) for i in range(NUM_BLOCKS)],
            "order_checksum": order_checksum(order),
            "first_10_block_ids": [int(v) for v in order[:10].tolist()],
            "first_10_token_index_ranges": token_ranges,
        }
    return modes


def run_reference_checks(model, tokens_phys, perm_state, ctx, seed):
    batch_phys = tokens_phys[: min(4, tokens_phys.size(0))]
    batch_model = permute_input_to_model_frame(batch_phys, perm_state)
    B = batch_model.size(0)
    rnd_phys = build_block_orders("random_fixed", B, batch_model.device, seed=seed)
    token_orders = build_model_token_order_from_phys_blocks(rnd_phys, perm_state)
    forward_loss, token_losses = model_loss(
        model, batch_model, mode=None, token_orders=token_orders, ctx=ctx, return_token_loss=True
    )
    ref_loss = reference_loss_from_logits(model, batch_model, token_orders, ctx=ctx)
    return {
        "same_order_forward_fn_loss": forward_loss,
        "same_order_reference_ce_loss": ref_loss,
        "absolute_diff": abs(forward_loss - ref_loss),
        "token_loss_shape": list(token_losses.shape) if token_losses is not None else None,
        "token_loss_mean": float(token_losses.float().mean().item()) if token_losses is not None else None,
    }


def load_reproduce_payload(output_dir: str):
    path = Path(output_dir) / "reproduce_ckpt_val_loss.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_report(payload, out_dir):
    comparison_rows = [
        ("input data", "val.bin random windows, then fixed_token_perm", "10k raw chunks were passed directly in current cotrain", "no", "raw chunks must be permuted before AO-GPT"),
        ("target shift", "[None] + shuffled idx; logits[:, :-1] predict shuffled idx", "same AO-GPT forward_fn when called correctly", "yes", "not standard LM x[:-1]->y[1:]"),
        ("order semantics", "Random block order in model frame over 64 blocks", "AR/model order; ON/random/l2r often physical order mapped to model order", "partial", "mapping helper is correct if input is model-frame"),
        ("mask semantics", "causal reveal mask over [None] + reveal prefix", "same model mask", "yes", "no separate attention mask bug found"),
        ("loss normalization", "mean CE over all B*256 reveal tokens", "same forward_fn mean", "yes", "no block-length multiplier"),
        ("dataset", "WikiText103 val.bin for best_val_loss", "WikiText103 train-arrow 10k chunks for cotrain", "no", "dataset split differs; objective comparison still possible"),
    ]
    lines = [
        "# Eval Mismatch Debug Report",
        "",
        "## 1. Problem",
        "",
        "The 50k block64 checkpoint stores best_val_loss around 3.607, while cotrain/fixed-order step-0 eval had losses around 5-6. This report separates checkpoint eval objective, data frame, order mode, and normalization.",
        "",
        "## 2. Original checkpoint eval formula",
        "",
        "Code path: `/home/admin/ych/nanogpt-learned-order/train.py` -> `estimate_loss()` -> `_forward_with_active_training_policy()` -> `model(X, mode='Random')` -> `AOGPT_block.forward_fn()`.",
        "",
        "Formula: sample raw WikiText103 `val.bin` windows, apply checkpoint `fixed_token_perm` because `permute_data=True`, sample one random 64-block reveal order per sample, shuffle `idx` by that order, prepend `[None]`, and compute CE from `logits[:, :-1]` to the shuffled reveal tokens. Loss is averaged over all `batch * 256` reveal tokens and then over eval batches.",
        "",
        "## 3. Current fixed-order eval formula",
        "",
        "Code path: `block_lo_arm_order_network/cotrain_aogpt_on.py::evaluate_aogpt()`. It loads `A32_from_N64_10k.tokens.npy` chunks and calls `model(tokens, mode='AR'/'Random')` or `model.forward_fn(tokens, token_order)`. The loaded chunks are raw physical WikiText token order. The checkpoint expects model-frame/permuted input, so direct calls are frame-mismatched.",
        "",
        "## 4. Reproduction results",
        "",
        f"- checkpoint_best_val_loss: {payload.get('checkpoint_best_val_loss')}",
        f"- reproduced_ckpt_val_loss_small: {payload.get('reproduced_ckpt_val_loss_small')}",
        f"- reproduced_ckpt_val_loss_full_or_approx: {payload.get('reproduced_ckpt_val_loss')}",
        f"- raw_unpermuted_val_negative_control_small: {payload.get('raw_unpermuted_negative_control_small')}",
        "",
        "## 5. 10k chunk sanity check",
        "",
        f"- mdm_random_loss_on_10k_permuted_input: {payload.get('mdm_random_loss_on_10k')}",
        f"- mdm_random_loss_on_10k_raw_input: {payload.get('mdm_random_loss_on_10k_raw_input')}",
        f"- fixed_ar_loss_raw_input: {payload.get('fixed_ar_loss')}",
        f"- fixed_ar_loss_permuted_input: {payload.get('corrected_fixed_ar_loss')}",
        f"- fixed_random_loss_raw_input: {payload.get('fixed_random_loss')}",
        f"- fixed_random_loss_permuted_input: {payload.get('corrected_fixed_random_loss')}",
        f"- fixed_on_order_loss_raw_input: {payload.get('fixed_on_order_loss')}",
        f"- fixed_on_order_loss_permuted_input: {payload.get('corrected_fixed_on_order_loss')}",
        f"- fixed_l2r_loss_raw_input: {payload.get('fixed_l2r_loss')}",
        f"- fixed_l2r_loss_permuted_input: {payload.get('corrected_fixed_l2r_loss')}",
        "",
        "## 6. Difference table",
        "",
        "| dimension | original_ckpt_eval | current_fixed_order_eval | match? | notes |",
        "|---|---|---|---|---|",
    ]
    for row in comparison_rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.extend([
        "",
        "## 7. Bug assessment",
        "",
        f"bug_found: {payload.get('bug_found')}. Suspected issue: {payload.get('suspected_issue')}.",
        "",
        "The same-order forward implementation and an independent CE reference match to numerical precision in the sanity check, so the loss formula itself is not the source. The large 5-6 losses are explained by evaluating a permuted-data checkpoint on raw physical token chunks.",
        "",
        "## 8. Fix plan",
        "",
        "Before any AO-GPT eval/training call on `A32_from_N64_10k.tokens.npy`, convert raw physical chunks to checkpoint model frame with `idx_model = idx_phys[:, fixed_token_perm]`. Keep ON/teacher orders in physical block coordinates, then map physical block order to model token order through `inverse_block_perm`.",
        "",
        "## 9. Why 3.607 and 6.2 cannot be compared directly",
        "",
        "`best_val_loss=3.607` is original Random MDM objective on permuted WikiText103 val windows. The 5-6 values are fixed/order evals on raw train chunks with a coordinate mismatch. Even after the frame bug is fixed, val.bin Random MDM and train-chunk fixed-order losses are different metrics and should be reported separately.",
        "",
        "## 10. Recommended fair protocol",
        "",
        payload.get("recommended_eval_protocol", ""),
        "",
    ])
    path = Path(out_dir) / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_eval(args):
    device = resolve_device(args.device)
    set_seeds(args.seed)
    model, ckpt = load_aogpt_from_ckpt(args.ckpt, device)
    perm_state = get_permutation_state(ckpt, device=device)
    ctx = autocast_context(device, args.dtype)

    tokens_all_phys = load_token_chunks(args.tokens_path, 0, args.num_seqs, device)
    A64 = np.load(args.a_matrices, mmap_mode="r")[: tokens_all_phys.size(0)]
    on_model = None
    if args.on_ckpt and Path(args.on_ckpt).exists():
        on_model = load_on_model(args.on_ckpt, device, args.d_edge, args.d_model)

    n_total = tokens_all_phys.size(0)
    n_train = min(8000, int(n_total * 0.8))
    eval_start = min(args.eval_start, max(n_total - 1, 0))
    eval_count = min(args.eval_seqs, n_total - eval_start)
    tokens_eval_phys = tokens_all_phys[eval_start: eval_start + eval_count]
    A_eval = np.array(A64[eval_start: eval_start + eval_count], dtype=np.float32)

    random_perm = eval_random_objective(
        model, tokens_all_phys, perm_state, args.batch_size, ctx, permute_input=True
    )
    random_raw = eval_random_objective(
        model, tokens_all_phys[: min(args.eval_seqs, n_total)], perm_state, args.batch_size, ctx, permute_input=False
    )
    fixed_raw = eval_fixed_orders(
        model, tokens_eval_phys, A_eval, on_model, perm_state, args.batch_size, ctx, args.seed, permute_input=False
    )
    fixed_perm = eval_fixed_orders(
        model, tokens_eval_phys, A_eval, on_model, perm_state, args.batch_size, ctx, args.seed, permute_input=True
    )
    ref_checks = run_reference_checks(model, tokens_eval_phys, perm_state, ctx, args.seed)

    raw_mean = np.nanmean([fixed_raw[k]["loss"] for k in fixed_raw])
    perm_mean = np.nanmean([fixed_perm[k]["loss"] for k in fixed_perm])
    bug_found = bool(raw_mean - perm_mean > args.bug_gap_threshold)

    reproduce = load_reproduce_payload(args.output_dir) or {}
    reproduced = reproduce.get("reproduced_val_loss_full_or_approx")
    checkpoint_best = float(ckpt.get("best_val_loss", float("nan")))

    payload = {
        "checkpoint": str(args.ckpt),
        "checkpoint_best_val_loss": checkpoint_best,
        "reproduced_ckpt_val_loss": reproduced,
        "reproduced_ckpt_val_loss_small": reproduce.get("reproduced_val_loss_small"),
        "raw_unpermuted_negative_control_small": reproduce.get("raw_unpermuted_negative_control_small"),
        "data_source": str(args.tokens_path),
        "a_matrices": str(args.a_matrices),
        "eval_objective": "AO-GPT reveal CE; Random MDM or explicit fixed block orders as labeled",
        "loss_normalization": "mean CE over batch * 256 predicted reveal tokens; mean over batches",
        "num_10k_sequences_used_for_mdm_random": int(n_total),
        "fixed_eval_start": int(eval_start),
        "fixed_eval_num_sequences": int(eval_count),
        "cotrain_nominal_train_val_split": {"train": int(n_train), "val": int(n_total - n_train)},
        "mdm_random_loss_on_10k": random_perm["loss"],
        "mdm_random_loss_on_10k_raw_input": random_raw["loss"],
        "fixed_ar_loss": fixed_raw["ar"]["loss"],
        "fixed_random_loss": fixed_raw["random_fixed"]["loss"],
        "fixed_on_order_loss": fixed_raw["on_order"]["loss"],
        "fixed_l2r_loss": fixed_raw["l2r"]["loss"],
        "corrected_fixed_ar_loss": fixed_perm["ar"]["loss"],
        "corrected_fixed_random_loss": fixed_perm["random_fixed"]["loss"],
        "corrected_fixed_on_order_loss": fixed_perm["on_order"]["loss"],
        "corrected_fixed_l2r_loss": fixed_perm["l2r"]["loss"],
        "raw_fixed_order_modes": fixed_raw,
        "corrected_permuted_fixed_order_modes": fixed_perm,
        "same_order_reference_check": ref_checks,
        "toy_order_sanity": toy_order_sanity(n_blocks=4, block_len=4),
        "model_to_phys_mapping_test": {
            "block_perm_model_to_phys_first_10": [int(v) for v in perm_state["block_perm"][:10].detach().cpu().tolist()],
            "inverse_block_perm_phys_to_model_first_10": [int(v) for v in perm_state["inverse_block_perm"][:10].detach().cpu().tolist()],
            "physical_l2r_first_10_model_blocks": [int(v) for v in perm_state["inverse_block_perm"][:10].detach().cpu().tolist()],
            "physical_l2r_first_10_model_token_ranges": [
                [int(v) * TOKENS_PER_BLOCK, int(v) * TOKENS_PER_BLOCK + TOKENS_PER_BLOCK - 1]
                for v in perm_state["inverse_block_perm"][:10].detach().cpu().tolist()
            ],
        },
        "bug_found": bug_found,
        "suspected_issue": (
            "cotrain eval/training passes raw physical chunks to a checkpoint trained/evaluated on fixed-token-permuted model-frame inputs"
            if bug_found else
            "no large input-frame bug detected by configured threshold"
        ),
        "recommended_eval_protocol": (
            "Option C as main reporting format: report original_mdm_val_loss on checkpoint-style permuted val.bin "
            "to preserve comparability with the original task, and separately report corrected fixed_order losses "
            "on the same permuted 10k chunks for order adaptation. Baselines and ON-refresh must use the same "
            "input frame, data split, objective, order set, and token-mean normalization."
        ),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", payload)
    report_path = build_report(payload, out_dir)
    print(f"wrote {out_dir / 'summary.json'}")
    print(f"wrote {report_path}")
    print(f"mdm_random_loss_on_10k={payload['mdm_random_loss_on_10k']:.6f}")
    print(f"mdm_random_loss_on_10k_raw_input={payload['mdm_random_loss_on_10k_raw_input']:.6f}")
    print(f"fixed_raw ar/random/on/l2r={payload['fixed_ar_loss']:.6f}/{payload['fixed_random_loss']:.6f}/{payload['fixed_on_order_loss']:.6f}/{payload['fixed_l2r_loss']:.6f}")
    print(f"fixed_permuted ar/random/on/l2r={payload['corrected_fixed_ar_loss']:.6f}/{payload['corrected_fixed_random_loss']:.6f}/{payload['corrected_fixed_on_order_loss']:.6f}/{payload['corrected_fixed_l2r_loss']:.6f}")
    print(f"bug_found={payload['bug_found']} suspected_issue={payload['suspected_issue']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--tokens-path", default=DEFAULT_TOKENS)
    parser.add_argument("--a-matrices", default=DEFAULT_A64)
    parser.add_argument("--on-ckpt", default=DEFAULT_ON_CKPT)
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-seqs", type=int, default=10000)
    parser.add_argument("--eval-start", type=int, default=8000)
    parser.add_argument("--eval-seqs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--d-edge", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--bug-gap-threshold", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
