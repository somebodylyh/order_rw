#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from attn_mlp_implicit_axis_common import (  # noqa: E402
    FORBIDDEN_SIGNALS,
    aggregate_layerhead_attention_to_current_blocks,
    append_jsonl,
    autocast_context,
    expand_orders_for_model,
    full_loss_for_block_orders,
    infer_data_dir,
    infer_data_record_mode,
    load_frozen_aogpt,
    load_permutation_state,
    load_tokens,
    mean_dicts,
    order_from_logits_desc,
    original_order_diagnostics,
    parse_int_list,
    random_block_orders,
    sample_batch,
    tensor_metrics_to_floats,
    write_csv,
    write_json,
)
from order_utils import token_losses_to_block_losses  # noqa: E402
from train_attn_mlp_order_distribution_energy import (  # noqa: E402
    AttentionDistributionOrderMLP,
    block_losses_to_current,
    logits_entropy,
    make_features,
    plackett_luce_log_prob,
    robust_z_vector,
)


LOSS_NAME = "on-policy frozen-NLL reward with insertion/segment-move preferences"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Attn-MLP with online frozen-NLL insertion/segment-move preferences.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--try_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="wikitext103")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--train_steps", type=int, default=500)
    parser.add_argument("--states_per_step", type=int, default=4)
    parser.add_argument("--probe_batch_size", type=int, default=64)
    parser.add_argument("--sampled_orders_per_state", type=int, default=2)
    parser.add_argument("--random_baseline_orders", type=int, default=4)
    parser.add_argument("--move_base_orders_per_state", type=int, default=1)
    parser.add_argument("--moves_per_base", type=int, default=8)
    parser.add_argument(
        "--move_mode",
        choices=("block_insertion", "segment_insertion", "mixed", "random_block"),
        default="mixed",
    )
    parser.add_argument("--min_move_distance", type=int, default=8)
    parser.add_argument("--max_move_distance", type=int, default=48)
    parser.add_argument("--max_segment_len", type=int, default=4)
    parser.add_argument("--max_pairs_per_move", type=int, default=128)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--head", type=int, default=3)
    parser.add_argument("--export_type", choices=("with_none", "without_none"), default="with_none")
    parser.add_argument("--feature_clip", type=float, default=8.0)
    parser.add_argument("--hidden_dims", type=str, default="1024,1024")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--init_policy_path", type=Path, default=None)
    parser.add_argument("--anchor_policy_path", type=Path, default=None)
    parser.add_argument("--lambda_anchor_l2", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_pg", type=float, default=1.0)
    parser.add_argument("--lambda_prefix_reward", type=float, default=0.0)
    parser.add_argument("--prefix_k", type=int, default=8)
    parser.add_argument("--lambda_attn_locality_reward", type=float, default=0.0)
    parser.add_argument("--attn_locality_sigma", type=float, default=4.0)
    parser.add_argument("--lambda_attn_path_reward", type=float, default=0.0)
    parser.add_argument("--attn_path_bandwidth", type=int, default=1)
    parser.add_argument("--lambda_move_pref", type=float, default=10.0)
    parser.add_argument("--lambda_std_floor", type=float, default=0.10)
    parser.add_argument("--lambda_entropy_ceiling", type=float, default=0.02)
    parser.add_argument("--lambda_entropy_floor", type=float, default=0.0)
    parser.add_argument("--lambda_attention_smooth", type=float, default=0.002)
    parser.add_argument("--lambda_center", type=float, default=0.001)
    parser.add_argument("--lambda_loss_axis_pair", type=float, default=0.0)
    parser.add_argument(
        "--loss_axis_mode",
        choices=("per_state_nll", "low_loss_first", "high_loss_first"),
        default="per_state_nll",
        help="How to orient the optional current-block-loss axis auxiliary.",
    )
    parser.add_argument("--min_std", type=float, default=1.0)
    parser.add_argument("--max_entropy", type=float, default=3.6)
    parser.add_argument("--min_entropy", type=float, default=0.0)
    parser.add_argument("--loss_axis_tau", type=float, default=1.0)
    parser.add_argument("--loss_axis_margin", type=float, default=0.5)
    parser.add_argument("--move_pref_tau", type=float, default=1.0)
    parser.add_argument("--move_pref_margin", type=float, default=0.005)
    parser.add_argument("--move_min_pair_weight", type=float, default=0.0)
    parser.add_argument("--reward_scale_floor", type=float, default=0.001)
    parser.add_argument("--advantage_clip", type=float, default=5.0)
    parser.add_argument("--eval_samples", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--random_eval_orders", type=int, default=16)
    parser.add_argument("--policy_sample_eval_orders", type=int, default=8)
    parser.add_argument("--success_random_margin", type=float, default=0.005)
    parser.add_argument("--success_tau_gate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=35791)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--progress_interval", type=int, default=25)
    parser.add_argument("--wandb_log", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="ao-gpt-experiments")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default="attn_mlp_loss_design")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="offline")
    parser.add_argument("--wandb_tags", type=str, default="attn-mlp,try20-repro,no-prior")
    parser.add_argument("--wandb_init_timeout", type=int, default=180)
    parser.add_argument("--allow_small_debug", action="store_true")
    return parser.parse_args()


def print_progress(text: str):
    print(text, flush=True)


def jsonable_for_wandb(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable_for_wandb(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_for_wandb(v) for v in value]
    return value


def init_wandb_if_requested(args, config_payload):
    if not bool(args.wandb_log) or str(args.wandb_mode) == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb logging was requested, but the wandb package is not installed") from exc

    run_name = str(args.wandb_run_name).strip() or args.try_dir.name
    tags = [tag.strip() for tag in str(args.wandb_tags).split(",") if tag.strip()]
    init_kwargs = {
        "project": str(args.wandb_project),
        "name": run_name,
        "group": str(args.wandb_group),
        "tags": tags,
        "config": jsonable_for_wandb(config_payload),
        "dir": str(args.try_dir),
        "mode": str(args.wandb_mode),
    }
    try:
        init_kwargs["settings"] = wandb.Settings(init_timeout=int(args.wandb_init_timeout))
    except TypeError:
        pass
    return wandb.init(**init_kwargs)


def log_wandb_scalars(wandb_run, prefix: str, payload: dict, *, step: int | None = None):
    if wandb_run is None:
        return
    logged = {}
    for key, value in payload.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            logged[f"{prefix}/{key}"] = float(value)
        elif isinstance(value, bool):
            logged[f"{prefix}/{key}"] = float(value)
    if logged:
        wandb_run.log(logged, step=step)


def command_invocation():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    prefix = f"CUDA_VISIBLE_DEVICES={cuda_visible} python " if cuda_visible else "python "
    return prefix + " ".join(sys.argv)


def train_states(args):
    return int(args.train_steps) * int(args.states_per_step)


def train_order_evals(args):
    loss_axis_evals = (
        2
        if float(getattr(args, "lambda_loss_axis_pair", 0.0)) > 0.0
        and str(getattr(args, "loss_axis_mode", "per_state_nll")) == "per_state_nll"
        else 0
    )
    move_evals = int(args.move_base_orders_per_state) * (1 + int(args.moves_per_base))
    return train_states(args) * (
        int(args.sampled_orders_per_state)
        + int(args.random_baseline_orders)
        + int(loss_axis_evals)
        + int(move_evals)
    )


def validate_budget(args):
    if bool(args.allow_small_debug):
        return
    checks = [
        ("train_states", train_states(args), 2000),
        ("train_order_evals", train_order_evals(args), 10000),
        ("eval_samples", int(args.eval_samples), 512),
        ("random_eval_orders", int(args.random_eval_orders), 16),
        ("policy_sample_eval_orders", int(args.policy_sample_eval_orders), 8),
    ]
    too_small = [(name, value, minimum) for name, value, minimum in checks if value < minimum]
    if too_small:
        text = ", ".join(f"{name}={value} < {minimum}" for name, value, minimum in too_small)
        raise ValueError(f"Formal try budget is too small: {text}. Use --allow_small_debug only for smoke.")


@torch.no_grad()
def collect_online_feature_batch(
    *,
    model,
    tokens,
    states: int,
    probe_batch_size: int,
    layer: int,
    head: int,
    export_type: str,
    rng: np.random.Generator,
    device: torch.device,
    dtype: str,
    token_perm: torch.Tensor | None,
    data_record_mode: str,
    feature_clip: float,
):
    matrices = []
    block_losses = []
    xs = []
    n = int(model.num_blocks)
    layer = int(layer)
    head = int(head)
    for _ in range(int(states)):
        x = sample_batch(
            tokens,
            batch_size=int(probe_batch_size),
            block_size=int(model.config.block_size),
            rng=rng,
            device=device,
            token_perm=token_perm,
            data_record_mode=data_record_mode,
        )
        reveal_orders = random_block_orders(int(probe_batch_size), n, device)
        token_orders = expand_orders_for_model(model, reveal_orders)
        with autocast_context(device, dtype):
            outputs = model(
                x,
                mode=None,
                orders=token_orders,
                return_token_loss=True,
                return_attentions=True,
                return_logits=False,
            )
        token_losses = outputs[2].detach().float()
        attentions = outputs[-1]
        local_layer = layer if layer >= 0 else len(attentions) + layer
        if local_layer < 0 or local_layer >= len(attentions):
            raise ValueError(f"layer={layer} is outside available layers 0..{len(attentions) - 1}")
        layer_heads = aggregate_layerhead_attention_to_current_blocks(
            attentions[local_layer].detach(),
            reveal_orders,
            block_len=int(model.block_order_block_len),
            export_type=export_type,
        )
        if head < 0:
            matrix = layer_heads.mean(dim=0)
        else:
            if head >= int(layer_heads.size(0)):
                raise ValueError(f"head={head} is outside available heads 0..{int(layer_heads.size(0)) - 1}")
            matrix = layer_heads[head]
        matrix = matrix.detach().float().cpu()
        matrix.fill_diagonal_(0.0)
        reveal_block_loss = token_losses.view(
            int(probe_batch_size),
            n,
            int(model.block_order_block_len),
        ).mean(dim=-1)
        current_loss = block_losses_to_current(reveal_block_loss, reveal_orders).mean(dim=0)
        matrices.append(matrix)
        block_losses.append(current_loss.detach().float().cpu())
        xs.append(x)
    block_loss_tensor = torch.stack(block_losses, dim=0)
    features = make_features(torch.stack(matrices, dim=0), block_loss_tensor, feature_clip)
    return features, xs, block_loss_tensor


def sample_pl_orders_and_logps(logits: torch.Tensor, num_samples: int):
    logits = logits.float()
    orders = []
    logps = []
    for _ in range(int(num_samples)):
        uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(uniform))
        order = torch.argsort(logits + gumbel, dim=-1, descending=True)
        logp = plackett_luce_log_prob(logits, order).squeeze(1)
        orders.append(order)
        logps.append(logp)
    return torch.stack(orders, dim=1).long(), torch.stack(logps, dim=1)


def regularization_terms(logits: torch.Tensor, features: torch.Tensor, args):
    s = logits.float()
    std = s.std(dim=1, unbiased=False)
    entropy = logits_entropy(s)
    loss_std_floor = torch.relu(s.new_tensor(float(args.min_std)) - std).pow(2).mean()
    loss_entropy_ceiling = torch.relu(entropy - s.new_tensor(float(args.max_entropy))).pow(2).mean()
    loss_entropy_floor = torch.relu(s.new_tensor(float(args.min_entropy)) - entropy).pow(2).mean()
    loss_center = s.mean(dim=1).pow(2).mean()

    n = int(s.size(1))
    x = (s - s.mean(dim=1, keepdim=True)) / s.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    smooth_weights = torch.relu(features[:, 1].to(device=s.device, dtype=torch.float32))
    eye = torch.eye(n, dtype=torch.bool, device=s.device)
    smooth_weights = smooth_weights.masked_fill(eye.view(1, n, n), 0.0)
    xdiff = x[:, :, None] - x[:, None, :]
    loss_attention_smooth = (smooth_weights * xdiff.pow(2)).sum() / smooth_weights.sum().clamp_min(1e-6)

    return {
        "loss_std_floor": loss_std_floor,
        "loss_entropy_ceiling": loss_entropy_ceiling,
        "loss_entropy_floor": loss_entropy_floor,
        "loss_center": loss_center,
        "loss_attention_smooth": loss_attention_smooth,
        "logit_std": std.mean().detach(),
        "logit_entropy": entropy.mean().detach(),
    }


@torch.no_grad()
def full_and_prefix_loss_for_block_orders(
    *,
    model,
    x: torch.Tensor,
    block_orders: torch.Tensor,
    device: torch.device | str,
    dtype: str,
    prefix_k: int,
):
    if block_orders.ndim == 1:
        block_orders = block_orders.unsqueeze(0).expand(x.size(0), -1)
    block_orders = block_orders.to(device=x.device, dtype=torch.long)
    token_orders = expand_orders_for_model(model, block_orders)
    with autocast_context(device, dtype):
        outputs = model(
            x,
            mode=None,
            orders=token_orders,
            return_token_loss=True,
            return_logits=False,
        )
    full_loss = float(outputs[1].detach().float().cpu().item())
    token_losses = outputs[2].detach().float()
    block_losses = token_losses_to_block_losses(
        token_losses,
        block_len=int(model.block_order_block_len),
    ).float()
    k = max(1, min(int(prefix_k), int(block_losses.size(1))))
    prefix_loss = float(block_losses[:, :k].mean(dim=1).mean().detach().cpu().item())
    return full_loss, prefix_loss


def attention_locality_scores(features_state: torch.Tensor, orders: torch.Tensor, sigma: float):
    if orders.ndim == 1:
        orders = orders.unsqueeze(0)
    orders = orders.long()
    device = features_state.device
    n = int(features_state.size(-1))
    weights = torch.relu(features_state[1].to(device=device, dtype=torch.float32))
    eye = torch.eye(n, dtype=torch.bool, device=device)
    weights = weights.masked_fill(eye, 0.0)
    denom = weights.sum().clamp_min(1e-6)
    positions = torch.empty((int(orders.size(0)), n), device=device, dtype=torch.float32)
    rank = torch.arange(n, device=device, dtype=torch.float32).view(1, n).expand_as(positions)
    positions.scatter_(1, orders.to(device=device), rank)
    distance = (positions[:, :, None] - positions[:, None, :]).abs()
    proximity = torch.exp(-distance / max(float(sigma), 1e-6))
    scores = (proximity * weights.view(1, n, n)).sum(dim=(1, 2)) / denom
    return scores


def attention_path_band_scores(features_state: torch.Tensor, orders: torch.Tensor, bandwidth: int):
    if orders.ndim == 1:
        orders = orders.unsqueeze(0)
    orders = orders.long()
    device = features_state.device
    weights = torch.relu(features_state[1].to(device=device, dtype=torch.float32))
    n = int(weights.size(0))
    band = max(1, min(int(bandwidth), n - 1))
    scores = []
    for order in orders.to(device=device):
        total = weights.new_tensor(0.0)
        denom = weights.new_tensor(0.0)
        for offset in range(1, band + 1):
            left = order[:-offset]
            right = order[offset:]
            if int(left.numel()) == 0:
                continue
            decay = 1.0 / float(offset)
            pair_weight = weights[left, right] + weights[right, left]
            total = total + float(decay) * pair_weight.sum()
            denom = denom + float(decay) * float(left.numel())
        scores.append(total / denom.clamp_min(1e-6))
    return torch.stack(scores, dim=0)


def pairwise_axis_from_scores(logits: torch.Tensor, target_scores: torch.Tensor, args):
    s = logits.float()
    target = target_scores.to(device=s.device, dtype=torch.float32)
    n = int(s.size(1))
    ii, jj = torch.triu_indices(n, n, offset=1, device=s.device)
    pred_logits = (s[:, ii] - s[:, jj]) / max(float(args.loss_axis_tau), 1e-8)
    target_diff = target[:, ii] - target[:, jj]
    target_prob = torch.sigmoid(target_diff / max(float(args.loss_axis_tau), 1e-8)).detach()
    weights = (target_diff.abs() / max(float(args.loss_axis_margin), 1e-8)).clamp(0.0, 1.0)
    raw = F.binary_cross_entropy_with_logits(pred_logits, target_prob, reduction="none")
    loss = (raw * weights).sum() / weights.sum().clamp_min(1e-6)
    with torch.no_grad():
        pred_sign = torch.sign((s[:, ii] - s[:, jj]).detach())
        truth_sign = torch.sign(target_diff.detach())
        active = weights > 0.0
        if bool(active.any()):
            acc = (pred_sign[active] == truth_sign[active]).float().mean()
        else:
            acc = s.new_tensor(float("nan"))
    return loss, acc.detach(), weights.detach().mean()


def normalized_logit_l2(logits: torch.Tensor, anchor_logits: torch.Tensor):
    pred = logits.float()
    target = anchor_logits.to(device=pred.device, dtype=torch.float32)
    pred = (pred - pred.mean(dim=1, keepdim=True)) / pred.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    target = (target - target.mean(dim=1, keepdim=True)) / target.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)
    loss = (pred - target).pow(2).mean()
    with torch.no_grad():
        pred_c = pred - pred.mean(dim=1, keepdim=True)
        target_c = target - target.mean(dim=1, keepdim=True)
        corr = (pred_c * target_c).mean(dim=1)
        corr = corr / (
            pred_c.std(dim=1, unbiased=False).clamp_min(1e-6)
            * target_c.std(dim=1, unbiased=False).clamp_min(1e-6)
        )
    return loss, corr.mean().detach()


def choose_move_specs(
    num_blocks: int,
    count: int,
    rng: np.random.Generator,
    mode: str,
    min_distance: int,
    max_distance: int,
    max_segment_len: int,
):
    specs = []
    num_blocks = int(num_blocks)
    max_segment_len = max(1, min(int(max_segment_len), num_blocks - 1))
    min_distance = max(1, min(int(min_distance), num_blocks - 1))
    max_distance = max(min_distance, min(int(max_distance), num_blocks - 1))
    for move_idx in range(int(count)):
        if str(mode) == "block_insertion" or (str(mode) == "mixed" and move_idx % 2 == 0):
            length = 1
        elif str(mode) in {"segment_insertion", "mixed"}:
            length = int(rng.integers(2, max_segment_len + 1)) if max_segment_len >= 2 else 1
        elif str(mode) == "random_block":
            length = 1
        else:
            raise ValueError(f"Unsupported move_mode={mode!r}")
        length = max(1, min(length, num_blocks - 1))
        start = int(rng.integers(0, num_blocks - length + 1))
        reduced_n = num_blocks - length
        if str(mode) == "random_block":
            candidates = [dest for dest in range(reduced_n + 1) if dest != start]
        else:
            lo = max(0, start - max_distance)
            hi = min(reduced_n, start + max_distance)
            candidates = [dest for dest in range(lo, hi + 1) if abs(dest - start) >= min_distance]
            if not candidates:
                candidates = [dest for dest in range(reduced_n + 1) if dest != start]
        dest = int(candidates[int(rng.integers(0, len(candidates)))])
        specs.append((start, length, dest))
    return specs


def make_moved_order(order: torch.Tensor, start: int, length: int, dest: int):
    start = int(start)
    length = int(length)
    segment = order[start : start + length]
    remainder = torch.cat([order[:start], order[start + length :]], dim=0)
    dest = max(0, min(int(dest), int(remainder.numel())))
    return torch.cat([remainder[:dest], segment, remainder[dest:]], dim=0)


def changed_pair_labels(base_order: torch.Tensor, moved_order: torch.Tensor, prefer_moved: bool):
    base = [int(v) for v in base_order.detach().cpu().tolist()]
    moved = [int(v) for v in moved_order.detach().cpu().tolist()]
    moved_pos = {block: idx for idx, block in enumerate(moved)}
    preferred_pos = moved_pos if bool(prefer_moved) else {block: idx for idx, block in enumerate(base)}
    labels = []
    for i, block_a in enumerate(base[:-1]):
        for block_b in base[i + 1 :]:
            if moved_pos[block_a] > moved_pos[block_b]:
                target = 1.0 if preferred_pos[block_a] < preferred_pos[block_b] else 0.0
                labels.append((block_a, block_b, target))
    return labels


@torch.no_grad()
def loss_axis_targets_from_block_losses(
    *,
    block_losses: torch.Tensor,
    xs: list[torch.Tensor],
    model,
    device,
    dtype,
    mode: str = "per_state_nll",
):
    targets = []
    margins = []
    asc_better = []
    for state_idx, x in enumerate(xs):
        loss_z = robust_z_vector(block_losses[state_idx].view(1, -1)).view(-1)
        low_loss_first_scores = -loss_z
        if mode == "low_loss_first":
            targets.append(low_loss_first_scores)
            continue
        if mode == "high_loss_first":
            targets.append(-low_loss_first_scores)
            continue
        if mode != "per_state_nll":
            raise ValueError(f"Unsupported loss_axis_mode={mode!r}")
        asc_order = torch.argsort(low_loss_first_scores, descending=True)
        desc_order = torch.flip(asc_order, dims=[0])
        asc_loss = full_loss_for_block_orders(
            model=model,
            x=x,
            block_orders=asc_order.to(device),
            device=device,
            dtype=dtype,
        )
        desc_loss = full_loss_for_block_orders(
            model=model,
            x=x,
            block_orders=desc_order.to(device),
            device=device,
            dtype=dtype,
        )
        if asc_loss <= desc_loss:
            targets.append(low_loss_first_scores)
            asc_better.append(1.0)
            margins.append(desc_loss - asc_loss)
        else:
            targets.append(-low_loss_first_scores)
            asc_better.append(0.0)
            margins.append(asc_loss - desc_loss)
    return (
        torch.stack(targets, dim=0).float(),
        float(np.mean(asc_better)) if asc_better else float("nan"),
        float(np.mean(margins)) if margins else float("nan"),
    )


def train_step(
    *,
    policy,
    optimizer,
    model,
    tokens,
    rng,
    device,
    dtype,
    token_perm,
    data_record_mode,
    args,
    anchor_policy=None,
):
    policy.train()
    features_cpu, xs, block_losses_cpu = collect_online_feature_batch(
        model=model,
        tokens=tokens,
        states=int(args.states_per_step),
        probe_batch_size=int(args.probe_batch_size),
        layer=int(args.layer),
        head=int(args.head),
        export_type=args.export_type,
        rng=rng,
        device=device,
        dtype=dtype,
        token_perm=token_perm,
        data_record_mode=data_record_mode,
        feature_clip=float(args.feature_clip),
    )
    features = features_cpu.to(device=device)
    logits = policy(features)
    sampled_orders, logps = sample_pl_orders_and_logps(logits, int(args.sampled_orders_per_state))

    sample_losses = torch.zeros_like(logps, dtype=torch.float32, device=device)
    sample_prefix_losses = torch.zeros_like(logps, dtype=torch.float32, device=device)
    sample_attn_scores = torch.zeros_like(logps, dtype=torch.float32, device=device)
    sample_attn_path_scores = torch.zeros_like(logps, dtype=torch.float32, device=device)
    baseline_losses = torch.zeros(int(args.states_per_step), dtype=torch.float32, device=device)
    baseline_prefix_losses = torch.zeros(int(args.states_per_step), dtype=torch.float32, device=device)
    baseline_attn_scores = torch.zeros(int(args.states_per_step), dtype=torch.float32, device=device)
    baseline_attn_path_scores = torch.zeros(int(args.states_per_step), dtype=torch.float32, device=device)
    move_pref_logits = []
    move_pref_targets = []
    move_pref_weights = []
    move_delta_values = []
    move_improved = []
    move_base_losses = []
    moved_losses = []
    move_pair_counts = []
    move_orders, _ = sample_pl_orders_and_logps(logits.detach(), int(args.move_base_orders_per_state))
    for state_idx, x in enumerate(xs):
        random_losses = []
        random_prefix_losses = []
        random_attn_scores = []
        random_attn_path_scores = []
        for _ in range(int(args.random_baseline_orders)):
            random_orders = random_block_orders(x.size(0), int(model.num_blocks), device)
            full_loss, prefix_loss = full_and_prefix_loss_for_block_orders(
                model=model,
                x=x,
                block_orders=random_orders,
                device=device,
                dtype=dtype,
                prefix_k=int(args.prefix_k),
            )
            random_losses.append(full_loss)
            random_prefix_losses.append(prefix_loss)
            random_attn_scores.append(
                float(
                    attention_locality_scores(
                        features[state_idx],
                        random_orders,
                        float(args.attn_locality_sigma),
                    )
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                )
            )
            random_attn_path_scores.append(
                float(
                    attention_path_band_scores(
                        features[state_idx],
                        random_orders,
                        int(args.attn_path_bandwidth),
                    )
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                )
            )
        baseline = float(np.mean(random_losses))
        baseline_losses[state_idx] = baseline
        baseline_prefix_losses[state_idx] = float(np.mean(random_prefix_losses))
        baseline_attn_scores[state_idx] = float(np.mean(random_attn_scores))
        baseline_attn_path_scores[state_idx] = float(np.mean(random_attn_path_scores))
        for sample_idx in range(int(args.sampled_orders_per_state)):
            full_loss, prefix_loss = full_and_prefix_loss_for_block_orders(
                model=model,
                x=x,
                block_orders=sampled_orders[state_idx, sample_idx],
                device=device,
                dtype=dtype,
                prefix_k=int(args.prefix_k),
            )
            sample_losses[state_idx, sample_idx] = full_loss
            sample_prefix_losses[state_idx, sample_idx] = prefix_loss
            sample_attn_scores[state_idx, sample_idx] = attention_locality_scores(
                features[state_idx],
                sampled_orders[state_idx, sample_idx],
                float(args.attn_locality_sigma),
            )[0]
            sample_attn_path_scores[state_idx, sample_idx] = attention_path_band_scores(
                features[state_idx],
                sampled_orders[state_idx, sample_idx],
                int(args.attn_path_bandwidth),
            )[0]
        for base_idx in range(int(args.move_base_orders_per_state)):
            base_order = move_orders[state_idx, base_idx].detach()
            base_loss = full_loss_for_block_orders(
                model=model,
                x=x,
                block_orders=base_order,
                device=device,
                dtype=dtype,
            )
            move_base_losses.append(base_loss)
            move_specs = choose_move_specs(
                int(model.num_blocks),
                int(args.moves_per_base),
                rng,
                str(args.move_mode),
                int(args.min_move_distance),
                int(args.max_move_distance),
                int(args.max_segment_len),
            )
            for start, length, dest in move_specs:
                moved = make_moved_order(base_order, start, length, dest)
                moved_loss = full_loss_for_block_orders(
                    model=model,
                    x=x,
                    block_orders=moved,
                    device=device,
                    dtype=dtype,
                )
                delta = float(moved_loss - base_loss)
                pair_labels = changed_pair_labels(base_order, moved, prefer_moved=delta < 0.0)
                if int(args.max_pairs_per_move) > 0 and len(pair_labels) > int(args.max_pairs_per_move):
                    keep = rng.choice(len(pair_labels), size=int(args.max_pairs_per_move), replace=False)
                    pair_labels = [pair_labels[int(idx)] for idx in keep]
                pair_weight = min(abs(delta) / max(float(args.move_pref_margin), 1e-8), 1.0)
                if pair_weight > 0.0:
                    pair_weight = max(pair_weight, float(args.move_min_pair_weight))
                for block_a, block_b, target in pair_labels:
                    move_pref_logits.append(
                        (logits[state_idx, block_a] - logits[state_idx, block_b]) / float(args.move_pref_tau)
                    )
                    move_pref_targets.append(float(target))
                    move_pref_weights.append(pair_weight)
                move_pair_counts.append(len(pair_labels))
                move_delta_values.append(abs(delta))
                move_improved.append(1.0 if delta < 0.0 else 0.0)
                moved_losses.append(moved_loss)

    full_advantages = baseline_losses[:, None] - sample_losses
    prefix_advantages = baseline_prefix_losses[:, None] - sample_prefix_losses
    attn_advantages = sample_attn_scores - baseline_attn_scores[:, None]
    attn_path_advantages = sample_attn_path_scores - baseline_attn_path_scores[:, None]
    advantages = (
        full_advantages
        + float(args.lambda_prefix_reward) * prefix_advantages
        + float(args.lambda_attn_locality_reward) * attn_advantages
        + float(args.lambda_attn_path_reward) * attn_path_advantages
    )
    reward_scale = advantages.detach().std(unbiased=False).clamp_min(float(args.reward_scale_floor))
    adv_norm = (advantages / reward_scale).clamp(-float(args.advantage_clip), float(args.advantage_clip))
    loss_pg = -(adv_norm.detach() * logps).mean()
    regs = regularization_terms(logits, features, args)
    loss_axis_pair = logits.new_tensor(0.0)
    loss_axis_acc = logits.new_tensor(float("nan"))
    loss_axis_weight_mean = logits.new_tensor(0.0)
    loss_axis_asc_better_rate = float("nan")
    loss_axis_margin = float("nan")
    if float(args.lambda_loss_axis_pair) > 0.0:
        target_scores, loss_axis_asc_better_rate, loss_axis_margin = loss_axis_targets_from_block_losses(
            block_losses=block_losses_cpu,
            xs=xs,
            model=model,
            device=device,
            dtype=dtype,
            mode=str(args.loss_axis_mode),
        )
        loss_axis_pair, loss_axis_acc, loss_axis_weight_mean = pairwise_axis_from_scores(
            logits,
            target_scores.to(device=device),
            args,
        )
    if move_pref_logits:
        pref_logits = torch.stack(move_pref_logits)
        pref_targets = torch.tensor(move_pref_targets, device=device, dtype=torch.float32)
        pref_weights = torch.tensor(move_pref_weights, device=device, dtype=torch.float32)
        raw_pref = F.binary_cross_entropy_with_logits(pref_logits, pref_targets, reduction="none")
        loss_move_pref = (raw_pref * pref_weights).sum() / pref_weights.sum().clamp_min(1e-6)
        with torch.no_grad():
            pref_pred = (pref_logits.detach() >= 0.0).float()
            move_pref_acc = (pref_pred == pref_targets).float().mean()
            move_pref_weight_mean = pref_weights.mean()
    else:
        loss_move_pref = logits.new_tensor(0.0)
        move_pref_acc = logits.new_tensor(float("nan"))
        move_pref_weight_mean = logits.new_tensor(0.0)
    loss_anchor_l2 = logits.new_tensor(0.0)
    anchor_corr = logits.new_tensor(float("nan"))
    if anchor_policy is not None and float(args.lambda_anchor_l2) > 0.0:
        with torch.no_grad():
            anchor_logits = anchor_policy(features).detach()
        loss_anchor_l2, anchor_corr = normalized_logit_l2(logits, anchor_logits)
    total = (
        float(args.lambda_pg) * loss_pg
        + float(args.lambda_std_floor) * regs["loss_std_floor"]
        + float(args.lambda_entropy_ceiling) * regs["loss_entropy_ceiling"]
        + float(args.lambda_entropy_floor) * regs["loss_entropy_floor"]
        + float(args.lambda_attention_smooth) * regs["loss_attention_smooth"]
        + float(args.lambda_center) * regs["loss_center"]
        + float(args.lambda_loss_axis_pair) * loss_axis_pair
        + float(args.lambda_move_pref) * loss_move_pref
        + float(args.lambda_anchor_l2) * loss_anchor_l2
    )

    optimizer.zero_grad(set_to_none=True)
    total.backward()
    if float(args.grad_clip) > 0.0:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), float(args.grad_clip))
    optimizer.step()

    metrics = {
        "loss_total": total,
        "loss_pg": loss_pg.detach(),
        "loss_axis_pair": loss_axis_pair.detach(),
        "loss_axis_acc": loss_axis_acc.detach(),
        "loss_axis_weight_mean": loss_axis_weight_mean.detach(),
        "loss_axis_asc_better_rate": loss_axis_asc_better_rate,
        "loss_axis_margin": loss_axis_margin,
        "loss_move_pref": loss_move_pref.detach(),
        "move_pref_acc": move_pref_acc.detach(),
        "move_pref_weight_mean": move_pref_weight_mean.detach(),
        "move_improved_rate": float(np.mean(move_improved)) if move_improved else float("nan"),
        "move_delta_abs_mean": float(np.mean(move_delta_values)) if move_delta_values else float("nan"),
        "move_pairs_per_compare": float(np.mean(move_pair_counts)) if move_pair_counts else float("nan"),
        "move_base_full_loss": float(np.mean(move_base_losses)) if move_base_losses else float("nan"),
        "moved_full_loss": float(np.mean(moved_losses)) if moved_losses else float("nan"),
        "loss_anchor_l2": loss_anchor_l2.detach(),
        "anchor_corr": anchor_corr.detach(),
        "advantage_mean": advantages.mean().detach(),
        "advantage_std": advantages.std(unbiased=False).detach(),
        "full_advantage_mean": full_advantages.mean().detach(),
        "prefix_advantage_mean": prefix_advantages.mean().detach(),
        "attn_locality_advantage_mean": attn_advantages.mean().detach(),
        "attn_path_advantage_mean": attn_path_advantages.mean().detach(),
        "sample_minus_random_mean": (sample_losses - baseline_losses[:, None]).mean().detach(),
        "sample_prefix_minus_random_mean": (sample_prefix_losses - baseline_prefix_losses[:, None]).mean().detach(),
        "sample_attn_locality_minus_random_mean": (sample_attn_scores - baseline_attn_scores[:, None]).mean().detach(),
        "sample_attn_path_minus_random_mean": (
            sample_attn_path_scores - baseline_attn_path_scores[:, None]
        ).mean().detach(),
        "sample_full_loss": sample_losses.mean().detach(),
        "sample_prefix_loss": sample_prefix_losses.mean().detach(),
        "sample_attn_locality": sample_attn_scores.mean().detach(),
        "sample_attn_path": sample_attn_path_scores.mean().detach(),
        "random_baseline_full_loss": baseline_losses.mean().detach(),
        "random_baseline_prefix_loss": baseline_prefix_losses.mean().detach(),
        "random_baseline_attn_locality": baseline_attn_scores.mean().detach(),
        "random_baseline_attn_path": baseline_attn_path_scores.mean().detach(),
        "pl_logp": logps.mean().detach(),
        "reward_scale": reward_scale.detach(),
    }
    metrics.update(regs)
    return tensor_metrics_to_floats(metrics)


@torch.no_grad()
def evaluate_policy_nll_online(
    *,
    policy,
    model,
    tokens,
    rng,
    device,
    dtype,
    token_perm,
    data_record_mode,
    permutation_state,
    args,
):
    policy.eval()
    rows = []
    for idx in range(int(args.eval_samples)):
        features_cpu, xs, _block_losses_cpu = collect_online_feature_batch(
            model=model,
            tokens=tokens,
            states=1,
            probe_batch_size=int(args.eval_batch_size),
            layer=int(args.layer),
            head=int(args.head),
            export_type=args.export_type,
            rng=rng,
            device=device,
            dtype=dtype,
            token_perm=token_perm,
            data_record_mode=data_record_mode,
            feature_clip=float(args.feature_clip),
        )
        x = xs[0]
        features = features_cpu.to(device=device)
        logits = policy(features).squeeze(0).detach().cpu()
        order = order_from_logits_desc(logits)
        reverse_order = torch.flip(order, dims=[0])
        map_loss = full_loss_for_block_orders(model=model, x=x, block_orders=order.to(device), device=device, dtype=dtype)
        reverse_loss = full_loss_for_block_orders(
            model=model,
            x=x,
            block_orders=reverse_order.to(device),
            device=device,
            dtype=dtype,
        )
        random_losses = []
        for _ in range(int(args.random_eval_orders)):
            random_orders = random_block_orders(x.size(0), int(model.num_blocks), device)
            random_losses.append(
                full_loss_for_block_orders(model=model, x=x, block_orders=random_orders, device=device, dtype=dtype)
            )
        policy_sample_losses = []
        sampled_orders, _ = sample_pl_orders_and_logps(logits.view(1, -1).to(device), int(args.policy_sample_eval_orders))
        for sample_idx in range(int(args.policy_sample_eval_orders)):
            policy_sample_losses.append(
                full_loss_for_block_orders(
                    model=model,
                    x=x,
                    block_orders=sampled_orders[0, sample_idx],
                    device=device,
                    dtype=dtype,
                )
            )
        diag = original_order_diagnostics(order, permutation_state)
        random_mean = float(np.mean(random_losses))
        random_best = float(np.min(random_losses))
        sample_mean = float(np.mean(policy_sample_losses)) if policy_sample_losses else float("nan")
        sample_best = float(np.min(policy_sample_losses)) if policy_sample_losses else float("nan")
        row = {
            "sample_id": idx,
            "mlp_map_full_loss": map_loss,
            "policy_sample_mean_full_loss": sample_mean,
            "policy_sample_best_full_loss": sample_best,
            "random_mean_full_loss": random_mean,
            "random_best_full_loss": random_best,
            "mlp_minus_random_mean": map_loss - random_mean,
            "policy_sample_mean_minus_random_mean": sample_mean - random_mean,
            "mlp_minus_random_best": map_loss - random_best,
            "reverse_mlp_full_loss": reverse_loss,
            "forward_minus_reverse_loss": map_loss - reverse_loss,
            "logit_std": float(logits.std(unbiased=False).item()),
            "logit_entropy": float(logits_entropy(logits.view(1, -1)).item()),
            "diagnostic/current_l2r_tau": diag["current_l2r_tau"],
            "diagnostic/within_unit_adjacent_rate": diag["within_unit_adjacent_rate"],
            "diagnostic/order_first16_current": json.dumps(diag["order_first16_current"]),
        }
        if "original_l2r_tau" in diag:
            row["diagnostic/original_l2r_tau"] = diag["original_l2r_tau"]
            row["diagnostic/original_id_tau"] = diag["original_id_tau"]
            row["diagnostic/within_unit_adjacent_rate_original"] = diag["within_unit_adjacent_rate_original"]
            row["diagnostic/order_first16_original"] = json.dumps(diag["order_first16_original"])
        rows.append(row)
        if (idx + 1) % max(1, int(args.progress_interval)) == 0:
            print_progress(f"evaluated_nll_samples={idx + 1}/{args.eval_samples}")
    summary = mean_dicts(rows)
    summary.update(
        {
            "num_eval_attention_samples": int(args.eval_samples),
            "random_eval_orders": int(args.random_eval_orders),
            "policy_sample_eval_orders": int(args.policy_sample_eval_orders),
            "eval_batch_size": int(args.eval_batch_size),
        }
    )
    return rows, summary


def make_method_md(args):
    return f"""# Method

## Loss Candidate

This try trains the Attn-MLP with an online sampled-order frozen-NLL reward
plus insertion/segment-move preferences. For each training state, the script samples one
token batch, uses a random reveal order only to extract current-frame attention
and current-frame block loss features, then evaluates MLP-sampled orders and
move perturbations on the same token batch.

The MLP outputs `s in R^64`, defining a Plackett-Luce policy. For each state:

```text
sample order o ~ PL(s)
baseline = mean frozen_NLL(random orders on the same x)
full_advantage = baseline_full_NLL - full_NLL(o on the same x)
prefix_advantage = baseline_prefix_NLL - prefix_NLL(o on the same x)
attn_advantage = attn_locality(o) - baseline_attn_locality
attn_path_advantage = attn_path_band(o) - baseline_attn_path_band
advantage = full_advantage + lambda_prefix_reward * prefix_advantage
          + lambda_attn_locality_reward * attn_advantage
          + lambda_attn_path_reward * attn_path_advantage
L_pg = - stopgrad(advantage / scale) * log PL_s(o)
```

For the move preference term, the script samples a base order from the current
policy, removes a block or short segment, inserts it at a distant position, and
evaluates both orders under the frozen AO-GPT:

```text
delta = frozen_NLL(moved_order) - frozen_NLL(base_order)
if delta >= 0: preserve the base relative order for every crossed pair
if delta < 0:  prefer the moved relative order for every crossed pair
L_move = weighted_BCE((s_a - s_b) / tau, preference)
```

The pair labels are defined only for block pairs whose relative order changes
between the base order and moved order. One frozen-NLL comparison can therefore
produce many pair labels. The target is the frozen-NLL delta from the same
current state, not an attention-derived order and not an original-order target.

The objective also includes non-collapse terms, a softmax-entropy ceiling,
centering, and a mild attention-graph smoothness regularizer. When enabled,
`lambda_loss_axis_pair` adds a pairwise auxiliary from current-frame block-loss
scores on the same sampled state. The reward is the frozen AO-GPT NLL itself,
not an attention-derived teacher order and not an original-order target.

## Allowed Signals

- current-frame attention and current-frame block losses as MLP input
- MLP-sampled current-frame orders
- random-order frozen-NLL baseline on the same token batch
- frozen AO-GPT full-NLL reward on the same token batch
- frozen AO-GPT prefix-NLL reward on the same token batch, when
  `lambda_prefix_reward > 0`
- attention-locality reward from the current attention matrix, when
  `lambda_attn_locality_reward > 0`
- attention path/band reward from the current attention matrix, when
  `lambda_attn_path_reward > 0`
- insertion/segment-move preference targets from frozen AO-GPT NLL deltas on
  the same token batch
- optional current-frame block-loss pair axis, when
  `lambda_loss_axis_pair > 0`
- optional anchor to a previous allowed learned policy, when
  `lambda_anchor_l2 > 0`

## Forbidden Signals

The following are not used in training, direction selection, model selection,
early stopping, or loss-design selection:

```text
{json.dumps(FORBIDDEN_SIGNALS + ["generated_teacher_order_supervision"], indent=2)}
```

`diagnostic/original_l2r_tau` is written only after the full try finishes.

## Formal Budget

- train steps: `{args.train_steps}`
- states per step: `{args.states_per_step}`
- train states: `{train_states(args)}`
- train order evals: `{train_order_evals(args)}`
- sampled orders per state: `{args.sampled_orders_per_state}`
- random baseline orders per state: `{args.random_baseline_orders}`
- move base orders per state: `{args.move_base_orders_per_state}`
- moves per base: `{args.moves_per_base}`
- move mode: `{args.move_mode}`
- min move distance: `{args.min_move_distance}`
- max move distance: `{args.max_move_distance}`
- max segment length: `{args.max_segment_len}`
- max pairs per move: `{args.max_pairs_per_move}`
- `lambda_prefix_reward`: `{args.lambda_prefix_reward}`
- prefix k: `{args.prefix_k}`
- `lambda_attn_locality_reward`: `{args.lambda_attn_locality_reward}`
- attention locality sigma: `{args.attn_locality_sigma}`
- `lambda_attn_path_reward`: `{args.lambda_attn_path_reward}`
- attention path bandwidth: `{args.attn_path_bandwidth}`
- `lambda_move_pref`: `{args.lambda_move_pref}`
- `lambda_loss_axis_pair`: `{args.lambda_loss_axis_pair}`
- loss axis mode: `{args.loss_axis_mode}`
- loss axis tau: `{args.loss_axis_tau}`
- loss axis margin: `{args.loss_axis_margin}`
- move preference tau: `{args.move_pref_tau}`
- move preference margin: `{args.move_pref_margin}`
- move min pair weight: `{args.move_min_pair_weight}`
- init policy path: `{args.init_policy_path}`
- anchor policy path: `{args.anchor_policy_path}`
- `lambda_anchor_l2`: `{args.lambda_anchor_l2}`
- `lambda_entropy_floor`: `{args.lambda_entropy_floor}`
- `min_entropy`: `{args.min_entropy}`
- final eval samples: `{args.eval_samples}`
- random eval orders: `{args.random_eval_orders}`
- policy sample eval orders: `{args.policy_sample_eval_orders}`

## Success Gate

The current continuation requires both:

- `diagnostic/original_l2r_tau > {args.success_tau_gate}`
- frozen-NLL margin vs random mean greater than `{args.success_random_margin}`
"""


def make_failure_analysis(summary):
    ev = summary["eval_summary"]
    opt = summary["optimization"]
    train_reward = opt["final_window"].get("advantage_mean", float("nan"))
    map_margin = -ev.get("mlp_minus_random_mean", float("inf"))
    sample_margin = -ev.get("policy_sample_mean_minus_random_mean", float("inf"))
    nll_ok = map_margin > summary["training_meta"]["success_random_margin"]
    dist_ok = sample_margin > summary["training_meta"]["success_random_margin"]
    reverse_ok = ev.get("forward_minus_reverse_loss", float("inf")) < 0.0
    tau_ok = ev.get("diagnostic/original_l2r_tau", float("-inf")) > summary["training_meta"]["success_tau_gate"]
    success = bool((nll_ok or dist_ok) and tau_ok)
    reasons = []
    if not nll_ok:
        reasons.append("MAP MLP order did not clearly beat random mean loss")
    if not dist_ok:
        reasons.append("sampled MLP distribution did not clearly beat random mean loss")
    if not tau_ok:
        reasons.append("post-hoc diagnostic original_l2r_tau did not exceed the current gate")
    next_step = "success gate reached; replicate with another seed and then test joint-training insertion"
    if not success:
        if train_reward > 0.0 and not (nll_ok or dist_ok):
            next_step = "increase online training states and lower entropy ceiling; training reward exists but does not transfer"
        elif tau_ok and not (nll_ok or dist_ok):
            next_step = "tau gate improved but frozen NLL did not; strengthen direct reward or add risk-sensitive best-of-samples objective"
        else:
            next_step = "increase move perturbation contrast or combine move preference with a stronger direct reward; this move objective did not satisfy both gates"
    return f"""# Failure Analysis

## Acceptance Checks

- final train advantage positive: `{train_reward > 0.0}` (`advantage_mean={train_reward:.6f}`)
- MAP loss clearly better than random mean: `{nll_ok}` (`margin={map_margin:.6f}`)
- sampled distribution loss clearly better than random mean: `{dist_ok}` (`margin={sample_margin:.6f}`)
- forward better than reverse: `{reverse_ok}` (`forward_minus_reverse_loss={ev.get('forward_minus_reverse_loss'):.6f}`)
- diagnostic original_l2r_tau > {summary['training_meta']['success_tau_gate']}: `{tau_ok}` (`tau={ev.get('diagnostic/original_l2r_tau'):.6f}`)

## Verdict

success: `{success}`

Failure reasons:

{chr(10).join(f'- {reason}' for reason in reasons) if reasons else '- none'}

## Next Try

{next_step}

Original-frame tau and OriginalL2R were not used in training, direction
selection, model selection, early stopping, or loss design.
"""


def write_try_docs(args, summary=None):
    args.try_dir.mkdir(parents=True, exist_ok=True)
    (args.try_dir / "method.md").write_text(make_method_md(args), encoding="utf-8")
    (args.try_dir / "commands.sh").write_text(command_invocation() + "\n", encoding="utf-8")
    readme = f"""# Attn-MLP Move Preference Try

- checkpoint: `{args.ckpt_path}`
- loss: `{LOSS_NAME}`
- frame: `current`
- train steps: `{args.train_steps}`
- states per step: `{args.states_per_step}`
- train states: `{train_states(args)}`
- train order evals: `{train_order_evals(args)}`
- final eval samples: `{args.eval_samples}`
- random eval orders: `{args.random_eval_orders}`
- policy sample eval orders: `{args.policy_sample_eval_orders}`
- model selection: fixed final step, no validation-PPL shortcut

Key files:

- `method.md`
- `failure_analysis.md`
- `train_log.jsonl`
- `metrics.csv`
- `eval_summary.json`
- `eval_rows.csv`
- `config.json`
- `policy.pt`
- `commands.sh`
"""
    (args.try_dir / "README.md").write_text(readme, encoding="utf-8")
    if summary is not None:
        (args.try_dir / "failure_analysis.md").write_text(make_failure_analysis(summary), encoding="utf-8")


def main():
    args = parse_args()
    validate_budget(args)
    args.try_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    device = torch.device(args.device)
    write_try_docs(args)

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

    hidden_dims = parse_int_list(args.hidden_dims, default=[1024, 1024])
    policy_config = {
        "num_blocks": int(model.num_blocks),
        "input_channels": 3,
        "hidden_dims": hidden_dims,
        "dropout": float(args.dropout),
        "activation": args.activation,
    }
    policy = AttentionDistributionOrderMLP(**policy_config).to(device)
    if args.init_policy_path is not None:
        init_payload = torch.load(args.init_policy_path, map_location="cpu")
        policy.load_state_dict(init_payload["model_state_dict"], strict=True)
        print_progress(f"loaded_init_policy={args.init_policy_path}")
    anchor_policy = None
    if args.anchor_policy_path is not None and float(args.lambda_anchor_l2) > 0.0:
        anchor_payload = torch.load(args.anchor_policy_path, map_location="cpu")
        anchor_config = dict(anchor_payload.get("config") or policy_config)
        anchor_policy = AttentionDistributionOrderMLP(**anchor_config).to(device)
        anchor_policy.load_state_dict(anchor_payload["model_state_dict"], strict=True)
        anchor_policy.eval()
        for param in anchor_policy.parameters():
            param.requires_grad_(False)
        print_progress(f"loaded_anchor_policy={args.anchor_policy_path}")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "frame": "current",
            "loss": LOSS_NAME,
            "forbidden_signals": FORBIDDEN_SIGNALS + ["generated_teacher_order_supervision"],
            "original_l2r_is_diagnostic_only": True,
            "tau_is_diagnostic_only": True,
            "model_selection": "fixed_final_step_no_val_ppl_shortcut",
            "checkpoint_iter": checkpoint.get("iter_num"),
            "checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
            "data_permutation_applied": permutation_state is not None,
            "train_states": train_states(args),
            "train_order_evals": train_order_evals(args),
        }
    )
    write_json(args.try_dir / "config.json", config_payload)
    wandb_run = init_wandb_if_requested(args, config_payload)

    train_log_path = args.try_dir / "train_log.jsonl"
    if train_log_path.exists():
        train_log_path.unlink()
    metrics_rows = []
    print_progress(f"loaded_ckpt={args.ckpt_path}")
    for step in range(1, int(args.train_steps) + 1):
        metrics = train_step(
            policy=policy,
            optimizer=optimizer,
            model=model,
            tokens=train_tokens,
            rng=rng,
            device=device,
            dtype=args.dtype,
            token_perm=token_perm,
            data_record_mode=data_record_mode,
            args=args,
            anchor_policy=anchor_policy,
        )
        row = {"step": step, "phase": "train"}
        row.update(metrics)
        metrics_rows.append(row)
        append_jsonl(train_log_path, row)
        log_wandb_scalars(wandb_run, "train", metrics, step=step)
        if step == 1 or step % max(1, int(args.progress_interval)) == 0:
            print_progress(json.dumps(row))

    write_csv(args.try_dir / "metrics.csv", metrics_rows)
    final_window = mean_dicts(metrics_rows[-min(len(metrics_rows), 25) :])

    print_progress(f"running_full_eval_samples={args.eval_samples}")
    eval_rows, eval_summary = evaluate_policy_nll_online(
        policy=policy,
        model=model,
        tokens=val_tokens,
        rng=rng,
        device=device,
        dtype=args.dtype,
        token_perm=token_perm,
        data_record_mode=data_record_mode,
        permutation_state=permutation_state,
        args=args,
    )
    write_csv(args.try_dir / "eval_rows.csv", eval_rows)

    policy_path = args.try_dir / "policy.pt"
    summary = {
        "script": "scripts/train/train_attn_mlp_move_preference.py",
        "command": command_invocation(),
        "policy_path": str(policy_path),
        "checkpoint": {
            "ckpt_path": str(args.ckpt_path),
            "iter_num": checkpoint.get("iter_num"),
            "best_val_loss": checkpoint.get("best_val_loss"),
        },
        "policy_config": policy_config,
        "training_meta": {
            "frame": "current",
            "loss": LOSS_NAME,
            "forbidden_signals": FORBIDDEN_SIGNALS + ["generated_teacher_order_supervision"],
            "original_l2r_is_diagnostic_only": True,
            "tau_is_diagnostic_only": True,
            "model_selection": "fixed_final_step_no_val_ppl_shortcut",
            "success_random_margin": float(args.success_random_margin),
            "success_tau_gate": float(args.success_tau_gate),
        },
        "samples": {
            "train_steps": int(args.train_steps),
            "states_per_step": int(args.states_per_step),
            "train_states": train_states(args),
            "train_order_evals": train_order_evals(args),
            "sampled_orders_per_state": int(args.sampled_orders_per_state),
            "random_baseline_orders": int(args.random_baseline_orders),
            "move_base_orders_per_state": int(args.move_base_orders_per_state),
            "moves_per_base": int(args.moves_per_base),
            "move_mode": str(args.move_mode),
            "min_move_distance": int(args.min_move_distance),
            "max_move_distance": int(args.max_move_distance),
            "max_segment_len": int(args.max_segment_len),
            "max_pairs_per_move": int(args.max_pairs_per_move),
            "lambda_prefix_reward": float(args.lambda_prefix_reward),
            "prefix_k": int(args.prefix_k),
            "lambda_attn_locality_reward": float(args.lambda_attn_locality_reward),
            "attn_locality_sigma": float(args.attn_locality_sigma),
            "lambda_attn_path_reward": float(args.lambda_attn_path_reward),
            "attn_path_bandwidth": int(args.attn_path_bandwidth),
            "eval_samples": int(args.eval_samples),
            "random_eval_orders": int(args.random_eval_orders),
            "policy_sample_eval_orders": int(args.policy_sample_eval_orders),
        },
        "optimization": {
            "steps": int(args.train_steps),
            "final_window": final_window,
        },
        "eval_summary": eval_summary,
        "diagnostic_acceptance": {
            "origin_tau_gate": float(args.success_tau_gate),
            "origin_tau_passed": bool(
                eval_summary.get("diagnostic/original_l2r_tau", float("-inf")) > float(args.success_tau_gate)
            ),
            "map_nll_vs_random_passed": bool(
                eval_summary.get("mlp_minus_random_mean", float("inf")) < -float(args.success_random_margin)
            ),
            "sample_mean_nll_vs_random_passed": bool(
                eval_summary.get("policy_sample_mean_minus_random_mean", float("inf"))
                < -float(args.success_random_margin)
            ),
            "forward_vs_reverse_passed": bool(eval_summary.get("forward_minus_reverse_loss", float("inf")) < 0.0),
        },
    }
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "config": policy_config,
            "policy_type": "attention_onpolicy_move_preference_mlp",
            "training_meta": summary["training_meta"],
            "metrics": {"train": final_window, "eval": eval_summary},
        },
        policy_path,
    )
    write_json(args.try_dir / "eval_summary.json", summary)
    write_try_docs(args, summary=summary)
    log_wandb_scalars(wandb_run, "eval", eval_summary, step=int(args.train_steps))
    log_wandb_scalars(wandb_run, "acceptance", summary["diagnostic_acceptance"], step=int(args.train_steps))
    if wandb_run is not None:
        wandb_run.summary.update(jsonable_for_wandb(summary["diagnostic_acceptance"]))
        wandb_run.summary.update(
            {
                "eval/mlp_minus_random_mean": eval_summary.get("mlp_minus_random_mean"),
                "eval/policy_sample_mean_minus_random_mean": eval_summary.get(
                    "policy_sample_mean_minus_random_mean"
                ),
                "eval/diagnostic_original_l2r_tau": eval_summary.get("diagnostic/original_l2r_tau"),
                "eval/forward_minus_reverse_loss": eval_summary.get("forward_minus_reverse_loss"),
                "policy_path": str(policy_path),
                "try_dir": str(args.try_dir),
            }
        )
        wandb_run.finish()
    print_progress(f"try_dir={args.try_dir}")
    print_progress(json.dumps(summary["diagnostic_acceptance"], indent=2))


if __name__ == "__main__":
    main()
