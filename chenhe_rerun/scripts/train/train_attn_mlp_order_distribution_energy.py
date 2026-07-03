#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from attn_mlp_implicit_axis_common import (  # noqa: E402
    FORBIDDEN_SIGNALS,
    aggregate_layerhead_attention_to_current_blocks,
    append_jsonl,
    autocast_context,
    expand_orders_for_model,
    full_loss_for_block_orders,
    infer_data_dir,
    infer_data_record_mode,
    kendall_tau_between_orders,
    load_frozen_aogpt,
    load_permutation_state,
    load_tokens,
    mean_dicts,
    order_from_logits_desc,
    original_order_diagnostics,
    parse_int_list,
    random_block_orders,
    robust_z_offdiag,
    sample_batch,
    tensor_metrics_to_floats,
    write_csv,
    write_json,
)


class AttentionDistributionOrderMLP(nn.Module):
    """Map current-frame attention/loss feature planes to 64 order logits."""

    def __init__(
        self,
        num_blocks: int,
        input_channels: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.num_blocks = int(num_blocks)
        self.input_channels = int(input_channels)
        hidden_dims = [int(value) for value in (hidden_dims or [1024, 1024])]
        dims = [self.num_blocks * self.num_blocks * self.input_channels] + hidden_dims + [self.num_blocks]
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(make_activation(activation))
                if float(dropout) > 0.0:
                    layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor):
        if features.ndim != 4:
            raise ValueError(f"Expected features shape (B,C,N,N), got {tuple(features.shape)}")
        expected = (self.input_channels, self.num_blocks, self.num_blocks)
        if tuple(features.shape[1:]) != expected:
            raise ValueError(f"Expected feature suffix {expected}, got {tuple(features.shape[1:])}")
        return self.net(features.float().reshape(features.size(0), -1))


def make_activation(name: str):
    name = str(name or "gelu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unsupported activation={name!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Attn-MLP with sampled-order energy distribution loss.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--try_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="wikitext103")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--num_train_samples", type=int, default=2000)
    parser.add_argument("--num_val_samples", type=int, default=512)
    parser.add_argument("--probe_batch_size", type=int, default=64)
    parser.add_argument("--candidate_orders_per_sample", type=int, default=5)
    parser.add_argument(
        "--candidate_bank_size",
        type=int,
        default=0,
        help="If >0, collect this many random candidate orders per state and train on resampled subsets.",
    )
    parser.add_argument(
        "--train_candidates_per_epoch",
        type=int,
        default=0,
        help="If >0 and smaller than the bank, sample this many candidates per state inside each training epoch.",
    )
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--head", type=int, default=3)
    parser.add_argument("--export_type", choices=("with_none", "without_none"), default="with_none")
    parser.add_argument("--feature_clip", type=float, default=8.0)
    parser.add_argument("--hidden_dims", type=str, default="1024,1024")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--target_loss_temp", type=float, default=0.005)
    parser.add_argument("--pl_temp", type=float, default=1.0)
    parser.add_argument("--pair_temp", type=float, default=1.0)
    parser.add_argument("--candidate_margin", type=float, default=0.003)
    parser.add_argument("--lambda_listwise", type=float, default=1.0)
    parser.add_argument("--lambda_pairwise", type=float, default=0.5)
    parser.add_argument("--lambda_expected_loss", type=float, default=0.25)
    parser.add_argument("--lambda_var", type=float, default=0.01)
    parser.add_argument("--lambda_entropy_floor", type=float, default=0.005)
    parser.add_argument("--lambda_candidate_entropy_floor", type=float, default=0.02)
    parser.add_argument("--lambda_diversity", type=float, default=0.01)
    parser.add_argument("--lambda_index_bias", type=float, default=0.01)
    parser.add_argument("--min_std", type=float, default=0.1)
    parser.add_argument("--min_entropy", type=float, default=1.0)
    parser.add_argument("--min_candidate_entropy", type=float, default=0.25)
    parser.add_argument("--eval_samples", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--random_eval_orders", type=int, default=16)
    parser.add_argument("--policy_sample_eval_orders", type=int, default=8)
    parser.add_argument("--success_random_margin", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--progress_interval", type=int, default=100)
    parser.add_argument("--recompute_dataset", action="store_true")
    parser.add_argument("--allow_small_debug", action="store_true")
    return parser.parse_args()


def print_progress(text: str):
    print(text, flush=True)


def validate_budget(args):
    if bool(args.allow_small_debug):
        return
    bank_size = effective_candidate_bank_size(args)
    checks = [
        ("num_train_samples", int(args.num_train_samples), 2000),
        ("num_val_samples", int(args.num_val_samples), 512),
        ("epochs", int(args.epochs), 50),
        ("eval_samples", int(args.eval_samples), 512),
        ("random_eval_orders", int(args.random_eval_orders), 16),
    ]
    train_order_probes = int(args.num_train_samples) * int(bank_size)
    checks.append(("train_order_probes", train_order_probes, 10000))
    too_small = [(name, value, minimum) for name, value, minimum in checks if value < minimum]
    if too_small:
        text = ", ".join(f"{name}={value} < {minimum}" for name, value, minimum in too_small)
        raise ValueError(f"Formal try budget is too small: {text}. Use --allow_small_debug only for smoke.")


def effective_candidate_bank_size(args):
    if int(args.candidate_bank_size) > 0:
        return int(args.candidate_bank_size)
    return int(args.candidate_orders_per_sample)


def effective_train_candidate_count(args):
    bank_size = effective_candidate_bank_size(args)
    if int(args.train_candidates_per_epoch) > 0:
        return min(int(args.train_candidates_per_epoch), bank_size)
    return min(int(args.candidate_orders_per_sample), bank_size)


def robust_z_vector(values: torch.Tensor, eps: float = 1e-6):
    values = values.float()
    median = values.median(dim=-1, keepdim=True).values
    q25 = torch.quantile(values, 0.25, dim=-1, keepdim=True)
    q75 = torch.quantile(values, 0.75, dim=-1, keepdim=True)
    scale = ((q75 - q25) / 1.349).clamp_min(eps)
    return (values - median) / scale


def block_losses_to_current(block_losses_reveal: torch.Tensor, block_orders: torch.Tensor):
    current = torch.empty_like(block_losses_reveal)
    current.scatter_(1, block_orders.long(), block_losses_reveal.float())
    return current


def make_features(matrices: torch.Tensor, block_losses: torch.Tensor, feature_clip: float = 0.0):
    rows = []
    n = int(matrices.size(-1))
    eye = torch.eye(n, dtype=torch.bool)
    for matrix, losses in zip(matrices, block_losses):
        a_z = robust_z_offdiag(matrix.float())
        sym = torch.maximum(a_z, a_z.t())
        loss_z = robust_z_vector(losses.view(1, -1)).view(-1)
        loss_diff = loss_z[:, None] - loss_z[None, :]
        planes = torch.stack([a_z, sym, loss_diff], dim=0)
        if float(feature_clip) > 0.0:
            clip = float(feature_clip)
            planes = planes.clamp(min=-clip, max=clip)
        planes[:, eye] = 0.0
        rows.append(planes)
    return torch.stack(rows, dim=0).float()


@torch.no_grad()
def collect_distribution_samples(
    *,
    model,
    tokens,
    num_samples: int,
    probe_batch_size: int,
    candidate_orders_per_sample: int,
    layer: int,
    head: int,
    export_type: str,
    rng: np.random.Generator,
    device: torch.device,
    dtype: str,
    token_perm: torch.Tensor | None,
    data_record_mode: str,
    progress_label: str,
    progress_interval: int,
):
    matrices = []
    block_losses = []
    candidate_orders = []
    candidate_losses = []
    first_probe_orders = []
    n = int(model.num_blocks)
    layer = int(layer)
    head = int(head)
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
        layer_heads = aggregate_layerhead_attention_to_current_blocks(
            attentions[local_layer].detach(),
            reveal_orders,
            block_len=int(model.block_order_block_len),
            export_type=export_type,
        )
        if head < 0:
            matrix = layer_heads.mean(dim=0)
        else:
            matrix = layer_heads[head]
        matrix = matrix.detach().float().cpu()
        matrix.fill_diagonal_(0.0)
        reveal_block_loss = token_losses.view(
            int(probe_batch_size),
            n,
            int(model.block_order_block_len),
        ).mean(dim=-1)
        current_loss = block_losses_to_current(reveal_block_loss, reveal_orders)
        candidates = []
        losses = []
        for _ in range(int(candidate_orders_per_sample)):
            order = torch.randperm(n, device=device)
            loss = full_loss_for_block_orders(model=model, x=x, block_orders=order, device=device, dtype=dtype)
            candidates.append(order.detach().cpu())
            losses.append(float(loss))
        matrices.append(matrix)
        block_losses.append(current_loss.mean(dim=0).detach().float().cpu())
        candidate_orders.append(torch.stack(candidates, dim=0))
        candidate_losses.append(torch.tensor(losses, dtype=torch.float32))
        if len(first_probe_orders) < 4:
            first_probe_orders.append(reveal_orders[0].detach().cpu().tolist())
        if (sample_idx + 1) % max(1, int(progress_interval)) == 0:
            print_progress(f"{progress_label}={sample_idx + 1}/{num_samples}")
    return {
        "matrices": torch.stack(matrices, dim=0),
        "block_losses": torch.stack(block_losses, dim=0),
        "candidate_orders": torch.stack(candidate_orders, dim=0).long(),
        "candidate_losses": torch.stack(candidate_losses, dim=0).float(),
        "probe_orders_first": first_probe_orders,
    }


def plackett_luce_log_prob(logits: torch.Tensor, orders: torch.Tensor):
    if orders.ndim == 2:
        orders = orders.unsqueeze(1)
    logits = logits.float()
    orders = orders.to(device=logits.device, dtype=torch.long)
    bsz, count, n = orders.shape
    expanded = logits.unsqueeze(1).expand(bsz, count, n)
    ordered_scores = torch.gather(expanded, 2, orders)
    denoms = [torch.logsumexp(ordered_scores[:, :, step:], dim=-1) for step in range(n)]
    denom = torch.stack(denoms, dim=-1)
    return (ordered_scores - denom).sum(dim=-1)


def logits_entropy(logits: torch.Tensor):
    probs = torch.softmax(logits.float(), dim=-1)
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)


def distribution_loss(
    logits: torch.Tensor,
    candidate_orders: torch.Tensor,
    candidate_losses: torch.Tensor,
    args,
):
    device = logits.device
    s = logits.float()
    orders = candidate_orders.to(device=device, dtype=torch.long)
    losses = candidate_losses.to(device=device, dtype=torch.float32)
    pl_logp = plackett_luce_log_prob(s, orders)
    centered_losses = losses - losses.mean(dim=1, keepdim=True)
    target_log = -centered_losses / max(float(args.target_loss_temp), 1e-8)
    target_prob = torch.softmax(target_log, dim=-1).detach()
    pred_log_prob = F.log_softmax(pl_logp / max(float(args.pl_temp), 1e-8), dim=-1)
    pred_prob = pred_log_prob.exp()
    loss_listwise = -(target_prob * pred_log_prob).sum(dim=-1).mean()

    k = int(losses.size(1))
    ii, jj = torch.triu_indices(k, k, offset=1, device=device)
    pair_logits = (pl_logp[:, ii] - pl_logp[:, jj]) / max(float(args.pair_temp), 1e-8)
    pair_target = torch.sigmoid((losses[:, jj] - losses[:, ii]) / max(float(args.target_loss_temp), 1e-8)).detach()
    pair_weight = ((losses[:, jj] - losses[:, ii]).abs() / max(float(args.candidate_margin), 1e-8)).clamp(0.0, 1.0)
    loss_pairwise = F.binary_cross_entropy_with_logits(pair_logits, pair_target, reduction="none")
    loss_pairwise = (loss_pairwise * pair_weight).sum() / pair_weight.sum().clamp_min(1e-6)

    loss_scale = losses.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized_losses = centered_losses / loss_scale
    loss_expected = (pred_prob * normalized_losses).sum(dim=-1).mean()

    std = s.std(dim=1, unbiased=False)
    loss_var = torch.relu(s.new_tensor(float(args.min_std)) - std).pow(2).mean()
    entropy = logits_entropy(s)
    loss_entropy_floor = torch.relu(s.new_tensor(float(args.min_entropy)) - entropy).pow(2).mean()
    if s.size(0) > 1:
        x = (s - s.mean(dim=1, keepdim=True)) / s.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)
        sim = x @ x.t()
        offdiag = ~torch.eye(s.size(0), dtype=torch.bool, device=device)
        loss_diversity = sim[offdiag].pow(2).mean()
    else:
        loss_diversity = s.new_tensor(0.0)
    index = torch.arange(s.size(1), device=device, dtype=torch.float32)
    index = (index - index.mean()) / index.std(unbiased=False).clamp_min(1e-6)
    x = (s - s.mean(dim=1, keepdim=True)) / s.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    loss_index_bias = (x * index.view(1, -1)).mean(dim=1).abs().mean()

    pred_best = pl_logp.argmax(dim=1)
    target_best = losses.argmin(dim=1)
    top1_match = (pred_best == target_best).float().mean()
    pred_entropy = -(pred_prob * pred_prob.clamp_min(1e-12).log()).sum(dim=-1).mean()
    target_entropy = -(target_prob * target_prob.clamp_min(1e-12).log()).sum(dim=-1).mean()
    loss_candidate_entropy_floor = torch.relu(
        s.new_tensor(float(args.min_candidate_entropy)) - pred_entropy
    ).pow(2)
    pred_expected_loss = (pred_prob.detach() * losses).sum(dim=-1)
    random_candidate_mean = losses.mean(dim=-1)
    expected_minus_candidate_mean = (pred_expected_loss - random_candidate_mean).mean()

    total = (
        float(args.lambda_listwise) * loss_listwise
        + float(args.lambda_pairwise) * loss_pairwise
        + float(args.lambda_expected_loss) * loss_expected
        + float(args.lambda_var) * loss_var
        + float(args.lambda_entropy_floor) * loss_entropy_floor
        + float(args.lambda_candidate_entropy_floor) * loss_candidate_entropy_floor
        + float(args.lambda_diversity) * loss_diversity
        + float(args.lambda_index_bias) * loss_index_bias
    )
    return {
        "loss_total": total,
        "loss_listwise_ce": loss_listwise,
        "loss_pairwise_candidate": loss_pairwise,
        "loss_expected_candidate": loss_expected,
        "loss_var": loss_var,
        "loss_entropy_floor": loss_entropy_floor,
        "loss_candidate_entropy_floor": loss_candidate_entropy_floor,
        "loss_diversity": loss_diversity,
        "loss_index_bias": loss_index_bias,
        "candidate_top1_match": top1_match.detach(),
        "candidate_expected_minus_mean_loss": expected_minus_candidate_mean.detach(),
        "candidate_pred_entropy": pred_entropy.detach(),
        "candidate_target_entropy": target_entropy.detach(),
        "candidate_loss_range": (losses.max(dim=1).values - losses.min(dim=1).values).mean().detach(),
        "logit_std": std.mean().detach(),
        "logit_entropy": entropy.mean().detach(),
    }


def select_candidate_subset(candidate_orders: torch.Tensor, candidate_losses: torch.Tensor, subset_size: int):
    subset_size = int(subset_size)
    if subset_size <= 0 or subset_size >= int(candidate_orders.size(1)):
        return candidate_orders, candidate_losses
    batch = int(candidate_orders.size(0))
    bank = int(candidate_orders.size(1))
    n = int(candidate_orders.size(2))
    selectors = torch.stack([torch.randperm(bank)[:subset_size] for _ in range(batch)], dim=0)
    selected_orders = torch.gather(candidate_orders, 1, selectors[:, :, None].expand(batch, subset_size, n))
    selected_losses = torch.gather(candidate_losses, 1, selectors)
    return selected_orders, selected_losses


@torch.no_grad()
def evaluate_objective(policy, features, candidate_orders, candidate_losses, args, device):
    policy.eval()
    rows = []
    orders = []
    for start in range(0, int(features.size(0)), int(args.batch_size)):
        end = min(int(features.size(0)), start + int(args.batch_size))
        feat = features[start:end].to(device=device)
        cand_orders = candidate_orders[start:end]
        cand_losses = candidate_losses[start:end]
        logits = policy(feat)
        metrics = distribution_loss(logits, cand_orders, cand_losses, args)
        rows.append(tensor_metrics_to_floats(metrics))
        orders.extend([order_from_logits_desc(row).detach().cpu() for row in logits])
    out = mean_dicts(rows)
    if len(orders) > 1:
        taus = [kendall_tau_between_orders(orders[idx - 1], orders[idx]) for idx in range(1, len(orders))]
        out["order_change_kendall_current"] = float(np.mean(taus))
    return out


def train_epoch(policy, optimizer, features, candidate_orders, candidate_losses, args, device):
    policy.train()
    idx_order = torch.randperm(int(features.size(0)))
    rows = []
    for start in range(0, int(idx_order.numel()), int(args.batch_size)):
        idx = idx_order[start : start + int(args.batch_size)]
        feat = features.index_select(0, idx).to(device=device)
        cand_orders = candidate_orders.index_select(0, idx)
        cand_losses = candidate_losses.index_select(0, idx)
        cand_orders, cand_losses = select_candidate_subset(
            cand_orders,
            cand_losses,
            subset_size=effective_train_candidate_count(args),
        )
        optimizer.zero_grad(set_to_none=True)
        logits = policy(feat)
        metrics = distribution_loss(logits, cand_orders, cand_losses, args)
        metrics["loss_total"].backward()
        if float(args.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(args.grad_clip))
        optimizer.step()
        rows.append(tensor_metrics_to_floats(metrics))
    return mean_dicts(rows)


def sample_plackett_luce_orders(logits: torch.Tensor, num_samples: int):
    logits = logits.detach().float()
    rows = []
    for _ in range(int(num_samples)):
        uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
        gumbel = -torch.log(-torch.log(uniform))
        rows.append(torch.argsort(logits + gumbel, descending=True))
    return torch.stack(rows, dim=0).long()


@torch.no_grad()
def evaluate_policy_nll(
    *,
    policy,
    features,
    candidate_orders,
    candidate_losses,
    model,
    tokens,
    rng,
    device,
    dtype,
    token_perm,
    data_record_mode,
    eval_batch_size,
    random_eval_orders,
    policy_sample_eval_orders,
    permutation_state,
    args,
):
    policy.eval()
    rows = []
    for idx in range(int(features.size(0))):
        feat = features[idx : idx + 1].to(device=device)
        logits = policy(feat).squeeze(0).detach().cpu()
        order = order_from_logits_desc(logits)
        reverse_order = torch.flip(order, dims=[0])
        x = sample_batch(
            tokens,
            batch_size=int(eval_batch_size),
            block_size=int(model.config.block_size),
            rng=rng,
            device=device,
            token_perm=token_perm,
            data_record_mode=data_record_mode,
        )
        map_loss = full_loss_for_block_orders(model=model, x=x, block_orders=order.to(device), device=device, dtype=dtype)
        reverse_loss = full_loss_for_block_orders(
            model=model,
            x=x,
            block_orders=reverse_order.to(device),
            device=device,
            dtype=dtype,
        )
        random_losses = []
        for _ in range(int(random_eval_orders)):
            random_orders = random_block_orders(x.size(0), int(model.num_blocks), device)
            random_losses.append(
                full_loss_for_block_orders(model=model, x=x, block_orders=random_orders, device=device, dtype=dtype)
            )
        policy_sample_losses = []
        sampled_orders = sample_plackett_luce_orders(logits, int(policy_sample_eval_orders))
        for sampled_order in sampled_orders:
            policy_sample_losses.append(
                full_loss_for_block_orders(
                    model=model,
                    x=x,
                    block_orders=sampled_order.to(device),
                    device=device,
                    dtype=dtype,
                )
            )
        objective = tensor_metrics_to_floats(
            distribution_loss(
                logits.view(1, -1).to(device),
                candidate_orders[idx : idx + 1],
                candidate_losses[idx : idx + 1],
                args,
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
            "heldout_objective": objective.get("loss_total", float("nan")),
            "candidate_top1_match": objective.get("candidate_top1_match", float("nan")),
            "candidate_expected_minus_mean_loss": objective.get("candidate_expected_minus_mean_loss", float("nan")),
            "candidate_pred_entropy": objective.get("candidate_pred_entropy", float("nan")),
            "candidate_target_entropy": objective.get("candidate_target_entropy", float("nan")),
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
            print_progress(f"evaluated_nll_samples={idx + 1}/{features.size(0)}")
    summary = mean_dicts(rows)
    summary.update(
        {
            "num_eval_attention_samples": int(features.size(0)),
            "random_eval_orders": int(random_eval_orders),
            "policy_sample_eval_orders": int(policy_sample_eval_orders),
            "eval_batch_size": int(eval_batch_size),
        }
    )
    return rows, summary


def make_method_md(args):
    bank_size = effective_candidate_bank_size(args)
    train_candidate_count = effective_train_candidate_count(args)
    return f"""# Method

## Loss Candidate

This try trains an Attn-MLP order distribution with random-candidate frozen-NLL
energy. The MLP outputs `s in R^64`, which defines a Plackett-Luce distribution:

```text
P_s(order) = product_t softmax(s over remaining blocks)[order_t]
```

For each current-frame state, the script samples random candidate orders and
evaluates frozen AO-GPT full loss `L_k` for each candidate. The training target
is the energy distribution over the sampled candidate set:

```text
q_k = softmax(-(L_k - mean(L)) / target_loss_temp)
p_k = softmax(log P_s(order_k) / pl_temp)
L_listwise = CE(q, p)
```

The objective also includes a pairwise candidate-energy contrast, expected
candidate-loss surrogate, non-collapse terms, diversity, and current-index bias.
If `candidate_bank_size` is larger than `train_candidates_per_epoch`, the
script stores a larger random candidate bank and resamples a smaller subset
inside each epoch. This tests whether the static five-candidate target was the
main blocker, but it is still a cached-bank design rather than full online
MLP-sampled candidate generation.

This is not attention-order distillation: no hard or soft order generated from
attention is used as a target. Attention is only an input feature.

## Allowed Signals

- current-frame attention from random reveal probes
- current-frame block losses from the same probes
- random current-frame candidate orders
- frozen AO-GPT NLL of those random candidates

## Forbidden Signals

The following are not used in training, direction selection, model selection,
early stopping, or loss-design selection:

```text
{json.dumps(FORBIDDEN_SIGNALS + ["generated_teacher_order_supervision"], indent=2)}
```

`diagnostic/original_l2r_tau` is written only after the full try finishes.

## Hyperparameters

- `candidate_orders_per_sample`: `{args.candidate_orders_per_sample}`
- `candidate_bank_size`: `{bank_size}`
- `train_candidates_per_epoch`: `{train_candidate_count}`
- `train_order_probes`: `{int(args.num_train_samples) * int(bank_size)}`
- `target_loss_temp`: `{args.target_loss_temp}`
- `pl_temp`: `{args.pl_temp}`
- `pair_temp`: `{args.pair_temp}`
- `feature_clip`: `{args.feature_clip}`
- `lambda_listwise`: `{args.lambda_listwise}`
- `lambda_pairwise`: `{args.lambda_pairwise}`
- `lambda_expected_loss`: `{args.lambda_expected_loss}`
- `lambda_var`: `{args.lambda_var}`
- `lambda_entropy_floor`: `{args.lambda_entropy_floor}`
- `lambda_candidate_entropy_floor`: `{args.lambda_candidate_entropy_floor}`
- `lambda_diversity`: `{args.lambda_diversity}`
- `lambda_index_bias`: `{args.lambda_index_bias}`

## Model Selection

No validation-PPL shortcut or early stop is used. The formal try trains for the
fixed requested epoch count and evaluates the final checkpoint.
"""


def make_failure_analysis(summary):
    ev = summary["eval_summary"]
    opt = summary["optimization"]
    objective_drop = opt["initial_train"]["loss_total"] - opt["final_train"]["loss_total"]
    noncollapsed = opt["final_train"].get("logit_std", 0.0) >= summary["training_meta"]["min_std"]
    map_margin = -ev.get("mlp_minus_random_mean", float("inf"))
    sample_margin = -ev.get("policy_sample_mean_minus_random_mean", float("inf"))
    nll_ok = map_margin > summary["training_meta"]["success_random_margin"]
    dist_ok = sample_margin > summary["training_meta"]["success_random_margin"]
    reverse_ok = ev.get("forward_minus_reverse_loss", float("inf")) < 0.0
    tau_ok = ev.get("diagnostic/original_l2r_tau", float("-inf")) > 0.4
    success = bool((nll_ok or dist_ok) and tau_ok)
    reasons = []
    if not nll_ok:
        reasons.append("MAP MLP order did not clearly beat random mode loss")
    if not dist_ok:
        reasons.append("sampled MLP distribution did not clearly beat random mode loss")
    if not tau_ok:
        reasons.append("post-hoc diagnostic original_l2r_tau did not exceed 0.4")
    next_step = "success gate reached; replicate with another seed and larger candidate set"
    if not success:
        if not (nll_ok or dist_ok):
            next_step = "current-frame candidate energy is not being converted into better frozen NLL; increase candidate contrast or move to policy-gradient sampling"
        elif not tau_ok:
            next_step = "NLL distribution improves but tau gate fails; next loss change should target distribution stability/augmentation, not original direction"
    return f"""# Failure Analysis

## Acceptance Checks

- train objective decreased: `{objective_drop > 0}` (`drop={objective_drop:.6f}`)
- logits non-collapsed: `{noncollapsed}` (`train/logit_std={opt['final_train'].get('logit_std'):.6f}`)
- MAP loss clearly better than random mean: `{nll_ok}` (`margin={map_margin:.6f}`)
- sampled distribution loss clearly better than random mean: `{dist_ok}` (`margin={sample_margin:.6f}`)
- forward better than reverse: `{reverse_ok}` (`forward_minus_reverse_loss={ev.get('forward_minus_reverse_loss'):.6f}`)
- diagnostic original_l2r_tau > 0.4: `{tau_ok}` (`tau={ev.get('diagnostic/original_l2r_tau'):.6f}`)

## Verdict

success: `{success}`

Failure reasons:

{chr(10).join(f'- {reason}' for reason in reasons) if reasons else '- none'}

## Next Try

{next_step}

Original-frame tau and OriginalL2R were not used in training, direction
selection, model selection, early stopping, or loss design.
"""


def format_command_invocation():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    command_prefix = f"CUDA_VISIBLE_DEVICES={cuda_visible} python " if cuda_visible else "python "
    return command_prefix + " ".join(sys.argv)


def write_try_docs(args, summary=None):
    bank_size = effective_candidate_bank_size(args)
    train_candidate_count = effective_train_candidate_count(args)
    args.try_dir.mkdir(parents=True, exist_ok=True)
    (args.try_dir / "method.md").write_text(make_method_md(args), encoding="utf-8")
    command = format_command_invocation()
    (args.try_dir / "commands.sh").write_text(command + "\n", encoding="utf-8")
    resampling_note = (
        "cached candidate bank with epoch-level subset resampling"
        if int(bank_size) > int(train_candidate_count)
        else "single cached candidate set"
    )
    readme = f"""# Attn-MLP Order Distribution Energy Try

- checkpoint: `{args.ckpt_path}`
- loss: random-candidate frozen-NLL energy listwise distribution
- frame: `current`
- candidate mode: `{resampling_note}`
- train samples: `{args.num_train_samples}`
- val samples: `{args.num_val_samples}`
- candidate bank size: `{bank_size}`
- train candidates per epoch: `{train_candidate_count}`
- train order probes: `{int(args.num_train_samples) * int(bank_size)}`
- epochs: `{args.epochs}`
- eval samples: `{args.eval_samples}`
- random eval orders: `{args.random_eval_orders}`
- policy sample eval orders: `{args.policy_sample_eval_orders}`
- model selection: fixed final epoch, no validation-PPL shortcut

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
- `distribution_dataset.pt`
"""
    (args.try_dir / "README.md").write_text(readme, encoding="utf-8")
    if summary is not None:
        (args.try_dir / "failure_analysis.md").write_text(make_failure_analysis(summary), encoding="utf-8")


def main():
    args = parse_args()
    validate_budget(args)
    bank_size = effective_candidate_bank_size(args)
    train_candidate_count = effective_train_candidate_count(args)
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

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "frame": "current",
            "forbidden_signals": FORBIDDEN_SIGNALS + ["generated_teacher_order_supervision"],
            "checkpoint_iter": checkpoint.get("iter_num"),
            "checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
            "data_permutation_applied": permutation_state is not None,
            "model_selection": "fixed_final_epoch_no_val_ppl_shortcut",
            "candidate_bank_size_effective": int(bank_size),
            "train_candidates_per_epoch_effective": int(train_candidate_count),
        }
    )
    write_json(args.try_dir / "config.json", config_payload)

    dataset_path = args.try_dir / "distribution_dataset.pt"
    print_progress(f"loaded_ckpt={args.ckpt_path}")
    if dataset_path.exists() and not bool(args.recompute_dataset):
        print_progress(f"loading_existing_dataset={dataset_path}")
        dataset = torch.load(dataset_path, map_location="cpu")
        train_payload = dataset["train"]
        val_payload = dataset["val"]
    else:
        print_progress(f"collecting_train_distribution_samples={args.num_train_samples}")
        train_payload = collect_distribution_samples(
            model=model,
            tokens=train_tokens,
            num_samples=int(args.num_train_samples),
            probe_batch_size=int(args.probe_batch_size),
            candidate_orders_per_sample=int(bank_size),
            layer=int(args.layer),
            head=int(args.head),
            export_type=args.export_type,
            rng=rng,
            device=device,
            dtype=args.dtype,
            token_perm=token_perm,
            data_record_mode=data_record_mode,
            progress_label="train_distribution_samples",
            progress_interval=int(args.progress_interval),
        )
        print_progress(f"collecting_val_distribution_samples={args.num_val_samples}")
        val_payload = collect_distribution_samples(
            model=model,
            tokens=val_tokens,
            num_samples=int(args.num_val_samples),
            probe_batch_size=int(args.probe_batch_size),
            candidate_orders_per_sample=int(bank_size),
            layer=int(args.layer),
            head=int(args.head),
            export_type=args.export_type,
            rng=rng,
            device=device,
            dtype=args.dtype,
            token_perm=token_perm,
            data_record_mode=data_record_mode,
            progress_label="val_distribution_samples",
            progress_interval=int(args.progress_interval),
        )
        torch.save({"train": train_payload, "val": val_payload, "meta": config_payload}, dataset_path)

    train_features = make_features(train_payload["matrices"], train_payload["block_losses"], float(args.feature_clip))
    val_features = make_features(val_payload["matrices"], val_payload["block_losses"], float(args.feature_clip))
    hidden_dims = parse_int_list(args.hidden_dims, default=[1024, 1024])
    policy_config = {
        "num_blocks": int(model.num_blocks),
        "input_channels": int(train_features.size(1)),
        "hidden_dims": hidden_dims,
        "dropout": float(args.dropout),
        "activation": args.activation,
    }
    policy = AttentionDistributionOrderMLP(**policy_config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_log_path = args.try_dir / "train_log.jsonl"
    if train_log_path.exists():
        train_log_path.unlink()
    metrics_rows = []
    init_train = evaluate_objective(
        policy,
        train_features,
        train_payload["candidate_orders"],
        train_payload["candidate_losses"],
        args,
        device,
    )
    init_val = evaluate_objective(
        policy,
        val_features,
        val_payload["candidate_orders"],
        val_payload["candidate_losses"],
        args,
        device,
    )
    row = {"epoch": 0, "phase": "initial"}
    row.update({f"train/{key}": value for key, value in init_train.items()})
    row.update({f"val/{key}": value for key, value in init_val.items()})
    metrics_rows.append(row)
    append_jsonl(train_log_path, row)
    print_progress(json.dumps(row))

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_epoch(
            policy,
            optimizer,
            train_features,
            train_payload["candidate_orders"],
            train_payload["candidate_losses"],
            args,
            device,
        )
        val_metrics = evaluate_objective(
            policy,
            val_features,
            val_payload["candidate_orders"],
            val_payload["candidate_losses"],
            args,
            device,
        )
        row = {"epoch": epoch, "phase": "train"}
        row.update({f"train/{key}": value for key, value in train_metrics.items()})
        row.update({f"val/{key}": value for key, value in val_metrics.items()})
        metrics_rows.append(row)
        append_jsonl(train_log_path, row)
        print_progress(json.dumps(row))

    final_train = evaluate_objective(
        policy,
        train_features,
        train_payload["candidate_orders"],
        train_payload["candidate_losses"],
        args,
        device,
    )
    final_val = evaluate_objective(
        policy,
        val_features,
        val_payload["candidate_orders"],
        val_payload["candidate_losses"],
        args,
        device,
    )
    write_csv(args.try_dir / "metrics.csv", metrics_rows)

    eval_n = min(int(args.eval_samples), int(val_features.size(0)))
    print_progress(f"running_full_eval_samples={eval_n}")
    eval_rows, eval_summary = evaluate_policy_nll(
        policy=policy,
        features=val_features[:eval_n],
        candidate_orders=val_payload["candidate_orders"][:eval_n],
        candidate_losses=val_payload["candidate_losses"][:eval_n],
        model=model,
        tokens=val_tokens,
        rng=rng,
        device=device,
        dtype=args.dtype,
        token_perm=token_perm,
        data_record_mode=data_record_mode,
        eval_batch_size=int(args.eval_batch_size),
        random_eval_orders=int(args.random_eval_orders),
        policy_sample_eval_orders=int(args.policy_sample_eval_orders),
        permutation_state=permutation_state,
        args=args,
    )
    write_csv(args.try_dir / "eval_rows.csv", eval_rows)

    policy_path = args.try_dir / "policy.pt"
    summary = {
        "script": "scripts/train/train_attn_mlp_order_distribution_energy.py",
        "command": format_command_invocation(),
        "policy_path": str(policy_path),
        "checkpoint": {
            "ckpt_path": str(args.ckpt_path),
            "iter_num": checkpoint.get("iter_num"),
            "best_val_loss": checkpoint.get("best_val_loss"),
        },
        "policy_config": policy_config,
        "training_meta": {
            "frame": "current",
            "loss": "random-candidate frozen-NLL energy Plackett-Luce distribution",
            "forbidden_signals": FORBIDDEN_SIGNALS + ["generated_teacher_order_supervision"],
            "original_l2r_is_diagnostic_only": True,
            "tau_is_diagnostic_only": True,
            "model_selection": "fixed_final_epoch_no_val_ppl_shortcut",
            "candidate_bank_size": int(bank_size),
            "train_candidates_per_epoch": int(train_candidate_count),
            "min_std": float(args.min_std),
            "success_random_margin": float(args.success_random_margin),
        },
        "samples": {
            "train_samples": int(args.num_train_samples),
            "val_samples": int(args.num_val_samples),
            "candidate_orders_per_sample": int(args.candidate_orders_per_sample),
            "candidate_bank_size": int(bank_size),
            "train_candidates_per_epoch": int(train_candidate_count),
            "train_order_probes": int(args.num_train_samples) * int(bank_size),
            "val_order_probes": int(args.num_val_samples) * int(bank_size),
            "eval_samples": int(eval_n),
            "random_eval_orders": int(args.random_eval_orders),
            "policy_sample_eval_orders": int(args.policy_sample_eval_orders),
            "train_random_probe_orders_first": train_payload.get("probe_orders_first", []),
            "val_random_probe_orders_first": val_payload.get("probe_orders_first", []),
        },
        "optimization": {
            "epochs": int(args.epochs),
            "initial_train": init_train,
            "initial_val": init_val,
            "final_train": final_train,
            "final_val": final_val,
        },
        "eval_summary": eval_summary,
        "diagnostic_acceptance": {
            "origin_tau_gate": 0.4,
            "origin_tau_passed": bool(eval_summary.get("diagnostic/original_l2r_tau", float("-inf")) > 0.4),
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
            "policy_type": "attention_order_distribution_energy_mlp",
            "training_meta": summary["training_meta"],
            "metrics": {"train": final_train, "val": final_val, "eval": eval_summary},
        },
        policy_path,
    )
    write_json(args.try_dir / "eval_summary.json", summary)
    write_try_docs(args, summary=summary)
    print_progress(f"try_dir={args.try_dir}")
    print_progress(json.dumps(summary["diagnostic_acceptance"], indent=2))
    print_progress(json.dumps(eval_summary, indent=2))


if __name__ == "__main__":
    main()
