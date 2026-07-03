#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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


class AttentionLossOrderMLP(nn.Module):
    """Predict current-frame block priority logits from attention/loss feature planes."""

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
        input_dim = self.input_channels * self.num_blocks * self.num_blocks
        hidden_dims = [int(value) for value in (hidden_dims or [1024, 1024])]
        dims = [input_dim] + hidden_dims + [self.num_blocks]
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
        if tuple(features.shape[1:]) != (self.input_channels, self.num_blocks, self.num_blocks):
            raise ValueError(
                f"Expected features shape (B,{self.input_channels},{self.num_blocks},{self.num_blocks}), "
                f"got {tuple(features.shape)}"
            )
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
    parser = argparse.ArgumentParser(description="Full-scale Attn-MLP loss-semantics try.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--try_dir", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="wikitext103")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--num_train_samples", type=int, default=2000)
    parser.add_argument("--num_val_samples", type=int, default=512)
    parser.add_argument("--probe_batch_size", type=int, default=64)
    parser.add_argument("--pair_probe_count", type=int, default=10000)
    parser.add_argument("--pair_probe_batch_size", type=int, default=64)
    parser.add_argument("--pair_probe_split", type=str, default="train")
    parser.add_argument("--pair_prefix_blocks", type=int, default=8)
    parser.add_argument("--pair_full_weight", type=float, default=0.7)
    parser.add_argument("--pair_prefix_weight", type=float, default=0.3)
    parser.add_argument(
        "--pair_signal_mode",
        choices=("full_prefix", "conditional_gain"),
        default="full_prefix",
        help=(
            "full_prefix uses full/prefix loss(order_ij)-loss(order_ji). "
            "conditional_gain uses second-block conditional loss improvement: "
            "(loss_j_first-loss_j_after_i) - (loss_i_first-loss_i_after_j)."
        ),
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=7)
    parser.add_argument("--export_type", choices=("with_none", "without_none"), default="with_none")
    parser.add_argument("--target_sign", type=float, default=1.0)
    parser.add_argument("--hidden_dims", type=str, default="1024,1024")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--tau_s", type=float, default=1.0)
    parser.add_argument("--tau_loss", type=float, default=0.25)
    parser.add_argument("--tau_block_loss", type=float, default=1.0)
    parser.add_argument(
        "--feature_clip",
        type=float,
        default=0.0,
        help="If >0, clamp current-frame feature planes to [-feature_clip, feature_clip] after robust normalization.",
    )
    parser.add_argument("--pair_margin", type=float, default=0.05)
    parser.add_argument("--min_pair_conf", type=float, default=0.05)
    parser.add_argument("--attn_gate_margin", type=float, default=1.0)
    parser.add_argument("--lambda_pair_bce", type=float, default=1.0)
    parser.add_argument("--lambda_expected_utility", type=float, default=0.25)
    parser.add_argument("--lambda_block_loss", type=float, default=0.15)
    parser.add_argument("--lambda_var", type=float, default=0.01)
    parser.add_argument("--lambda_diversity", type=float, default=0.01)
    parser.add_argument("--lambda_index_bias", type=float, default=0.01)
    parser.add_argument("--min_std", type=float, default=0.1)
    parser.add_argument("--eval_samples", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--random_eval_orders", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--progress_interval", type=int, default=100)
    parser.add_argument(
        "--recompute_dataset",
        action="store_true",
        help="Ignore an existing loss_semantics_dataset.pt and recollect all attention/loss/pair probes.",
    )
    return parser.parse_args()


def print_progress(text: str):
    print(text, flush=True)


def zscore_vector(values: torch.Tensor, eps: float = 1e-6):
    values = values.float()
    return (values - values.mean(dim=-1, keepdim=True)) / values.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)


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


@torch.no_grad()
def collect_attention_loss_samples(
    *,
    model,
    tokens,
    num_samples: int,
    probe_batch_size: int,
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
    first_orders = []
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
        block_orders = random_block_orders(int(probe_batch_size), int(model.num_blocks), device)
        token_orders = expand_orders_for_model(model, block_orders)
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
            block_orders,
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
            int(model.num_blocks),
            int(model.block_order_block_len),
        ).mean(dim=-1)
        current_loss = block_losses_to_current(reveal_block_loss, block_orders)
        matrices.append(matrix)
        block_losses.append(current_loss.mean(dim=0).detach().float().cpu())
        if len(first_orders) < 4:
            first_orders.append(block_orders[0].detach().cpu().tolist())
        if (sample_idx + 1) % max(1, int(progress_interval)) == 0:
            print_progress(f"{progress_label}={sample_idx + 1}/{num_samples}")
    return torch.stack(matrices), torch.stack(block_losses), first_orders


def make_pair_orders(pair_i: torch.Tensor, pair_j: torch.Tensor, num_blocks: int, device: torch.device, reverse: bool):
    rows = []
    all_blocks = torch.arange(int(num_blocks), device=device)
    for i_value, j_value in zip(pair_i.tolist(), pair_j.tolist()):
        i = int(i_value)
        j = int(j_value)
        first, second = (j, i) if reverse else (i, j)
        mask = (all_blocks != i) & (all_blocks != j)
        rest = all_blocks[mask]
        rest = rest[torch.randperm(rest.numel(), device=device)]
        rows.append(torch.cat([torch.tensor([first, second], device=device), rest]))
    return torch.stack(rows)


def make_pair_order_pair_same_rest(pair_i: torch.Tensor, pair_j: torch.Tensor, num_blocks: int, device: torch.device):
    rows_ij = []
    rows_ji = []
    all_blocks = torch.arange(int(num_blocks), device=device)
    for i_value, j_value in zip(pair_i.tolist(), pair_j.tolist()):
        i = int(i_value)
        j = int(j_value)
        mask = (all_blocks != i) & (all_blocks != j)
        rest = all_blocks[mask]
        rest = rest[torch.randperm(rest.numel(), device=device)]
        rows_ij.append(torch.cat([torch.tensor([i, j], device=device), rest]))
        rows_ji.append(torch.cat([torch.tensor([j, i], device=device), rest]))
    return torch.stack(rows_ij), torch.stack(rows_ji)


@torch.no_grad()
def per_sample_full_and_prefix_loss(
    *,
    model,
    x: torch.Tensor,
    block_orders: torch.Tensor,
    prefix_blocks: int,
    device: torch.device,
    dtype: str,
):
    token_orders = expand_orders_for_model(model, block_orders)
    with autocast_context(device, dtype):
        outputs = model(
            x,
            mode=None,
            orders=token_orders,
            return_token_loss=True,
            return_logits=False,
        )
    token_losses = outputs[2].detach().float()
    block_losses = token_losses.view(x.size(0), int(model.num_blocks), int(model.block_order_block_len)).mean(dim=-1)
    full_loss = block_losses.mean(dim=1)
    k = max(1, min(int(prefix_blocks), int(model.num_blocks)))
    prefix_loss = block_losses[:, :k].mean(dim=1)
    return full_loss, prefix_loss, block_losses


@torch.no_grad()
def collect_pair_loss_drop_probes(
    *,
    model,
    tokens,
    pair_probe_count: int,
    pair_probe_batch_size: int,
    prefix_blocks: int,
    full_weight: float,
    prefix_weight: float,
    signal_mode: str,
    rng: np.random.Generator,
    device: torch.device,
    dtype: str,
    token_perm: torch.Tensor | None,
    data_record_mode: str,
    progress_interval: int,
):
    n = int(model.num_blocks)
    delta_sum = torch.zeros(n, n, dtype=torch.float64)
    full_delta_sum = torch.zeros(n, n, dtype=torch.float64)
    prefix_delta_sum = torch.zeros(n, n, dtype=torch.float64)
    count = torch.zeros(n, n, dtype=torch.float64)
    rows = []
    done = 0
    while done < int(pair_probe_count):
        bsz = min(int(pair_probe_batch_size), int(pair_probe_count) - done)
        x = sample_batch(
            tokens,
            batch_size=bsz,
            block_size=int(model.config.block_size),
            rng=rng,
            device=device,
            token_perm=token_perm,
            data_record_mode=data_record_mode,
        )
        pair_i = torch.randint(0, n, (bsz,), device=device)
        pair_j = torch.randint(0, n - 1, (bsz,), device=device)
        pair_j = pair_j + (pair_j >= pair_i).long()
        orders_ij, orders_ji = make_pair_order_pair_same_rest(pair_i, pair_j, n, device)
        full_ij, prefix_ij, blocks_ij = per_sample_full_and_prefix_loss(
            model=model,
            x=x,
            block_orders=orders_ij,
            prefix_blocks=prefix_blocks,
            device=device,
            dtype=dtype,
        )
        full_ji, prefix_ji, blocks_ji = per_sample_full_and_prefix_loss(
            model=model,
            x=x,
            block_orders=orders_ji,
            prefix_blocks=prefix_blocks,
            device=device,
            dtype=dtype,
        )
        signal_mode = str(signal_mode)
        if signal_mode == "conditional_gain":
            gain_i_to_j = blocks_ji[:, 0] - blocks_ij[:, 1]
            gain_j_to_i = blocks_ij[:, 0] - blocks_ji[:, 1]
            full_delta = (gain_i_to_j - gain_j_to_i).detach().cpu()
            prefix_delta = (prefix_ij - prefix_ji).detach().cpu()
            delta = full_delta
        elif signal_mode == "full_prefix":
            full_delta = (full_ij - full_ji).detach().cpu()
            prefix_delta = (prefix_ij - prefix_ji).detach().cpu()
            delta = float(full_weight) * full_delta + float(prefix_weight) * prefix_delta
        else:
            raise ValueError(f"Unsupported pair_signal_mode={signal_mode!r}")
        for offset, (i_value, j_value) in enumerate(zip(pair_i.detach().cpu().tolist(), pair_j.detach().cpu().tolist())):
            i = int(i_value)
            j = int(j_value)
            d = float(delta[offset].item())
            df = float(full_delta[offset].item())
            dp = float(prefix_delta[offset].item())
            delta_sum[i, j] += d
            delta_sum[j, i] -= d
            full_delta_sum[i, j] += df
            full_delta_sum[j, i] -= df
            prefix_delta_sum[i, j] += dp
            prefix_delta_sum[j, i] -= dp
            count[i, j] += 1.0
            count[j, i] += 1.0
            if len(rows) < 20000:
                rows.append(
                    {
                        "probe_id": done + offset,
                        "i": i,
                        "j": j,
                        "delta_mixed_ij_minus_ji": d,
                        "delta_full_ij_minus_ji": df,
                        "delta_prefix_ij_minus_ji": dp,
                        "pair_signal_mode": signal_mode,
                    }
                )
        done += bsz
        if done % max(1, int(progress_interval)) == 0 or done == int(pair_probe_count):
            print_progress(f"pair_loss_drop_probes={done}/{pair_probe_count}")
    avg = torch.zeros_like(delta_sum)
    avg_full = torch.zeros_like(delta_sum)
    avg_prefix = torch.zeros_like(delta_sum)
    mask = count > 0
    avg[mask] = delta_sum[mask] / count[mask]
    avg_full[mask] = full_delta_sum[mask] / count[mask]
    avg_prefix[mask] = prefix_delta_sum[mask] / count[mask]
    avg.fill_diagonal_(0.0)
    avg_full.fill_diagonal_(0.0)
    avg_prefix.fill_diagonal_(0.0)
    observed = int((count > 0).sum().item() // 2)
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    abs_delta = avg[upper].abs()
    summary = {
        "pair_probe_count": int(pair_probe_count),
        "observed_undirected_pairs": observed,
        "total_undirected_pairs": int(n * (n - 1) // 2),
        "coverage_fraction": float(observed / max(1, n * (n - 1) // 2)),
        "delta_abs_mean": float(abs_delta.mean().item()),
        "delta_abs_p90": float(torch.quantile(abs_delta, 0.9).item()),
        "delta_abs_max": float(abs_delta.max().item()),
        "pair_signal_mode": str(signal_mode),
    }
    return {
        "delta": avg.float(),
        "delta_full": avg_full.float(),
        "delta_prefix": avg_prefix.float(),
        "count": count.float(),
        "rows": rows,
        "summary": summary,
    }


def robust_z_pair_matrix(matrix: torch.Tensor):
    z = robust_z_offdiag(matrix.float())
    z.fill_diagonal_(0.0)
    return z


def make_features(
    matrices: torch.Tensor,
    block_losses: torch.Tensor,
    pair_delta: torch.Tensor,
    pair_conf: torch.Tensor,
    feature_clip: float = 0.0,
):
    feature_rows = []
    pair_delta_z = robust_z_pair_matrix(pair_delta).float()
    pair_conf = pair_conf.float()
    n = int(pair_delta.size(0))
    eye = torch.eye(n, dtype=torch.bool)
    pair_delta_z = pair_delta_z.masked_fill(eye, 0.0)
    pair_conf = pair_conf.masked_fill(eye, 0.0)
    for matrix, losses in zip(matrices, block_losses):
        a_z = robust_z_offdiag(matrix.float())
        sym = torch.maximum(a_z, a_z.t())
        loss_z = robust_z_vector(losses.view(1, -1)).view(-1)
        loss_diff = loss_z[:, None] - loss_z[None, :]
        planes = torch.stack([a_z, sym, loss_diff, pair_delta_z, pair_conf], dim=0)
        if float(feature_clip) > 0.0:
            clip = float(feature_clip)
            planes = planes.clamp(min=-clip, max=clip)
        planes[:, eye] = 0.0
        feature_rows.append(planes)
    return torch.stack(feature_rows, dim=0).float()


def pair_confidence(pair_delta: torch.Tensor, pair_count: torch.Tensor, margin: float):
    conf = (pair_delta.abs() / max(float(margin), 1e-8)).clamp(0.0, 1.0)
    count_conf = (pair_count / pair_count.max().clamp_min(1.0)).clamp(0.0, 1.0).sqrt()
    out = conf * count_conf
    out.fill_diagonal_(0.0)
    return out.float()


def semantic_loss(
    logits: torch.Tensor,
    features: torch.Tensor,
    block_losses: torch.Tensor,
    *,
    pair_delta: torch.Tensor,
    pair_conf: torch.Tensor,
    args,
):
    device = logits.device
    s = logits.float()
    x = (s - s.mean(dim=1, keepdim=True)) / s.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    batch, n = x.shape
    ii, jj = torch.triu_indices(n, n, offset=1, device=device)
    pair_delta = pair_delta.to(device=device, dtype=torch.float32)
    pair_conf = pair_conf.to(device=device, dtype=torch.float32)
    effective_delta = float(args.target_sign) * pair_delta[ii, jj]
    pair_conf_upper = pair_conf[ii, jj]
    base_mask = pair_conf_upper >= float(args.min_pair_conf)
    if not bool(base_mask.any()):
        base_mask = pair_conf_upper > 0
    pair_logits = (x[:, ii] - x[:, jj]) / max(float(args.tau_s), 1e-6)
    attn_z = features[:, 0].to(device=device, dtype=torch.float32)
    attn_conf = 0.5 * (attn_z[:, ii, jj].abs() + attn_z[:, jj, ii].abs())
    attn_gate = 1.0 + (attn_conf / max(float(args.attn_gate_margin), 1e-6)).clamp(0.0, 1.0)
    weights = pair_conf_upper.view(1, -1) * attn_gate
    weights = weights[:, base_mask]
    selected_logits = pair_logits[:, base_mask]
    selected_delta = effective_delta[base_mask]
    target = torch.sigmoid(-selected_delta / max(float(args.tau_loss), 1e-6)).view(1, -1).expand_as(selected_logits)
    bce = F.binary_cross_entropy_with_logits(selected_logits, target, reduction="none")
    loss_pair = (bce * weights).sum() / weights.sum().clamp_min(1e-6)
    prob = torch.sigmoid(selected_logits)
    delta_scale = selected_delta.abs().median().clamp_min(1e-6)
    delta_norm = (selected_delta / delta_scale).view(1, -1)
    loss_expected = (weights * ((2.0 * prob - 1.0) * delta_norm)).sum() / weights.sum().clamp_min(1e-6)

    loss_z = zscore_vector(block_losses.to(device=device, dtype=torch.float32))
    loss_diff = loss_z[:, ii] - loss_z[:, jj]
    block_target = torch.sigmoid(-loss_diff / max(float(args.tau_block_loss), 1e-6))
    block_weight = attn_gate.detach()
    loss_block = F.binary_cross_entropy_with_logits(pair_logits, block_target, reduction="none")
    loss_block = (loss_block * block_weight).sum() / block_weight.sum().clamp_min(1e-6)

    std = s.std(dim=1, unbiased=False)
    loss_var = torch.relu(s.new_tensor(float(args.min_std)) - std).pow(2).mean()
    if batch > 1:
        x_norm = x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)
        sim = x_norm @ x_norm.t()
        offdiag = ~torch.eye(batch, dtype=torch.bool, device=device)
        loss_diversity = sim[offdiag].pow(2).mean()
    else:
        loss_diversity = s.new_tensor(0.0)
    index = torch.arange(n, device=device, dtype=torch.float32)
    index = (index - index.mean()) / index.std(unbiased=False).clamp_min(1e-6)
    corr = (x * index.view(1, -1)).mean(dim=1).abs().mean()
    pred_sign = torch.sign(selected_logits.detach())
    truth_sign = torch.sign(-selected_delta.detach()).view(1, -1)
    valid = truth_sign != 0
    if bool(valid.any()):
        pair_agree = ((pred_sign == truth_sign).float() * weights)[valid.expand_as(weights)].sum()
        pair_agree = pair_agree / weights[valid.expand_as(weights)].sum().clamp_min(1e-6)
    else:
        pair_agree = s.new_tensor(float("nan"))
    loss_total = (
        float(args.lambda_pair_bce) * loss_pair
        + float(args.lambda_expected_utility) * loss_expected
        + float(args.lambda_block_loss) * loss_block
        + float(args.lambda_var) * loss_var
        + float(args.lambda_diversity) * loss_diversity
        + float(args.lambda_index_bias) * corr
    )
    entropy = -(torch.softmax(s, dim=-1) * torch.log_softmax(s, dim=-1)).sum(dim=-1).mean()
    return {
        "loss_total": loss_total,
        "loss_pair_bce": loss_pair,
        "loss_expected_utility": loss_expected,
        "loss_block_consistency": loss_block,
        "loss_var": loss_var,
        "loss_diversity": loss_diversity,
        "loss_index_bias": corr,
        "logit_std": std.mean().detach(),
        "logit_entropy": entropy.detach(),
        "pairwise_loss_derived_agreement": pair_agree.detach(),
        "pair_conf_mean": weights.detach().mean(),
    }


@torch.no_grad()
def evaluate_objective(policy, features, block_losses, pair_delta, pair_conf, args, device):
    policy.eval()
    rows = []
    orders = []
    for start in range(0, int(features.size(0)), int(args.batch_size)):
        end = min(int(features.size(0)), start + int(args.batch_size))
        feat = features[start:end].to(device=device)
        losses = block_losses[start:end].to(device=device)
        logits = policy(feat)
        metrics = semantic_loss(logits, feat, losses, pair_delta=pair_delta, pair_conf=pair_conf, args=args)
        rows.append(tensor_metrics_to_floats(metrics))
        orders.extend([order_from_logits_desc(row).cpu() for row in logits])
    summary = mean_dicts(rows)
    if len(orders) > 1:
        taus = [kendall_tau_between_orders(orders[idx - 1], orders[idx]) for idx in range(1, len(orders))]
        summary["order_change_kendall_current"] = float(np.mean(taus))
    return summary


def train_epoch(policy, optimizer, features, block_losses, pair_delta, pair_conf, args, device):
    policy.train()
    order = torch.randperm(int(features.size(0)))
    rows = []
    for start in range(0, int(order.numel()), int(args.batch_size)):
        idx = order[start : start + int(args.batch_size)]
        feat = features.index_select(0, idx).to(device=device)
        losses = block_losses.index_select(0, idx).to(device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = policy(feat)
        metrics = semantic_loss(logits, feat, losses, pair_delta=pair_delta, pair_conf=pair_conf, args=args)
        metrics["loss_total"].backward()
        if float(args.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(args.grad_clip))
        optimizer.step()
        rows.append(tensor_metrics_to_floats(metrics))
    return mean_dicts(rows)


@torch.no_grad()
def evaluate_policy_nll(
    *,
    policy,
    features,
    matrices,
    block_losses,
    model,
    tokens,
    rng,
    device,
    dtype,
    token_perm,
    data_record_mode,
    eval_batch_size,
    random_eval_orders,
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
        forward_loss = full_loss_for_block_orders(model=model, x=x, block_orders=order.to(device), device=device, dtype=dtype)
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
                full_loss_for_block_orders(
                    model=model,
                    x=x,
                    block_orders=random_orders,
                    device=device,
                    dtype=dtype,
                )
            )
        objective = tensor_metrics_to_floats(
            semantic_loss(
                logits.view(1, -1).to(device),
                feat,
                block_losses[idx : idx + 1].to(device),
                pair_delta=args._pair_delta,
                pair_conf=args._pair_conf,
                args=args,
            )
        )
        diag = original_order_diagnostics(order, permutation_state)
        row = {
            "sample_id": idx,
            "mlp_map_full_loss": forward_loss,
            "random_mean_full_loss": float(np.mean(random_losses)),
            "random_best_full_loss": float(np.min(random_losses)),
            "mlp_minus_random_mean": forward_loss - float(np.mean(random_losses)),
            "mlp_minus_random_best": forward_loss - float(np.min(random_losses)),
            "reverse_mlp_full_loss": reverse_loss,
            "forward_minus_reverse_loss": forward_loss - reverse_loss,
            "logit_std": float(logits.std(unbiased=False).item()),
            "logit_entropy": float((-(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum()).item()),
            "pairwise_loss_derived_agreement": objective.get("pairwise_loss_derived_agreement", float("nan")),
            "heldout_objective": objective.get("loss_total", float("nan")),
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
            "eval_batch_size": int(eval_batch_size),
        }
    )
    return rows, summary


def make_method_md(args):
    target_formula = "sigmoid(-target_sign * DeltaL_ij / tau_loss)"
    return f"""# Method

## Goal

Train an Attn-MLP policy from current-frame AO-GPT model-side signals, without
supervising on a generated teacher order.

The policy consumes current-frame feature planes built from:

- robust-normalized `A_current`
- symmetric attention affinity from `A_current`
- current-frame random-probe block-loss differences
- current-frame pairwise loss-drop matrix from frozen AO-GPT probes
- pairwise loss-drop confidence

It outputs `s in R^64`; larger `s_i` reveals current-frame block `i` earlier.

## Pairwise Loss-Drop Signal

For a sampled current-frame pair `(i,j)`, the frozen AO-GPT is evaluated with:

```text
order_ij = [i, j] + random_remaining_blocks
order_ji = [j, i] + same_random_remaining_blocks
DeltaL_ij = pair_full_weight * (full_loss(order_ij) - full_loss(order_ji))
          + pair_prefix_weight * (prefix_loss(order_ij) - prefix_loss(order_ji))
```

If `pair_signal_mode=conditional_gain`, the two orders still share the same
random remainder, but the pair signal is:

```text
gain_i_to_j = loss(j first) - loss(j after i)
gain_j_to_i = loss(i first) - loss(i after j)
DeltaL_ij = gain_i_to_j - gain_j_to_i
```

This keeps the target current-frame and model-side, while focusing on directed
conditional dependence rather than total-order NLL.

The target pair probability is:

```text
Y_ij = {target_formula}
P_ij = sigmoid((s_i - s_j) / tau_s)
```

This is not teacher-order supervision: no hard order is searched, selected, or
used as target. The signal is a local current-frame loss-derived preference.

## Loss

```text
L = lambda_pair_bce * BCEWithLogits((s_i - s_j)/tau_s, Y_ij)
  + lambda_expected_utility * E[(2 * P_ij - 1) * target_sign * DeltaL_ij]
  + lambda_block_loss * BCEWithLogits((s_i - s_j)/tau_s,
                                      sigmoid(-(block_loss_i - block_loss_j)/tau_block_loss))
  + lambda_var * relu(min_std - std(s))^2
  + lambda_diversity * batch_order_similarity^2
  + lambda_index_bias * abs(corr(s, current_frame_index))
```

The pair BCE and expected utility are gated by pairwise loss-drop confidence and
sample-level attention confidence. The block-loss consistency term is gated by
attention confidence.

## Current Try Hyperparameters

- `target_sign`: `{args.target_sign}`
- `pair_signal_mode`: `{args.pair_signal_mode}`
- `pair_full_weight`: `{args.pair_full_weight}`
- `pair_prefix_weight`: `{args.pair_prefix_weight}`
- `feature_clip`: `{args.feature_clip}`
- `lambda_pair_bce`: `{args.lambda_pair_bce}`
- `lambda_expected_utility`: `{args.lambda_expected_utility}`
- `lambda_block_loss`: `{args.lambda_block_loss}`
- `lambda_var`: `{args.lambda_var}`
- `lambda_diversity`: `{args.lambda_diversity}`
- `lambda_index_bias`: `{args.lambda_index_bias}`

## Forbidden Signals

The following are not used in training, direction selection, hyperparameter
selection, or early stop:

```text
{json.dumps(FORBIDDEN_SIGNALS, indent=2)}
```

`block_perm`, `*_original`, and original-frame tau are written only after the
try finishes as diagnostics / acceptance gates.

## Difference From Previous Implicit-Axis Loss

The previous loss mainly optimized graph smoothness plus attention asymmetry.
This try adds current-frame frozen-AO-GPT token/block-loss semantics and direct
pairwise loss-drop probes, while still avoiding hard teacher order supervision.
"""


def make_failure_analysis(summary):
    ev = summary["eval_summary"]
    opt = summary["optimization"]
    objective_drop = opt["initial_val"]["loss_total"] - opt["final_val"]["loss_total"]
    noncollapsed = opt["final_val"].get("logit_std", 0.0) >= summary["training_meta"]["min_std"]
    nll_ok = ev.get("mlp_minus_random_mean", float("inf")) < 0.0
    reverse_ok = ev.get("forward_minus_reverse_loss", float("inf")) < 0.0
    tau_ok = ev.get("diagnostic/original_l2r_tau", float("-inf")) >= 0.4
    success = objective_drop > 0 and noncollapsed and nll_ok and reverse_ok and tau_ok
    reasons = []
    if objective_drop <= 0:
        reasons.append("heldout current-frame objective did not decrease")
    if not noncollapsed:
        reasons.append("logits collapsed or stayed below min_std")
    if not nll_ok:
        reasons.append("MLP MAP full_loss did not beat random_mean_full_loss")
    if not reverse_ok:
        reasons.append("forward order did not beat reverse order")
    if not tau_ok:
        reasons.append("diagnostic original_l2r_tau did not reach 0.4 acceptance gate")
    next_step = "success gate reached; next step is replicate with another seed and larger pair probes"
    if not success:
        if not reverse_ok:
            next_step = "direction failure: next try should flip target_sign using current-frame reverse contrast, not original tau"
        elif not nll_ok:
            next_step = "NLL failure: next try should increase loss-drop weight or reduce block-loss/attention-only terms"
        elif not tau_ok:
            next_step = "diagnostic tau failure: keep training-selection unchanged, but inspect whether current-frame loss signal recovers a useful non-L2R order"
    return f"""# Failure Analysis

## Acceptance Checks

- objective decreased: `{objective_drop > 0}` (`drop={objective_drop:.6f}`)
- logits non-collapsed: `{noncollapsed}` (`val/logit_std={opt['final_val'].get('logit_std'):.6f}`)
- MLP MAP full_loss better than random mean: `{nll_ok}` (`mlp_minus_random_mean={ev.get('mlp_minus_random_mean'):.6f}`)
- forward better than reverse: `{reverse_ok}` (`forward_minus_reverse_loss={ev.get('forward_minus_reverse_loss'):.6f}`)
- diagnostic original_l2r_tau >= 0.4: `{tau_ok}` (`tau={ev.get('diagnostic/original_l2r_tau'):.6f}`)

## Verdict

success: `{success}`

Failure reasons:

{chr(10).join(f'- {reason}' for reason in reasons) if reasons else '- none'}

## Next Try

{next_step}

Original-frame tau and OriginalL2R were not used in training, selection,
direction selection, or early stopping.
"""


def write_try_docs(args, summary=None):
    args.try_dir.mkdir(parents=True, exist_ok=True)
    (args.try_dir / "method.md").write_text(make_method_md(args), encoding="utf-8")
    readme = f"""# Attn-MLP Loss Semantics Try

- checkpoint: `{args.ckpt_path}`
- frame: `current`
- train samples: `{args.num_train_samples}`
- val samples: `{args.num_val_samples}`
- pair probes: `{args.pair_probe_count}`
- pair signal mode: `{args.pair_signal_mode}`
- feature clip: `{args.feature_clip}`
- epochs: `{args.epochs}`
- eval samples: `{args.eval_samples}`
- random eval orders: `{args.random_eval_orders}`

Key files:

- `method.md`
- `failure_analysis.md`
- `train_log.jsonl`
- `metrics.csv`
- `eval_summary.json`
- `config.json`
- `policy.pt`
- `commands.sh`
"""
    (args.try_dir / "README.md").write_text(readme, encoding="utf-8")
    command = "CUDA_VISIBLE_DEVICES=1 python " + " ".join(sys.argv)
    (args.try_dir / "commands.sh").write_text(command + "\n", encoding="utf-8")
    if summary is not None:
        (args.try_dir / "failure_analysis.md").write_text(make_failure_analysis(summary), encoding="utf-8")


def main():
    args = parse_args()
    args.try_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    device = torch.device(args.device)
    write_try_docs(args)

    model, checkpoint = load_frozen_aogpt(args.ckpt_path, device)
    data_dir = infer_data_dir(args.dataset, args.data_dir)
    train_tokens = load_tokens(data_dir, args.train_split)
    val_tokens = load_tokens(data_dir, args.val_split)
    pair_tokens = load_tokens(data_dir, args.pair_probe_split)
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
            "forbidden_signals": FORBIDDEN_SIGNALS,
            "checkpoint_iter": checkpoint.get("iter_num"),
            "checkpoint_best_val_loss": checkpoint.get("best_val_loss"),
            "data_permutation_applied": permutation_state is not None,
        }
    )
    write_json(args.try_dir / "config.json", config_payload)

    print_progress(f"loaded_ckpt={args.ckpt_path}")
    dataset_path = args.try_dir / "loss_semantics_dataset.pt"
    if dataset_path.exists() and not bool(args.recompute_dataset):
        print_progress(f"loading_existing_dataset={dataset_path}")
        dataset_payload = torch.load(dataset_path, map_location="cpu")
        train_mats = dataset_payload["train_matrices"]
        train_block_losses = dataset_payload["train_block_losses"]
        val_mats = dataset_payload["val_matrices"]
        val_block_losses = dataset_payload["val_block_losses"]
        pair_delta = dataset_payload["pair_delta"]
        pair_conf = dataset_payload["pair_conf"]
        pair_count = dataset_payload["pair_count"]
        pair_summary_path = args.try_dir / "pair_probe_summary.json"
        if pair_summary_path.exists():
            pair_summary = json.loads(pair_summary_path.read_text(encoding="utf-8"))
        else:
            pair_summary = {"pair_probe_count": int(pair_count.sum().item() // 2)}
        pair_payload = {"count": pair_count, "summary": pair_summary, "rows": []}
        train_orders_first = (dataset_payload.get("meta") or {}).get("train_random_probe_orders_first", [])
        val_orders_first = (dataset_payload.get("meta") or {}).get("val_random_probe_orders_first", [])
    else:
        print_progress(f"collecting_train_attention_loss_samples={args.num_train_samples}")
        train_mats, train_block_losses, train_orders_first = collect_attention_loss_samples(
            model=model,
            tokens=train_tokens,
            num_samples=int(args.num_train_samples),
            probe_batch_size=int(args.probe_batch_size),
            layer=int(args.layer),
            head=int(args.head),
            export_type=args.export_type,
            rng=rng,
            device=device,
            dtype=args.dtype,
            token_perm=token_perm,
            data_record_mode=data_record_mode,
            progress_label="train_attention_loss_samples",
            progress_interval=int(args.progress_interval),
        )
        print_progress(f"collecting_val_attention_loss_samples={args.num_val_samples}")
        val_mats, val_block_losses, val_orders_first = collect_attention_loss_samples(
            model=model,
            tokens=val_tokens,
            num_samples=int(args.num_val_samples),
            probe_batch_size=int(args.probe_batch_size),
            layer=int(args.layer),
            head=int(args.head),
            export_type=args.export_type,
            rng=rng,
            device=device,
            dtype=args.dtype,
            token_perm=token_perm,
            data_record_mode=data_record_mode,
            progress_label="val_attention_loss_samples",
            progress_interval=int(args.progress_interval),
        )
        print_progress(f"collecting_pair_loss_drop_probes={args.pair_probe_count}")
        pair_payload = collect_pair_loss_drop_probes(
            model=model,
            tokens=pair_tokens,
            pair_probe_count=int(args.pair_probe_count),
            pair_probe_batch_size=int(args.pair_probe_batch_size),
            prefix_blocks=int(args.pair_prefix_blocks),
            full_weight=float(args.pair_full_weight),
            prefix_weight=float(args.pair_prefix_weight),
            signal_mode=str(args.pair_signal_mode),
            rng=rng,
            device=device,
            dtype=args.dtype,
            token_perm=token_perm,
            data_record_mode=data_record_mode,
            progress_interval=max(1, int(args.progress_interval)),
        )
        pair_delta = pair_payload["delta"]
        pair_conf = pair_confidence(pair_delta, pair_payload["count"], margin=float(args.pair_margin))
        write_csv(args.try_dir / "pair_probe_rows.csv", pair_payload["rows"])
        write_json(args.try_dir / "pair_probe_summary.json", pair_payload["summary"])
        config_payload["train_random_probe_orders_first"] = train_orders_first
        config_payload["val_random_probe_orders_first"] = val_orders_first
        torch.save(
            {
                "train_matrices": train_mats,
                "train_block_losses": train_block_losses,
                "val_matrices": val_mats,
                "val_block_losses": val_block_losses,
                "pair_delta": pair_delta,
                "pair_conf": pair_conf,
                "pair_count": pair_payload["count"],
                "meta": config_payload,
            },
            dataset_path,
        )

    train_features = make_features(
        train_mats,
        train_block_losses,
        pair_delta,
        pair_conf,
        feature_clip=float(args.feature_clip),
    )
    val_features = make_features(
        val_mats,
        val_block_losses,
        pair_delta,
        pair_conf,
        feature_clip=float(args.feature_clip),
    )

    hidden_dims = parse_int_list(args.hidden_dims, default=[1024, 1024])
    policy_config = {
        "num_blocks": int(model.num_blocks),
        "input_channels": int(train_features.size(1)),
        "hidden_dims": hidden_dims,
        "dropout": float(args.dropout),
        "activation": args.activation,
    }
    policy = AttentionLossOrderMLP(**policy_config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    train_log_path = args.try_dir / "train_log.jsonl"
    if train_log_path.exists():
        train_log_path.unlink()
    metrics_rows = []

    init_train = evaluate_objective(policy, train_features, train_block_losses, pair_delta, pair_conf, args, device)
    init_val = evaluate_objective(policy, val_features, val_block_losses, pair_delta, pair_conf, args, device)
    row = {"epoch": 0, "phase": "initial"}
    row.update({f"train/{k}": v for k, v in init_train.items()})
    row.update({f"val/{k}": v for k, v in init_val.items()})
    metrics_rows.append(row)
    append_jsonl(train_log_path, row)
    print_progress(json.dumps(row))

    best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    best_val = float(init_val.get("loss_total", float("inf")))
    best_epoch = 0
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_epoch(policy, optimizer, train_features, train_block_losses, pair_delta, pair_conf, args, device)
        val_metrics = evaluate_objective(policy, val_features, val_block_losses, pair_delta, pair_conf, args, device)
        row = {"epoch": epoch, "phase": "train"}
        row.update({f"train/{k}": v for k, v in train_metrics.items()})
        row.update({f"val/{k}": v for k, v in val_metrics.items()})
        metrics_rows.append(row)
        append_jsonl(train_log_path, row)
        print_progress(json.dumps(row))
        val_loss = float(val_metrics.get("loss_total", float("inf")))
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}

    policy.load_state_dict(best_state, strict=True)
    final_train = evaluate_objective(policy, train_features, train_block_losses, pair_delta, pair_conf, args, device)
    final_val = evaluate_objective(policy, val_features, val_block_losses, pair_delta, pair_conf, args, device)
    write_csv(args.try_dir / "metrics.csv", metrics_rows)

    eval_n = min(int(args.eval_samples), int(val_features.size(0)))
    args._pair_delta = pair_delta
    args._pair_conf = pair_conf
    print_progress(f"running_full_eval_samples={eval_n}")
    eval_rows, eval_summary = evaluate_policy_nll(
        policy=policy,
        features=val_features[:eval_n],
        matrices=val_mats[:eval_n],
        block_losses=val_block_losses[:eval_n],
        model=model,
        tokens=val_tokens,
        rng=rng,
        device=device,
        dtype=args.dtype,
        token_perm=token_perm,
        data_record_mode=data_record_mode,
        eval_batch_size=int(args.eval_batch_size),
        random_eval_orders=int(args.random_eval_orders),
        permutation_state=permutation_state,
        args=args,
    )
    write_csv(args.try_dir / "eval_rows.csv", eval_rows)

    policy_path = args.try_dir / "policy.pt"
    summary = {
        "script": "scripts/train/train_attn_mlp_loss_semantics.py",
        "command": "CUDA_VISIBLE_DEVICES=1 python " + " ".join(sys.argv),
        "policy_path": str(policy_path),
        "checkpoint": {
            "ckpt_path": str(args.ckpt_path),
            "iter_num": checkpoint.get("iter_num"),
            "best_val_loss": checkpoint.get("best_val_loss"),
        },
        "policy_config": policy_config,
        "training_meta": {
            "frame": "current",
            "target_direction": "larger_logit_reveals_earlier",
            "loss": "attention/loss semantic pairwise loss-drop objective",
            "pair_signal_mode": str(args.pair_signal_mode),
            "feature_clip": float(args.feature_clip),
            "target_sign": float(args.target_sign),
            "min_std": float(args.min_std),
            "forbidden_signals": FORBIDDEN_SIGNALS,
            "tau_is_diagnostic_only": True,
            "original_l2r_is_diagnostic_only": True,
        },
        "samples": {
            "train_attention_loss_samples": int(args.num_train_samples),
            "val_attention_loss_samples": int(args.num_val_samples),
            "probe_batch_size": int(args.probe_batch_size),
            "pair_probe_count": int(args.pair_probe_count),
            "pair_signal_mode": str(args.pair_signal_mode),
            "pair_probe_summary": pair_payload["summary"],
            "eval_samples": int(eval_n),
            "random_eval_orders": int(args.random_eval_orders),
            "train_random_probe_orders_first": train_orders_first,
            "val_random_probe_orders_first": val_orders_first,
        },
        "optimization": {
            "epochs": int(args.epochs),
            "best_epoch_by_val_objective": int(best_epoch),
            "initial_train": init_train,
            "initial_val": init_val,
            "final_train": final_train,
            "final_val": final_val,
        },
        "eval_summary": eval_summary,
        "diagnostic_acceptance": {
            "origin_tau_gate": 0.4,
            "origin_tau_passed": bool(eval_summary.get("diagnostic/original_l2r_tau", float("-inf")) >= 0.4),
            "nll_vs_random_passed": bool(eval_summary.get("mlp_minus_random_mean", float("inf")) < 0.0),
            "forward_vs_reverse_passed": bool(eval_summary.get("forward_minus_reverse_loss", float("inf")) < 0.0),
        },
    }
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "config": policy_config,
            "policy_type": "attention_loss_semantics_mlp",
            "target_direction": "larger_logit_reveals_earlier",
            "training_meta": summary["training_meta"],
            "metrics": {
                "train": final_train,
                "val": final_val,
                "eval": eval_summary,
            },
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
