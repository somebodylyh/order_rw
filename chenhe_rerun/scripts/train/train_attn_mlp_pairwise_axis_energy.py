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
    append_jsonl,
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
from train_attn_mlp_order_distribution_energy import (  # noqa: E402
    AttentionDistributionOrderMLP,
    logits_entropy,
    make_features,
    sample_plackett_luce_orders,
)


LOSS_NAME = "block-pair frozen-NLL axis energy"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Attn-MLP with block-pair frozen-NLL axis energy.")
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--try_dir", type=Path, required=True)
    parser.add_argument(
        "--source_dataset_path",
        type=Path,
        default=Path("Report/attn_mlp_loss_design_exploration/try2_resampled_candidate_energy/distribution_dataset.pt"),
        help="Full-scale random-candidate frozen-NLL bank used to derive block-pair energy targets.",
    )
    parser.add_argument("--dataset", type=str, default="wikitext103")
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--num_train_samples", type=int, default=2000)
    parser.add_argument("--num_val_samples", type=int, default=512)
    parser.add_argument("--feature_clip", type=float, default=8.0)
    parser.add_argument("--hidden_dims", type=str, default="1024,1024")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--tau_q", type=float, default=0.20)
    parser.add_argument("--tau_s", type=float, default=1.0)
    parser.add_argument("--q_margin", type=float, default=0.20)
    parser.add_argument("--min_pair_abs", type=float, default=0.03)
    parser.add_argument("--lambda_pair_bce", type=float, default=1.0)
    parser.add_argument("--lambda_axis_energy", type=float, default=0.25)
    parser.add_argument("--lambda_std_floor", type=float, default=0.20)
    parser.add_argument("--lambda_entropy_ceiling", type=float, default=0.02)
    parser.add_argument("--lambda_attention_smooth", type=float, default=0.005)
    parser.add_argument("--lambda_center", type=float, default=0.001)
    parser.add_argument("--min_std", type=float, default=0.75)
    parser.add_argument("--max_entropy", type=float, default=3.8)
    parser.add_argument("--eval_samples", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--random_eval_orders", type=int, default=16)
    parser.add_argument("--policy_sample_eval_orders", type=int, default=8)
    parser.add_argument("--success_random_margin", type=float, default=0.005)
    parser.add_argument("--success_tau_gate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--progress_interval", type=int, default=100)
    parser.add_argument("--recompute_targets", action="store_true")
    parser.add_argument("--allow_small_debug", action="store_true")
    return parser.parse_args()


def print_progress(text: str):
    print(text, flush=True)


def command_invocation():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    prefix = f"CUDA_VISIBLE_DEVICES={cuda_visible} python " if cuda_visible else "python "
    return prefix + " ".join(sys.argv)


def validate_budget(args, train_payload, val_payload):
    if bool(args.allow_small_debug):
        return
    train_n = min(int(args.num_train_samples), int(train_payload["candidate_orders"].size(0)))
    val_n = min(int(args.num_val_samples), int(val_payload["candidate_orders"].size(0)))
    bank = int(train_payload["candidate_orders"].size(1))
    checks = [
        ("num_train_samples", train_n, 2000),
        ("num_val_samples", val_n, 512),
        ("epochs", int(args.epochs), 50),
        ("eval_samples", int(args.eval_samples), 512),
        ("random_eval_orders", int(args.random_eval_orders), 16),
        ("train_order_probes", train_n * bank, 10000),
    ]
    too_small = [(name, value, minimum) for name, value, minimum in checks if value < minimum]
    if too_small:
        text = ", ".join(f"{name}={value} < {minimum}" for name, value, minimum in too_small)
        raise ValueError(f"Formal try budget is too small: {text}.")


def trim_payload(payload: dict, count: int):
    count = int(count)
    out = {}
    for key, value in payload.items():
        if torch.is_tensor(value) and value.size(0) >= count:
            out[key] = value[:count].clone()
        else:
            out[key] = value
    return out


def compute_pairwise_axis_targets(candidate_orders: torch.Tensor, candidate_losses: torch.Tensor, chunk_size: int = 64):
    """Estimate Q[i,j] from random candidate NLL: positive means i before j lowers loss."""
    orders = candidate_orders.long()
    losses = candidate_losses.float()
    total, bank, num_blocks = orders.shape
    outputs = []
    arange_blocks = torch.arange(num_blocks, dtype=torch.long).view(1, 1, num_blocks)
    eye = torch.eye(num_blocks, dtype=torch.bool)
    for start in range(0, total, int(chunk_size)):
        end = min(total, start + int(chunk_size))
        chunk_orders = orders[start:end]
        chunk_losses = losses[start:end]
        bsz = int(chunk_orders.size(0))
        positions = torch.empty_like(chunk_orders)
        positions.scatter_(2, chunk_orders, arange_blocks.expand(bsz, bank, num_blocks))
        centered = chunk_losses - chunk_losses.mean(dim=1, keepdim=True)
        scale = chunk_losses.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        advantage = -(centered / scale)
        pair_sign = (positions[:, :, None, :] - positions[:, :, :, None]).sign().float()
        q = (advantage[:, :, None, None] * pair_sign).mean(dim=1)
        q[:, eye] = 0.0
        outputs.append(q)
    return torch.cat(outputs, dim=0).float()


def pairwise_axis_loss(logits: torch.Tensor, q_targets: torch.Tensor, features: torch.Tensor, args):
    device = logits.device
    s = logits.float()
    q = q_targets.to(device=device, dtype=torch.float32)
    n = int(s.size(1))
    ii, jj = torch.triu_indices(n, n, offset=1, device=device)
    qv = q[:, ii, jj]
    diff = s[:, ii] - s[:, jj]
    pair_logits = diff / max(float(args.tau_s), 1e-8)
    target = torch.sigmoid(qv / max(float(args.tau_q), 1e-8)).detach()
    weights = ((qv.abs() - float(args.min_pair_abs)) / max(float(args.q_margin), 1e-8)).clamp(0.0, 1.0)
    raw_bce = F.binary_cross_entropy_with_logits(pair_logits, target, reduction="none")
    loss_pair = (raw_bce * weights).sum() / weights.sum().clamp_min(1e-6)

    axis_alignment = torch.tanh(pair_logits) * torch.sign(qv)
    loss_axis_energy = -(axis_alignment * weights).sum() / weights.sum().clamp_min(1e-6)

    std = s.std(dim=1, unbiased=False)
    loss_std_floor = torch.relu(s.new_tensor(float(args.min_std)) - std).pow(2).mean()
    entropy = logits_entropy(s)
    loss_entropy_ceiling = torch.relu(entropy - s.new_tensor(float(args.max_entropy))).pow(2).mean()
    loss_center = s.mean(dim=1).pow(2).mean()

    x = (s - s.mean(dim=1, keepdim=True)) / s.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    smooth_weights = torch.relu(features[:, 1].to(device=device, dtype=torch.float32))
    eye = torch.eye(n, dtype=torch.bool, device=device)
    smooth_weights = smooth_weights.masked_fill(eye.view(1, n, n), 0.0)
    xdiff = x[:, :, None] - x[:, None, :]
    loss_attention_smooth = (smooth_weights * xdiff.pow(2)).sum() / smooth_weights.sum().clamp_min(1e-6)

    with torch.no_grad():
        active = weights > 0.0
        if bool(active.any()):
            pred_sign = torch.sign(diff[active])
            truth_sign = torch.sign(qv[active])
            pair_acc = (pred_sign == truth_sign).float().mean()
            q_abs = qv[active].abs().mean()
        else:
            pair_acc = s.new_tensor(float("nan"))
            q_abs = s.new_tensor(0.0)
        pred = diff
        pred_center = pred - pred.mean(dim=1, keepdim=True)
        q_center = qv - qv.mean(dim=1, keepdim=True)
        corr = (pred_center * q_center).mean(dim=1)
        corr = corr / (
            pred_center.std(dim=1, unbiased=False).clamp_min(1e-6)
            * q_center.std(dim=1, unbiased=False).clamp_min(1e-6)
        )

    total = (
        float(args.lambda_pair_bce) * loss_pair
        + float(args.lambda_axis_energy) * loss_axis_energy
        + float(args.lambda_std_floor) * loss_std_floor
        + float(args.lambda_entropy_ceiling) * loss_entropy_ceiling
        + float(args.lambda_attention_smooth) * loss_attention_smooth
        + float(args.lambda_center) * loss_center
    )
    return {
        "loss_total": total,
        "loss_pair_bce": loss_pair,
        "loss_axis_energy": loss_axis_energy,
        "loss_std_floor": loss_std_floor,
        "loss_entropy_ceiling": loss_entropy_ceiling,
        "loss_attention_smooth": loss_attention_smooth,
        "loss_center": loss_center,
        "pairwise_axis_acc": pair_acc.detach(),
        "pairwise_axis_corr": corr.mean().detach(),
        "pairwise_q_abs": q_abs.detach(),
        "logit_std": std.mean().detach(),
        "logit_entropy": entropy.mean().detach(),
    }


@torch.no_grad()
def evaluate_objective(policy, features, q_targets, args, device):
    policy.eval()
    rows = []
    orders = []
    for start in range(0, int(features.size(0)), int(args.batch_size)):
        end = min(int(features.size(0)), start + int(args.batch_size))
        feat = features[start:end].to(device=device)
        q = q_targets[start:end]
        logits = policy(feat)
        metrics = pairwise_axis_loss(logits, q, feat, args)
        rows.append(tensor_metrics_to_floats(metrics))
        orders.extend([order_from_logits_desc(row).detach().cpu() for row in logits])
    out = mean_dicts(rows)
    if len(orders) > 1:
        from attn_mlp_implicit_axis_common import kendall_tau_between_orders

        taus = [kendall_tau_between_orders(orders[idx - 1], orders[idx]) for idx in range(1, len(orders))]
        out["order_change_kendall_current"] = float(np.mean(taus))
    return out


def train_epoch(policy, optimizer, features, q_targets, args, device):
    policy.train()
    idx_order = torch.randperm(int(features.size(0)))
    rows = []
    for start in range(0, int(idx_order.numel()), int(args.batch_size)):
        idx = idx_order[start : start + int(args.batch_size)]
        feat = features.index_select(0, idx).to(device=device)
        q = q_targets.index_select(0, idx)
        optimizer.zero_grad(set_to_none=True)
        logits = policy(feat)
        metrics = pairwise_axis_loss(logits, q, feat, args)
        metrics["loss_total"].backward()
        if float(args.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(args.grad_clip))
        optimizer.step()
        rows.append(tensor_metrics_to_floats(metrics))
    return mean_dicts(rows)


@torch.no_grad()
def evaluate_policy_nll_axis(
    *,
    policy,
    features,
    q_targets,
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
            pairwise_axis_loss(
                logits.view(1, -1).to(device),
                q_targets[idx : idx + 1],
                feat,
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
            "pairwise_axis_acc": objective.get("pairwise_axis_acc", float("nan")),
            "pairwise_axis_corr": objective.get("pairwise_axis_corr", float("nan")),
            "pairwise_q_abs": objective.get("pairwise_q_abs", float("nan")),
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


def make_method_md(args, samples):
    return f"""# Method

## Loss Candidate

This try trains an Attn-MLP order distribution with a block-pair frozen-NLL
axis-energy loss. The MLP outputs `s in R^64`, interpreted as a learned
low-dimensional axis over current-frame blocks.

The supervision is derived only from random candidate orders and frozen AO-GPT
NLL. For each state and block pair `(i,j)`, the script estimates:

```text
Q_ij = mean_k advantage_k * sign(pos_k(j) - pos_k(i))
advantage_k = -(L_k - mean(L)) / std(L)
```

So `Q_ij > 0` means random candidates where block `i` appears before block `j`
tended to have lower frozen AO-GPT NLL. This is not an attention-derived order
and not an original-order target.

The MLP axis is trained with:

```text
target_ij = sigmoid(Q_ij / tau_q)
pred_ij   = sigmoid((s_i - s_j) / tau_s)
L_pair    = weighted_BCE(pred_ij, target_ij)
L_axis    = - weighted_mean(tanh((s_i - s_j)/tau_s) * sign(Q_ij))
```

The objective also includes non-collapse terms, a mild attention-graph
smoothness regularizer, and centering. Attention and current block losses are
input features only; no hard or soft order generated from attention is used as
a target.

## Allowed Signals

- current-frame attention and current-frame block losses as MLP input
- random current-frame candidate orders from the source bank
- frozen AO-GPT NLL of those random candidates
- a block-pair axis target derived from random-candidate NLL correlations

## Forbidden Signals

The following are not used in training, direction selection, model selection,
early stopping, or loss-design selection:

```text
{json.dumps(FORBIDDEN_SIGNALS + ["generated_teacher_order_supervision"], indent=2)}
```

`diagnostic/original_l2r_tau` is written only after the full try finishes.

## Hyperparameters

- source dataset: `{args.source_dataset_path}`
- train samples: `{samples['train_samples']}`
- val samples: `{samples['val_samples']}`
- candidate bank size: `{samples['candidate_bank_size']}`
- train order probes: `{samples['train_order_probes']}`
- `tau_q`: `{args.tau_q}`
- `tau_s`: `{args.tau_s}`
- `q_margin`: `{args.q_margin}`
- `min_pair_abs`: `{args.min_pair_abs}`
- `lambda_pair_bce`: `{args.lambda_pair_bce}`
- `lambda_axis_energy`: `{args.lambda_axis_energy}`
- `lambda_std_floor`: `{args.lambda_std_floor}`
- `lambda_entropy_ceiling`: `{args.lambda_entropy_ceiling}`
- `lambda_attention_smooth`: `{args.lambda_attention_smooth}`
- `min_std`: `{args.min_std}`
- `max_entropy`: `{args.max_entropy}`

## Model Selection

No validation-PPL shortcut or early stop is used. The formal try trains for the
fixed requested epoch count and evaluates the final checkpoint. The success gate
for this continuation is `diagnostic/original_l2r_tau > {args.success_tau_gate}`
and a frozen-NLL margin greater than `{args.success_random_margin}`.
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
    tau_ok = ev.get("diagnostic/original_l2r_tau", float("-inf")) > summary["training_meta"]["success_tau_gate"]
    success = bool((nll_ok or dist_ok) and tau_ok)
    reasons = []
    if not nll_ok:
        reasons.append("MAP MLP order did not clearly beat random mode loss")
    if not dist_ok:
        reasons.append("sampled MLP distribution did not clearly beat random mode loss")
    if not tau_ok:
        reasons.append("post-hoc diagnostic original_l2r_tau did not exceed the current gate")
    if not noncollapsed:
        reasons.append("logits remained below the configured non-collapse std floor")
    next_step = "success gate reached; replicate with another seed and then test joint-training insertion"
    if not success:
        if tau_ok and not (nll_ok or dist_ok):
            next_step = "axis structure improved tau but not NLL; combine with direct policy-gradient frozen-NLL reward"
        elif (nll_ok or dist_ok) and not tau_ok:
            next_step = "NLL improved but tau gate failed; add stability/axis constraints without original-order signals"
        else:
            next_step = "move to direct on-policy or REINFORCE frozen-NLL reward; cached random-candidate energy is still too indirect"
    return f"""# Failure Analysis

## Acceptance Checks

- train objective decreased: `{objective_drop > 0}` (`drop={objective_drop:.6f}`)
- logits non-collapsed: `{noncollapsed}` (`train/logit_std={opt['final_train'].get('logit_std'):.6f}`)
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


def write_try_docs(args, samples, summary=None):
    args.try_dir.mkdir(parents=True, exist_ok=True)
    (args.try_dir / "method.md").write_text(make_method_md(args, samples), encoding="utf-8")
    (args.try_dir / "commands.sh").write_text(command_invocation() + "\n", encoding="utf-8")
    readme = f"""# Attn-MLP Pairwise Axis Energy Try

- checkpoint: `{args.ckpt_path}`
- source dataset: `{args.source_dataset_path}`
- loss: `{LOSS_NAME}`
- frame: `current`
- train samples: `{samples['train_samples']}`
- val samples: `{samples['val_samples']}`
- candidate bank size: `{samples['candidate_bank_size']}`
- train order probes: `{samples['train_order_probes']}`
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
- `pairwise_axis_targets.pt`
"""
    (args.try_dir / "README.md").write_text(readme, encoding="utf-8")
    if summary is not None:
        (args.try_dir / "failure_analysis.md").write_text(make_failure_analysis(summary), encoding="utf-8")


def main():
    args = parse_args()
    args.try_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    device = torch.device(args.device)

    if not args.source_dataset_path.exists():
        raise FileNotFoundError(f"Missing source dataset: {args.source_dataset_path}")
    source = torch.load(args.source_dataset_path, map_location="cpu")
    train_payload = trim_payload(source["train"], int(args.num_train_samples))
    val_payload = trim_payload(source["val"], int(args.num_val_samples))
    validate_budget(args, train_payload, val_payload)

    bank = int(train_payload["candidate_orders"].size(1))
    samples = {
        "train_samples": int(train_payload["candidate_orders"].size(0)),
        "val_samples": int(val_payload["candidate_orders"].size(0)),
        "candidate_bank_size": bank,
        "train_order_probes": int(train_payload["candidate_orders"].size(0)) * bank,
        "val_order_probes": int(val_payload["candidate_orders"].size(0)) * int(val_payload["candidate_orders"].size(1)),
    }
    write_try_docs(args, samples)

    targets_path = args.try_dir / "pairwise_axis_targets.pt"
    if targets_path.exists() and not bool(args.recompute_targets):
        print_progress(f"loading_existing_targets={targets_path}")
        targets = torch.load(targets_path, map_location="cpu")
        train_q = targets["train_q"]
        val_q = targets["val_q"]
    else:
        print_progress("computing_pairwise_axis_targets=train")
        train_q = compute_pairwise_axis_targets(train_payload["candidate_orders"], train_payload["candidate_losses"])
        print_progress("computing_pairwise_axis_targets=val")
        val_q = compute_pairwise_axis_targets(val_payload["candidate_orders"], val_payload["candidate_losses"])
        torch.save({"train_q": train_q, "val_q": val_q, "meta": samples}, targets_path)

    train_features = make_features(train_payload["matrices"], train_payload["block_losses"], float(args.feature_clip))
    val_features = make_features(val_payload["matrices"], val_payload["block_losses"], float(args.feature_clip))
    hidden_dims = parse_int_list(args.hidden_dims, default=[1024, 1024])
    policy_config = {
        "num_blocks": int(train_features.size(-1)),
        "input_channels": int(train_features.size(1)),
        "hidden_dims": hidden_dims,
        "dropout": float(args.dropout),
        "activation": args.activation,
    }
    policy = AttentionDistributionOrderMLP(**policy_config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "frame": "current",
            "loss": LOSS_NAME,
            "forbidden_signals": FORBIDDEN_SIGNALS + ["generated_teacher_order_supervision"],
            "original_l2r_is_diagnostic_only": True,
            "tau_is_diagnostic_only": True,
            "model_selection": "fixed_final_epoch_no_val_ppl_shortcut",
            "samples": samples,
        }
    )
    write_json(args.try_dir / "config.json", config_payload)

    train_log_path = args.try_dir / "train_log.jsonl"
    if train_log_path.exists():
        train_log_path.unlink()
    metrics_rows = []
    init_train = evaluate_objective(policy, train_features, train_q, args, device)
    init_val = evaluate_objective(policy, val_features, val_q, args, device)
    row = {"epoch": 0, "phase": "initial"}
    row.update({f"train/{key}": value for key, value in init_train.items()})
    row.update({f"val/{key}": value for key, value in init_val.items()})
    metrics_rows.append(row)
    append_jsonl(train_log_path, row)
    print_progress(json.dumps(row))

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_epoch(policy, optimizer, train_features, train_q, args, device)
        val_metrics = evaluate_objective(policy, val_features, val_q, args, device)
        row = {"epoch": epoch, "phase": "train"}
        row.update({f"train/{key}": value for key, value in train_metrics.items()})
        row.update({f"val/{key}": value for key, value in val_metrics.items()})
        metrics_rows.append(row)
        append_jsonl(train_log_path, row)
        print_progress(json.dumps(row))

    final_train = evaluate_objective(policy, train_features, train_q, args, device)
    final_val = evaluate_objective(policy, val_features, val_q, args, device)
    write_csv(args.try_dir / "metrics.csv", metrics_rows)

    print_progress(f"loading_ckpt_for_full_eval={args.ckpt_path}")
    model, checkpoint = load_frozen_aogpt(args.ckpt_path, device)
    data_dir = infer_data_dir(args.dataset, args.data_dir)
    val_tokens = load_tokens(data_dir, args.val_split)
    data_record_mode = infer_data_record_mode(checkpoint)
    permutation_state = load_permutation_state(
        checkpoint,
        num_blocks=int(model.num_blocks),
        block_len=int(model.block_order_block_len),
    )
    token_perm = None if permutation_state is None else permutation_state["token_perm"]

    eval_n = min(int(args.eval_samples), int(val_features.size(0)))
    print_progress(f"running_full_eval_samples={eval_n}")
    eval_rows, eval_summary = evaluate_policy_nll_axis(
        policy=policy,
        features=val_features[:eval_n],
        q_targets=val_q[:eval_n],
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
        "script": "scripts/train/train_attn_mlp_pairwise_axis_energy.py",
        "command": command_invocation(),
        "policy_path": str(policy_path),
        "source_dataset_path": str(args.source_dataset_path),
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
            "model_selection": "fixed_final_epoch_no_val_ppl_shortcut",
            "min_std": float(args.min_std),
            "success_random_margin": float(args.success_random_margin),
            "success_tau_gate": float(args.success_tau_gate),
        },
        "samples": {
            **samples,
            "epochs": int(args.epochs),
            "eval_samples": int(eval_n),
            "random_eval_orders": int(args.random_eval_orders),
            "policy_sample_eval_orders": int(args.policy_sample_eval_orders),
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
            "policy_type": "attention_pairwise_axis_energy_mlp",
            "training_meta": summary["training_meta"],
            "metrics": {"train": final_train, "val": final_val, "eval": eval_summary},
        },
        policy_path,
    )
    write_json(args.try_dir / "eval_summary.json", summary)
    write_try_docs(args, samples, summary=summary)
    print_progress(f"try_dir={args.try_dir}")
    print_progress(json.dumps(summary["diagnostic_acceptance"], indent=2))
    print_progress(json.dumps(eval_summary, indent=2))


if __name__ == "__main__":
    main()
