"""Unified V3 group-credit trainer for the four Phase-1 arms.

The policy path is deliberately separated from the language-model path:
attention features and rewards are detached, so LM loss updates only the AO-GPT
backbone while policy-gradient loss updates only the OrderHead.  Orders emitted
by g_beta are in model-block coordinates; physical L2R is converted directly
from physical blocks to model-token coordinates.
"""

from __future__ import annotations

import inspect
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
BLOCK_ROOT = ROOT / "block_lo_arm_order_network"
for path in (ROOT, BLOCK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyses.order_head_module import AOGPTWithOrderHead, OrderHeadModule
from analyses.p5_utility_controller import BLOCK_LEN, N, load_p5_ckpt
from analyses.p7_gbeta_policy import GBETA_CKPT, sample_pl
from analyses.v3_group_credit import GroupEMA, group_ids_for, group_rewards, per_sample_loss
from batch_readout.hook_order_provider import random_probe_token_orders
from clean_training_protocol import (
    build_phys_to_model_token_gather,
    physical_blocks_to_model_token_order,
    sample_stream_batch,
)


ARMS = {
    "l2r": {"default_m": None, "use_orderhead": False, "train_orderhead": False},
    "frozen_gbeta": {"default_m": 64, "use_orderhead": True, "train_orderhead": False},
    "joint_group": {"default_m": 16, "use_orderhead": True, "train_orderhead": True},
    "joint_batch": {"default_m": 64, "use_orderhead": True, "train_orderhead": True},
}


def _validate_request(arm, n_steps, batch_size, m, tau):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(float(tau)) or tau <= 0:
        raise ValueError("tau must be finite and positive")

    cfg = ARMS[arm]
    if not cfg["use_orderhead"]:
        if m is not None:
            raise ValueError("m is not used by the l2r arm and must be None")
        return None
    resolved_m = cfg["default_m"] if m is None else m
    # group_ids_for owns the exact positive-integer/divisibility validation.
    group_ids_for(batch_size, resolved_m)
    return resolved_m


@dataclass
class _ContinuousBatchSource:
    stream: object
    gather: torch.Tensor
    heldout: torch.Tensor
    batch_size: int
    block_size: int
    seed: int
    start_step: int

    def __post_init__(self):
        self._heldout_keys = {
            row.contiguous().numpy().tobytes() for row in self.heldout.cpu()
        }

    def batch(self, local_step):
        """Lazily sample the continuation stream; never materialize all steps."""
        micro = 0
        while True:
            physical = sample_stream_batch(
                self.stream,
                self.batch_size,
                self.block_size,
                self.seed,
                self.start_step + int(local_step),
                micro,
            )
            model = physical[:, self.gather]
            # The fixed held-out batch is never returned for training.  Exact row
            # comparison also protects tiny synthetic streams used by tests.
            collision = any(
                row.contiguous().numpy().tobytes() in self._heldout_keys
                for row in model
            )
            if not collision:
                return model
            micro += 1
            if micro > 1024:
                raise RuntimeError("could not draw a training batch disjoint from held-out data")


@dataclass
class _FiniteBatchSource:
    chunks: torch.Tensor
    heldout: torch.Tensor
    batch_size: int

    def batch(self, local_step):
        start = (int(local_step) * self.batch_size) % len(self.chunks)
        ids = (torch.arange(self.batch_size) + start) % len(self.chunks)
        return self.chunks.index_select(0, ids)


def _load_training_context(ckpt_path, batch_size, n_steps, device):
    """Load a model plus a lazy, deterministic source and one reserved batch.

    The production 10k checkpoint uses a continuous memmap, so a 20k-step run
    keeps only one training batch and one fixed held-out batch in memory.
    Classic chunk checkpoints fall back to a bounded deterministic pool; the
    result metadata makes that fallback explicit.
    """
    model, heldout, clean_perm, dev = load_p5_ckpt(ckpt_path, batch_size, device=device)
    heldout = torch.stack(list(heldout)) if not isinstance(heldout, torch.Tensor) else heldout
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = {
        "args": raw.get("args", {}),
        "global_step": raw.get("global_step", raw.get("iter_num", 0)),
        "iter_num": raw.get("iter_num", 0),
        "optimizer": raw.get("optimizer"),
    }
    del raw
    args = meta.get("args", {})
    start_step = int(meta.get("global_step", meta.get("iter_num", 0)))

    if args.get("data_source") == "continuous":
        from clean_training_protocol import load_token_stream

        stream = load_token_stream(args["train_bin"])
        gather = build_phys_to_model_token_gather(clean_perm, BLOCK_LEN)
        source = _ContinuousBatchSource(
            stream=stream,
            gather=gather,
            heldout=heldout.cpu(),
            batch_size=batch_size,
            block_size=N * BLOCK_LEN,
            seed=int(args.get("seed", 0)),
            start_step=start_step,
        )
        meta["data_source_mode"] = "continuous_lazy"
    else:
        # A bounded pool avoids an n_steps*batch_size allocation.  This path is
        # not the Phase-1 production protocol and is identified in the result.
        pool_size = max(batch_size, min(max(n_steps * batch_size, batch_size), 4096))
        _, pool, _, _ = load_p5_ckpt(ckpt_path, pool_size + batch_size, device=device)
        pool = torch.stack(list(pool)) if not isinstance(pool, torch.Tensor) else pool
        source = _FiniteBatchSource(pool[batch_size:], pool[:batch_size], batch_size)
        heldout = source.heldout
        meta["data_source_mode"] = "bounded_chunk_pool"
    return model, source, clean_perm, dev, meta


def _forward_with_token_losses(model, idx, token_order):
    """Return ``(loss, per-token loss, contract name)``.

    Newer AO-GPT variants expose ``return_token_loss=True``.  One legacy family
    exposes the same values as ``return_probe_data=True``/``loss_per_step``.
    The production 10k handoff class exposes neither, so its exact-logit fallback
    is handled below without deriving token loss from a batch scalar.
    """
    params = inspect.signature(model.forward_fn).parameters
    if "return_token_loss" in params:
        _logits, loss, token_losses = model.forward_fn(
            idx, token_order, return_token_loss=True
        )
        return loss, token_losses, "return_token_loss"
    if "return_probe_data" in params:
        _logits, loss, probe = model.forward_fn(
            idx, token_order, return_probe_data=True
        )
        if "loss_per_step" not in probe:
            raise RuntimeError("forward_fn probe data did not contain loss_per_step")
        return loss, probe["loss_per_step"], "return_probe_data.loss_per_step"
    # The production 10k class predates both flags.  Derive the exact same
    # unreduced CE from its returned logits and its own shuffle operation,
    # while preserving the model-returned scalar loss for backbone backward.
    if not hasattr(model, "shuffle"):
        raise TypeError(
            "forward_fn exposes neither token-loss contract and model has no shuffle() fallback"
        )
    logits, loss = model.forward_fn(idx, token_order)
    targets = model.shuffle(idx, token_order)
    shift_logits = logits[..., :-1, :].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-1,
        reduction="none",
    ).view_as(targets)
    if not torch.allclose(token_losses.mean(), loss, rtol=1e-6, atol=1e-7):
        raise RuntimeError("derived per-token losses do not reproduce forward_fn scalar loss")
    return loss, token_losses, "derived_from_logits_and_model.shuffle"


def _scheduled_lr(global_step, args, base_lr):
    warmup = int(args.get("warmup_iters", 0))
    decay_steps = int(args.get("lr_decay_steps", 50000))
    original_base = float(args.get("lr", base_lr))
    original_min = float(args.get("min_lr", min(base_lr, 1e-4)))
    min_lr = original_min if base_lr == original_base else min(original_min, base_lr)
    if global_step < warmup:
        return base_lr * (global_step + 1) / max(warmup, 1)
    if global_step >= decay_steps:
        return min_lr
    ratio = (global_step - warmup) / max(decay_steps - warmup, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (base_lr - min_lr)


def _build_optimizer(model, order_head, train_orderhead, meta, lr_backbone, lr_orderhead, dev):
    args = meta.get("args", {})
    base_lr = float(args.get("lr", 1e-4) if lr_backbone is None else lr_backbone)
    weight_decay = float(args.get("weight_decay", 0.1))
    betas = (float(args.get("beta1", 0.9)), float(args.get("beta2", 0.99)))
    if hasattr(model, "configure_optimizers"):
        optimizer = model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=base_lr,
            betas=betas,
            device_type=dev.type,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=betas
        )

    resume_status = "checkpoint_has_no_optimizer"
    state = meta.pop("optimizer", None)
    if state is not None:
        try:
            optimizer.load_state_dict(state)
            if dev.type != "cuda":
                for group in optimizer.param_groups:
                    group["fused"] = False
            resume_status = "loaded"
        except (ValueError, RuntimeError) as exc:
            # All arms take this same path.  Never silently claim an exact resume.
            resume_status = f"not_loaded:{type(exc).__name__}:{exc}"

    if train_orderhead:
        optimizer.add_param_group({
            "params": list(order_head.gbeta.parameters()),
            "lr": float(lr_orderhead),
            "weight_decay": 0.0,
            "is_orderhead": True,
        })
    for group in optimizer.param_groups:
        group.setdefault("is_orderhead", False)
    return optimizer, base_lr, resume_status


def _grad_norm(parameters):
    squared = sum(
        p.grad.detach().float().square().sum().item()
        for p in parameters
        if p.grad is not None
    )
    return float(math.sqrt(squared))


def _snapshot(parameters):
    return [p.detach().cpu().clone() for p in parameters]


def _parameter_delta(parameters, before):
    return float(sum(
        (p.detach().cpu() - old).abs().sum().item()
        for p, old in zip(parameters, before)
    ))


def _deterministic_order(scores):
    return np.argsort(-scores.detach().cpu().numpy(), kind="stable").astype(np.int64)


def train_arm(
    ckpt_path,
    arm,
    *,
    n_steps,
    batch_size=64,
    m=None,
    lr_backbone=None,
    lr_orderhead=3e-4,
    tau=1.0,
    beta=3e-3,
    ema_alpha=0.9,
    adv_clip=0.3,
    device="cpu",
    out_dir="runs/v3_sweep",
    tag="phase1",
    eval_steps=(),
    evaluator=None,
):
    """Train one Phase-1 arm and return a JSON-serializable result.

    ``eval_steps`` are 1-based local continuation steps.  Task 5's runner is
    responsible for converting its global 10k/15k/... schedule to local steps.
    """
    resolved_m = _validate_request(arm, n_steps, batch_size, m, tau)
    cfg = ARMS[arm]
    model, source, clean_perm, dev, meta = _load_training_context(
        ckpt_path, batch_size, n_steps, device
    )
    model.train()

    order_head = None
    if cfg["use_orderhead"]:
        order_head = OrderHeadModule(GBETA_CKPT, device=str(dev))
        order_head.to(dev)
        for parameter in order_head.gbeta.parameters():
            parameter.requires_grad_(cfg["train_orderhead"])
        wrap = AOGPTWithOrderHead(model, order_head, clean_perm, device=str(dev))
    else:
        wrap = SimpleNamespace(backbone=model, order_head=None, clean_perm=clean_perm)
    wrap.heldout_chunks = source.heldout.detach().clone()

    optimizer, base_lr, optimizer_resume = _build_optimizer(
        model, order_head, cfg["train_orderhead"], meta,
        lr_backbone, lr_orderhead, dev,
    )
    backbone_params = list(model.parameters())
    orderhead_params = list(order_head.gbeta.parameters()) if order_head is not None else []
    backbone_before = _snapshot(backbone_params)
    orderhead_before = _snapshot(orderhead_params)

    groups = group_ids_for(batch_size, resolved_m) if cfg["use_orderhead"] else []
    ema = GroupEMA(len(groups), ema_alpha) if cfg["train_orderhead"] else None
    start_step = int(meta.get("global_step", meta.get("iter_num", 0)))
    policy_seed = int(meta.get("args", {}).get("seed", 0))
    torch.manual_seed(policy_seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(policy_seed)
    eval_step_set = {int(step) for step in eval_steps}
    logs, evals = [], []
    pg_only_backbone_grad = 0.0
    token_loss_contract = None

    for local_step in range(n_steps):
        global_step = start_step + local_step
        idx = source.batch(local_step).to(dev)
        logps, entropies = [], []

        if arm == "l2r":
            physical = torch.arange(N, dtype=torch.long).repeat(batch_size, 1)
            token_order = physical_blocks_to_model_token_order(
                physical, clean_perm, BLOCK_LEN
            ).to(dev)
        else:
            model_orders = []
            for group in groups:
                group_index = torch.as_tensor(group, dtype=torch.long, device=dev)
                idx_group = idx.index_select(0, group_index)
                probe = random_probe_token_orders(
                    len(group), 0, global_step, dev
                )
                scores = wrap.compute_order_logits(idx_group, probe, per_sample=False)[0]
                # The attention extractor enters eval mode; continuation training
                # must explicitly return to train mode before the LM forward.
                model.train()
                if arm == "frozen_gbeta":
                    order = _deterministic_order(scores)
                else:
                    order, logp, entropy = sample_pl(scores, tau=tau)
                    logps.append(logp)
                    entropies.append(entropy)
                model_orders.append(np.repeat(order[None, :], len(group), axis=0))
            token_order = wrap.token_orders_from_model_blocks(
                np.concatenate(model_orders, axis=0)
            ).to(dev)

        lm_loss, token_losses, contract = _forward_with_token_losses(
            model, idx, token_order
        )
        token_loss_contract = contract
        if cfg["train_orderhead"]:
            logp = torch.stack(logps)
            entropy = torch.stack(entropies)
            # Reward and advantage are numerical feedback, never a differentiable
            # path into AO-GPT.
            ell_i = per_sample_loss(token_losses.detach())
            ell_g = group_rewards(ell_i, groups)
            baseline = ema.update(ell_g)
            advantage = (baseline - ell_g).detach().clamp(-adv_clip, adv_clip)
            pg = -(advantage * logp).mean() - float(beta) * entropy.mean()
            pg_grads = torch.autograd.grad(
                pg,
                backbone_params,
                retain_graph=True,
                allow_unused=True,
            )
            pg_only_backbone_grad = max(
                pg_only_backbone_grad,
                float(sum(
                    grad.detach().abs().sum().item()
                    for grad in pg_grads
                    if grad is not None
                )),
            )
            total_loss = lm_loss + pg
            entropy_value = float(entropy.detach().mean().item())
        else:
            pg = lm_loss.new_zeros(())
            total_loss = lm_loss
            entropy_value = 0.0

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        backbone_grad_norm = _grad_norm(backbone_params)
        orderhead_grad_norm = _grad_norm(orderhead_params)
        grad_clip = float(meta.get("args", {}).get("grad_clip", 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(backbone_params, grad_clip)
        lr = _scheduled_lr(global_step, meta.get("args", {}), base_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = (
                float(lr_orderhead) if param_group.get("is_orderhead") else lr
            )
        optimizer.step()

        entry = {
            "step": local_step + 1,
            "global_step": global_step + 1,
            "lm_loss": float(lm_loss.detach().item()),
            "pg": float(pg.detach().item()),
            "entropy": entropy_value,
            "orderhead_grad_norm": orderhead_grad_norm,
            "backbone_grad_norm": backbone_grad_norm,
            "lr_backbone": float(lr),
        }
        logs.append(entry)

        callback_step = local_step + 1
        if evaluator is not None and callback_step in eval_step_set:
            model.eval()
            try:
                metrics = evaluator(callback_step, model, wrap)
            finally:
                model.train()
            if not isinstance(metrics, dict):
                raise TypeError("evaluator must return a dict")
            evals.append({"step": callback_step, **metrics})

    orderhead_delta = _parameter_delta(orderhead_params, orderhead_before)
    backbone_delta = _parameter_delta(backbone_params, backbone_before)
    numeric_log_keys = (
        "lm_loss", "pg", "entropy", "orderhead_grad_norm", "backbone_grad_norm"
    )
    nan = any(
        not math.isfinite(float(entry[key]))
        for entry in logs
        for key in numeric_log_keys
    )
    result = {
        "ckpt_path": str(ckpt_path),
        "arm": arm,
        "m": resolved_m,
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "start_global_step": start_step,
        "policy_seed": policy_seed,
        "tau": float(tau),
        "beta": float(beta),
        "ema_alpha": float(ema_alpha),
        "adv_clip": float(adv_clip),
        "lr_backbone_base": float(base_lr),
        "lr_orderhead": float(lr_orderhead),
        "nan": bool(nan),
        "orderhead_param_delta": orderhead_delta,
        "backbone_param_delta": backbone_delta,
        "pg_only_backbone_grad": float(pg_only_backbone_grad),
        "optimizer_resume": optimizer_resume,
        "data_source_mode": meta.get("data_source_mode", "injected"),
        "token_loss_contract": token_loss_contract,
        "log": logs,
        "evals": evals,
    }
    output = pathlib.Path(out_dir) / str(tag) / arm
    output.mkdir(parents=True, exist_ok=True)
    (output / "train.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


__all__ = ["ARMS", "train_arm"]
