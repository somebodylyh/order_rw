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
import os
import pathlib
import random
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
from analyses.p7_gbeta_policy import GBETA_CKPT
from analyses.v3_group_credit import GroupEMA, group_ids_for, group_rewards, per_sample_loss
from batch_readout.hook_order_provider import random_probe_token_orders
from clean_training_protocol import (
    batch_indices_for_step,
    build_phys_to_model_token_gather,
    physical_blocks_to_model_token_order,
    phys_to_model_idx_clean,
    sample_stream_batch,
)
from training_utils import load_train_chunks


ARMS = {
    "l2r":           {"default_m": None, "use_orderhead": False, "train_orderhead": False, "freeze_backbone": False},
    "frozen_gbeta":  {"default_m": 64,  "use_orderhead": True,  "train_orderhead": False, "freeze_backbone": False},
    "joint_group":   {"default_m": 16,  "use_orderhead": True,  "train_orderhead": True,  "freeze_backbone": False},
    "joint_batch":   {"default_m": 64,  "use_orderhead": True,  "train_orderhead": True,  "freeze_backbone": False},
    "train_gbeta":   {"default_m": 16,  "use_orderhead": True,  "train_orderhead": True,  "freeze_backbone": True},
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

    def batch(self, global_step, micro_step):
        """Exact production sampler used by ``train_clean_aogpt`` continuation."""
        physical = sample_stream_batch(
            self.stream,
            self.batch_size,
            self.block_size,
            self.seed,
            int(global_step),
            int(micro_step),
        )
        return physical[:, self.gather]


@dataclass
class _ClassicBatchSource:
    chunks: torch.Tensor
    train_shuffle_order: np.ndarray
    heldout: torch.Tensor
    batch_size: int
    grad_accum: int

    def batch(self, global_step, micro_step):
        ids = batch_indices_for_step(
            self.train_shuffle_order,
            global_step,
            micro_step,
            self.batch_size,
            self.grad_accum,
        )
        return self.chunks.index_select(0, torch.as_tensor(ids, dtype=torch.long))


def _load_training_context(ckpt_path, batch_size, n_steps, device):
    """Load a model plus a lazy, deterministic source and one reserved batch.

    The production 10k checkpoint uses a continuous memmap, so a 20k-step run
    keeps only one training batch and one fixed held-out batch in memory.
    Classic chunk checkpoints fall back to a bounded deterministic pool; the
    result metadata makes that fallback explicit.
    """
    model, _fallback, clean_perm, dev = load_p5_ckpt(ckpt_path, 1, device=device)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = {
        "args": raw.get("args", {}),
        "global_step": raw.get("global_step", raw.get("iter_num", 0)),
        "iter_num": raw.get("iter_num", 0),
        "optimizer": raw.get("optimizer"),
        "clean_protocol": raw.get("clean_protocol", {}),
    }
    del raw
    args = meta.get("args", {})
    start_step = int(meta.get("global_step", meta.get("iter_num", 0)))
    grad_accum = int(args.get("grad_accum", 1))

    if args.get("data_source") == "continuous":
        from clean_training_protocol import load_token_stream

        stream = load_token_stream(args["train_bin"])
        gather = build_phys_to_model_token_gather(clean_perm, BLOCK_LEN)
        val_stream = load_token_stream(args["val_bin"])
        eval_windows = sample_stream_batch(
            val_stream,
            int(args.get("stream_eval_windows", batch_size)),
            N * BLOCK_LEN,
            int(args.get("permute_seed", 0)),
            -1,
            0,
        )[:, gather]
        if eval_windows.shape[0] < batch_size:
            raise ValueError("continuous checkpoint lacks enough fixed held-out eval windows")
        heldout = eval_windows[:batch_size].clone()
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
        protocol = meta["clean_protocol"]
        idx_phys = load_train_chunks(n_chunks=None)
        idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
        eval_ids = torch.as_tensor(protocol["eval_indices"], dtype=torch.long)
        if eval_ids.numel() < batch_size:
            raise ValueError("checkpoint lacks enough fixed held-out eval chunks")
        source = _ClassicBatchSource(
            idx_model,
            np.asarray(protocol["train_shuffle_order"], dtype=np.int64),
            idx_model.index_select(0, eval_ids[:batch_size]),
            batch_size,
            grad_accum,
        )
        heldout = source.heldout
        meta["data_source_mode"] = "classic_exact_cursor"
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


def _build_optimizer(model, order_head, train_orderhead, meta, lr_backbone, lr_orderhead,
                     dev, freeze_backbone=False):
    args = meta.get("args", {})
    base_lr = float(args.get("lr", 1e-4) if lr_backbone is None else lr_backbone)
    weight_decay = float(args.get("weight_decay", 0.1))
    betas = (float(args.get("beta1", 0.9)), float(args.get("beta2", 0.99)))

    if freeze_backbone:
        # Backbone frozen — only OrderHead params in the optimizer.
        for p in model.parameters():
            p.requires_grad_(False)
        oh_params = list(order_head.gbeta.parameters()) if order_head is not None else []
        optimizer = torch.optim.AdamW(oh_params, lr=float(lr_orderhead),
                                      weight_decay=0.0, betas=betas)
        for group in optimizer.param_groups:
            group["is_orderhead"] = True
        resume_status = "backbone_frozen"
        return optimizer, float(lr_orderhead), resume_status
    elif hasattr(model, "configure_optimizers"):
        optimizer = model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=base_lr,
            betas=betas,
            device_type=dev.type,
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
            except (KeyError, ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    "source checkpoint contains optimizer state but exact restore failed"
                ) from exc
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
            except (KeyError, ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    "source checkpoint contains optimizer state but exact restore failed"
                ) from exc

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
        p.grad.detach().double().square().sum().item()
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


def _sample_pl_order(scores, tau):
    """Sample a PL permutation without moving scores or selected indices to CPU."""
    available = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
    order = []
    logp = scores.new_zeros(())
    entropy = scores.new_zeros(())
    for _ in range(scores.shape[0]):
        logits = (scores / tau).masked_fill(~available, -torch.inf)
        distribution = torch.distributions.Categorical(logits=logits)
        selected = distribution.sample()
        logp = logp + distribution.log_prob(selected)
        entropy = entropy + distribution.entropy()
        order.append(selected)
        available[selected] = False
    return torch.stack(order), logp, entropy


def _atomic_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_torch_save(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _finite_scalar(value):
    return bool(torch.isfinite(value.detach()).all().item())


def _finite_gradients(parameters):
    return all(
        p.grad is None or bool(torch.isfinite(p.grad.detach()).all().item())
        for p in parameters
    )


def _resume_config(ckpt_path, arm, batch_size, m, grad_accum, tau, beta,
                   ema_alpha, adv_clip, lr_backbone, lr_orderhead,
                   policy_seed, eval_steps):
    return {
        "ckpt_path": str(ckpt_path),
        "arm": arm,
        "batch_size": int(batch_size),
        "m": m,
        "grad_accum": int(grad_accum),
        "tau": float(tau),
        "beta": float(beta),
        "ema_alpha": float(ema_alpha),
        "adv_clip": float(adv_clip),
        "lr_backbone": float(lr_backbone),
        "lr_orderhead": float(lr_orderhead),
        "policy_seed": int(policy_seed),
        "eval_steps": sorted(int(step) for step in eval_steps),
    }


def _checkpoint_payload(model, order_head, optimizer, ema, *, config,
                        next_local_step, start_global_step, logs, evals,
                        token_loss_contract, pg_only_backbone_grad, failure):
    return {
        "version": 1,
        "backbone": model.state_dict(),
        "orderhead": order_head.gbeta.state_dict() if order_head is not None else None,
        "optimizer": optimizer.state_dict(),
        "group_ema": None if ema is None or ema.b is None else ema.b.detach().cpu(),
        "rng": _rng_state(),
        "cursor": {
            "next_local_step": int(next_local_step),
            "next_global_step": int(start_global_step + next_local_step),
            "next_micro_step": 0,
        },
        "config": config,
        "log": logs,
        "evals": evals,
        "token_loss_contract": token_loss_contract,
        "pg_only_backbone_grad": float(pg_only_backbone_grad),
        "failure": failure,
    }


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
    resume_from=None,
    save_checkpoints=True,
    allow_batch_size_override=False,
    gbeta_ckpt=None,
):
    """Train one Phase-1 arm and return a JSON-serializable result.

    ``n_steps`` is the target number of completed optimizer steps relative to
    the source checkpoint (not an additional count after ``resume_from``).
    ``eval_steps`` are 1-based local continuation steps.  Task 5's runner is
    responsible for converting its global 10k/15k/... schedule to local steps.
    ``gbeta_ckpt`` overrides the default P7 gβ checkpoint (used by
    ``frozen_gbeta`` to deploy a Phase-A-trained OrderHead).
    """
    resolved_m = _validate_request(arm, n_steps, batch_size, m, tau)
    cfg = ARMS[arm]
    freeze_backbone = bool(cfg.get("freeze_backbone", False))
    model, source, clean_perm, dev, meta = _load_training_context(
        ckpt_path, batch_size, n_steps, device
    )
    args = meta.get("args", {})
    checkpoint_batch_size = int(args.get("batch_size", batch_size))
    batch_override = checkpoint_batch_size != batch_size
    if batch_override and not allow_batch_size_override:
        raise ValueError(
            f"batch_size={batch_size} conflicts with checkpoint batch_size="
            f"{checkpoint_batch_size}; pass allow_batch_size_override=True explicitly"
        )
    grad_accum = int(args.get("grad_accum", 1))
    if grad_accum <= 0:
        raise ValueError("checkpoint grad_accum must be positive")

    if freeze_backbone:
        model.eval()
    else:
        model.train()

    order_head = None
    _gbeta_src = gbeta_ckpt if gbeta_ckpt is not None else GBETA_CKPT
    if cfg["use_orderhead"]:
        order_head = OrderHeadModule(_gbeta_src, device=str(dev)).to(dev)
        for parameter in order_head.gbeta.parameters():
            parameter.requires_grad_(cfg["train_orderhead"])
        wrap = AOGPTWithOrderHead(model, order_head, clean_perm, device=str(dev))
    else:
        wrap = SimpleNamespace(backbone=model, order_head=None, clean_perm=clean_perm)
    wrap.heldout_chunks = source.heldout.detach().clone()

    optimizer, base_lr, optimizer_resume = _build_optimizer(
        model, order_head, cfg["train_orderhead"], meta,
        lr_backbone, lr_orderhead, dev, freeze_backbone=freeze_backbone,
    )
    backbone_params = list(model.parameters())
    orderhead_params = list(order_head.gbeta.parameters()) if order_head is not None else []
    # These snapshots are deliberately taken from the original source checkpoint,
    # before an optional run-resume state is loaded.
    backbone_before = _snapshot(backbone_params)
    orderhead_before = _snapshot(orderhead_params)

    groups = group_ids_for(batch_size, resolved_m) if cfg["use_orderhead"] else []
    ema = GroupEMA(len(groups), ema_alpha) if cfg["train_orderhead"] else None
    start_step = int(meta.get("global_step", meta.get("iter_num", 0)))
    policy_seed = int(args.get("seed", 0))
    eval_step_set = {int(step) for step in eval_steps}
    output = pathlib.Path(out_dir) / str(tag) / arm
    checkpoint_dir = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    config = _resume_config(
        ckpt_path, arm, batch_size, resolved_m, grad_accum, tau, beta,
        ema_alpha, adv_clip, base_lr, lr_orderhead, policy_seed, eval_step_set,
    )

    torch.manual_seed(policy_seed)
    np.random.seed(policy_seed)
    random.seed(policy_seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(policy_seed)
    logs, evals = [], []
    pg_only_backbone_grad = 0.0
    token_loss_contract = None
    next_local_step = 0
    failure = None

    print(
        f"[{arm}] starting {n_steps} steps, batch_size={batch_size}, "
        f"m={resolved_m}, grad_accum={grad_accum}, device={dev}",
        flush=True,
    )

    if resume_from is not None:
        resume = torch.load(resume_from, map_location=dev, weights_only=False)
        if resume.get("config") != config:
            raise ValueError("resume checkpoint config does not match requested run")
        if resume.get("failure") is not None:
            raise ValueError("refusing to resume a failed/non-finite checkpoint")
        model.load_state_dict(resume["backbone"])
        if order_head is None:
            if resume.get("orderhead") is not None:
                raise ValueError("resume unexpectedly contains an OrderHead")
        else:
            order_head.gbeta.load_state_dict(resume["orderhead"])
        try:
            optimizer.load_state_dict(resume["optimizer"])
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError("failed to restore run optimizer exactly") from exc
        next_local_step = int(resume["cursor"]["next_local_step"])
        if int(resume["cursor"].get("next_micro_step", 0)) != 0:
            raise ValueError("only optimizer-step-boundary resumes are supported")
        if next_local_step > n_steps:
            raise ValueError("resume cursor is beyond requested target n_steps")
        logs = list(resume.get("log", []))
        evals = list(resume.get("evals", []))
        token_loss_contract = resume.get("token_loss_contract")
        pg_only_backbone_grad = float(resume.get("pg_only_backbone_grad", 0.0))
        if ema is not None and resume.get("group_ema") is not None:
            ema.b = resume["group_ema"].to(dev)
        _restore_rng_state(resume["rng"])
        optimizer_resume = f"{optimizer_resume}+run_resume"

    def save_run_checkpoint(label, cursor, checkpoint_failure=None):
        if not save_checkpoints:
            return
        payload = _checkpoint_payload(
            model, order_head, optimizer, ema, config=config,
            next_local_step=cursor, start_global_step=start_step,
            logs=logs, evals=evals, token_loss_contract=token_loss_contract,
            pg_only_backbone_grad=pg_only_backbone_grad,
            failure=checkpoint_failure,
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save(checkpoint_dir / f"{label}.pt", payload)
        _atomic_torch_save(checkpoint_dir / "latest.pt", payload)

    for local_step in range(next_local_step, n_steps):
        global_step = start_step + local_step
        optimizer.zero_grad(set_to_none=True)
        lm_values, pg_values, entropy_values = [], [], []

        for micro_step in range(grad_accum):
            idx = source.batch(global_step, micro_step).to(dev)
            logps, entropies = [], []

            if arm == "l2r":
                physical = torch.arange(N, dtype=torch.long).repeat(batch_size, 1)
                token_order = physical_blocks_to_model_token_order(
                    physical, clean_perm, BLOCK_LEN
                ).to(dev)
            else:
                probe_step = global_step * grad_accum + micro_step
                probe = random_probe_token_orders(batch_size, 0, probe_step, dev)
                selected_A = wrap.extract_B(idx, probe).detach()
                if not freeze_backbone:
                    model.train()
                model_orders = []
                for group in groups:
                    group_index = torch.as_tensor(group, dtype=torch.long, device=dev)
                    group_A = selected_A.index_select(0, group_index)
                    scores = order_head.scores(group_A, per_sample=False)[0]
                    if arm == "frozen_gbeta":
                        order = torch.argsort(scores.detach(), descending=True, stable=True)
                    else:
                        order, logp, entropy = _sample_pl_order(scores, tau=tau)
                        logps.append(logp)
                        entropies.append(entropy)
                    model_orders.append(order.unsqueeze(0).expand(len(group), -1))
                order_model = torch.cat(model_orders, dim=0)
                inv_perm = clean_perm.inv_perm_model_to_phys.to(dev)
                order_phys = inv_perm[order_model]
                token_order = physical_blocks_to_model_token_order(
                    order_phys, clean_perm, BLOCK_LEN
                ).to(dev)

            lm_loss, token_losses, contract = _forward_with_token_losses(
                model, idx, token_order
            )
            token_loss_contract = contract
            if not _finite_scalar(lm_loss):
                failure = {
                    "component": "lm_loss", "global_step": global_step,
                    "local_step": local_step, "micro_step": micro_step,
                }
                break

            if cfg["train_orderhead"]:
                logp = torch.stack(logps)
                entropy = torch.stack(entropies)
                ell_i = per_sample_loss(token_losses.detach())
                ell_g = group_rewards(ell_i, groups)
                baseline = ema.update(ell_g)
                advantage = (baseline - ell_g).detach().clamp(-adv_clip, adv_clip)
                pg = -(advantage * logp).mean() - float(beta) * entropy.mean()
                if not _finite_scalar(pg):
                    failure = {
                        "component": "pg", "global_step": global_step,
                        "local_step": local_step, "micro_step": micro_step,
                    }
                    break
                if not freeze_backbone:
                    pg_grads = torch.autograd.grad(
                        pg, backbone_params, retain_graph=True, allow_unused=True
                    )
                    pg_only_backbone_grad = max(
                        pg_only_backbone_grad,
                        float(sum(
                            grad.detach().abs().sum().item()
                            for grad in pg_grads if grad is not None
                        )),
                    )
                entropy_value = float(entropy.detach().mean().item())
            else:
                pg = lm_loss.new_zeros(())
                entropy_value = 0.0

            if freeze_backbone:
                total_loss = pg  # backbone frozen — LM loss not backprop'd
            else:
                total_loss = lm_loss + pg
            if not _finite_scalar(total_loss):
                failure = {
                    "component": "total_loss", "global_step": global_step,
                    "local_step": local_step, "micro_step": micro_step,
                }
                break
            (total_loss / grad_accum).backward()
            lm_values.append(float(lm_loss.detach().item()))
            pg_values.append(float(pg.detach().item()))
            entropy_values.append(entropy_value)

        if failure is not None:
            optimizer.zero_grad(set_to_none=True)
            break
        if not _finite_gradients(backbone_params + orderhead_params):
            failure = {
                "component": "gradients", "global_step": global_step,
                "local_step": local_step, "micro_step": grad_accum - 1,
            }
            optimizer.zero_grad(set_to_none=True)
            break

        backbone_grad_norm = _grad_norm(backbone_params)
        orderhead_grad_norm = _grad_norm(orderhead_params)
        grad_clip = float(args.get("grad_clip", 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(backbone_params, grad_clip)
        lr = _scheduled_lr(global_step, args, base_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = (
                float(lr_orderhead) if param_group.get("is_orderhead") else lr
            )
        optimizer.step()
        next_local_step = local_step + 1

        if next_local_step <= 10 or next_local_step % 500 == 0:
            print(
                f"[{arm}] step {next_local_step}/{n_steps} "
                f"(global {global_step + 1}) "
                f"lm={float(np.mean(lm_values)):.4f} "
                f"pg={float(np.mean(pg_values)):.4f} "
                f"entropy={float(np.mean(entropy_values)):.3f}",
                flush=True,
            )

        entry = {
            "step": next_local_step,
            "global_step": global_step + 1,
            "lm_loss": float(np.mean(lm_values)),
            "pg": float(np.mean(pg_values)),
            "entropy": float(np.mean(entropy_values)),
            "orderhead_grad_norm": orderhead_grad_norm,
            "backbone_grad_norm": backbone_grad_norm,
            "lr_backbone": float(lr),
        }
        logs.append(entry)

        if evaluator is not None and next_local_step in eval_step_set:
            model.eval()
            try:
                metrics = evaluator(next_local_step, model, wrap)
                if not isinstance(metrics, dict):
                    raise TypeError("evaluator must return a dict")
            except Exception as exc:
                model.train()
                evaluator_failure = {
                    "component": "evaluator_exception",
                    "global_step": global_step + 1,
                    "local_step": next_local_step,
                    "micro_step": 0,
                    "exception_type": type(exc).__name__,
                }
                _atomic_json(output / "failure.json", evaluator_failure)
                save_run_checkpoint(
                    f"failure_step_{global_step + 1}", next_local_step,
                    evaluator_failure,
                )
                raise
            finally:
                model.train()
            evals.append({"step": next_local_step, **metrics})
            save_run_checkpoint(f"step_{global_step + 1}", next_local_step)

    if failure is not None:
        _atomic_json(output / "failure.json", failure)
        save_run_checkpoint(
            f"failure_step_{failure['global_step']}", next_local_step, failure
        )
    else:
        save_run_checkpoint(
            f"step_{start_step + next_local_step}", next_local_step
        )

    orderhead_delta = _parameter_delta(orderhead_params, orderhead_before)
    backbone_delta = _parameter_delta(backbone_params, backbone_before)
    result = {
        "ckpt_path": str(ckpt_path),
        "arm": arm,
        "m": resolved_m,
        "n_steps": int(n_steps),
        "completed_steps": int(next_local_step),
        "batch_size": int(batch_size),
        "checkpoint_batch_size": checkpoint_batch_size,
        "batch_size_override": bool(batch_override),
        "grad_accum": grad_accum,
        "start_global_step": start_step,
        "policy_seed": policy_seed,
        "tau": float(tau),
        "beta": float(beta),
        "ema_alpha": float(ema_alpha),
        "adv_clip": float(adv_clip),
        "lr_backbone_base": float(base_lr),
        "lr_orderhead": float(lr_orderhead),
        "nan": failure is not None,
        "failure": failure,
        "orderhead_param_delta": orderhead_delta,
        "backbone_param_delta": backbone_delta,
        "pg_only_backbone_grad": float(pg_only_backbone_grad),
        "optimizer_resume": optimizer_resume,
        "data_source_mode": meta.get("data_source_mode", "injected"),
        "token_loss_contract": token_loss_contract,
        "log": logs,
        "evals": evals,
    }
    _atomic_json(output / "train.json", result)

    # After a successful train_gbeta run, export the trained gβ for Phase B.
    # Save in FrozenBetaHook-compatible format: {"model": state_dict, "config": {...}}.
    if cfg["train_orderhead"] and failure is None and order_head is not None:
        import torch as _torch
        _src = _torch.load(_gbeta_src, map_location="cpu", weights_only=False)
        gbeta_out = output / "gbeta_trained.pt"
        _atomic_torch_save(gbeta_out, {
            "model": {k: v.cpu() for k, v in order_head.gbeta.state_dict().items()},
            "config": _src.get("config", {}),
        })
        del _src
        result["gbeta_ckpt"] = str(gbeta_out)

    return result


__all__ = ["ARMS", "train_arm"]
