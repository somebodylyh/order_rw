#!/usr/bin/env python3
"""Overnight 60k-step frozen g_beta hook comparison.

Aligns with existing 60k baseline config:
  lr=1e-3, cosine decay to min_lr=1e-4 over 50k steps
  val_ori_l2r_block as primary metric
  wandb logging

Policies:
  frozen_gbeta     (METHOD, batch_mean_probes=4)
  gbeta_destroyed  (SANITY)
  gbeta_uniform    (ABLATION)
  single_head_CDL  (ORACLE, if not already available)
"""

import json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import wandb
from scipy.stats import kendalltau

torch.set_float32_matmul_precision("high")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"))

from training_utils import SEQ_LEN, N, BLOCK_LEN, AOGPT, AOGPTConfig
from clean_training_protocol import (
    CleanPermutation, expand_model_blocks_to_token_order,
    physical_blocks_to_model_token_order, sample_stream_batch,
    load_token_stream, build_phys_to_model_token_gather,
)
from batch_readout.frozen_gbeta_hook import (
    FrozenGBetaModelFrameProvider, random_model_frame_token_order,
)
from batch_readout.label_free_cdl_teacher import destroy_strict65, order_to_rank
from batch_readout.l0_strict65 import build_model_frame_strict65
from none_separated_block_graph import build_none_separated_B, rollout_from_none
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_model_vec


# ── Config ──
STEPS = 60000
BATCH_SIZE = 16  # reduced from 64 to fit g_beta on GPU
LR = 1e-3
MIN_LR = 1e-4
LR_DECAY_STEPS = 50000
REFRESH_EVERY = 10
EVAL_EVERY = 2000
BATCH_MEAN_PROBES = 4
SEED = 2
DEVICE = "cuda:0"

CKPT_PATH = "block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2/ckpt_step20000.pt"
G_BETA_CKPT = "/tmp/l0_dynamic_gbeta_m2000_train/g_beta_best.pt"
WANDB_PROJECT = "frozen-gbeta-60k"


def get_lr(step):
    if step >= LR_DECAY_STEPS:
        return MIN_LR
    ratio = step / max(LR_DECAY_STEPS, 1)
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (LR - MIN_LR)


def _tau(a, b):
    ra = order_to_rank(np.asarray(a, dtype=np.int64))
    rb = order_to_rank(np.asarray(b, dtype=np.int64))
    t, _ = kendalltau(ra, rb)
    return float(t) if not np.isnan(t) else float("nan")


@torch.no_grad()
def _single_head_cdl_order(aogpt_model, idx_batch, dev, head_h=2):
    gen = torch.Generator(device="cpu"); gen.manual_seed(42)
    blocks = torch.randperm(N, generator=gen, device="cpu")
    probe = expand_model_blocks_to_token_order(blocks.unsqueeze(0), BLOCK_LEN).to(dev)
    aogpt_model.eval()
    _, _, attn_list = aogpt_model.forward_fn(idx_batch, probe, return_attentions=True)
    if dev.type == "cuda": torch.cuda.synchronize(dev)
    attn_np = attn_list[0].detach().cpu().numpy()
    A = _attn_to_A_block_loss_aligned_with_none_model_vec(attn_np[0, head_h], probe[0].cpu().numpy())
    B65 = build_none_separated_B(A)
    order_blocks = rollout_from_none(B65)
    sigma = torch.from_numpy(order_blocks).long().unsqueeze(0).expand(idx_batch.shape[0], -1)
    return expand_model_blocks_to_token_order(sigma, BLOCK_LEN).to(dev)


# ── Providers ──
class _DestroyedProvider(FrozenGBetaModelFrameProvider):
    @torch.no_grad()
    def model_frame_token_order(self, aogpt_model, idx_batch, global_step):
        Bsz = idx_batch.shape[0]; H = self.model.H; P = self.batch_mean_probes
        B_samples = np.empty((P, Bsz, H, 65, 65), dtype=np.float32)
        for p in range(P):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed * 100_000_000 + int(global_step) * 1000 + p)
            blocks = torch.randperm(N, generator=gen, device="cpu")
            probe = expand_model_blocks_to_token_order(blocks.unsqueeze(0), BLOCK_LEN).to(self.device)
            aogpt_model.eval()
            _, _, attn = aogpt_model.forward_fn(idx_batch, probe, return_attentions=True)
            if self.device.type == "cuda": torch.cuda.synchronize(self.device)
            B_samples[p] = build_model_frame_strict65(attn[0].detach().cpu().numpy(), probe.cpu().numpy())
        B_mean = B_samples.mean(axis=0)
        for b in range(Bsz):
            for h in range(H):
                rng = np.random.default_rng(self.seed * 1000 + int(global_step) * 100 + b * 10 + h)
                B_mean[b, h] = destroy_strict65(B_mean[b, h], rng)
        B_t = torch.from_numpy(B_mean).float().to(self.device)
        scores, _ = self.model(B_t, apply_head_dropout=False)
        sigma = scores.argsort(dim=1, descending=True)
        return expand_model_blocks_to_token_order(sigma, BLOCK_LEN).to(self.device)


class _UniformProvider(FrozenGBetaModelFrameProvider):
    @torch.no_grad()
    def model_frame_token_order(self, aogpt_model, idx_batch, global_step):
        Bsz = idx_batch.shape[0]; H = self.model.H; P = self.batch_mean_probes
        B_samples = np.empty((P, Bsz, H, 65, 65), dtype=np.float32)
        for p in range(P):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed * 100_000_000 + int(global_step) * 1000 + p)
            blocks = torch.randperm(N, generator=gen, device="cpu")
            probe = expand_model_blocks_to_token_order(blocks.unsqueeze(0), BLOCK_LEN).to(self.device)
            aogpt_model.eval()
            _, _, attn = aogpt_model.forward_fn(idx_batch, probe, return_attentions=True)
            if self.device.type == "cuda": torch.cuda.synchronize(self.device)
            B_samples[p] = build_model_frame_strict65(attn[0].detach().cpu().numpy(), probe.cpu().numpy())
        B_mean = B_samples.mean(axis=0)
        B_t = torch.from_numpy(B_mean).float().to(self.device)
        _, aux = self.model(B_t, apply_head_dropout=False)
        alpha_u = torch.full((Bsz, H), 1.0 / H, device=self.device)
        scores = (aux["scores_per_head"] * alpha_u[:, :, None]).sum(dim=1)
        sigma = scores.argsort(dim=1, descending=True)
        return expand_model_blocks_to_token_order(sigma, BLOCK_LEN).to(self.device)


def run_policy(policy_name, policy_fn):
    """Run one policy for 60k steps with wandb logging."""
    print(f"\n{'='*60}\nPolicy: {policy_name}\n{'='*60}")

    wandb.init(project=WANDB_PROJECT, name=policy_name, config={
        "steps": STEPS, "batch_size": BATCH_SIZE, "lr": LR, "min_lr": MIN_LR,
        "lr_decay_steps": LR_DECAY_STEPS, "refresh_every": REFRESH_EVERY,
        "batch_mean_probes": BATCH_MEAN_PROBES, "seed": SEED,
        "source_ckpt": CKPT_PATH, "g_beta_ckpt": G_BETA_CKPT,
    })

    # Load model fresh
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    protocol = ckpt["clean_protocol"]; run_args = ckpt.get("args", {})
    clean_perm = CleanPermutation(
        block_perm_phys_to_model=torch.tensor(protocol["block_perm_phys_to_model"], dtype=torch.long),
        inv_perm_model_to_phys=torch.tensor(protocol["inv_perm_model_to_phys"], dtype=torch.long))
    model_args = dict(ckpt["model_args"]); model_args["block_size"] = SEQ_LEN
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid = {k: v for k, v in model_args.items() if k in sig}
    dev = torch.device(DEVICE)

    model = AOGPT(AOGPTConfig(**valid)); model.crop_block_size(SEQ_LEN); model.to(dev)
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(clean_sd)

    stream = load_token_stream(run_args["train_bin"])
    g_gather = build_phys_to_model_token_gather(clean_perm, BLOCK_LEN)
    bp = clean_perm.block_perm_phys_to_model.numpy()

    gbeta = FrozenGBetaModelFrameProvider(g_beta_ckpt=G_BETA_CKPT, device=DEVICE, seed=SEED, batch_mean_probes=BATCH_MEAN_PROBES)
    gbeta_d = _DestroyedProvider(g_beta_ckpt=G_BETA_CKPT, device=DEVICE, seed=SEED, batch_mean_probes=BATCH_MEAN_PROBES)
    gbeta_u = _UniformProvider(g_beta_ckpt=G_BETA_CKPT, device=DEVICE, seed=SEED, batch_mean_probes=BATCH_MEAN_PROBES)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    t0 = time.time(); n_refresh = 0; t_refresh_total = 0.0

    for step in range(STEPS):
        lr = get_lr(step)
        opt.param_groups[0]["lr"] = lr
        micro = step % 256; glo = step // 256
        batch = sample_stream_batch(stream, BATCH_SIZE, SEQ_LEN, seed=SEED, step=glo, micro=micro)
        idx = batch[:, g_gather].to(dev)

        # Order
        if step % REFRESH_EVERY == 0 or step == 0:
            t_r0 = time.time()
            if policy_name == "single_head_CDL":
                order = _single_head_cdl_order(model, idx[:1], dev).expand(BATCH_SIZE, -1)
            elif policy_name == "frozen_gbeta":
                order = gbeta.model_frame_token_order(model, idx[:1], step).expand(BATCH_SIZE, -1)
            elif policy_name == "gbeta_destroyed":
                order = gbeta_d.model_frame_token_order(model, idx[:1], step).expand(BATCH_SIZE, -1)
            elif policy_name == "gbeta_uniform":
                order = gbeta_u.model_frame_token_order(model, idx[:1], step).expand(BATCH_SIZE, -1)
            else:
                raise ValueError(policy_name)
            refresh_ms = (time.time() - t_r0) * 1000
            t_refresh_total += refresh_ms; n_refresh += 1
        else:
            refresh_ms = 0.0

        # Train step
        opt.zero_grad()
        model.train()
        _, loss = model.forward_fn(idx, order)
        loss.backward(); opt.step()

        # Eval
        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            model.eval()
            phys = torch.arange(N).unsqueeze(0).expand(BATCH_SIZE, -1)
            val_order = physical_blocks_to_model_token_order(phys, clean_perm, BLOCK_LEN).to(dev)
            _, val_loss = model.forward_fn(idx, val_order)
            model.train()

            log_dict = {"step": step, "train_loss": loss.item(),
                        "val_ori_l2r_block": val_loss.item(),
                        "lr": lr, "refresh_ms": refresh_ms}
            # Tau for method policies
            if policy_name in ("frozen_gbeta", "gbeta_uniform"):
                blocks = (order[0] // BLOCK_LEN).cpu().numpy()
                seen = set(); block_order = []
                for b in blocks:
                    if b not in seen: seen.add(b); block_order.append(int(b))
                log_dict["tau_vs_bp"] = _tau(np.array(block_order, dtype=np.int64), bp)

            wandb.log(log_dict, step=step)
            tau_str = f' τ={log_dict.get("tau_vs_bp", 0):.3f}' if "tau_vs_bp" in log_dict else ''
            print(f'  step {step:5d}: train={loss.item():.4f} val_l2r={val_loss.item():.4f}'
                  f' lr={lr:.2e} refresh={refresh_ms:.0f}ms{tau_str}', flush=True)

    elapsed = time.time() - t0
    avg_refresh = t_refresh_total / max(1, n_refresh)
    print(f'DONE: {elapsed:.0f}s  n_refresh={n_refresh}  avg_refresh={avg_refresh:.1f}ms')
    wandb.finish()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--policies", default="frozen_gbeta,gbeta_destroyed,gbeta_uniform,single_head_CDL")
    p.add_argument("--device", default=DEVICE)
    args = p.parse_args()

    policies = [x.strip() for x in args.policies.split(",")]
    print(f"Overnight 60k run: {policies}")
    for policy in policies:
        run_policy(policy, None)
    print("\nAll policies complete.")
