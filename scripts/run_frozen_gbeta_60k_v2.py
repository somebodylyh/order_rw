#!/usr/bin/env python3
"""Overnight 60k-step frozen g_beta hook comparison (v2 — fixed eval set).

Aligns with baseline: lr=1e-3→1e-4 cosine over 50k steps, batch_size=16.
Fixed eval set for consistent val_l2r measurement.
All policies start from step 20k checkpoint, train 60k additional steps.
"""

import math, sys, time
from pathlib import Path
import numpy as np
import torch
import wandb
torch.set_float32_matmul_precision("high")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"))
from training_utils import SEQ_LEN, N, BLOCK_LEN, AOGPT, AOGPTConfig
from clean_training_protocol import (CleanPermutation, expand_model_blocks_to_token_order,
    physical_blocks_to_model_token_order, sample_stream_batch, load_token_stream, build_phys_to_model_token_gather)
from batch_readout.frozen_gbeta_hook import (FrozenGBetaModelFrameProvider, random_model_frame_token_order)
from batch_readout.label_free_cdl_teacher import destroy_strict65, order_to_rank
from batch_readout.l0_strict65 import build_model_frame_strict65
from none_separated_block_graph import build_none_separated_B, rollout_from_none
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_model_vec
from scipy.stats import kendalltau

# ── Fixed config ──
STEPS = 60000; BATCH_SIZE = 64
LR = 1e-3; MIN_LR = 1e-4; LR_DECAY_STEPS = 50000
REFRESH_EVERY = 10; EVAL_EVERY = 2000; BATCH_MEAN_PROBES = 4; SEED = 2
CKPT = "block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2/ckpt_step20000.pt"
G_BETA = "/tmp/l0_dynamic_gbeta_m2000_train/g_beta_best.pt"
DEVICE = "cuda:0"

def get_lr(step):
    if step >= LR_DECAY_STEPS: return MIN_LR
    r = step / max(LR_DECAY_STEPS, 1)
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * r)) * (LR - MIN_LR)

def _tau(a, b):
    ra = order_to_rank(np.asarray(a, dtype=np.int64))
    rb = order_to_rank(np.asarray(b, dtype=np.int64))
    t, _ = kendalltau(ra, rb)
    return float(t) if not np.isnan(t) else float("nan")

@torch.no_grad()
def _cdl_order(model, idx_batch, dev, head_h=2):
    gen = torch.Generator(device="cpu"); gen.manual_seed(42)
    blocks = torch.randperm(N, generator=gen, device="cpu")
    probe = expand_model_blocks_to_token_order(blocks.unsqueeze(0), BLOCK_LEN).to(dev)
    model.eval()
    _, _, attn = model.forward_fn(idx_batch, probe, return_attentions=True)
    if dev.type == "cuda": torch.cuda.synchronize(dev)
    A = _attn_to_A_block_loss_aligned_with_none_model_vec(attn[0][0, head_h].detach().cpu().numpy(), probe[0].cpu().numpy())
    sigma = torch.from_numpy(rollout_from_none(build_none_separated_B(A))).long()
    return expand_model_blocks_to_token_order(sigma.unsqueeze(0).expand(idx_batch.shape[0], -1), BLOCK_LEN).to(dev)

class _DestroyedProvider(FrozenGBetaModelFrameProvider):
    @torch.no_grad()
    def model_frame_token_order(self, aogpt_model, idx_batch, step):
        Bsz, H, P = idx_batch.shape[0], self.model.H, self.batch_mean_probes
        B_samples = np.empty((P, Bsz, H, 65, 65), dtype=np.float32)
        for p in range(P):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed * 100_000_000 + int(step) * 1000 + p)
            blocks = torch.randperm(N, generator=gen, device="cpu")
            probe = expand_model_blocks_to_token_order(blocks.unsqueeze(0), BLOCK_LEN).to(self.device)
            aogpt_model.eval()
            _, _, attn = aogpt_model.forward_fn(idx_batch, probe, return_attentions=True)
            if self.device.type == "cuda": torch.cuda.synchronize(self.device)
            B_samples[p] = build_model_frame_strict65(attn[0].detach().cpu().numpy(), probe.cpu().numpy())
        B_mean = B_samples.mean(axis=0)
        for b in range(Bsz):
            for h in range(H):
                rng = np.random.default_rng(self.seed * 1000 + int(step) * 100 + b * 10 + h)
                B_mean[b, h] = destroy_strict65(B_mean[b, h], rng)
        scores, _ = self.model(torch.from_numpy(B_mean).float().to(self.device), apply_head_dropout=False)
        return expand_model_blocks_to_token_order(scores.argsort(dim=1, descending=True), BLOCK_LEN).to(self.device)

class _UniformProvider(FrozenGBetaModelFrameProvider):
    @torch.no_grad()
    def model_frame_token_order(self, aogpt_model, idx_batch, step):
        Bsz, H, P = idx_batch.shape[0], self.model.H, self.batch_mean_probes
        B_samples = np.empty((P, Bsz, H, 65, 65), dtype=np.float32)
        for p in range(P):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed * 100_000_000 + int(step) * 1000 + p)
            blocks = torch.randperm(N, generator=gen, device="cpu")
            probe = expand_model_blocks_to_token_order(blocks.unsqueeze(0), BLOCK_LEN).to(self.device)
            aogpt_model.eval()
            _, _, attn = aogpt_model.forward_fn(idx_batch, probe, return_attentions=True)
            if self.device.type == "cuda": torch.cuda.synchronize(self.device)
            B_samples[p] = build_model_frame_strict65(attn[0].detach().cpu().numpy(), probe.cpu().numpy())
        B_mean = B_samples.mean(axis=0)
        _, aux = self.model(torch.from_numpy(B_mean).float().to(self.device), apply_head_dropout=False)
        alpha_u = torch.full((Bsz, H), 1.0 / H, device=self.device)
        scores = (aux["scores_per_head"] * alpha_u[:, :, None]).sum(dim=1)
        return expand_model_blocks_to_token_order(scores.argsort(dim=1, descending=True), BLOCK_LEN).to(self.device)


def run_policy(name):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    wandb.init(project="frozen-gbeta-60k", name=name, config={
        "steps": STEPS, "batch_size": BATCH_SIZE, "lr": LR, "min_lr": MIN_LR,
        "lr_decay": LR_DECAY_STEPS, "refresh": REFRESH_EVERY, "probes": BATCH_MEAN_PROBES,
        "source_ckpt": CKPT, "seed": SEED,
    })

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    protocol = ckpt["clean_protocol"]; args = ckpt.get("args", {})
    clean_perm = CleanPermutation(
        block_perm_phys_to_model=torch.tensor(protocol["block_perm_phys_to_model"], dtype=torch.long),
        inv_perm_model_to_phys=torch.tensor(protocol["inv_perm_model_to_phys"], dtype=torch.long))
    dev = torch.device(DEVICE)

    ma = dict(ckpt["model_args"]); ma["block_size"] = SEQ_LEN
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    model = AOGPT(AOGPTConfig(**{k: v for k, v in ma.items() if k in sig}))
    model.crop_block_size(SEQ_LEN); model.to(dev)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()})

    stream = load_token_stream(args["train_bin"])
    g_gather = build_phys_to_model_token_gather(clean_perm, BLOCK_LEN)
    bp = clean_perm.block_perm_phys_to_model.numpy()

    # Fixed eval set
    eval_batches = []
    for ei in range(5):
        b = sample_stream_batch(stream, BATCH_SIZE, SEQ_LEN, seed=999, step=ei, micro=0)
        eval_batches.append(b[:, g_gather].to(dev))

    @torch.no_grad()
    def fixed_eval():
        model.eval(); total = 0.0
        for eb in eval_batches:
            phys = torch.arange(N).unsqueeze(0).expand(eb.shape[0], -1)
            vo = physical_blocks_to_model_token_order(phys, clean_perm, BLOCK_LEN).to(dev)
            _, loss = model.forward_fn(eb, vo)
            total += loss.item()
        return total / len(eval_batches)

    gbeta = FrozenGBetaModelFrameProvider(g_beta_ckpt=G_BETA, device=DEVICE, seed=SEED, batch_mean_probes=BATCH_MEAN_PROBES)
    gbeta_d = _DestroyedProvider(g_beta_ckpt=G_BETA, device=DEVICE, seed=SEED, batch_mean_probes=BATCH_MEAN_PROBES)
    gbeta_u = _UniformProvider(g_beta_ckpt=G_BETA, device=DEVICE, seed=SEED, batch_mean_probes=BATCH_MEAN_PROBES)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    t0 = time.time(); n_refresh = 0; t_refresh = 0.0; tau_log = []

    # Baseline eval
    base_eval = fixed_eval()
    print(f"  baseline val_l2r (fixed eval): {base_eval:.4f}")
    wandb.log({"val_ori_l2r_block": base_eval, "train_loss": float("nan")}, step=0)

    for step in range(1, STEPS + 1):
        lr = get_lr(step)
        opt.param_groups[0]["lr"] = lr
        micro = (step - 1) % 256; glo = (step - 1) // 256
        batch = sample_stream_batch(stream, BATCH_SIZE, SEQ_LEN, seed=SEED, step=glo, micro=micro)
        idx = batch[:, g_gather].to(dev)

        if (step - 1) % REFRESH_EVERY == 0 or step == 1:
            t_r0 = time.time()
            idx1 = idx[:1]
            if name == "single_head_CDL":
                order = _cdl_order(model, idx1, dev).expand(BATCH_SIZE, -1)
            elif name == "frozen_gbeta":
                order = gbeta.model_frame_token_order(model, idx1, step).expand(BATCH_SIZE, -1)
            elif name == "gbeta_destroyed":
                order = gbeta_d.model_frame_token_order(model, idx1, step).expand(BATCH_SIZE, -1)
            elif name == "gbeta_uniform":
                order = gbeta_u.model_frame_token_order(model, idx1, step).expand(BATCH_SIZE, -1)
            else:
                raise ValueError(name)
            rms = (time.time() - t_r0) * 1000; t_refresh += rms; n_refresh += 1
            if name in ("frozen_gbeta", "gbeta_uniform"):
                blocks = (order[0] // BLOCK_LEN).cpu().numpy()
                seen = set(); bo = [int(b) for b in blocks if not (b in seen or seen.add(b))]
                tau_log.append(_tau(np.array(bo, dtype=np.int64), bp))
        else:
            rms = 0.0

        opt.zero_grad(); model.train()
        _, loss = model.forward_fn(idx, order)
        loss.backward(); opt.step()

        if step % EVAL_EVERY == 0 or step == STEPS:
            val = fixed_eval()
            log = {"step": step, "train_loss": loss.item(), "val_ori_l2r_block": val, "lr": lr}
            if tau_log: log["tau_vs_bp"] = tau_log[-1]
            wandb.log(log, step=step)
            tstr = f' τ={tau_log[-1]:.3f}' if tau_log else ''
            print(f'  step {step:5d}: train={loss.item():.4f} val={val:.4f} lr={lr:.2e}{tstr}', flush=True)

    elapsed = time.time() - t0
    final_val = fixed_eval()
    print(f'  DONE: final_val={final_val:.4f} Δ={final_val-base_eval:+.4f} '
          f'{elapsed:.0f}s avg_refresh={t_refresh/max(1,n_refresh):.1f}ms')
    wandb.log({"final_val_l2r": final_val, "delta_val": final_val - base_eval,
               "avg_tau": np.mean(tau_log) if tau_log else 0.0}, step=STEPS)
    wandb.finish()
    return final_val


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--policies", default="frozen_gbeta,gbeta_destroyed,gbeta_uniform,single_head_CDL")
    args = p.parse_args()
    policies = [x.strip() for x in args.policies.split(",")]
    print(f"Overnight 60k v2: {policies}")
    results = {}
    for pol in policies:
        results[pol] = run_policy(pol)
    print(f"\n{'='*60}\nFinal results:")
    for pol, val in results.items():
        print(f"  {pol:25s}: val_l2r={val:.4f}")
