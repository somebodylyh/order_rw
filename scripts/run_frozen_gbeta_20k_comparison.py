#!/usr/bin/env python3
"""20k-step label-free frozen g_beta hook policy comparison.

Policies (6):
  random           — label-free baseline
  layout_path      — oracle (uses clean_perm)
  single_head_CDL  — L0H2 CDL oracle-ish baseline
  frozen_gbeta     — METHOD (batch_mean_probes=4)
  gbeta_destroyed  — method sanity
  gbeta_uniform    — method ablation
"""

import json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

torch.set_float32_matmul_precision("high")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"))

from training_utils import SEQ_LEN, N, BLOCK_LEN, AOGPT, AOGPTConfig
from clean_training_protocol import (
    CleanPermutation, expand_model_blocks_to_token_order,
    physical_blocks_to_model_token_order, sample_stream_batch,
    load_token_stream, build_phys_to_model_token_gather,
    model_blocks_to_physical_blocks,
)
from batch_readout.frozen_gbeta_hook import (
    FrozenGBetaModelFrameProvider, random_model_frame_token_order,
)
from batch_readout.label_free_cdl_teacher import destroy_strict65, order_to_rank
from batch_readout.l0_strict65 import build_model_frame_strict65
from none_separated_block_graph import build_none_separated_B, rollout_from_none
from per_head_order_scan import _attn_to_A_block_loss_aligned_with_none_model_vec


# ---------------------------------------------------------------------------
# Single-head CDL order provider
# ---------------------------------------------------------------------------

@torch.no_grad()
def _single_head_cdl_order(aogpt_model, idx_batch, dev, head_h=2):
    """Extract L0 attention, run CDL on head h's B, return model-frame token order."""
    Bsz = idx_batch.shape[0]
    # Probe with arbitrary random order (CDL operates on B regardless).
    gen = torch.Generator(device="cpu")
    gen.manual_seed(42)
    rand_blocks = torch.randperm(N, generator=gen, device="cpu")
    probe = expand_model_blocks_to_token_order(rand_blocks.unsqueeze(0), BLOCK_LEN).to(dev)

    aogpt_model.eval()
    _, _, attn_list = aogpt_model.forward_fn(idx_batch, probe, return_attentions=True)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    attn_np = attn_list[0].detach().cpu().numpy()  # (B, H, 257, 257)
    probe_np = probe.cpu().numpy()

    # Build strict-65 B for head_h
    A_model = _attn_to_A_block_loss_aligned_with_none_model_vec(
        attn_np[0, head_h], probe_np[0],
    )  # (64, 65)
    B65 = build_none_separated_B(A_model)  # (65, 65)
    order_1indexed = rollout_from_none(B65)  # content blocks 0..63
    # order_1indexed is already 0-indexed after rollout_from_none returns (node_order - 1)
    sigma = torch.from_numpy(order_1indexed).long().unsqueeze(0).expand(Bsz, -1)

    return expand_model_blocks_to_token_order(sigma, BLOCK_LEN).to(dev)


# ---------------------------------------------------------------------------
# Destroyed provider (batch-mean aware)
# ---------------------------------------------------------------------------

class _DestroyedProvider(FrozenGBetaModelFrameProvider):
    @torch.no_grad()
    def model_frame_token_order(self, aogpt_model, idx_batch, global_step):
        Bsz = idx_batch.shape[0]; H = self.model.H
        N_probes = self.batch_mean_probes
        B_samples = np.empty((N_probes, Bsz, H, 65, 65), dtype=np.float32)
        for p in range(N_probes):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed * 100_000_000 + int(global_step) * 1000 + p)
            blocks = torch.randperm(N, generator=gen, device="cpu")
            probe = expand_model_blocks_to_token_order(blocks.unsqueeze(0), BLOCK_LEN).to(self.device)
            aogpt_model.eval()
            _, _, attn = aogpt_model.forward_fn(idx_batch, probe, return_attentions=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            attn_np = attn[0].detach().cpu().numpy()
            B_samples[p] = build_model_frame_strict65(attn_np, probe.cpu().numpy())
        B_mean = B_samples.mean(axis=0)
        # DESTROY each head
        for b in range(Bsz):
            for h in range(H):
                rng = np.random.default_rng(self.seed * 1000 + int(global_step) * 100 + b * 10 + h)
                B_mean[b, h] = destroy_strict65(B_mean[b, h], rng)
        B_t = torch.from_numpy(B_mean).float().to(self.device)
        scores, _ = self.model(B_t, apply_head_dropout=False)
        sigma = scores.argsort(dim=1, descending=True)
        return expand_model_blocks_to_token_order(sigma, BLOCK_LEN).to(self.device)


# ---------------------------------------------------------------------------
# Uniform-alpha provider
# ---------------------------------------------------------------------------

class _UniformProvider(FrozenGBetaModelFrameProvider):
    @torch.no_grad()
    def model_frame_token_order(self, aogpt_model, idx_batch, global_step):
        Bsz = idx_batch.shape[0]; H = self.model.H
        N_probes = self.batch_mean_probes
        B_samples = np.empty((N_probes, Bsz, H, 65, 65), dtype=np.float32)
        for p in range(N_probes):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed * 100_000_000 + int(global_step) * 1000 + p)
            blocks = torch.randperm(N, generator=gen, device="cpu")
            probe = expand_model_blocks_to_token_order(blocks.unsqueeze(0), BLOCK_LEN).to(self.device)
            aogpt_model.eval()
            _, _, attn = aogpt_model.forward_fn(idx_batch, probe, return_attentions=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            B_samples[p] = build_model_frame_strict65(attn[0].detach().cpu().numpy(), probe.cpu().numpy())
        B_mean = B_samples.mean(axis=0)
        B_t = torch.from_numpy(B_mean).float().to(self.device)
        _, aux = self.model(B_t, apply_head_dropout=False)
        scores_h = aux["scores_per_head"]
        alpha_u = torch.full((Bsz, H), 1.0 / H, device=self.device)
        scores = (scores_h * alpha_u[:, :, None]).sum(dim=1)
        sigma = scores.argsort(dim=1, descending=True)
        return expand_model_blocks_to_token_order(sigma, BLOCK_LEN).to(self.device)


# ---------------------------------------------------------------------------
# Tau helper
# ---------------------------------------------------------------------------

def _tau(a, b):
    ra = order_to_rank(np.asarray(a, dtype=np.int64))
    rb = order_to_rank(np.asarray(b, dtype=np.int64))
    t, _ = kendalltau(ra, rb)
    return float(t) if not np.isnan(t) else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--g-beta-ckpt", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--refresh-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--batch-mean-probes", type=int, default=4)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--policies", default="random,layout_path,single_head_CDL,frozen_gbeta,gbeta_destroyed,gbeta_uniform")
    p.add_argument("--out-dir", default="/tmp/frozen_gbeta_20k")
    args = p.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Load AO-GPT
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    protocol = ckpt["clean_protocol"]; run_args = ckpt.get("args", {})
    clean_perm = CleanPermutation(
        block_perm_phys_to_model=torch.tensor(protocol["block_perm_phys_to_model"], dtype=torch.long),
        inv_perm_model_to_phys=torch.tensor(protocol["inv_perm_model_to_phys"], dtype=torch.long))
    bp = clean_perm.block_perm_phys_to_model.numpy()
    model_args = dict(ckpt["model_args"]); model_args["block_size"] = SEQ_LEN
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid = {k: v for k, v in model_args.items() if k in sig}

    stream = load_token_stream(run_args["train_bin"])
    g_gather = build_phys_to_model_token_gather(clean_perm, BLOCK_LEN)

    # Init providers
    gbeta = FrozenGBetaModelFrameProvider(
        g_beta_ckpt=args.g_beta_ckpt, device=str(dev),
        seed=args.seed, batch_mean_probes=args.batch_mean_probes)
    gbeta_destroyed = _DestroyedProvider(
        g_beta_ckpt=args.g_beta_ckpt, device=str(dev),
        seed=args.seed, batch_mean_probes=args.batch_mean_probes)
    gbeta_uniform = _UniformProvider(
        g_beta_ckpt=args.g_beta_ckpt, device=str(dev),
        seed=args.seed, batch_mean_probes=args.batch_mean_probes)

    policies = [x.strip() for x in args.policies.split(",")]
    all_results = {}

    for policy in policies:
        print(f"\n{'='*60}\nPolicy: {policy}\n{'='*60}")

        model = AOGPT(AOGPTConfig(**valid)); model.crop_block_size(SEQ_LEN); model.to(dev)
        clean_sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(clean_sd)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        records = []; t0 = time.time(); taus_vs_bp = []; H_alphas = []
        n_refresh = 0; t_refresh_total = 0.0

        for step in range(args.steps):
            micro = step % 256; glo = step // 256
            batch = sample_stream_batch(stream, args.batch_size, SEQ_LEN, seed=args.seed, step=glo, micro=micro)
            idx = batch[:, g_gather].to(dev)

            # Get order
            if step % args.refresh_every == 0 or step == 0:
                t_r0 = time.time()
                if policy == "random":
                    order = random_model_frame_token_order(args.batch_size, args.seed, step, dev)
                elif policy == "layout_path":
                    mb = clean_perm.block_perm_phys_to_model.unsqueeze(0).expand(args.batch_size, -1)
                    order = expand_model_blocks_to_token_order(mb, BLOCK_LEN).to(dev)
                elif policy == "single_head_CDL":
                    order = _single_head_cdl_order(model, idx[:1], dev, head_h=2).expand(args.batch_size, -1)
                elif policy == "frozen_gbeta":
                    order = gbeta.model_frame_token_order(model, idx[:1], step).expand(args.batch_size, -1)
                elif policy == "gbeta_destroyed":
                    order = gbeta_destroyed.model_frame_token_order(model, idx[:1], step).expand(args.batch_size, -1)
                elif policy == "gbeta_uniform":
                    order = gbeta_uniform.model_frame_token_order(model, idx[:1], step).expand(args.batch_size, -1)
                else:
                    raise ValueError(policy)
                refresh_ms = (time.time() - t_r0) * 1000
                t_refresh_total += refresh_ms; n_refresh += 1

                # Hook-time τ vs layout path (for method policies)
                if policy in ("frozen_gbeta", "gbeta_uniform"):
                    blocks = (order[0] // BLOCK_LEN).cpu().numpy()
                    seen = set(); block_order = []
                    for b in blocks:
                        if b not in seen: seen.add(b); block_order.append(int(b))
                    taus_vs_bp.append(_tau(np.array(block_order, dtype=np.int64), bp))
            else:
                refresh_ms = 0.0

            # Train step
            model.train(); opt.zero_grad()
            _, loss = model.forward_fn(idx, order)
            loss.backward(); opt.step()

            # Eval
            if step % args.eval_every == 0 or step == args.steps - 1:
                model.eval()
                # val_ori_l2r_block: physical L2R eval (oracle eval metric)
                phys_order = torch.arange(N).unsqueeze(0).expand(args.batch_size, -1)
                val_order = physical_blocks_to_model_token_order(phys_order, clean_perm, BLOCK_LEN).to(dev)
                _, val_loss = model.forward_fn(idx, val_order)
                model.train()

                rec = {"step": step, "train_loss": loss.item(),
                       "val_ori_l2r_block": val_loss.item(),
                       "refresh_ms": refresh_ms, "n_refresh": n_refresh}
                if policy in ("frozen_gbeta", "gbeta_uniform") and taus_vs_bp:
                    rec["tau_vs_bp"] = taus_vs_bp[-1]
                records.append(rec)
                tau_str = f' τ={taus_vs_bp[-1]:.3f}' if taus_vs_bp else ''
                print(f'  step {step:5d}: train={loss.item():.4f} val_l2r={val_loss.item():.4f}'
                      f' refresh={refresh_ms:.0f}ms{tau_str}', flush=True)

        elapsed = time.time() - t0
        final = {k: np.mean([r[k] for r in records[-3:]]) for k in records[-1] if k != "step"}
        print(f'  DONE: train≈{final["train_loss"]:.4f} val_l2r≈{final["val_ori_l2r_block"]:.4f} '
              f'{elapsed:.0f}s  refreshes={n_refresh}  avg_refresh={t_refresh_total/max(1,n_refresh):.1f}ms')
        all_results[policy] = {"records": records, "elapsed_s": elapsed,
                               "final": final, "n_refresh": n_refresh,
                               "avg_tau_vs_bp": np.mean(taus_vs_bp) if taus_vs_bp else None}

    # Summary
    print(f"\n{'='*60}\n{'Policy':25s} {'val_l2r':>8s} {'train':>8s} {'Δrandom':>8s}  {'τ_bp':>7s}  time")
    print('-'*70)
    rand_final = all_results["random"]["final"]["val_ori_l2r_block"]
    for policy in policies:
        r = all_results[policy]
        f = r["final"]
        delta = f["val_ori_l2r_block"] - rand_final
        tau_str = f'{r["avg_tau_vs_bp"]:.3f}' if r["avg_tau_vs_bp"] else '-'
        tag = "METHOD" if policy == "frozen_gbeta" else \
              "ORACLE" if policy in ("layout_path","single_head_CDL") else \
              "SANITY" if "destroyed" in policy else \
              "ABLATION" if "uniform" in policy else "BASELINE"
        print(f'[{tag:>9s}] {policy:25s} {f["val_ori_l2r_block"]:8.4f} {f["train_loss"]:8.4f} '
              f'{delta:+8.4f}  {tau_str:>7s}  {r["elapsed_s"]:.0f}s')

    with open(out / "20k_summary.json", "w") as f:
        json.dump({p: {"final": all_results[p]["final"],
                       "avg_tau_vs_bp": all_results[p]["avg_tau_vs_bp"],
                       "elapsed_s": all_results[p]["elapsed_s"]}
                   for p in policies}, f, indent=2)
    print(f"\nSaved: {out}/20k_summary.json")


if __name__ == "__main__":
    main()
