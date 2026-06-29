#!/usr/bin/env python3
"""Label-free frozen g_beta hook: policy comparison training run.

Policies:
  random          — label-free baseline
  layout_path     — oracle baseline (uses clean_perm, explicitly labeled)
  l2r             — oracle baseline (uses clean_perm)
  frozen_gbeta    — METHOD (label-free model-frame)
  gbeta_destroyed — method sanity
  gbeta_uniform   — method ablation
"""

from __future__ import annotations

import argparse, json, time, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
from batch_readout.label_free_cdl_teacher import destroy_strict65
from batch_readout.l0_strict65 import build_model_frame_strict65


def _token_order(policy, step, aogpt_model, idx_batch, dev, clean_perm,
                 gbeta, gbeta_destroyed, provider_seed):
    """Return (token_order, meta).  Label-free for method policies."""
    B = idx_batch.shape[0]
    meta = {"policy": policy}

    if policy == "random":
        order = random_model_frame_token_order(B, provider_seed, step, dev)
        meta["label_free"] = True
    elif policy == "layout_path":
        mb = clean_perm.block_perm_phys_to_model.unsqueeze(0).expand(B, -1)
        order = expand_model_blocks_to_token_order(mb, BLOCK_LEN).to(dev)
        meta["label_free"] = False; meta["oracle"] = True
    elif policy == "l2r":
        phys = torch.arange(N).unsqueeze(0).expand(B, -1)
        order = physical_blocks_to_model_token_order(phys, clean_perm, BLOCK_LEN).to(dev)
        meta["label_free"] = False; meta["oracle"] = True
    elif policy == "frozen_gbeta":
        order = gbeta.model_frame_token_order(aogpt_model, idx_batch[:1], step)
        order = order.expand(B, -1)
        meta["label_free"] = True; meta["method"] = True
    elif policy == "gbeta_destroyed":
        order = gbeta_destroyed.model_frame_token_order(aogpt_model, idx_batch[:1], step)
        order = order.expand(B, -1)
        meta["label_free"] = True; meta["sanity"] = True
    elif policy == "gbeta_uniform":
        # g_beta with uniform alpha (no gate)
        order, _ = _uniform_gbeta_order(
            aogpt_model, idx_batch[:1], dev, gbeta, step, provider_seed,
        )
        order = order.expand(B, -1)
        meta["label_free"] = True; meta["ablation"] = True
    else:
        raise ValueError(f"unknown policy: {policy}")
    return order, meta


@torch.no_grad()
def _uniform_gbeta_order(aogpt_model, idx_batch, dev, gbeta_provider, step, seed):
    """g_beta with uniform alpha = 1/H (no gate)."""
    B = idx_batch.shape[0]
    H = gbeta_provider.model.H
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed * 100_000_000 + int(step) * 1000)
    rand_blocks = torch.randperm(N, generator=gen, device="cpu")
    probe = expand_model_blocks_to_token_order(rand_blocks.unsqueeze(0), BLOCK_LEN).to(dev)

    aogpt_model.eval()
    _, _, attn_list = aogpt_model.forward_fn(idx_batch, probe, return_attentions=True)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    attn_l0 = attn_list[0].cpu().numpy()
    B_np = build_model_frame_strict65(attn_l0, probe.cpu().numpy())
    B_t = torch.from_numpy(B_np).float().to(dev)

    _, aux = gbeta_provider.model(B_t, apply_head_dropout=False)
    scores_h = aux["scores_per_head"]  # (B, H, 64)
    alpha_u = torch.full((B, H), 1.0 / H, device=dev)
    scores = (scores_h * alpha_u[:, :, None]).sum(dim=1)

    sigma = scores.argsort(dim=1, descending=True)
    return expand_model_blocks_to_token_order(sigma, BLOCK_LEN).to(dev), {}


class _DestroyedProvider(FrozenGBetaModelFrameProvider):
    @torch.no_grad()
    def model_frame_token_order(self, aogpt_model, idx_batch, global_step):
        Bsz = idx_batch.shape[0]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed * 100_000_000 + int(global_step) * 1000)
        rand_blocks = torch.randperm(N, generator=gen, device="cpu")
        probe = expand_model_blocks_to_token_order(rand_blocks.unsqueeze(0), BLOCK_LEN).to(self.device)
        aogpt_model.eval()
        _, _, attn_list = aogpt_model.forward_fn(idx_batch, probe, return_attentions=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        attn_l0 = attn_list[0].cpu().numpy()
        B = build_model_frame_strict65(attn_l0, probe.cpu().numpy())
        B_d = B.copy()
        for b in range(B_d.shape[0]):
            for h in range(B_d.shape[1]):
                rng = np.random.default_rng(self.seed * 1000 + int(global_step) * 100 + b * 10 + h)
                B_d[b, h] = destroy_strict65(B[b, h], rng)
        B_t = torch.from_numpy(B_d).float().to(self.device)
        scores, _ = self.model(B_t, apply_head_dropout=False)
        sigma = scores.argsort(dim=1, descending=True)
        return expand_model_blocks_to_token_order(sigma, BLOCK_LEN).to(self.device)


@torch.no_grad()
def _eval_loss(model, idx_batch, token_order):
    model.eval()
    _, loss = model.forward_fn(idx_batch, token_order)
    return loss.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--g-beta-ckpt", required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--refresh-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--policies", default="random,layout_path,frozen_gbeta,gbeta_destroyed,gbeta_uniform")
    p.add_argument("--out-dir", default="/tmp/frozen_gbeta_comparison")
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
        inv_perm_model_to_phys=torch.tensor(protocol["inv_perm_model_to_phys"], dtype=torch.long),
    )
    model_args = dict(ckpt["model_args"]); model_args["block_size"] = SEQ_LEN
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid = {k: v for k, v in model_args.items() if k in sig}
    base_model = AOGPT(AOGPTConfig(**valid))
    base_model.crop_block_size(SEQ_LEN); base_model.to(dev)
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    base_model.load_state_dict(clean_sd)

    train_bin = run_args.get("train_bin")
    if not train_bin: raise ValueError("missing args.train_bin")
    stream = load_token_stream(train_bin)
    g_gather = build_phys_to_model_token_gather(clean_perm, BLOCK_LEN)

    # Init g_beta providers
    gbeta = FrozenGBetaModelFrameProvider(g_beta_ckpt=args.g_beta_ckpt, device=str(dev), seed=args.seed)
    gbeta_destroyed = _DestroyedProvider(g_beta_ckpt=args.g_beta_ckpt, device=str(dev), seed=args.seed)

    # Fixed eval batch
    eval_batch = sample_stream_batch(stream, args.batch_size, SEQ_LEN, seed=args.seed, step=0, micro=999)
    eval_idx = eval_batch[:, g_gather].to(dev)

    policies = [p.strip() for p in args.policies.split(",")]
    all_results = {}

    for policy in policies:
        print(f"\n{'='*60}\nPolicy: {policy}\n{'='*60}")

        # Fresh model copy
        model = AOGPT(AOGPTConfig(**valid)); model.crop_block_size(SEQ_LEN); model.to(dev)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()})
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        records = []; t0 = time.time()
        cached_order = None; n_refresh = 0

        for step in range(args.steps):
            micro = step % 256; glo = step // 256
            batch = sample_stream_batch(stream, args.batch_size, SEQ_LEN, seed=args.seed, step=glo, micro=micro)
            idx_batch = batch[:, g_gather].to(dev)

            if step % args.refresh_every == 0 or cached_order is None:
                t_r = time.time()
                cached_order, meta = _token_order(
                    policy, step, model, idx_batch, dev, clean_perm,
                    gbeta, gbeta_destroyed, args.seed,
                )
                n_refresh += 1; refresh_ms = (time.time() - t_r) * 1000
            else:
                refresh_ms = 0.0

            model.train(); optimizer.zero_grad()
            _, loss = model.forward_fn(idx_batch, cached_order)
            loss.backward(); optimizer.step()

            if step % args.eval_every == 0 or step == args.steps - 1:
                # Eval: use same policy's order on fixed eval batch
                eval_order, _ = _token_order(
                    policy, step, model, eval_idx, dev, clean_perm,
                    gbeta, gbeta_destroyed, args.seed,
                )
                val_loss = _eval_loss(model, eval_idx, eval_order)
                rec = {"step": step, "train_loss": loss.item(), "val_loss": val_loss,
                       "refresh_ms": refresh_ms, "n_refresh": n_refresh}
                records.append(rec)
                print(f"  step {step:5d}: train={loss.item():.4f} val={val_loss:.4f} "
                      f"refresh={refresh_ms:.1f}ms n_refresh={n_refresh}", flush=True)

        elapsed = time.time() - t0
        final_train = np.mean([r["train_loss"] for r in records[-3:]])
        final_val = np.mean([r["val_loss"] for r in records[-3:]])
        print(f"  final: train_loss≈{final_train:.4f} val_loss≈{final_val:.4f} "
              f"time={elapsed:.0f}s refreshes={n_refresh}")
        all_results[policy] = {"records": records, "elapsed_s": elapsed,
                               "n_refresh": n_refresh, "final_train": final_train,
                               "final_val": final_val, "meta": meta}

    # Summary
    print(f"\n{'='*60}\nSummary\n{'='*60}")
    summary = {}
    for policy, res in all_results.items():
        r = res["records"]
        summary[policy] = {
            "final_val_loss": res["final_val"],
            "final_train_loss": res["final_train"],
            "elapsed_s": res["elapsed_s"],
            "n_refresh": res["n_refresh"],
            "label_free": res.get("meta", {}).get("label_free", "N/A"),
        }
        tag = "METHOD" if res.get("meta", {}).get("method") else \
              "ORACLE" if res.get("meta", {}).get("oracle") else \
              "SANITY" if res.get("meta", {}).get("sanity") else \
              "ABLATION" if res.get("meta", {}).get("ablation") else "BASELINE"
        print(f"  [{tag:>9s}] {policy:25s} val_loss={res['final_val']:.4f}  "
              f"train_loss={res['final_train']:.4f}  {res['elapsed_s']:.0f}s")

    with open(out / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out / "comparison_full.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}/comparison_*.json")


if __name__ == "__main__":
    main()
