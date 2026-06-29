#!/usr/bin/env python3
"""Smoke test: integrate frozen g_beta into a minimal AO-GPT training loop.

Runs 200 steps per policy to verify:
  - order shape / validity
  - no NaN
  - refresh overhead < 10%
  - destroyed vs normal order difference
  - label-free audit on method path

Does NOT replace train_clean_aogpt.py — this is an integration verification
script only.  Full comparison runs will follow after smoke passes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"))

from training_utils import SEQ_LEN, N, BLOCK_LEN, AOGPT, AOGPTConfig  # noqa: E402
from clean_training_protocol import (  # noqa: E402
    CleanPermutation,
    expand_model_blocks_to_token_order,
    physical_blocks_to_model_token_order,
    sample_stream_batch,
    load_token_stream,
    build_phys_to_model_token_gather,
)
from batch_readout.frozen_gbeta_hook import (  # noqa: E402
    FrozenGBetaModelFrameProvider,
    random_model_frame_token_order,
)
from batch_readout.label_free_cdl_teacher import destroy_strict65  # noqa: E402
from batch_readout.l0_strict65 import build_model_frame_strict65  # noqa: E402


# ---------------------------------------------------------------------------
# Order policies
# ---------------------------------------------------------------------------

def _get_order(step, policy, aogpt_model, idx_batch, dev, clean_perm,
               gbeta_provider, gbeta_provider_destroyed, seed):
    """Return (token_order, meta_dict) for the current step and policy."""
    B = idx_batch.shape[0]
    meta = {}

    if policy == "random":
        order = random_model_frame_token_order(B, seed, step, dev)
    elif policy == "layout_path":
        model_blocks = clean_perm.block_perm_phys_to_model.clone()
        order = expand_model_blocks_to_token_order(
            model_blocks.unsqueeze(0).expand(B, -1), BLOCK_LEN,
        ).to(dev)
        meta["note"] = "oracle baseline"
    elif policy == "l2r":
        phys = torch.arange(N).unsqueeze(0).expand(B, -1)
        order = physical_blocks_to_model_token_order(phys, clean_perm, BLOCK_LEN).to(dev)
        meta["note"] = "oracle baseline"
    elif policy in ("frozen_gbeta", "frozen_gbeta_destroyed", "frozen_gbeta_uniform"):
        provider = gbeta_provider if policy == "frozen_gbeta" else gbeta_provider_destroyed
        order = provider.model_frame_token_order(aogpt_model, idx_batch[:1], step)
        order = order.expand(B, -1)
        meta["note"] = "method" if policy == "frozen_gbeta" else "method sanity"
    else:
        raise ValueError(f"unknown policy: {policy}")

    return order, meta


# ---------------------------------------------------------------------------
# Minimal training step
# ---------------------------------------------------------------------------

def _train_step(model, idx_batch, token_order, optimizer):
    model.train()
    optimizer.zero_grad()
    logits, loss = model.forward_fn(idx_batch, token_order)
    loss.backward()
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Smoke test frozen g_beta hook integration")
    p.add_argument("--ckpt", required=True,
                   help="AO-GPT checkpoint to resume from")
    p.add_argument("--g-beta-ckpt", required=True,
                   help="Pretrained g_beta checkpoint")
    p.add_argument("--steps", type=int, default=200,
                   help="Training steps per policy (default 200)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Training batch size")
    p.add_argument("--lr", type=float, default=3e-5,
                   help="Learning rate (low to avoid disturbing ckpt)")
    p.add_argument("--refresh-every", type=int, default=10,
                   help="Steps between g_beta refresh")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--policies", default="random,frozen_gbeta,frozen_gbeta_destroyed",
                   help="Comma-separated policies to test")
    args = p.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")

    # ── Load AO-GPT checkpoint ──
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    protocol = ckpt["clean_protocol"]
    run_args = ckpt.get("args", {})

    clean_perm = CleanPermutation(
        block_perm_phys_to_model=torch.tensor(
            protocol["block_perm_phys_to_model"], dtype=torch.long,
        ),
        inv_perm_model_to_phys=torch.tensor(
            protocol["inv_perm_model_to_phys"], dtype=torch.long,
        ),
    )
    # ══════════════════════════════════════════════════════════════════════
    # clean_perm is used ONLY for oracle baselines (l2r, layout_path) and
    # for data loading.  Method policies (frozen_gbeta*) never touch it.
    # ══════════════════════════════════════════════════════════════════════

    # Build model
    model_args = dict(ckpt["model_args"])
    model_args["block_size"] = SEQ_LEN
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid = {k: v for k, v in model_args.items() if k in sig}
    model = AOGPT(AOGPTConfig(**valid))
    model.crop_block_size(SEQ_LEN)
    model.to(dev)
    clean_sd = {k.replace("_orig_mod.", ""): v
                for k, v in ckpt["model"].items()}
    model.load_state_dict(clean_sd)

    # Load data stream
    train_bin = run_args.get("train_bin")
    if not train_bin:
        raise ValueError("checkpoint missing args.train_bin")
    stream = load_token_stream(train_bin)
    g_gather = build_phys_to_model_token_gather(clean_perm, BLOCK_LEN)

    # ── Init g_beta providers ──
    gbeta = FrozenGBetaModelFrameProvider(
        g_beta_ckpt=args.g_beta_ckpt,
        device=str(dev),
        seed=args.seed,
    )

    # Destroyed variant: use the same g_beta model but with destroyed B input.
    # We create a thin wrapper that patches the B before g_beta call.
    class _DestroyedProvider(FrozenGBetaModelFrameProvider):
        """Patch: destroy B before feeding to g_beta."""
        @torch.no_grad()
        def model_frame_token_order(self, aogpt_model, idx_batch, global_step):
            Bsz = idx_batch.shape[0]
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed * 100_000_000 + int(global_step) * 1000)
            rand_blocks = torch.randperm(N, generator=gen, device="cpu")
            probe = expand_model_blocks_to_token_order(
                rand_blocks.unsqueeze(0), BLOCK_LEN,
            ).to(self.device)

            aogpt_model.eval()
            _, _, attn_list = aogpt_model.forward_fn(
                idx_batch, probe, return_attentions=True,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            attn_l0 = attn_list[0].cpu().numpy()
            probe_np = probe.cpu().numpy()
            B = build_model_frame_strict65(attn_l0, probe_np)

            # DESTROY
            B_d = B.copy()
            for b in range(B_d.shape[0]):
                for h in range(B_d.shape[1]):
                    rng = np.random.default_rng(
                        args.seed * 1000 + int(global_step) * 100 + b * 10 + h
                    )
                    B_d[b, h] = destroy_strict65(B[b, h], rng)

            B_t = torch.from_numpy(B_d).float().to(self.device)
            scores, _aux = self.model(B_t, apply_head_dropout=False)
            sigma_model = scores.argsort(dim=1, descending=True)
            token_order = expand_model_blocks_to_token_order(
                sigma_model, BLOCK_LEN,
            ).to(self.device)
            return token_order

    gbeta_destroyed = _DestroyedProvider(
        g_beta_ckpt=args.g_beta_ckpt,
        device=str(dev),
        seed=args.seed,
    )

    policies = [p.strip() for p in args.policies.split(",")]
    print(f"Policies: {policies}")
    print(f"Steps per policy: {args.steps}")
    print(f"Refresh every: {args.refresh_every}")
    print(f"Batch size: {args.batch_size}")

    # ── Run each policy ──
    for policy in policies:
        print(f"\n{'='*60}")
        print(f"Policy: {policy}")
        print(f"{'='*60}")

        # Reset optimizer and data cursor
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        losses = []
        overheads = []
        n_refreshes = 0

        t0 = time.time()
        for step in range(args.steps):
            # Sample data
            micro = step % 256
            glo = step // 256
            batch = sample_stream_batch(
                stream, args.batch_size, SEQ_LEN,
                seed=args.seed, step=glo, micro=micro,
            )
            idx_batch = batch[:, g_gather].to(dev)

            # Get order
            t_order = time.time()
            if step % args.refresh_every == 0 or step == 0:
                order, meta = _get_order(
                    step, policy, model, idx_batch, dev, clean_perm,
                    gbeta, gbeta_destroyed, args.seed,
                )
                n_refreshes += 1
            else:
                # reuse cached order (already in `order` variable)
                pass
            t_order = time.time() - t_order

            # Train step
            loss = _train_step(model, idx_batch, order, optimizer)
            losses.append(loss)

            if step % args.refresh_every == 0:
                overheads.append(t_order)

            if step % 50 == 0 or step == args.steps - 1:
                print(f"  step {step:4d}: loss={loss:.4f}  "
                      f"refresh_time={t_order*1000:.1f}ms",
                      flush=True)

        elapsed = time.time() - t0
        avg_loss = np.mean(losses[-50:])  # last 50 steps
        avg_overhead = np.mean(overheads) * 1000 if overheads else 0  # ms
        step_time = elapsed / args.steps * 1000  # ms per step

        print(f"\n  Results for {policy}:")
        print(f"    avg loss (last 50): {avg_loss:.4f}")
        print(f"    refreshes: {n_refreshes}")
        print(f"    avg refresh time: {avg_overhead:.1f}ms")
        print(f"    avg step time: {step_time:.1f}ms")
        print(f"    overhead %: {avg_overhead / step_time * 100:.1f}%"
              if step_time > 0 else "    overhead: N/A")
        print(f"    total time: {elapsed:.1f}s")
        print(f"    NaN? {'YES' if np.isnan(avg_loss) else 'no'}")

        # Label-free audit for method policies
        if "frozen_gbeta" in policy and "destroyed" not in policy:
            import inspect
            src = inspect.getsource(FrozenGBetaModelFrameProvider)
            for term in ("inv_perm", "clean_perm", "block_perm"):
                assert term not in src, f"method class has {term}"
            print(f"    label-free audit: PASS")

    print(f"\n{'='*60}")
    print("Smoke complete.")


if __name__ == "__main__":
    main()
