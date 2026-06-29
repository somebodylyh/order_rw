#!/usr/bin/env python3
"""FID 50k for 3 checkpoints (alt-mlp, fixed-random, fixed-raster) under shared orders.

Runs: alt ckpt × {mlp,raster,random} + fixed_random × {raster,random} + fixed_raster × {raster,random}
Output: probe_results_image/fid_3ckpt_50k/{ckpt_name}/{order}/metrics.json + summary.tsv
"""
from __future__ import annotations
import sys, time, math, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
sys.path.insert(0, str(_REPO / "nanogpt-learned-order"))
sys.path.insert(0, str(_REPO / "scripts"))

from AOGPT import AOGPTConfig, AOGPT
from train_attn_order_mlp import OrderMLP
import attn_order_mlp_policy as P
from sample_quality_fid_imagenet64_vq import (
    load_vae, decode_all, compute_fid, iter_real_tokens,
    save_grid, image_quality_stats, save_json, now,
)

N_BLOCKS = 64
DEFAULT_VAE = Path("/home/admin/.cache/huggingface/hub/models--xvjiarui--ldm-vq-f4/snapshots/bd90e23e11bfa26c96e09e86a4b95595c8ca5379")

CKPTS = [
    {
        "name": "alt_mlp",
        "run_dir": _REPO / "probe_results_image/vq64_alt_from0_mlp_patch2x2_l8h8e512",
        "step": 30000,
        "has_mlp": True,
    },
    {
        "name": "fixed_random",
        "run_dir": _REPO / "probe_results_image/vq64_fixed_random_l8h8e512",
        "step": 30000,
        "has_mlp": False,
    },
    {
        "name": "fixed_raster",
        "run_dir": _REPO / "probe_results_image/vq64_fixed_raster_l8h8e512",
        "step": 30000,
        "has_mlp": False,
    },
]

DATA_DIR = _REPO / "nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2"
OUT_BASE = _REPO / "probe_results_image/fid_3ckpt_50k"
NUM_SAMPLES = 50000
FID_REAL_SAMPLES = 50000
BATCH_SIZE = 32
DECODE_BATCH_SIZE = 64
TEMPERATURE = 1.0


def load_model_and_policy(cfg, device):
    """Load AOGPT model, A_global, and MLP policy (if available)."""
    run_dir = cfg["run_dir"]
    step = cfg["step"]

    ckpt = torch.load(run_dir / f"ckpt_step{step}.pt", map_location=device, weights_only=False)
    model_args = ckpt["model_args"]
    model = AOGPT(AOGPTConfig(**model_args)).to(device)
    sd = ckpt["model"]
    if all(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    B_np, mlp = None, None
    if cfg["has_mlp"]:
        B_np = np.load(run_dir / f"A_global_step{step}.npy").T.copy()
        np.fill_diagonal(B_np, 0.0)
        B_np = B_np.astype(np.float32)
        mlp = OrderMLP().to(device)
        mlp.load_state_dict(torch.load(run_dir / f"beta_step{step}.pt",
                                       map_location=device, weights_only=False))
        mlp.eval()

    return model, ckpt, B_np, mlp


def build_block_orders(policy, bs, batch_id, seed, device, B_np, mlp):
    if policy == "raster":
        return torch.arange(N_BLOCKS, device=device).view(1, -1).expand(bs, -1).contiguous()
    if policy == "random":
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed) * 1000003 + int(batch_id))
        return torch.stack([torch.randperm(N_BLOCKS, device=device, generator=gen) for _ in range(bs)])
    if policy == "mlp":
        return P.sample_orders_batched_mlp(
            B_np, bs, mlp, "original",
            base_seed=int(seed) * 100003 + int(batch_id),
            device=device, tau=0.5, top_k=4)
    raise ValueError(f"unknown policy {policy!r}")


@torch.no_grad()
def generate_tokens(model, policy, B_np, mlp, device, num_samples):
    bsz = model.config.block_size
    out = torch.empty(num_samples, bsz, dtype=torch.long, device="cpu")
    produced, batch_id = 0, 0
    while produced < num_samples:
        bs = min(BATCH_SIZE, num_samples - produced)
        phys_block_orders = build_block_orders(policy, bs, batch_id, 42, device, B_np, mlp)
        token_orders = model._expand_block_orders_to_token_orders(phys_block_orders)
        x = torch.zeros(bs, bsz, dtype=torch.long, device=device)
        for rank in range(bsz):
            result = model(x, mode=None, orders=token_orders)
            logits = result[0][:, rank, :] / TEMPERATURE
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
            pos = token_orders[:, rank]
            x[torch.arange(bs, device=device), pos] = nxt
        out[produced:produced + bs] = x.cpu()
        produced += bs
        batch_id += 1
        if batch_id % 50 == 0:
            print(f"    [{policy}] generated {produced}/{num_samples}", flush=True)
    return out


def main():
    device = "cuda:1"  # GPU 1 has image training; GPU 0 has text training

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # Load VAE and real tokens once
    print("Loading VAE...", flush=True)
    vae = load_vae(str(DEFAULT_VAE), device)
    print("Loading real tokens...", flush=True)
    real_tokens = iter_real_tokens(DATA_DIR, FID_REAL_SAMPLES)
    inverse_patch_order = torch.from_numpy(
        np.load(DATA_DIR / "inverse_patch_order_indices.npy")).long()
    print(f"Decoding {FID_REAL_SAMPLES} real images...", flush=True)
    real_images = decode_all(vae, real_tokens, inverse_patch_order,
                             type("Args", (), {"decode_batch_size": DECODE_BATCH_SIZE, "device": device}))

    all_rows = []
    for cfg in CKPTS:
        orders = ["mlp", "raster", "random"] if cfg["has_mlp"] else ["raster", "random"]
        print(f"\n{'='*60}")
        print(f"Ckpt: {cfg['name']} (step={cfg['step']})  orders={orders}")
        print(f"{'='*60}", flush=True)

        model, ckpt, B_np, mlp = load_model_and_policy(cfg, device)

        for policy in orders:
            print(f"--- order={policy} ---", flush=True)
            t0 = time.time()
            fake_tokens = generate_tokens(model, policy, B_np, mlp, device, NUM_SAMPLES)
            gen_s = time.time() - t0
            print(f"  Generation: {gen_s:.0f}s ({NUM_SAMPLES} images)", flush=True)

            fake_images = decode_all(vae, fake_tokens, inverse_patch_order,
                                     type("Args", (), {"decode_batch_size": DECODE_BATCH_SIZE, "device": device}))

            pol_dir = OUT_BASE / cfg["name"] / policy
            pol_dir.mkdir(parents=True, exist_ok=True)
            save_grid(fake_images[:64], pol_dir / "sample_grid.png")
            stats = image_quality_stats(fake_images, fake_tokens)

            fid_val = compute_fid(fake_images, real_images, device, DECODE_BATCH_SIZE)
            row = {
                "ckpt": cfg["name"], "step": cfg["step"], "order": policy,
                "fid": round(float(fid_val), 3), "samples": NUM_SAMPLES,
                "generation_seconds": round(gen_s, 1),
                "pixel_std": round(stats["pixel_std"], 4),
                "duplicate_image_rate": round(stats["duplicate_image_rate"], 4),
            }
            save_json(pol_dir / "metrics.json", row)
            all_rows.append(row)
            print(f"  -> FID={fid_val:.3f}  dup={stats['duplicate_image_rate']:.3f}  "
                  f"pix_std={stats['pixel_std']:.3f}  ({gen_s:.0f}s)", flush=True)

        del model, ckpt
        torch.cuda.empty_cache()

    # Summary
    with (OUT_BASE / "summary.tsv").open("w") as f:
        keys = ["ckpt", "order", "fid", "pixel_std", "duplicate_image_rate", "generation_seconds"]
        f.write("\t".join(keys) + "\n")
        for r in all_rows:
            f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")

    print("\n" + "=" * 60)
    print("FID SUMMARY (50k images)")
    print("=" * 60)
    print(f"{'ckpt':<16}{'order':<10}{'FID':>8}{'dup':>8}{'pix_std':>10}")
    for r in all_rows:
        print(f"{r['ckpt']:<16}{r['order']:<10}{r['fid']:>8.2f}{r['duplicate_image_rate']:>8.3f}{r['pixel_std']:>10.4f}")
    print(f"\nWrote {OUT_BASE}/summary.tsv")
    save_json(OUT_BASE / "summary.json", {"created_at": now(), "rows": all_rows, "num_samples": NUM_SAMPLES})


if __name__ == "__main__":
    main()
