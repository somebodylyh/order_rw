#!/usr/bin/env python3
"""FID for the AMOR VQ-image three arms (random / frozen / pgonly) under raster order.

Each arm's deploy ckpt (20k, continued from the 30k random VQ backbone) is sampled
autoregressively under a shared raster block order, decoded via the LDM VQ-f4 VAE,
and scored by FID against real Imagenet64-VQ images. Relative comparison only
(model is weak; absolute FID not meaningful per project memory).
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
sys.path.insert(0, str(_REPO / "nanogpt-learned-order"))
sys.path.insert(0, str(_REPO / "scripts"))

from AOGPT import AOGPTConfig, AOGPT
from sample_quality_fid_imagenet64_vq import (
    load_vae, decode_all, compute_fid, iter_real_tokens, save_grid, save_json,
)

N_BLOCKS = 64
DEFAULT_VAE = Path("/home/admin/.cache/huggingface/hub/models--xvjiarui--ldm-vq-f4/"
                   "snapshots/bd90e23e11bfa26c96e09e86a4b95595c8ca5379")
DATA_DIR = _REPO / "nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2"
OUT_BASE = _REPO / "out/rerun_vq/fid_three_arms_raster"

NUM_SAMPLES = 10000
FID_REAL_SAMPLES = 10000
BATCH_SIZE = 64
DECODE_BATCH_SIZE = 64
TEMPERATURE = 1.0

ARMS = [
    {"name": "random", "ckpt": _REPO / "out/rerun_vq/deploy_random/ckpt.pt"},
    {"name": "frozen", "ckpt": _REPO / "out/rerun_vq/deploy_frozen/ckpt.pt"},
    {"name": "pgonly", "ckpt": _REPO / "out/rerun_vq/deploy_pgonly/ckpt.pt"},
]


def load_arm(ckpt_path, device):
    import inspect
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    valid = set(inspect.signature(AOGPTConfig).parameters)
    margs = {k: v for k, v in ck["model_args"].items() if k in valid}
    model = AOGPT(AOGPTConfig(**margs)).to(device)
    sd = ck["model"]
    if all(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def generate_raster(model, device, num_samples):
    bsz = model.config.block_size
    out = torch.empty(num_samples, bsz, dtype=torch.long, device="cpu")
    produced, batch_id = 0, 0
    while produced < num_samples:
        bs = min(BATCH_SIZE, num_samples - produced)
        block_orders = torch.arange(N_BLOCKS, device=device).view(1, -1).expand(bs, -1).contiguous()
        token_orders = model._expand_block_orders_to_token_orders(block_orders)
        x = torch.zeros(bs, bsz, dtype=torch.long, device=device)
        for rank in range(bsz):
            logits = model(x, mode=None, orders=token_orders)[0][:, rank, :] / TEMPERATURE
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
            pos = token_orders[:, rank]
            x[torch.arange(bs, device=device), pos] = nxt
        out[produced:produced + bs] = x.cpu()
        produced += bs
        batch_id += 1
        if batch_id % 20 == 0:
            print(f"    generated {produced}/{num_samples}", flush=True)
    return out


def main():
    device = "cuda"
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print("Loading VAE...", flush=True)
    vae = load_vae(str(DEFAULT_VAE), device)
    real_tokens = iter_real_tokens(DATA_DIR, FID_REAL_SAMPLES)
    inverse_patch_order = torch.from_numpy(
        np.load(DATA_DIR / "inverse_patch_order_indices.npy")).long()
    print(f"Decoding {FID_REAL_SAMPLES} real images...", flush=True)
    args = type("Args", (), {"decode_batch_size": DECODE_BATCH_SIZE, "device": device})
    real_images = decode_all(vae, real_tokens, inverse_patch_order, args)

    rows = []
    for arm in ARMS:
        print(f"\n{'='*50}\nARM: {arm['name']}\n{'='*50}", flush=True)
        model = load_arm(arm["ckpt"], device)
        t0 = time.time()
        fake_tokens = generate_raster(model, device, NUM_SAMPLES)
        gen_s = time.time() - t0
        fake_images = decode_all(vae, fake_tokens, inverse_patch_order, args)
        d = OUT_BASE / arm["name"]; d.mkdir(parents=True, exist_ok=True)
        save_grid(fake_images[:64], d / "sample_grid.png")
        fid_val = compute_fid(fake_images, real_images, device, DECODE_BATCH_SIZE)
        row = {"arm": arm["name"], "order": "raster", "fid": round(float(fid_val), 3),
               "samples": NUM_SAMPLES, "gen_seconds": round(gen_s, 1)}
        save_json(d / "metrics.json", row)
        rows.append(row)
        print(f"  >>> {arm['name']} raster FID = {fid_val:.3f}  ({gen_s:.0f}s gen)", flush=True)

    print("\n==================== SUMMARY (raster FID) ====================")
    for r in rows:
        print(f"  {r['arm']:<8} FID={r['fid']:.3f}", flush=True)
    save_json(OUT_BASE / "summary.json", rows)


if __name__ == "__main__":
    main()
