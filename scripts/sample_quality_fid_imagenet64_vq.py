#!/usr/bin/env python3
"""Generate ImageNet64 VQ samples from AO-GPT checkpoints and compute FID.

This is an evaluation script, not a training script. It keeps the Stage-1
coordinate convention:
  physical order -> inverse_block_perm -> model-frame order for generation,
  generated model-frame tokens -> physical/data frame -> 16x16 VQ row-major.

FID is computed between VQ-decoded generated samples and VQ-decoded validation
tokens, so the metric is a reconstruction-space proxy unless raw ImageNet
reference images are added later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
NANOGPT_ROOT = ROOT / "nanogpt-learned-order"
if str(NANOGPT_ROOT) not in sys.path:
    sys.path.insert(0, str(NANOGPT_ROOT))
if str(ROOT / "block_lo_arm_order_network") not in sys.path:
    sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))

from AOGPT import AOGPT, AOGPTConfig  # noqa: E402
from order_utils import block_permutation_to_token_permutation  # noqa: E402
from readout_order_diagnostic import hilbert_order  # noqa: E402
from train_imagelarge_round2 import (  # noqa: E402
    make_shuffled_Bcov_control,
    sample_coverage_batched_torch,
    sample_progressive_rw_v1_batched_torch,
)


DEFAULT_VAE = (
    "/home/admin/.cache/huggingface/hub/models--xvjiarui--ldm-vq-f4/"
    "snapshots/bd90e23e11bfa26c96e09e86a4b95595c8ca5379"
)

POLICY_BY_ARM = {
    "cont_random": "random",
    "cont_v1_graph_rw": "graph_rw",
    "cont_hilbert": "hilbert",
    "cont_Bcov_balanced": "Bcov_balanced",
    "cont_distance_only_coverage": "distance_only_coverage",
    "cont_shuffled_Bcov_balanced": "shuffled_Bcov_balanced",
    "cont_raster": "raster",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round2-root", type=Path, required=True)
    p.add_argument("--arms", type=str, default="cont_random,cont_Bcov_balanced,cont_hilbert,cont_distance_only_coverage,cont_shuffled_Bcov_balanced")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8")
    p.add_argument("--a-block-path", type=Path, default=ROOT / "probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy")
    p.add_argument("--vae-path", type=str, default=DEFAULT_VAE)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--fid-real-samples", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--decode-batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--force-policy", type=str, default=None,
                   choices=["random", "graph_rw", "hilbert", "Bcov_balanced",
                            "distance_only_coverage", "shuffled_Bcov_balanced", "raster"],
                   help="Generate every checkpoint with this common readout instead of each arm's matched readout.")
    p.add_argument("--skip-fid", action="store_true")
    p.add_argument("--save-individual", action="store_true")
    return p.parse_args()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def git_summary():
    try:
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True, timeout=10)
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, timeout=10).strip()
        lines = [line for line in status.splitlines() if line.strip()]
        return {
            "head": rev,
            "dirty_count": len(lines),
            "status_short_preview": lines[:80],
            "truncated": len(lines) > 80,
        }
    except Exception as exc:
        return {"error": repr(exc)}


def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = AOGPT(AOGPTConfig(**ckpt["model_args"]))
    state = ckpt["model"]
    for key in list(state.keys()):
        if key.startswith("_orig_mod."):
            state[key[len("_orig_mod."):]] = state.pop(key)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt


def load_perm_from_baseline(baseline_ckpt: Path, block_len: int):
    ckpt = torch.load(baseline_ckpt, map_location="cpu")
    dp = ckpt.get("data_permutation", {})
    block_perm = torch.tensor(dp["block_perm"], dtype=torch.long)
    inv_block_perm = torch.tensor(dp["inverse_block_perm"], dtype=torch.long)
    fixed_token_perm = block_permutation_to_token_permutation(block_perm, block_len)
    inv_token_perm = torch.empty_like(fixed_token_perm)
    inv_token_perm[fixed_token_perm] = torch.arange(fixed_token_perm.numel(), dtype=torch.long)
    return {
        "block_perm": block_perm,
        "inverse_block_perm": inv_block_perm,
        "fixed_token_perm": fixed_token_perm,
        "inverse_token_perm": inv_token_perm,
    }


def expand_block_orders(model, block_orders):
    return model._expand_block_orders_to_token_orders(block_orders)


def raster_order(device):
    return torch.arange(64, device=device, dtype=torch.long)


def random_block_orders(batch_size, device, generator):
    return torch.stack([torch.randperm(64, device=device, generator=generator) for _ in range(batch_size)], dim=0)


def build_physical_block_orders(policy: str, batch_size: int, step: int, seed: int, device: str, B_np: np.ndarray):
    if policy == "random":
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed * 1000003 + step))
        return random_block_orders(batch_size, device, gen)
    if policy == "raster":
        return raster_order(device).view(1, -1).expand(batch_size, -1)
    if policy == "hilbert":
        return torch.tensor(hilbert_order().astype(np.int64), device=device, dtype=torch.long).view(1, -1).expand(batch_size, -1)
    if policy == "graph_rw":
        params = dict(
            tau_start=1.0,
            tau_step=1.0,
            alpha_dep=0.5,
            top_k=4,
            epsilon_uniform=0.0,
            beta_sup=1.0,
            beta_fut=0.5,
            beta_src=0.2,
            beta_loc=0.5,
        )
        return sample_progressive_rw_v1_batched_torch(B_np, params, batch_size, seed, step, device)
    if policy == "Bcov_balanced":
        return sample_coverage_batched_torch(B_np, batch_size, step, seed, device, gamma_B=1.0, gamma_d=1.0)
    if policy == "distance_only_coverage":
        return sample_coverage_batched_torch(B_np, batch_size, step, seed, device, gamma_B=0.01, gamma_d=1.0)
    if policy == "shuffled_Bcov_balanced":
        shuf = make_shuffled_Bcov_control(B_np, seed + 424242)
        return sample_coverage_batched_torch(shuf, batch_size, step, seed, device, gamma_B=1.0, gamma_d=1.0)
    raise ValueError(f"unknown policy {policy!r}")


@torch.no_grad()
def generate_tokens(model, policy: str, B_np: np.ndarray, perms: Dict[str, torch.Tensor], args, num_samples: int):
    device = args.device
    out = torch.empty(num_samples, model.config.block_size, dtype=torch.long, device="cpu")
    inv_block_perm = perms["inverse_block_perm"].to(device)
    inv_token_perm = perms["inverse_token_perm"].to(device)
    produced = 0
    batch_id = 0
    while produced < num_samples:
        bs = min(args.batch_size, num_samples - produced)
        phys_block_orders = build_physical_block_orders(policy, bs, batch_id, args.seed, device, B_np)
        model_block_orders = inv_block_perm[phys_block_orders]
        token_orders = expand_block_orders(model, model_block_orders)
        x_model = torch.zeros(bs, model.config.block_size, dtype=torch.long, device=device)
        for rank in range(model.config.block_size):
            logits, _ = model(x_model, mode=None, orders=token_orders)
            logits = logits[:, rank, :] / float(args.temperature)
            if args.top_k and args.top_k > 0:
                vals, _ = torch.topk(logits, min(int(args.top_k), logits.size(-1)), dim=-1)
                logits = logits.masked_fill(logits < vals[:, -1:], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
            pos = token_orders[:, rank]
            x_model[torch.arange(bs, device=device), pos] = nxt
        x_phys_patch = x_model[:, inv_token_perm]
        out[produced: produced + bs] = x_phys_patch.cpu()
        produced += bs
        batch_id += 1
        print(f"  generated {produced}/{num_samples}", flush=True)
    return out


def load_vae(path: str, device: str):
    from diffusers import VQModel

    vae = VQModel.from_pretrained(path)
    vae.to(device).eval()
    return vae


@torch.no_grad()
def decode_vq_tokens(vae, tokens_patch_order: torch.Tensor, inverse_patch_order: torch.Tensor, device: str):
    tokens_patch_order = tokens_patch_order.to(device=device, dtype=torch.long)
    inverse_patch_order = inverse_patch_order.to(device=device, dtype=torch.long)
    tokens_row = tokens_patch_order[:, inverse_patch_order]
    b = tokens_row.size(0)
    indices = tokens_row.reshape(-1)
    embed_dim = int(vae.quantize.embedding.embedding_dim)
    z = vae.quantize.get_codebook_entry(indices, shape=(b, 16, 16, embed_dim))
    dec = vae.decode(z, force_not_quantize=True).sample
    img = ((dec.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return img


def save_grid(images_uint8: torch.Tensor, path: Path, nrow: int = 8):
    from torchvision.utils import make_grid

    grid = make_grid(images_uint8.float() / 255.0, nrow=nrow, padding=2)
    arr = (grid.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy())
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def save_individual(images_uint8: torch.Tensor, out_dir: Path, offset: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = images_uint8.permute(0, 2, 3, 1).cpu().numpy()
    for i, img in enumerate(arr):
        Image.fromarray(img).save(out_dir / f"{offset + i:06d}.png")


def image_quality_stats(images_uint8: torch.Tensor, tokens: torch.Tensor):
    imgs = images_uint8.float() / 255.0
    flat_tokens = tokens.reshape(-1).numpy()
    counts = np.bincount(flat_tokens, minlength=8192).astype(np.float64)
    probs = counts / max(1.0, counts.sum())
    entropy = -float(np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
    hashes = []
    arr = images_uint8.permute(0, 2, 3, 1).cpu().numpy()
    for img in arr:
        hashes.append(hashlib.sha1(img.tobytes()).hexdigest())
    return {
        "num_images": int(images_uint8.size(0)),
        "pixel_mean": float(imgs.mean().item()),
        "pixel_std": float(imgs.std().item()),
        "pixel_min": int(images_uint8.min().item()),
        "pixel_max": int(images_uint8.max().item()),
        "token_entropy_nats": entropy,
        "token_entropy_bits": entropy / math.log(2.0),
        "token_unique": int((counts > 0).sum()),
        "unique_image_hashes": int(len(set(hashes))),
        "duplicate_image_rate": float(1.0 - len(set(hashes)) / max(1, len(hashes))),
    }


def iter_real_tokens(data_dir: Path, n: int):
    val = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    tokens_per_image = 256
    total = len(val) // tokens_per_image
    n = min(int(n), int(total))
    arr = np.asarray(val[: n * tokens_per_image], dtype=np.int64).reshape(n, tokens_per_image)
    return torch.from_numpy(arr)


def compute_fid(fake_images: torch.Tensor, real_images: torch.Tensor, device: str, batch_size: int):
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    for start in range(0, real_images.size(0), batch_size):
        metric.update(real_images[start:start + batch_size].to(device), real=True)
    for start in range(0, fake_images.size(0), batch_size):
        metric.update(fake_images[start:start + batch_size].to(device), real=False)
    return float(metric.compute().detach().cpu().item())


def decode_all(vae, tokens: torch.Tensor, inverse_patch_order: torch.Tensor, args):
    parts = []
    for start in range(0, tokens.size(0), args.decode_batch_size):
        imgs = decode_vq_tokens(vae, tokens[start:start + args.decode_batch_size], inverse_patch_order, args.device)
        parts.append(imgs.cpu())
    return torch.cat(parts, dim=0)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    B_np = np.load(args.a_block_path).astype(np.float32)
    inverse_patch_order = torch.from_numpy(np.load(args.data_dir / "inverse_patch_order_indices.npy")).long()
    vae = load_vae(args.vae_path, args.device)
    real_tokens = iter_real_tokens(args.data_dir, args.fid_real_samples)
    real_images = decode_all(vae, real_tokens, inverse_patch_order, args)
    save_grid(real_images[: min(64, real_images.size(0))], args.out_dir / "real_val_recon_grid.png")

    metadata = {
        "created_at": now(),
        "round2_root": str(args.round2_root),
        "arms": arms,
        "data_dir": str(args.data_dir),
        "a_block_path": str(args.a_block_path),
        "vae_path": str(args.vae_path),
        "device": args.device,
        "num_samples": int(args.num_samples),
        "fid_real_samples": int(real_images.size(0)),
        "batch_size": int(args.batch_size),
        "decode_batch_size": int(args.decode_batch_size),
        "seed": int(args.seed),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "force_policy": args.force_policy,
        "fid_reference": "VQ-decoded validation tokens, not raw ImageNet images",
        "frame_note": "physical readout order remapped via inverse_block_perm; generated model-frame tokens unpermuted with inverse_token_perm before VQ decode",
        "git": git_summary(),
    }
    save_json(args.out_dir / "metadata.json", metadata)

    rows = []
    for arm in arms:
        arm_dir = args.round2_root / arm
        ckpt_path = arm_dir / "ckpt_final.pt"
        if not ckpt_path.exists():
            print(f"[skip] missing {ckpt_path}", flush=True)
            continue
        matched_policy = POLICY_BY_ARM.get(arm, arm.replace("cont_", ""))
        policy = args.force_policy or matched_policy
        print(f"=== {arm} policy={policy} ===", flush=True)
        model, ckpt = load_model(ckpt_path, args.device)
        baseline_ckpt = Path(ckpt.get("config", {}).get("baseline_ckpt", ""))
        perms = load_perm_from_baseline(baseline_ckpt, model.block_order_block_len)
        t0 = time.time()
        fake_tokens = generate_tokens(model, policy, B_np, perms, args, args.num_samples)
        gen_seconds = time.time() - t0
        fake_images = decode_all(vae, fake_tokens, inverse_patch_order, args)

        arm_out = args.out_dir / arm
        arm_out.mkdir(parents=True, exist_ok=True)
        torch.save(fake_tokens.to(torch.int16), arm_out / "samples_tokens_patch_order.pt")
        save_grid(fake_images[: min(64, fake_images.size(0))], arm_out / "sample_grid.png")
        if args.save_individual:
            save_individual(fake_images, arm_out / "images")

        stats = image_quality_stats(fake_images, fake_tokens)
        fid = None
        fid_error = None
        if not args.skip_fid:
            try:
                fid = compute_fid(fake_images, real_images, args.device, args.decode_batch_size)
            except Exception as exc:
                fid_error = repr(exc)
                print(f"  FID failed for {arm}: {fid_error}", flush=True)
        row = {
            "arm": arm,
            "policy": policy,
            "matched_policy": matched_policy,
            "force_policy": args.force_policy,
            "ckpt": str(ckpt_path),
            "step": int(ckpt.get("step", -1)),
            "samples": int(fake_images.size(0)),
            "generation_seconds": float(gen_seconds),
            "fid_vq_val": fid,
            "fid_error": fid_error,
            **stats,
        }
        rows.append(row)
        save_json(arm_out / "metrics.json", row)
        del model
        torch.cuda.empty_cache()

    save_json(args.out_dir / "metrics.json", rows)
    tsv = args.out_dir / "metrics.tsv"
    if rows:
        keys = list(rows[0].keys())
        with tsv.open("w", encoding="utf-8") as f:
            f.write("\t".join(keys) + "\n")
            for row in rows:
                f.write("\t".join("" if row.get(k) is None else str(row.get(k)) for k in keys) + "\n")

    md = ["# Round-2 Sample Quality / FID", ""]
    md.append("FID is computed against VQ-decoded validation tokens, not raw ImageNet images.")
    md.append("")
    md.append("| arm | policy | samples | FID_vq_val | pixel_std | token_entropy_bits | duplicate_rate |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        fid_s = "NA" if row["fid_vq_val"] is None else f"{row['fid_vq_val']:.4f}"
        md.append(
            f"| {row['arm']} | {row['policy']} | {row['samples']} | {fid_s} | "
            f"{row['pixel_std']:.4f} | {row['token_entropy_bits']:.3f} | {row['duplicate_image_rate']:.4f} |"
        )
    md.extend([
        "",
        "## Metadata",
        "",
        f"- source round2 root: `{args.round2_root}`",
        f"- source A/B path: `{args.a_block_path}`",
        "- physical order is remapped via `inverse_block_perm`; generated tokens are unpermuted before decoding.",
        "- B construction method: frozen Stage-1 E3-control-small attention-derived A_block/B readout input.",
        f"- readout: {'common ' + args.force_policy if args.force_policy else 'matched per arm'}; sampling temperature={args.temperature}, top_k={args.top_k}",
        f"- random seed: {args.seed}",
        f"- git: `{metadata['git']}`",
    ])
    (args.out_dir / "SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_dir / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
