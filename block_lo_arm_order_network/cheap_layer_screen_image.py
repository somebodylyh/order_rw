#!/usr/bin/env python3
"""Cheap-score → layer screening diagnostic for image AOGPT checkpoints.

Loads an image-trained AOGPT ckpt, forwards image data with random block-reveal
orders, extracts per-(layer,head) physical-frame block graphs (N=64), runs CDL-
source-start readout to get per-head tau_vs_raster (expensive ground truth), and
computes cheap scores C1-C4 per head. Then reports per-layer aggregation and
cheap-vs-expensive alignment (Spearman ρ, recall@k).

Motivation: on text, L0H0 is the canonical order-signal head. On image, which
layer carries the order signal is unknown a priori. This script answers whether
cheap scores (C1-C4, computed purely from A_lh without CDL rollout) can predict
which layer/head has the strongest order signal on image data.

Usage:
    python cheap_layer_screen_image.py \
        --ckpt probe_results_image/vq64_fixed_raster_from0_40k/ckpt_step40000.pt \
        --data nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2/train.bin \
        --n-images 200 --device cuda:0 \
        --out-dir probe_results_image/cheap_screen
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

# -- project imports ----------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from per_head_order_scan import _per_sample_A, SEQ_LEN, N, BLOCK_LEN

# Add nanogpt-learned-order for AOGPT
_NANO_ROOT = _ROOT.parent / "nanogpt-learned-order"
sys.path.insert(0, str(_NANO_ROOT))
from AOGPT import AOGPTConfig, AOGPT

from quick_head_selector import (
    cheap_head_scores, calibrate_sign, spearman_cheap_expensive, recall_at_k,
)

# Add teacher_labels (nested path)
_TEACHER_ROOT = _ROOT / "neural_readout"
sys.path.insert(0, str(_TEACHER_ROOT))
from teacher_labels import generate_teacher_label


# ── model loading (from extract_image_attention.load_model) ──────────────────
def load_image_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_args = ckpt.get("model_args", {})
    cfg = ckpt["config"]
    model_args = {}
    for k in ["block_size", "vocab_size", "n_layer", "n_head", "n_embd",
              "dropout", "bias", "block_order_block_len", "order_impl"]:
        if k in m_args:
            model_args[k] = m_args[k]
        elif k in cfg:
            model_args[k] = cfg[k]
    model = AOGPT(AOGPTConfig(**model_args))
    sd = ckpt["model"]
    if all(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys: {missing[:5]}...")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:5]}")
    model.crop_block_size(model_args["block_size"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, model_args


# ── per-head CDL tau vs raster ──────────────────────────────────────────────
def per_head_tau_vs_raster(B_lh, raster_ref=None):
    """Compute mean Kendall tau between each head's CDL order and raster ref.

    Args:
        B_lh: (L, H, N, N) float32 — per-head batch-mean directed graphs.
        raster_ref: (N,) int, default arange(N).

    Returns:
        tau_lh: (L, H) float64, tau_vs_raster per head.
        orders: dict (l,h) -> sigma (N,) int64.
    """
    L, H, Nn = B_lh.shape[:3]
    if raster_ref is None:
        raster_ref = np.arange(Nn)
    tau_lh = np.full((L, H), np.nan, dtype=np.float64)
    orders = {}
    for l in range(L):
        for h in range(H):
            B = B_lh[l, h]
            sigma, _rank, _Y = generate_teacher_label(B, alpha_dep=0.5)
            orders[(l, h)] = sigma
            t, _ = kendalltau(sigma, raster_ref)
            if not np.isnan(t):
                tau_lh[l, h] = float(t)
    return tau_lh, orders


# ── main diagnostic ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="Image AOGPT checkpoint .pt")
    p.add_argument("--data", required=True, help="Image .bin data file (VQ tokens, uint16)")
    p.add_argument("--n-images", type=int, default=200, help="Number of images to process")
    p.add_argument("--m-orders", type=int, default=3, help="Random orders per image")
    p.add_argument("--fwd-batch", type=int, default=16, help="Forward batch size")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--none-mode", default="b1", choices=["old", "b1"])
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    print(f"[load] ckpt={args.ckpt}", flush=True)
    model, model_args = load_image_model(args.ckpt, args.device)
    Ln, Hn = model_args["n_layer"], model_args["n_head"]
    T = model_args["block_size"]
    Nn = T // BLOCK_LEN  # should be 64 for ImageNet64 patch2x2
    print(f"[load] L={Ln} H={Hn} T={T} N={Nn} vocab={model_args['vocab_size']}", flush=True)

    # 2. Load image data
    data_tokens = np.memmap(args.data, dtype=np.uint16, mode="r")
    n_available = len(data_tokens) // T
    n_images = min(args.n_images, n_available)
    total_forward = n_images * args.m_orders
    print(f"[data] {n_available} images available, using {n_images} x {args.m_orders} orders = {total_forward} forwards", flush=True)

    # 3. Accumulate per-head A_lh over all samples
    A_lh_acc = np.zeros((Ln, Hn, Nn, Nn), dtype=np.float64)
    A_heavy_acc = np.zeros((Nn, Nn), dtype=np.float64)
    n_accum = 0

    inv_perm = np.arange(Nn, dtype=np.int64)  # identity: physical=model for image data
    rng = np.random.RandomState(args.seed)

    n_batches = (total_forward + args.fwd_batch - 1) // args.fwd_batch
    fwd_idx = 0

    print(f"[extract] {total_forward} forwards in {n_batches} batches...", flush=True)

    for img_idx in range(n_images):
        start = img_idx * T
        tokens_np = data_tokens[start:start + T].astype(np.int64)

        for _ord in range(args.m_orders):
            # Seeded random block permutation (deterministic, reproducible)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(args.seed) * 10000 + int(img_idx) * 100 + int(_ord))
            rand_blocks = torch.randperm(Nn, generator=gen, device="cpu")  # (N,)
            # Expand blocks to token order: block b → tokens [b*4, b*4+1, b*4+2, b*4+3]
            token_order = (rand_blocks * BLOCK_LEN).unsqueeze(1) + torch.arange(BLOCK_LEN)  # (N, 4)
            token_order = token_order.reshape(-1)  # (T,)

            tokens = torch.from_numpy(tokens_np).to(args.device).unsqueeze(0)  # (1, T)
            order = token_order.unsqueeze(0).to(args.device)  # (1, T)

            with torch.no_grad():
                _, _, attn_list = model.forward_fn(tokens, order, return_attentions=True)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()

            # attn_list: list of (B, nh, T+1, T+1), one per layer
            attn_stack = torch.stack(attn_list).squeeze(1).cpu().numpy()  # (L, H, T+1, T+1)
            reveal_tokens = token_order.numpy().astype(np.int64)  # already in physical coords

            A_lh_i, A_heavy_i = _per_sample_A(
                attn_stack, reveal_tokens, inv_perm,
                n_top=4, none_mode=args.none_mode,
            )  # A_lh_i: (L, H, N, N), A_heavy_i: (N, N)

            A_lh_acc += A_lh_i.astype(np.float64)
            A_heavy_acc += A_heavy_i.astype(np.float64)
            n_accum += 1

            fwd_idx += 1
            if fwd_idx % 50 == 0:
                print(f"  [{fwd_idx}/{total_forward}] done", flush=True)

    print(f"[extract] done, {n_accum} samples accumulated", flush=True)

    # 4. Batch-mean graphs
    A_lh_mean = (A_lh_acc / n_accum).astype(np.float32)  # (L, H, N, N)
    A_heavy_mean = (A_heavy_acc / n_accum).astype(np.float32)

    # B = A.T with zero diagonal
    B_lh = np.transpose(A_lh_mean, (0, 1, 3, 2)).copy()
    for l in range(Ln):
        for h in range(Hn):
            np.fill_diagonal(B_lh[l, h], 0.0)

    # 5. Cheap scores C1-C4
    cheap = cheap_head_scores(A_lh_mean, alpha_dep=0.5)  # dict of (L,H) arrays

    # 6. Expensive tau_vs_raster via CDL readout
    print("[cdl] computing per-head CDL tau_vs_raster...", flush=True)
    raster_ref = np.arange(Nn)
    tau_lh, orders = per_head_tau_vs_raster(B_lh, raster_ref)

    # 7. Per-layer summary
    print("\n=== Per-layer summary ===")
    layer_rows = []
    for l in range(Ln):
        taus = [tau_lh[l, h] for h in range(Hn) if not np.isnan(tau_lh[l, h])]
        mean_tau = float(np.mean(taus)) if taus else float("nan")
        mean_abs_tau = float(np.mean(np.abs(taus))) if taus else float("nan")
        c1s = [cheap["C1"][l, h] for h in range(Hn)]
        c2s = [cheap["C2"][l, h] for h in range(Hn)]
        c3s = [cheap["C3"][l, h] for h in range(Hn)]
        c4s = [cheap["C4"][l, h] for h in range(Hn)]
        row = {
            "layer": l,
            "mean_tau": round(mean_tau, 4),
            "mean_abs_tau": round(mean_abs_tau, 4),
            "best_head": int(np.nanargmax([tau_lh[l, h] for h in range(Hn)])),
            "best_tau": round(float(np.nanmax([tau_lh[l, h] for h in range(Hn)])), 4),
            "C1_mean": round(float(np.mean(c1s)), 4),
            "C2_mean": round(float(np.mean(c2s)), 4),
            "C3_mean": round(float(np.mean(c3s)), 6),
            "C4_mean": round(float(np.mean(c4s)), 4),
        }
        layer_rows.append(row)
        print(f"  L{l}: mean_tau={mean_tau:+.4f} |tau|={mean_abs_tau:.4f}  "
              f"best=H{row['best_head']}({row['best_tau']:+.4f})  "
              f"C1={row['C1_mean']:+.4f} C2={row['C2_mean']:+.4f} "
              f"C3={row['C3_mean']:.6f} C4={row['C4_mean']:.4f}")

    # 8. Cheap-vs-expensive alignment
    print("\n=== Cheap vs expensive alignment ===")
    align_rows = []
    # Determine the expensive winner (per full L*H space, not per layer)
    tau_flat = tau_lh.ravel()
    exp_winner_idx = int(np.nanargmax(tau_flat))
    exp_winner = (exp_winner_idx // Hn, exp_winner_idx % Hn, tau_flat[exp_winner_idx])
    print(f"  Expensive winner: L{exp_winner[0]}H{exp_winner[1]} tau={exp_winner[2]:+.4f}")

    for ck in ("C1", "C2", "C3", "C4"):
        raw = cheap[ck].copy()
        sign, signed = calibrate_sign(raw, tau_lh)
        rho = spearman_cheap_expensive(signed, tau_lh)
        r1 = recall_at_k(signed, tau_lh, k=1)
        r3 = recall_at_k(signed, tau_lh, k=3)

        # Which head does this cheap score pick?
        signed_flat = signed.ravel()
        cheap_winner_idx = int(np.nanargmax(signed_flat))
        cheap_winner = (cheap_winner_idx // Hn, cheap_winner_idx % Hn)

        print(f"  {ck}: sign={sign:+.0f} ρ={rho:+.4f} recall@1={'Y' if r1 else 'N'} recall@3={'Y' if r3 else 'N'}  "
              f"winner=L{cheap_winner[0]}H{cheap_winner[1]} (exp=L{exp_winner[0]}H{exp_winner[1]})")
        align_rows.append({
            "score": ck, "sign": int(sign), "spearman_rho": round(float(rho), 4),
            "recall_at_1": bool(r1), "recall_at_3": bool(r3),
            "cheap_winner_layer": cheap_winner[0], "cheap_winner_head": cheap_winner[1],
            "expensive_winner_layer": exp_winner[0], "expensive_winner_head": exp_winner[1],
        })

    # Also try all cheap scores pooled — take the head with max (normalized & signed) score
    print("\n  --- pooled cheap → expensive per-layer analysis ---")
    pool_rows = []
    for l in range(Ln):
        best_tau = float(np.nanmax([tau_lh[l, h] for h in range(Hn)]))
        # Which cheap score picks the right layer?
        cheap_layer_picks = {}
        for ck in ("C1", "C2", "C3", "C4"):
            sign, signed = calibrate_sign(cheap[ck].copy(), tau_lh)
            layer_scores = signed[l, :]  # all heads in this layer
            best_in_layer = float(np.nanmax(layer_scores))
            cheap_layer_picks[ck] = best_in_layer
        best_cheap_layer = max(cheap_layer_picks, key=cheap_layer_picks.get)
        pool_rows.append({
            "layer": l, "best_tau": round(best_tau, 4),
            "C1_max": round(cheap_layer_picks["C1"], 4),
            "C2_max": round(cheap_layer_picks["C2"], 4),
            "C3_max": round(cheap_layer_picks["C3"], 6),
            "C4_max": round(cheap_layer_picks["C4"], 4),
            "best_cheap": best_cheap_layer,
        })

    # Which layer wins by expensive tau?
    layer_taus = [float(np.nanmax([tau_lh[l, h] for h in range(Hn)])) for l in range(Ln)]
    exp_layer_winner = int(np.argmax(layer_taus))
    cheap_layer_winners = {}
    for ck in ("C1", "C2", "C3", "C4"):
        sign, signed = calibrate_sign(cheap[ck].copy(), tau_lh)
        layer_maxes = [float(np.nanmax(signed[l, :])) for l in range(Ln)]
        cheap_layer_winners[ck] = int(np.argmax(layer_maxes))
    print(f"  Expensive best layer: L{exp_layer_winner}")
    for ck in ("C1", "C2", "C3", "C4"):
        match = "✓" if cheap_layer_winners[ck] == exp_layer_winner else "✗"
        print(f"  {ck} best layer: L{cheap_layer_winners[ck]} {match}")

    # 9. Per-head detail table
    print("\n=== Per-head detail (sorted by tau) ===")
    head_details = []
    for l in range(Ln):
        for h in range(Hn):
            head_details.append({
                "layer": l, "head": h,
                "tau_vs_raster": round(float(tau_lh[l, h]), 4),
                "C1": round(float(cheap["C1"][l, h]), 4),
                "C2": round(float(cheap["C2"][l, h]), 4),
                "C3": round(float(cheap["C3"][l, h]), 6),
                "C4": round(float(cheap["C4"][l, h]), 4),
            })
    head_details.sort(key=lambda d: abs(d["tau_vs_raster"]), reverse=True)
    for d in head_details[:8]:
        print(f"  L{d['layer']}H{d['head']}: tau={d['tau_vs_raster']:+.4f}  "
              f"C1={d['C1']:+.4f} C2={d['C2']:+.4f} C3={d['C3']:.6f} C4={d['C4']:.4f}")

    # 10. Save results
    result = {
        "config": {"ckpt": args.ckpt, "data": args.data,
                   "n_images": n_images, "m_orders": args.m_orders,
                   "L": Ln, "H": Hn, "N": Nn, "T": T,
                   "none_mode": args.none_mode, "seed": args.seed},
        "layer_summary": layer_rows,
        "cheap_vs_expensive_alignment": align_rows,
        "expensive_winner": {"layer": exp_winner[0], "head": exp_winner[1], "tau": round(float(exp_winner[2]), 4)},
        "expensive_best_layer": exp_layer_winner,
        "cheap_best_layers": cheap_layer_winners,
        "layer_pool": pool_rows,
        "per_head_sorted": head_details,
    }
    out_json = out_dir / "cheap_screen_result.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nSaved → {out_json}")

    # Also save the mean block graphs for offline analysis
    np.save(out_dir / "A_lh_mean.npy", A_lh_mean)
    np.save(out_dir / "B_lh_mean.npy", B_lh)
    np.save(out_dir / "tau_lh.npy", tau_lh)
    for ck in ("C1", "C2", "C3", "C4"):
        np.save(out_dir / f"cheap_{ck}.npy", cheap[ck])
    print(f"Saved arrays → {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
