#!/usr/bin/env python3
"""FID for the from-0 alternating VQ64 run (patch2x2), comparing readout orders.

This run trains with `_forward_with_block_orders(..., fixed_token_perm=None,
inv_block_perm=None)` — i.e. the data is already in physical patch frame and
block orders are physical block indices 0..63 with NO permutation. So generation
is the round2 FID `generate_tokens` path WITHOUT the inv_block_perm / inv_token_perm
remap. We reuse the VAE-decode / FID / real-token helpers from the round2 script.

Orders compared (same ckpt, different sampling order):
  - mlp     : distilled OrderMLP policy (beta_step{N}.pt) over B=A_global.T
  - raster  : arange(64)
  - random  : per-sample randperm(64)
"""
from __future__ import annotations
import argparse, json, sys, time
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
# reuse round2 FID helpers verbatim (decode / FID / IO):
from sample_quality_fid_imagenet64_vq import (
    load_vae, decode_all, compute_fid, iter_real_tokens,
    save_grid, image_quality_stats, save_json, now,
)

N_BLOCKS = 64
DEFAULT_MODEL_ARGS = dict(vocab_size=8192, n_layer=8, n_head=8,
                          n_embd=512, dropout=0.0, bias=False, order_impl="block")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path,
                   default=_REPO / "probe_results_image/vq64_alt_from0_mlp_patch2x2")
    p.add_argument("--step", type=int, default=30000)
    p.add_argument("--data-dir", type=Path,
                   default=_REPO / "nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch2x2")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--orders", type=str, default="mlp,raster,random")
    p.add_argument("--num-samples", type=int, default=1024)
    p.add_argument("--fid-real-samples", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--decode-batch-size", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--mlp-tau", type=float, default=0.5)
    p.add_argument("--mlp-top-k", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip-fid", action="store_true")
    p.add_argument("--mode", choices=["sample", "teacher_forced"], default="sample",
                   help="sample=free-running AR generation; teacher_forced=one-pass reconstruction of real val tokens conditioned on true in-order prefix")
    p.add_argument("--tf-pred", choices=["sample", "argmax"], default="sample",
                   help="teacher_forced only: how to turn per-position logits into the reconstructed token")
    return p.parse_args()


def load_model(run_dir: Path, step: int, device: str):
    ckpt = torch.load(run_dir / f"ckpt_step{step}.pt", map_location=device, weights_only=False)
    model_args = ckpt["model_args"]
    model = AOGPT(AOGPTConfig(**model_args)).to(device)
    sd = ckpt["model"]
    if all(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, ckpt


def build_block_orders(policy, bs, batch_id, seed, device, B_np, mlp, mlp_tau, mlp_top_k):
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
            device=device, tau=mlp_tau, top_k=mlp_top_k)
    raise ValueError(f"unknown policy {policy!r}")


@torch.no_grad()
def generate_tokens(model, policy, B_np, mlp, args, num_samples):
    """Physical-frame AR generation (no block/token perm — matches training)."""
    device = args.device
    bsz = model.config.block_size
    out = torch.empty(num_samples, bsz, dtype=torch.long, device="cpu")
    produced, batch_id = 0, 0
    while produced < num_samples:
        bs = min(args.batch_size, num_samples - produced)
        phys_block_orders = build_block_orders(policy, bs, batch_id, args.seed, device,
                                               B_np, mlp, args.mlp_tau, args.mlp_top_k)
        token_orders = model._expand_block_orders_to_token_orders(phys_block_orders)  # (bs, 256)
        x = torch.zeros(bs, bsz, dtype=torch.long, device=device)
        for rank in range(bsz):
            result = model(x, mode=None, orders=token_orders)
            logits = result[0][:, rank, :] / float(args.temperature)
            if args.top_k and args.top_k > 0:
                vals, _ = torch.topk(logits, min(int(args.top_k), logits.size(-1)), dim=-1)
                logits = logits.masked_fill(logits < vals[:, -1:], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
            pos = token_orders[:, rank]
            x[torch.arange(bs, device=device), pos] = nxt
        out[produced:produced + bs] = x.cpu()
        produced += bs
        batch_id += 1
        print(f"  [{policy}] generated {produced}/{num_samples}", flush=True)
    return out


@torch.no_grad()
def teacher_forced_reconstruct(model, real_tokens, policy, B_np, mlp, args):
    """One forward pass per batch: reconstruct each real image by replacing every
    position with the model's prediction conditioned on the TRUE in-order prefix
    (logits[:, rank] predicts position token_orders[:, rank] given true tokens at
    earlier ranks — exactly the training-time teacher-forced forward). No AR loop."""
    device = args.device
    N = real_tokens.size(0)
    out = torch.empty_like(real_tokens)
    batch_id = 0
    for s in range(0, N, args.batch_size):
        e = min(s + args.batch_size, N)
        x = real_tokens[s:e].to(device=device, dtype=torch.long)
        bs = x.size(0)
        phys_block_orders = build_block_orders(policy, bs, batch_id, args.seed, device,
                                               B_np, mlp, args.mlp_tau, args.mlp_top_k)
        token_orders = model._expand_block_orders_to_token_orders(phys_block_orders)  # (bs, 256)
        logits = model(x, mode=None, orders=token_orders)[0] / float(args.temperature)  # (bs,256,V)
        if args.tf_pred == "argmax":
            preds = logits.argmax(dim=-1)                       # (bs, 256), indexed by rank
        else:
            probs = F.softmax(logits.reshape(-1, logits.size(-1)), dim=-1)
            preds = torch.multinomial(probs, 1).squeeze(-1).reshape(bs, -1)
        recon = x.clone()
        recon.scatter_(1, token_orders, preds)                  # recon[b, order[b,rank]] = preds[b,rank]
        out[s:e] = recon.cpu()
        batch_id += 1
        print(f"  [{policy}/TF] reconstructed {e}/{N}", flush=True)
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = args.out_dir or (args.run_dir / f"fid_step{args.step}")
    out_dir.mkdir(parents=True, exist_ok=True)
    orders = [o.strip() for o in args.orders.split(",") if o.strip()]

    model, ckpt = load_model(args.run_dir, args.step, args.device)
    B_np = np.load(args.run_dir / f"A_global_step{args.step}.npy").T.copy()
    np.fill_diagonal(B_np, 0.0)
    B_np = B_np.astype(np.float32)
    mlp = OrderMLP()
    mlp.load_state_dict(torch.load(args.run_dir / f"beta_step{args.step}.pt",
                                   map_location="cpu", weights_only=False))
    mlp.to(args.device).eval()

    inverse_patch_order = torch.from_numpy(
        np.load(args.data_dir / "inverse_patch_order_indices.npy")).long()
    vae = load_vae(str(_default_vae()), args.device)
    real_tokens = iter_real_tokens(args.data_dir, args.fid_real_samples)
    real_images = decode_all(vae, real_tokens, inverse_patch_order, args)
    save_grid(real_images[:min(64, real_images.size(0))], out_dir / "real_val_recon_grid.png")

    rows = []
    for policy in orders:
        print(f"=== order={policy} mode={args.mode} ===", flush=True)
        t0 = time.time()
        if args.mode == "teacher_forced":
            fake_tokens = teacher_forced_reconstruct(model, real_tokens, policy, B_np, mlp, args)
        else:
            fake_tokens = generate_tokens(model, policy, B_np, mlp, args, args.num_samples)
        gen_s = time.time() - t0
        fake_images = decode_all(vae, fake_tokens, inverse_patch_order, args)
        pol_dir = out_dir / policy; pol_dir.mkdir(parents=True, exist_ok=True)
        save_grid(fake_images[:min(64, fake_images.size(0))], pol_dir / "sample_grid.png")
        stats = image_quality_stats(fake_images, fake_tokens)
        fid, fid_err = None, None
        if not args.skip_fid:
            try:
                fid = compute_fid(fake_images, real_images, args.device, args.decode_batch_size)
            except Exception as exc:
                fid_err = repr(exc); print(f"  FID failed {policy}: {fid_err}", flush=True)
        row = {"order": policy, "mode": args.mode,
               "tf_pred": args.tf_pred if args.mode == "teacher_forced" else None,
               "step": int(ckpt.get("step", args.step)),
               "samples": int(fake_images.size(0)), "generation_seconds": round(gen_s, 1),
               "fid_vq_val": fid, "fid_error": fid_err, **stats}
        rows.append(row); save_json(pol_dir / "metrics.json", row)
        print(f"  -> FID={fid} dup={stats['duplicate_image_rate']:.3f} "
              f"pix_std={stats['pixel_std']:.3f} ({gen_s:.0f}s)", flush=True)

    save_json(out_dir / "metrics.json",
              {"created_at": now(), "run_dir": str(args.run_dir), "step": args.step,
               "mode": args.mode, "tf_pred": args.tf_pred,
               "num_samples": args.num_samples, "fid_real_samples": int(real_images.size(0)),
               "temperature": args.temperature, "top_k": args.top_k, "seed": args.seed,
               "fid_reference": "VQ-decoded validation tokens (reconstruction-space proxy)",
               "rows": rows})
    with (out_dir / "metrics.tsv").open("w") as f:
        keys = ["order", "fid_vq_val", "pixel_std", "token_entropy_bits",
                "duplicate_image_rate", "samples", "generation_seconds"]
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + "\n")
    print("\n#### FID SUMMARY ####")
    print(f"{'order':<10}{'FID':>10}{'dup_rate':>10}{'pix_std':>10}")
    for r in rows:
        fid_s = "NA" if r["fid_vq_val"] is None else f"{r['fid_vq_val']:.3f}"
        print(f"{r['order']:<10}{fid_s:>10}{r['duplicate_image_rate']:>10.3f}{r['pixel_std']:>10.3f}")
    print(f"\nWrote {out_dir}")


def _default_vae():
    import sample_quality_fid_imagenet64_vq as M
    return M.DEFAULT_VAE


if __name__ == "__main__":
    main()
