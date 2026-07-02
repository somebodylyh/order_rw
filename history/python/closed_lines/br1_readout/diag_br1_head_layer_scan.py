"""BR-1 mismatch diagnostic v2: per-(layer, head) scan vs ori-L2R.

Rather than head-mean (which the v1 diag showed washes the order signal from
tau=0.53 down to ~0.05), scan EVERY single (layer, head) attention map and ask:
which single head, on its own, induces a teacher order closest to the text's
intrinsic structure (ori-L2R)?

For each (layer ell, head h):
  single-head attention -> physical remap -> 256->64 block-agg -> B = A^T
  -> batch-mean over `batch_size` -> teacher CDL sigma per graph.
Metrics:
  tau_vs_l2r        = mean_m Kendall tau(sigma_T[m], [0,1,...,63])   <- the ask
  mean_pairwise_tau = teacher diversity across graphs
  tau_vs_heavy      = agreement with the heavy (top-4 var + all-layer + [None]) sigma
  first_step_entropy

Baseline row "heavy" reproduces the Phase-1 training B definition.

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=block_lo_arm_order_network \
    python scripts/diag_br1_head_layer_scan.py --M 100 --batch_size 32 --seed 0
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from neural_readout.extract_b import _load_model_and_chunks
from train_clean_aogpt import expand_model_blocks_to_token_order, N, SEQ_LEN, BLOCK_LEN
from neural_readout.teacher_labels import generate_teacher_label
from batch_readout.diversity_batch import teacher_diversity_stats
from batch_readout.eval_metrics import kendall_tau_batch


def _block_agg(attn_phys):
    """(.., 256, 256) -> (.., 64, 64) by mean over 4x4 token blocks."""
    shp = attn_phys.shape[:-2]
    x = attn_phys.reshape(*shp, N, BLOCK_LEN, N, BLOCK_LEN)
    return x.mean(axis=(-3, -1))


@torch.no_grad()
def scan(model, idx_chunks, clean_perm, device, n_chunks, batch_size, seed):
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    M = n_chunks // batch_size

    # accumulate batch-mean B for every (layer, head) and for heavy
    A_lh = None         # lazily sized once we know (L, H)
    A_heavy_sum = np.zeros((M, N, N), dtype=np.float64)

    model.eval()
    for i in range(n_chunks):
        tokens = idx_chunks[i:i + 1].to(device)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + int(i))
        rand_blocks = torch.randperm(N, generator=gen, device="cpu")
        token_order = expand_model_blocks_to_token_order(
            rand_blocks.unsqueeze(0), BLOCK_LEN
        ).to(device)
        _, _, attn_list = model.forward_fn(tokens, token_order, return_attentions=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        attn_stack = torch.stack(attn_list).squeeze(1).cpu().numpy()  # (L, H, 257, 257)
        L, H = attn_stack.shape[:2]
        if A_lh is None:
            A_lh = np.zeros((L, H, M, N, N), dtype=np.float64)

        reveal_tokens = token_order[0].cpu().numpy()
        model_blocks = reveal_tokens // BLOCK_LEN
        phys_blocks = inv_perm[model_blocks]
        phys_tokens = phys_blocks * BLOCK_LEN + (reveal_tokens % BLOCK_LEN)  # perm of 0..255

        m = i // batch_size

        # --- all (L, H) single heads, vectorized remap + block-agg ---
        content = attn_stack[:, :, 1:, 1:]                     # (L, H, 256, 256), reveal coords
        phys = np.empty_like(content)
        phys[:, :, phys_tokens[:, None], phys_tokens[None, :]] = content   # remap (perm, no collision)
        A_blocks = _block_agg(phys)                            # (L, H, 64, 64)
        # B = A^T, zero diagonal, accumulate into batch m
        B_blocks = np.swapaxes(A_blocks, -1, -2)
        di = np.arange(N)
        B_blocks[..., di, di] = 0.0
        A_lh[:, :, m] += B_blocks

        # --- heavy baseline: top-4 variance head, all-layer mean, +0.1*[None] ---
        head_vars = np.zeros(H)
        mask = ~np.eye(SEQ_LEN, dtype=bool)
        for h in range(H):
            offdiag = content[:, h][:, mask].reshape(L, SEQ_LEN, SEQ_LEN - 1)
            head_vars[h] = float(np.var(offdiag))
        top_heads = np.argsort(head_vars)[-4:]
        avg_heavy = attn_stack[:, top_heads, :, :].mean(axis=(0, 1))   # (257, 257)
        # remap heavy content
        hc = avg_heavy[1:, 1:]
        hp = np.empty_like(hc)
        hp[phys_tokens[:, None], phys_tokens[None, :]] = hc
        A_h = _block_agg(hp)
        none_attn = avg_heavy[1:, 0]
        none_block = np.array(
            [none_attn[b * BLOCK_LEN:(b + 1) * BLOCK_LEN].mean() for b in range(N)]
        )
        A_h = A_h + none_block[np.newaxis, :] * 0.1
        B_h = A_h.T.copy()
        np.fill_diagonal(B_h, 0.0)
        A_heavy_sum[m] += B_h

    # finalize batch means
    Bb_lh = A_lh / batch_size                                  # (L, H, M, 64, 64)
    di = np.arange(N)
    Bb_lh[..., di, di] = 0.0
    Bb_heavy = (A_heavy_sum / batch_size).astype(np.float32)
    np.fill_diagonal  # noop reference
    for mm in range(M):
        np.fill_diagonal(Bb_heavy[mm], 0.0)
    return Bb_lh.astype(np.float32), Bb_heavy, L, H


def _sigmas(Bb):
    return np.stack([generate_teacher_label(Bb[m], alpha_dep=0.5)[0] for m in range(len(Bb))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="probe_results/attention_order_mlp/alt_from0_random/ckpt_step5000.pt")
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="block_lo_arm_order_network/batch_readout/logs/diag_head_layer_scan.json")
    args = ap.parse_args()

    total = args.M * args.batch_size
    model, chunks, clean_perm, dev, _ci = _load_model_and_chunks(
        args.ckpt, total, args.seed, args.device, "train"
    )
    print(f"[scan] extracting {total} graphs (M={args.M} x B={args.batch_size}) ...", flush=True)
    Bb_lh, Bb_heavy, L, H = scan(model, chunks, clean_perm, dev, total, args.batch_size, args.seed)

    l2r = np.tile(np.arange(N), (args.M, 1))
    sig_heavy = _sigmas(Bb_heavy)
    heavy_tau_l2r = kendall_tau_batch(sig_heavy, l2r)

    table = []
    for ell in range(L):
        for h in range(H):
            sig = _sigmas(Bb_lh[ell, h])
            div = teacher_diversity_stats(sig)
            table.append({
                "layer": ell, "head": h,
                "tau_vs_l2r": kendall_tau_batch(sig, l2r),
                "mean_pairwise_tau": div["mean_pairwise_tau"],
                "first_step_entropy": div["first_step_entropy"],
                "tau_vs_heavy": kendall_tau_batch(sig, sig_heavy),
            })

    table.sort(key=lambda r: abs(r["tau_vs_l2r"]), reverse=True)
    report = {
        "config": {"M": args.M, "batch_size": args.batch_size, "seed": args.seed,
                   "ckpt": args.ckpt, "L": L, "H": H},
        "heavy_baseline": {
            "tau_vs_l2r": heavy_tau_l2r,
            "diversity": teacher_diversity_stats(sig_heavy),
        },
        "per_head_layer_sorted_by_abs_tau_vs_l2r": table,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nheavy baseline: tau_vs_l2r={heavy_tau_l2r:+.3f}")
    print(f"\n{'layer':>5} {'head':>4} {'tau_vs_l2r':>11} {'pairwise_tau':>13} {'first_H':>8} {'tau_vs_heavy':>13}")
    for r in table:
        print(f"{r['layer']:>5} {r['head']:>4} {r['tau_vs_l2r']:>+11.3f} "
              f"{r['mean_pairwise_tau']:>+13.3f} {r['first_step_entropy']:>8.3f} {r['tau_vs_heavy']:>+13.3f}")


if __name__ == "__main__":
    main()
