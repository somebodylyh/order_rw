#!/usr/bin/env python3
"""Block granularity post-hoc analysis: does τ_vs_L2R survive when attention is
aggregated at different block granularities (32/64/128 blocks)?

Reuses the per_head_order_scan extraction pipeline but aggregates attention at
three resolutions:
  - 32 blocks × 8 tokens  (coarse)
  - 64 blocks × 4 tokens  (current, baseline)
  - 128 blocks × 2 tokens (fine)

No training required — uses existing clean_base_random_perm checkpoints.

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=block_lo_arm_order_network \
    python analyses/block_granularity_scan.py --ckpt-step 5000 --M 100 --batch-size 32
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
import torch
from scipy.stats import kendalltau

_ROOT = pathlib.Path(__file__).resolve().parents[1] / "block_lo_arm_order_network"
sys.path.insert(0, str(_ROOT))

from training_utils import SEQ_LEN, N as N_DEFAULT, BLOCK_LEN as BL_DEFAULT
from clean_training_protocol import expand_model_blocks_to_token_order
from neural_readout.extract_b import _load_model_and_chunks
from neural_readout.teacher_labels import generate_teacher_label
from batch_readout.diversity_batch import teacher_diversity_stats
from batch_readout.eval_metrics import kendall_tau_batch
from quick_head_selector import cheap_head_scores

# ── granularity presets ───────────────────────────────────────────────────────
GRANULARITIES = [
    ("32blk_x_8tok", 32, 8),
    ("64blk_x_4tok", 64, 4),   # baseline / current
    ("128blk_x_2tok", 128, 2),
]


# ── multi-granularity attention → block aggregation ───────────────────────────
def _phys_attn_to_multi_granularity(attn_phys, none_col, lead, granularities):
    """Aggregate physical-frame token-level attention (..., 256, 256) to multiple
    (num_blocks, block_len) pairs.

    Args:
        attn_phys: (..., 256, 256) float32, physical-frame token-level attention.
        none_col: (..., 256) float32, attention from each token to the [None] sink.
        lead: tuple of leading dims (e.g. (L,H) for per-head, () for heavy).
        granularities: list of (name, num_blocks, block_len) tuples.

    Returns:
        dict: name → A array (..., num_blocks, num_blocks) float32.
    """
    attn_phys = np.asarray(attn_phys, dtype=np.float32)
    none_col = np.asarray(none_col, dtype=np.float32)
    # Flatten leading dims to (K, 256, 256) / (K, 256)
    lead_dims = attn_phys.shape[:-2]
    K = int(np.prod(lead_dims)) if lead_dims else 1
    a = attn_phys.reshape(K, SEQ_LEN, SEQ_LEN)
    n = none_col.reshape(K, SEQ_LEN)
    results = {}
    for gname, nb, bl in granularities:
        # Block-mean aggregate: (K, nb, bl, nb, bl) → (K, nb, nb)
        A_blk = a.reshape(K, nb, bl, nb, bl).mean(axis=(2, 4))

        # None-source term: mean per-block attention to [None], shape (K, nb)
        none_block = n.reshape(K, nb, bl).mean(axis=2)
        A_blk = A_blk + none_block[:, None, :] * 0.1  # same none_weight as canonical

        # Zero diagonal
        di = np.arange(nb)
        A_blk[:, di, di] = 0.0
        A_blk = A_blk.astype(np.float32, copy=False)

        # Reshape back to original leading dims
        if lead:
            A_blk = A_blk.reshape(lead + (nb, nb))
        else:
            A_blk = A_blk[0]
        results[gname] = A_blk
    return results


# ── per-sample physical-frame attention extraction (modified from per_head_order_scan) ──
def _per_sample_phys_attn(attn_stack, reveal_tokens, inv_perm_64, n_top=4):
    """Extract per-(L,H) physical-frame token-level attention (256×256) + heavy.

    Returns:
        attn_phys_lh: (L, H, 256, 256) float32 — per-head physical-frame token attn
        none_col_lh:  (L, H, 256) float32 — per-head [None] column
        attn_phys_heavy: (256, 256) float32
        none_col_heavy: (256,) float32
    """
    L, H = attn_stack.shape[:2]
    attn = np.asarray(attn_stack, dtype=np.float32)  # (L, H, 257, 257)

    reveal_tokens = np.asarray(reveal_tokens, dtype=np.int64)
    inv_perm_64 = np.asarray(inv_perm_64, dtype=np.int64)
    model_blocks = reveal_tokens // BL_DEFAULT
    phys_blocks = inv_perm_64[model_blocks]
    phys_tokens = phys_blocks * BL_DEFAULT + (reveal_tokens % BL_DEFAULT)

    # Heavy: top-n_top heads by off-diagonal variance
    head_vars = np.zeros(H)
    mask = ~np.eye(SEQ_LEN, dtype=bool)
    for h in range(H):
        content = attn[:, h, 1:, 1:]
        offdiag = content[:, mask].reshape(L, SEQ_LEN, SEQ_LEN - 1)
        head_vars[h] = float(np.var(offdiag))
    top_heads = np.argsort(head_vars)[-n_top:]

    # ── Per-head physical-frame attention ──
    attn_phys_lh = np.zeros((L, H, SEQ_LEN, SEQ_LEN), dtype=np.float32)
    none_col_lh = np.zeros((L, H, SEQ_LEN), dtype=np.float32)
    for l in range(L):
        for h in range(H):
            a_lh = attn[l, h]  # (257, 257)
            p = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32)
            p[phys_tokens[:, None], phys_tokens[None, :]] = a_lh[1:, 1:]
            attn_phys_lh[l, h] = p
            none_col_lh[l, h] = a_lh[1:, 0]

    # ── Heavy physical-frame attention ──
    avg_attn_heavy = attn[:, top_heads, :, :].mean(axis=(0, 1))  # (257, 257)
    attn_phys_heavy = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32)
    attn_phys_heavy[phys_tokens[:, None], phys_tokens[None, :]] = avg_attn_heavy[1:, 1:]
    none_col_heavy = np.asarray(avg_attn_heavy[1:, 0], dtype=np.float32)

    return attn_phys_lh, none_col_lh, attn_phys_heavy, none_col_heavy


# ── main scan ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def scan_multi_granularity(ckpt_path, M, batch_size, seed, device="cuda:0",
                           split="train", alpha_dep=0.5, fwd_batch=64):
    """Full multi-granularity per-head scan.

    Returns:
        dict: {
            "config": {...},
            "per_granularity": {
                "32blk_x_8tok": {
                    "heavy": {...},
                    "per_head_sorted": [...],
                },
                ...
            },
            "comparison": {  # top-head comparison across granularities
                "head": "LxHy",
                "32blk_x_8tok": tau,
                ...
            },
        }
    """
    total = M * batch_size
    model, chunks, clean_perm, dev, _ci = _load_model_and_chunks(
        ckpt_path, total, seed, device, split
    )
    device = torch.device(dev) if isinstance(dev, str) else dev
    inv_perm_64 = clean_perm.inv_perm_model_to_phys.cpu().numpy()

    n_chunks = len(chunks)
    model.eval()

    # Pre-build seeded reveal token_orders
    token_orders = torch.empty((n_chunks, SEQ_LEN), dtype=torch.long)
    for i in range(n_chunks):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + int(i))
        rand_blocks = torch.randperm(N_DEFAULT, generator=gen, device="cpu")
        token_orders[i] = expand_model_blocks_to_token_order(
            rand_blocks.unsqueeze(0), BL_DEFAULT
        )[0]

    # ── Accumulators: batch-mean A for each granularity, per (L,H) ──
    Ln, Hn = None, None
    # Dict[gname] → (M, L, H, N_g, N_g) or (M, N_g, N_g) for heavy
    A_lh_acc = {}
    A_heavy_acc = {}

    def _init_acc(nb):
        return np.zeros((M, Ln, Hn, nb, nb), dtype=np.float64)

    sample_idx = 0
    for start in range(0, n_chunks, max(1, int(fwd_batch))):
        stop = min(start + max(1, int(fwd_batch)), n_chunks)
        tokens = chunks[start:stop].to(device)
        order = token_orders[start:stop].to(device)

        _, _, attn_list = model.forward_fn(tokens, order, return_attentions=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        attn_batch = torch.stack(attn_list).cpu().numpy()  # (L, B, H, 257, 257)

        for bi in range(stop - start):
            i = start + bi
            m = i // batch_size
            attn_stack = attn_batch[:, bi]  # (L, H, 257, 257)
            reveal_tokens = token_orders[i].numpy()

            if Ln is None:
                Ln, Hn = attn_stack.shape[:2]
                for gname, nb, bl in GRANULARITIES:
                    A_lh_acc[gname] = _init_acc(nb)
                    A_heavy_acc[gname] = np.zeros((M, nb, nb), dtype=np.float64)

            # Extract per-head physical-frame token-level attention
            attn_phys_lh, none_col_lh, attn_phys_heavy, none_col_heavy = \
                _per_sample_phys_attn(attn_stack, reveal_tokens, inv_perm_64)

            # ── Per-head: aggregate at each granularity ──
            for gname, nb, bl in GRANULARITIES:
                # Per-head
                A_per_head = _phys_attn_to_multi_granularity(
                    attn_phys_lh, none_col_lh,
                    lead=(Ln, Hn),  # L,H are leading dims
                    granularities=[(gname, nb, bl)]
                )[gname]  # (L, H, nb, nb)
                for l in range(Ln):
                    for h in range(Hn):
                        A_lh_acc[gname][m, l, h] += A_per_head[l, h].astype(np.float64)

                # Heavy
                A_heavy = _phys_attn_to_multi_granularity(
                    attn_phys_heavy[None, :, :], none_col_heavy[None, :],
                    lead=(),
                    granularities=[(gname, nb, bl)]
                )[gname]  # (nb, nb)
                A_heavy_acc[gname][m] += A_heavy.astype(np.float64)

            sample_idx += 1

        if start % (max(1, int(fwd_batch)) * 10) == 0 or stop >= n_chunks:
            print(f"  [extract] {stop}/{n_chunks} chunks ({100*stop//n_chunks}%)", flush=True)

    # ── Finalise batch-means: divide by batch_size ──
    for gname, nb, bl in GRANULARITIES:
        A_lh_acc[gname] /= batch_size
        A_heavy_acc[gname] /= batch_size

    # ── Per-granularity analysis ──
    per_granularity = {}
    comparison_rows = []  # for the cross-granularity top-head table

    for gname, nb, bl in GRANULARITIES:
        print(f"\n─── {gname} (N={nb}, BL={bl}) ───", flush=True)

        # Heavy baseline
        A_heavy_m = A_heavy_acc[gname].astype(np.float32)  # (M, nb, nb)
        B_heavy = np.transpose(A_heavy_m, (0, 2, 1)).astype(np.float32)
        di = np.arange(nb)
        B_heavy[:, di, di] = 0.0
        sigma_heavy = {}
        for m in range(M):
            s, _, _ = generate_teacher_label(B_heavy[m], alpha_dep=alpha_dep)
            sigma_heavy[m] = s
        sigma_heavy = np.array([sigma_heavy[m] for m in range(M)])
        heavy_tau = float(np.mean([
            t for t in [kendalltau(s, np.arange(nb))[0] for s in sigma_heavy]
            if not np.isnan(t)
        ]))
        heavy_div = teacher_diversity_stats(sigma_heavy)

        print(f"  heavy tau_vs_l2r={heavy_tau:.4f}  "
              f"mean_pairwise_tau={heavy_div.get('mean_pairwise_tau', float('nan')):.4f}  "
              f"first_step_H={heavy_div.get('first_step_entropy', float('nan')):.3f}",
              flush=True)

        # Per-head analysis
        per_head = []
        for l in range(Ln):
            for h in range(Hn):
                A_m = A_lh_acc[gname][:, l, h].astype(np.float32)  # (M, nb, nb)
                B_m = np.transpose(A_m, (0, 2, 1)).astype(np.float32)
                B_m[:, di, di] = 0.0
                sigmas = {}
                for m in range(M):
                    s, _, _ = generate_teacher_label(B_m[m], alpha_dep=alpha_dep)
                    sigmas[m] = s
                sigmas = np.array([sigmas[m] for m in range(M)])
                tau = float(np.mean([
                    t for t in [kendalltau(s, np.arange(nb))[0] for s in sigmas]
                    if not np.isnan(t)
                ]))
                div = teacher_diversity_stats(sigmas)
                per_head.append({
                    "layer": l, "head": h,
                    "tau_vs_l2r": tau,
                    "mean_pairwise_tau": div.get("mean_pairwise_tau", float("nan")),
                    "first_step_entropy": div.get("first_step_entropy", float("nan")),
                })

        per_head.sort(key=lambda d: abs(d["tau_vs_l2r"]), reverse=True)

        # Top-5 heads
        print(f"  top-5 heads by |τ_vs_l2r|:", flush=True)
        for d in per_head[:5]:
            print(f"    L{d['layer']}H{d['head']}: τ={d['tau_vs_l2r']:.4f}  "
                  f"pw_τ={d['mean_pairwise_tau']:.4f}  H1={d['first_step_entropy']:.3f}",
                  flush=True)

        per_granularity[gname] = {
            "N": nb, "BL": bl,
            "heavy": {
                "tau_vs_l2r": heavy_tau,
                "diversity": heavy_div,
            },
            "per_head_sorted": per_head,
        }

        # Build comparison rows for top heads
        for d in per_head[:3]:
            comparison_rows.append({
                "head": f"L{d['layer']}H{d['head']}",
                "granularity": gname,
                "tau_vs_l2r": d["tau_vs_l2r"],
            })

    # ── Cross-granularity comparison ──
    # For each unique head found in top-3 of any granularity, show tau across all
    all_top_heads = sorted(set(r["head"] for r in comparison_rows))
    print(f"\n─── Cross-granularity comparison (top heads) ───", flush=True)
    header = f"{'Head':>8}"
    for gname, nb, bl in GRANULARITIES:
        header += f"  {gname:>16}"
    print(header)
    print("-" * len(header))
    for head_name in all_top_heads:
        l_str, h_str = head_name[1:].split("H")
        l, h = int(l_str), int(h_str)
        row = f"{head_name:>8}"
        for gname, nb, bl in GRANULARITIES:
            # Find this head in the per_granularity results
            entry = next((d for d in per_granularity[gname]["per_head_sorted"]
                         if d["layer"] == l and d["head"] == h), None)
            if entry:
                row += f"  τ={entry['tau_vs_l2r']:+.4f}"
            else:
                row += f"  {'—':>16}"
        print(row)

    # ── Per-granularity: best head signal ──
    print(f"\n─── Best |τ| per granularity ───", flush=True)
    for gname, nb, bl in GRANULARITIES:
        best = per_granularity[gname]["per_head_sorted"][0]
        print(f"  {gname}: L{best['layer']}H{best['head']} |τ|={abs(best['tau_vs_l2r']):.4f}",
              flush=True)

    return {
        "config": {
            "M": M, "batch_size": batch_size, "seed": seed,
            "ckpt": str(ckpt_path), "L": Ln, "H": Hn, "alpha_dep": alpha_dep,
            "granularities": [(gname, nb, bl) for gname, nb, bl in GRANULARITIES],
        },
        "per_granularity": per_granularity,
        "cross_comparison": comparison_rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", default=None,
                   help="dir containing ckpt_step{N}.pt (default: clean_base_random_perm)")
    p.add_argument("--ckpt-step", type=int, default=5000)
    p.add_argument("--M", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", default="train")
    p.add_argument("--alpha-dep", type=float, default=0.5)
    p.add_argument("--fwd-batch", type=int, default=64)
    p.add_argument("--out", required=True, help="output JSON path")
    args = p.parse_args()

    if args.ckpt_dir:
        ckpt_dir = pathlib.Path(args.ckpt_dir)
    else:
        ckpt_dir = _ROOT / "probe_results" / "clean_base_random_perm"
    ckpt_path = ckpt_dir / f"ckpt_step{args.ckpt_step}.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"ckpt not found: {ckpt_path}")

    print(f"Loading ckpt: {ckpt_path}")
    print(f"M={args.M} batch_size={args.batch_size} total={args.M * args.batch_size} chunks")
    print(f"Granularities: {[(g, n, bl) for g, n, bl in GRANULARITIES]}")

    res = scan_multi_granularity(
        ckpt_path, args.M, args.batch_size, args.seed,
        device=args.device, split=args.split, alpha_dep=args.alpha_dep,
        fwd_batch=args.fwd_batch,
    )

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, default=lambda x: float(x) if isinstance(x, (np.floating,)) else str(x))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
