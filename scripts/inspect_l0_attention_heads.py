#!/usr/bin/env python3
"""Extract and visualise Layer 0 attention maps from an AO-GPT checkpoint.

Usage:
  python3 scripts/inspect_l0_attention_heads.py \
    --ckpt block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2/ckpt_step20000.pt \
    --device cuda:0
"""
from __future__ import annotations

import argparse, json, math, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "block_lo_arm_order_network"))

from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig
from training_utils import SEQ_LEN, N, BLOCK_LEN
from clean_training_protocol import expand_model_blocks_to_token_order

OUT_BASE = ROOT / "outputs" / "attention_l0_heads"


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str):
    dev = torch.device(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_args" in state:
        # Filter to only AOGPTConfig fields, discard extra keys like
        # block_order_block_len, order_impl, etc.
        valid_keys = {"n_layer", "n_head", "n_embd", "block_size", "bias",
                       "vocab_size", "dropout"}
        ma = {k: v for k, v in state["model_args"].items() if k in valid_keys}
        cfg = AOGPTConfig(**ma)
    elif "config" in state and isinstance(state["config"], dict):
        cfg = AOGPTConfig(**state["config"])
    else:
        cfg = AOGPTConfig()
    model = AOGPT(cfg).to(dev)
    # Handle torch.compile prefix (_orig_mod.)
    sd = state["model"] if "model" in state else state["model_state_dict"]
    fixed = {}
    for k, v in sd.items():
        fixed[k.replace("_orig_mod.", "")] = v
    model.load_state_dict(fixed, strict=False)
    model.eval()
    return model, dev


def token_to_block_B(attn_token: np.ndarray, block_size: int = 4,
                     strip_none: bool = True):
    """Aggregate token-level attention → block-level B matrix.

    B = A^T with zero diagonal (row = source block, col = target block).
    This is the same representation used by g_beta and CDL.

    attn_token: (H, T, T)  with T = 1 + block_size * N_blocks
    returns:    (H, N, N)  B matrices, diag=0
    """
    H, T, _ = attn_token.shape
    if strip_none and T == 1 + block_size * 64:
        attn_content = attn_token[:, 1:, 1:]  # (H, 256, 256)
        T = attn_content.shape[1]
    else:
        attn_content = attn_token
    N_blk = T // block_size
    # Aggregate token→block: mean over token positions
    A_block = attn_content.reshape(H, N_blk, block_size, N_blk, block_size)
    A_block = A_block.mean(axis=(2, 4))  # (H, N, N)
    # B = A^T: row=source (query attended FROM), col=target (key attended TO)
    B_block = A_block.transpose(0, 2, 1).copy()
    for h in range(H):
        np.fill_diagonal(B_block[h], 0.0)
    return B_block


def head_stats(attn: np.ndarray) -> dict:
    """Compute per-head summary statistics.  attn: (H, N, N)."""
    H = attn.shape[0]
    stats = {}
    for h in range(H):
        a = attn[h]
        # entropy per row (key-attended distribution per query block)
        row_ent = -np.sum(a * np.log(a.clip(1e-12)), axis=-1)  # (N,)
        stats[f"H{h:02d}"] = {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "max": float(a.max()),
            "min": float(a.min()),
            "sparsity_lt_001": float((a < 0.01).mean()),  # fraction < 1%
            "sparsity_lt_0001": float((a < 0.001).mean()),
            "row_entropy_mean": float(row_ent.mean()),
            "row_entropy_std": float(row_ent.std()),
            "diagonal_mean": float(np.diag(a).mean()),  # self-attention weight
            "triu_mean": float(np.triu(a, k=1).mean()),  # future-looking (causal)
        }
    return stats


# ──────────────────────────────────────────────────────────────────────────
# plotting
# ──────────────────────────────────────────────────────────────────────────

def plot_head_heatmap(attn: np.ndarray, out_path: pathlib.Path,
                      title: str, log_scale: bool = False):
    """Save a single-head block attention heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    data = np.log1p(attn) if log_scale else attn
    im = ax.imshow(data, aspect="auto", cmap="viridis", origin="upper")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("key block")
    ax.set_ylabel("query block")
    plt.colorbar(im, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_grid(attns: list[np.ndarray], heads: list[int],
              out_path: pathlib.Path, layer: int = 0):
    """Grid of all heads' block attention."""
    n = len(heads)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.6))
    axes = np.atleast_2d(axes)
    for i, h in enumerate(heads):
        ax = axes[i // cols, i % cols]
        im = ax.imshow(attns[i], aspect="auto", cmap="viridis", origin="upper")
        ax.set_title(f"L{layer} H{h}", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
    # hide unused axes
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].set_visible(False)
    fig.suptitle(f"Layer {layer} — all heads B (A^T, diag=0)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="path to .pt checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default=str(OUT_BASE))
    p.add_argument("--batch", type=int, default=1, help="number of samples")
    p.add_argument("--layer", type=int, default=0, help="layer to inspect")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── load model ──
    print(f"Loading {args.ckpt} ...")
    model, dev = load_model(args.ckpt, args.device)
    n_layers = len(model.transformer.h)
    n_heads = model.transformer.h[0].attn.n_head
    print(f"  layers={n_layers}  heads={n_heads}  d={model.config.n_embd}")

    # ── create a dummy batch (random tokens) ──
    torch.manual_seed(args.seed)
    idx = torch.randint(0, 1000, (args.batch, SEQ_LEN), device=dev)
    # random probe order (model-frame)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    rand_blocks = torch.randperm(N, generator=gen, device="cpu")
    probe = expand_model_blocks_to_token_order(
        rand_blocks.unsqueeze(0), BLOCK_LEN,
    ).to(dev)  # (1, 256)

    # ── forward with attentions ──
    print(f"Running forward (B={args.batch}, T={SEQ_LEN}) ...")
    with torch.no_grad():
        logits, loss, attn_list = model.forward_fn(
            idx, probe, return_attentions=True,
        )
    print(f"  returned {len(attn_list)} layer attention tensors")
    print(f"  each shape: {attn_list[0].shape}")  # (B, nh, 257, 257)

    # ── extract layer L attention ──
    attn_l = attn_list[args.layer]  # (B, nh, T, T)
    B, nh, T, _ = attn_l.shape
    print(f"\nLayer {args.layer}:")
    print(f"  T (token dim) = {T}")
    print(f"  nh = {nh}")
    print(f"  block_size = {BLOCK_LEN}")
    print(f"  N (blocks)  = {N}")

    # Use first sample
    attn_sample = attn_l[0].cpu().numpy()  # (nh, T, T)

    # ── Coordinate frames ──
    # Reveal order: attention map row/col i = reveal step i
    #   Block i in reveal = model-frame block rand_blocks[i]
    # Model order: row/col = model-frame block index (0..63)
    # Physical order: row/col = physical block index (inverse of block_perm)
    reveal_blocks = rand_blocks.cpu().numpy()  # (64,)  model-frame blocks in reveal order
    inv_reveal = np.argsort(reveal_blocks)       # model-frame block → reveal position

    # Load physical mappings from checkpoint
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    phys_to_model = None
    if "data_permutation" in state:
        phys_to_model = state["data_permutation"].get("block_perm_phys_to_model")
    elif "clean_protocol" in state:
        phys_to_model = state["clean_protocol"].get("block_perm_phys_to_model")
    if phys_to_model is not None:
        phys_to_model = phys_to_model.cpu().numpy() if hasattr(phys_to_model, 'cpu') else np.asarray(phys_to_model)
        print(f"  loaded block_perm_phys_to_model: {phys_to_model.shape}")

    # ── block-level aggregation (in reveal order) ──
    attn_reveal = token_to_block_B(attn_sample, block_size=BLOCK_LEN)  # (nh, 64, 64)
    N_blk = attn_reveal.shape[1]
    print(f"  block attn shape: {attn_reveal.shape}  (expected N={N})")

    # Remap to model-frame order
    attn_model = attn_reveal[:, :, :][:, inv_reveal, :][:, :, inv_reveal]  # (nh, 64, 64)

    # Remap to physical order if available
    attn_phys = None
    if phys_to_model is not None:
        # phys_to_model[i] = model-frame block index for physical block i
        # model_pos = phys_to_model[phys_i] → phys_i = inv_phys_to_model[model_pos]
        inv_phys = np.argsort(phys_to_model)
        attn_phys = attn_model[:, inv_phys, :][:, :, inv_phys]

    # ── per-head stats (on model-frame attention) ──
    stats = head_stats(attn_model)
    print(f"\nPer-head block attention statistics:")
    for h in range(nh):
        s = stats[f"H{h:02d}"]
        print(f"  H{h:02d}  mean={s['mean']:.5f}  max={s['max']:.5f}  "
              f"diag={s['diagonal_mean']:.5f}  row_ent={s['row_entropy_mean']:.3f}  "
              f"sparsity<1%={s['sparsity_lt_001']:.3f}")

    # Save stats JSON
    stats_path = out / "summary.json"
    stats_path.write_text(json.dumps({
        "ckpt": args.ckpt,
        "layer": args.layer,
        "n_heads": nh,
        "T": T,
        "N": N_blk,
        "block_size": BLOCK_LEN,
        "reveal_blocks": reveal_blocks.tolist(),
        "has_physical": phys_to_model is not None,
        "heads": stats,
    }, indent=2))
    print(f"\nSaved {stats_path}")

    # ── plot: three coordinate frames per head ──
    frames = [("model", attn_model)]
    if attn_phys is not None:
        frames.append(("phys", attn_phys))

    print("\nPlotting per-head block attention (model + physical frames) ...")
    for h in range(nh):
        for tag, attn in frames:
            name = f"l{args.layer}_head{h:02d}_block_{tag}.png"
            plot_head_heatmap(attn[h], out / name,
                              f"Layer {args.layer} Head {h} B (A^T, diag=0) ({tag} order)")
            name_log = f"l{args.layer}_head{h:02d}_block_{tag}_log1p.png"
            plot_head_heatmap(attn[h], out / name_log,
                              f"Layer {args.layer} Head {h} B (A^T, diag=0) ({tag}, log1p)",
                              log_scale=True)

    # ── grid per frame ──
    print("Plotting grids ...")
    for tag, attn in frames:
        plot_grid([attn[h] for h in range(nh)], list(range(nh)),
                  out / f"l{args.layer}_heads_grid_block_{tag}.png", layer=args.layer)

    print(f"\nAll outputs → {out}/")
    print("Done.")


if __name__ == "__main__":
    main()
