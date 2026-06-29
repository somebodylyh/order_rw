#!/usr/bin/env python3
"""Extract L0 B = A^T (diag=0) block graphs in reveal / model / physical frames.

B is the same representation used by g_beta and CDL:
  B[i,j] = attention FROM block i TO block j, diagonal zeroed.

Usage:
  python3 scripts/inspect_l0_B_heads.py \
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


# ── helpers ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str):
    dev = torch.device(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    valid = {"n_layer", "n_head", "n_embd", "block_size", "bias",
             "vocab_size", "dropout"}
    if "model_args" in state:
        ma = {k: v for k, v in state["model_args"].items() if k in valid}
        cfg = AOGPTConfig(**ma)
    else:
        cfg = AOGPTConfig()
    model = AOGPT(cfg).to(dev)
    sd = state.get("model", state.get("model_state_dict", {}))
    model.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in sd.items()}, strict=False,
    )
    model.eval()
    return model, dev, state


def token_to_B(attn_token: np.ndarray, block_size: int = 4):
    """Token attention → block B = A^T with diag=0.

    attn_token: (H, T, T)  with T = 1 + block_size * N_blocks
    returns:    (H, N, N)  B matrix, diag=0, row=source col=target
    """
    H, T, _ = attn_token.shape
    # Strip None token (position 0)
    content = attn_token[:, 1:, 1:]  # (H, 256, 256)
    N_blk = content.shape[1] // block_size
    A = content.reshape(H, N_blk, block_size, N_blk, block_size).mean(axis=(2, 4))
    # B = A^T, then zero diag
    B = A.transpose(0, 2, 1).copy()
    for h in range(H):
        np.fill_diagonal(B[h], 0.0)
    return B


def head_stats(B: np.ndarray) -> dict:
    """Per-head B statistics.  B: (H, N, N), diag=0."""
    H = B.shape[0]
    stats = {}
    for h in range(H):
        b = B[h]
        total = b.sum()
        off = b[~np.eye(b.shape[0], dtype=bool)]
        row_ent = -np.sum(b * np.log(b.clip(1e-12)), axis=-1)
        stats[f"H{h:02d}"] = {
            "mean": float(b.mean()),
            "max_off_diag": float(off.max()),
            "sparsity_lt_1e-4": float((b < 1e-4).mean()),
            "row_entropy_mean": float(row_ent.mean()),
            "source_concentration": float(
                (b.max(axis=1) / b.sum(axis=1).clip(1e-12)).mean()
            ),
        }
    return stats


# ── plotting ─────────────────────────────────────────────────────────────

def plot_heatmap(B: np.ndarray, out_path: pathlib.Path,
                 title: str, log_scale: bool = False):
    fig, ax = plt.subplots(figsize=(7, 6))
    if log_scale:
        data = np.log1p(B * 1000)
    else:
        # Clip at 95th percentile of off-diagonal to avoid one bright spot
        off = B[~np.eye(B.shape[0], dtype=bool)]
        vmax = np.percentile(off, 95)
        data = B
    im = ax.imshow(data, aspect="auto", cmap="viridis", origin="upper",
                   vmin=0, vmax=vmax if not log_scale else None)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("target block"); ax.set_ylabel("source block")
    plt.colorbar(im, ax=ax, shrink=0.82)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_grid(Bs: list[np.ndarray], heads: list[int],
              out_path: pathlib.Path, layer: int = 0, tag: str = ""):
    n = len(heads)
    ncols, nrows = 4, math.ceil(n / 4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.6))
    axes = np.atleast_2d(axes)
    for i, h in enumerate(heads):
        ax = axes[i // ncols, i % ncols]
        im = ax.imshow(Bs[i], aspect="auto", cmap="viridis", origin="upper")
        ax.set_title(f"L{layer} H{h}", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    fig.suptitle(f"Layer {layer} B (A^T, diag=0) — {tag} order", fontsize=13)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default=str(OUT_BASE))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layer", type=int, default=0)
    args = p.parse_args()
    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ── load ──
    print(f"Loading {args.ckpt} ...")
    model, dev, state = load_model(args.ckpt, args.device)
    n_heads = model.config.n_head
    print(f"  layers={len(model.transformer.h)}  heads={n_heads}  d={model.config.n_embd}")

    # ── load real text data ──
    import mmap
    data_path = pathlib.Path("/home/admin/ych/nanogpt-learned-order/data/wikitext103/val.bin")
    if not data_path.exists():
        data_path = pathlib.Path("/home/admin/ych/nanogpt-learned-order/data/wikitext103/train.bin")
    tokens = np.memmap(str(data_path), dtype=np.uint16, mode="r")
    torch.manual_seed(args.seed)
    start = torch.randint(0, len(tokens) - SEQ_LEN - 1, (1,)).item()
    idx = torch.from_numpy(tokens[start:start + SEQ_LEN].astype(np.int64)).unsqueeze(0).to(dev)
    print(f"  data: {data_path.name}[{start}:{start+SEQ_LEN}]")
    gen = torch.Generator(device="cpu"); gen.manual_seed(args.seed)
    rand_blocks = torch.randperm(N, generator=gen, device="cpu")
    probe = expand_model_blocks_to_token_order(
        rand_blocks.unsqueeze(0), BLOCK_LEN,
    ).to(dev)

    print(f"Forward (B=1, T={SEQ_LEN}) ...")
    with torch.no_grad():
        _, _, attn_list = model.forward_fn(idx, probe, return_attentions=True)
    print(f"  attn_list[{args.layer}] shape: {attn_list[args.layer].shape}")

    # ── B in reveal order ──
    attn_l0 = attn_list[args.layer][0].cpu().numpy()  # (H, 257, 257)
    B_reveal = token_to_B(attn_l0, BLOCK_LEN)  # (H, 64, 64), diag=0
    print(f"  B_reveal: {B_reveal.shape}  diag={B_reveal[0].diagonal().sum():.6f}")

    # ── coordinate remapping ──
    reveal_blocks = rand_blocks.cpu().numpy()  # (64,) model-block-id in reveal order
    inv_reveal = np.argsort(reveal_blocks)      # model→reveal position

    B_model = B_reveal[:, inv_reveal, :][:, :, inv_reveal]

    pm = state.get("data_permutation", state.get("clean_protocol", {}))
    phys_to_model = pm.get("block_perm_phys_to_model", None)
    B_phys = None
    if phys_to_model is not None:
        phys_to_model = (phys_to_model.cpu().numpy() if hasattr(phys_to_model, "cpu")
                         else np.asarray(phys_to_model))
        inv_phys = np.argsort(phys_to_model)
        B_phys = B_model[:, inv_phys, :][:, :, inv_phys]
        print(f"  B_phys:   {B_phys.shape}")

    # ── stats ──
    print("\nPer-head B statistics (model frame):")
    stats = head_stats(B_model)
    for h in range(n_heads):
        s = stats[f"H{h:02d}"]
        print(f"  H{h:02d}  mean={s['mean']:.6f}  off_max={s['max_off_diag']:.4f}  "
              f"sparsity={s['sparsity_lt_1e-4']:.3f}  row_ent={s['row_entropy_mean']:.3f}  "
              f"src_conc={s['source_concentration']:.3f}")

    # ── model vs phys difference ──
    if B_phys is not None:
        print("\nModel vs physical B difference:")
        for h in range(n_heads):
            rel_diff = np.abs(B_model[h] - B_phys[h]).sum() / (B_model[h].sum() + 1e-12)
            print(f"  H{h:02d}  relative diff: {rel_diff:.3f}")

    # ── save stats ──
    (out / "summary.json").write_text(json.dumps({
        "ckpt": args.ckpt, "layer": args.layer, "n_heads": n_heads,
        "N": N, "block_size": BLOCK_LEN, "heads": stats,
    }, indent=2))
    print(f"\nSaved summary.json")

    # ── plot ──
    frames = [("model", B_model)]
    if B_phys is not None:
        frames.append(("phys", B_phys))

    print("Plotting per-head B matrices ...")
    for h in range(n_heads):
        for tag, Bf in frames:
            name = f"l{args.layer}_H{h:02d}_B_{tag}.png"
            plot_heatmap(Bf[h], out / name,
                         f"L{args.layer} H{h}  B={{{{A^T, diag=0}}}} ({tag})")
            name_log = f"l{args.layer}_H{h:02d}_B_{tag}_log1p.png"
            plot_heatmap(Bf[h], out / name_log,
                         f"L{args.layer} H{h}  B ({tag}, log1p)", log_scale=True)

    # ── row-normalised contrast (B / row_mean) to see relative structure ──
    print("Plotting row-contrast ...")
    for tag, Bf in frames:
        B_contrast = np.empty_like(Bf)
        for h in range(n_heads):
            row_mean = Bf[h].mean(axis=1, keepdims=True).clip(1e-12)
            B_contrast[h] = Bf[h] / row_mean
        plot_grid([B_contrast[h] for h in range(n_heads)], list(range(n_heads)),
                  out / f"l{args.layer}_B_grid_{tag}_contrast.png",
                  layer=args.layer, tag=f"{tag} (row-norm)")

    print("Plotting grids ...")
    for tag, Bf in frames:
        plot_grid([Bf[h] for h in range(n_heads)], list(range(n_heads)),
                  out / f"l{args.layer}_B_grid_{tag}.png",
                  layer=args.layer, tag=tag)

    print(f"\nDone → {out}/")


if __name__ == "__main__":
    main()
