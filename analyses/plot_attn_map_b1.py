#!/usr/bin/env python3
"""B1 block-level B — L0H2, model-order vs physical-order.

Replicates L0H2_B_model_vs_phys.png parameters, with extraction mode changed B0→B1.

Left:  B in model order (模型收到的打乱后数据顺序)
Right: B in physical order (原始未打乱顺序)
B = A^T, diagonal zeroed.

Colormap: black → purple → bright green.
"""
import argparse, json, os, pathlib, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

_ROOT = pathlib.Path(__file__).resolve().parent.parent / "block_lo_arm_order_network"
sys.path.insert(0, str(_ROOT))

from training_utils import SEQ_LEN, N, BLOCK_LEN
from neural_readout.extract_b import _load_model_and_chunks
from clean_training_protocol import expand_model_blocks_to_token_order
from per_head_order_scan import _attn_to_A_block_b1_vec, _attn_to_A_block_predictor_vec

CMAP = LinearSegmentedColormap.from_list("black_purple_green_div", [
    (0.00, "#4a148c"),
    (0.20, "#1a0a2e"),
    (0.50, "#0a0a0a"),
    (0.72, "#33691e"),
    (0.88, "#76ff03"),
    (1.00, "#00e676"),
])

LAYER, HEAD = 0, 2  # L0H2 — match reference figure
DEFAULT_CKPT = (
    "/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/probe_results"
    "/random_baseline_continuous_jun08_seed2/ckpt_step30000.pt"
)


def A_to_B(A):
    B = np.asarray(A.T, dtype=np.float64).copy()
    np.fill_diagonal(B, 0.0)
    return B


@torch.no_grad()
def extract_one_sample(model, chunks, clean_perm, dev, seed, layer, sample_idx):
    """Return raw attention (H,T+1,T+1), reveal_blocks, inv_perm for one sample."""
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    n_chunks = len(chunks)
    model.eval()
    rng = np.random.default_rng(seed)
    chunk_indices = rng.choice(n_chunks, size=sample_idx + 1, replace=False)
    ci = chunk_indices[sample_idx]

    tokens = chunks[ci:ci+1].to(dev)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + int(ci))
    rand_blocks = torch.randperm(N, generator=gen, device="cpu")
    token_order = expand_model_blocks_to_token_order(
        rand_blocks.unsqueeze(0), BLOCK_LEN).to(dev)

    _, _, attn_list = model.forward_fn(tokens, token_order, return_attentions=True)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    attn_stack = torch.stack(attn_list).squeeze(1).detach().cpu().numpy()
    reveal_blocks = rand_blocks.cpu().numpy()
    reveal_tokens = token_order[0].cpu().numpy()
    return attn_stack, reveal_blocks, reveal_tokens, inv_perm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", default="train")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir or os.path.join(
        os.path.dirname(__file__), "figures", "attn_map_b1"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.ckpt}")
    t0 = time.time()
    model, chunks, clean_perm, dev, _ci = _load_model_and_chunks(
        args.ckpt, 1, args.seed, args.device, args.split)
    Ln, Hn = model.config.n_layer, model.config.n_head
    print(f"  Model: {Ln}L × {Hn}H  |  head: L{LAYER}H{HEAD}  |  mode: B1 (predictor)")

    attn_stack, reveal_blocks, reveal_tokens, inv_perm = extract_one_sample(
        model, chunks, clean_perm, dev, args.seed, LAYER, sample_idx=0)

    # B1 without physical remap (predictor frame, no remap)
    A_b1_raw = _attn_to_A_block_predictor_vec(
        attn_stack[LAYER, HEAD], np.arange(SEQ_LEN), inv_perm)
    # B1 with physical remap (predictor frame + segment-mean remap)
    A_b1_phys = _attn_to_A_block_b1_vec(
        attn_stack[LAYER, HEAD], reveal_tokens, inv_perm)

    B_b1_raw = A_to_B(A_b1_raw)
    B_b1_phys = A_to_B(A_b1_phys)

    del model
    if args.device != "cpu":
        torch.cuda.empty_cache()

    print(f"  Done in {time.time() - t0:.1f}s")

    vlim = max(abs(B_b1_raw).max(), abs(B_b1_phys).max()) * 1.05

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 6), facecolor="#0a0a0a")

    im0 = ax0.imshow(B_b1_raw, cmap=CMAP, aspect="equal", vmin=-vlim, vmax=vlim,
                     interpolation="nearest")
    ax0.set_title("B1 — predictor frame (no remap)", fontsize=13, fontweight="bold", color="#e0e0e0")
    ax0.set_xlabel("block j", color="#999999", fontsize=11)
    ax0.set_ylabel("block i", color="#999999", fontsize=11)
    ax0.tick_params(color="#555555", labelcolor="#999999")
    ax0.set_facecolor("#0a0a0a")

    im1 = ax1.imshow(B_b1_phys, cmap=CMAP, aspect="equal", vmin=-vlim, vmax=vlim,
                     interpolation="nearest")
    ax1.set_title("B1 — predictor frame + physical remap", fontsize=13, fontweight="bold", color="#e0e0e0")
    ax1.set_xlabel("physical block j", color="#999999", fontsize=11)
    ax1.set_ylabel("physical block i", color="#999999", fontsize=11)
    ax1.tick_params(color="#555555", labelcolor="#999999")
    ax1.set_facecolor("#0a0a0a")

    cbar = fig.colorbar(im0, ax=[ax0, ax1], fraction=0.018, pad=0.02)
    cbar.set_label("B = A^T", fontsize=11, color="#cccccc")
    cbar.ax.yaxis.set_tick_params(color="#cccccc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#cccccc")

    step = pathlib.Path(args.ckpt).stem.replace("ckpt_step", "")
    fig.suptitle(f"B1  L{LAYER}H{HEAD}  |  seed{args.seed}  step{step}  |  black→purple→green",
                 fontsize=14, fontweight="bold", color="#e0e0e0", y=1.02)
    png = out_dir / f"b1_L{LAYER}_H{HEAD}_step{step}_model_vs_phys.png"
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)
    print(f"Saved: {png}")


if __name__ == "__main__":
    main()
