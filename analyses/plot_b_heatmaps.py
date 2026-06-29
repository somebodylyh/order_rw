"""Sanity-check visualisation: B heatmap for best CDL head of each model.

Saves B matrices as heatmaps so we can verify:
  - random L0H7: near-diagonal / forward dependency?
  - shuffled L1H5: random or aligned with wrong layout?
  - ori-L2R L1H3: not just a position-sink artifact?

Loads pre-computed JSON from attention_diagnostic_20260609.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AOGPT_DIR = os.path.join(_SCRIPT_DIR, "..", "block_lo_arm_order_network")
sys.path.insert(0, _AOGPT_DIR)

from per_head_order_scan import extract_per_head_and_heavy_A
from neural_readout.extract_b import _load_model_and_chunks

N_BLOCKS = 64
OUT_DIR = os.path.join(_SCRIPT_DIR, "attention_diagnostic_20260609")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(OUT_DIR, "cdl_tau_diagnostic.json")) as f:
    tau_data = json.load(f)

BASE = "/home/admin/lyuyuhuan/order_lyu/block_lo_arm_order_network/probe_results"
MODELS = {
    "shuffled-L2R":   f"{BASE}/shuffled_l2r_continuous_jun05/ckpt_step50000.pt",
    "random-order":   f"{BASE}/random_baseline_continuous_jun08_seed2/ckpt_step50000.pt",
    "ori-L2R":        f"{BASE}/l2r_continuous_jun05/ckpt_step50000.pt",
}
TITLES = {
    "shuffled-L2R": "Shuffled-L2R — Fixed wrong order",
    "random-order": "Random-order — Spontaneous L2R emerges",
    "ori-L2R": "Ori-L2R — Natural order",
}


def _A_to_B(A):
    B = np.asarray(A.T, dtype=np.float64).copy()
    np.fill_diagonal(B, 0.0)
    return B


def extract_one_head_B(ckpt_path, head, device="cuda:0", n_samples=200):
    """Extract single-head batch-mean B matrix."""
    import torch
    model, chunks, clean_perm, dev, chunk_index = _load_model_and_chunks(
        ckpt_path=ckpt_path, M=n_samples, seed=42, device=device, split="train")
    A_lh, _ = extract_per_head_and_heavy_A(
        model, chunks, clean_perm, dev, seed=42, fwd_batch=64, none_mode="b0")
    l, h = head
    A_mean = A_lh[:, l, h].mean(axis=0)
    B = _A_to_B(A_mean)
    del model, A_lh
    if device != "cpu":
        torch.cuda.empty_cache()
    return B


def main():
    device = "cuda:0"
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    for ax, name in zip(axes, ["shuffled-L2R", "random-order", "ori-L2R"]):
        d = tau_data[name]
        bl, bh = d["best_head"]
        print(f"  {name}: extracting L{bl}H{bh} ...", flush=True)
        B = extract_one_head_B(MODELS[name], (bl, bh), device=device)

        im = ax.imshow(B, cmap="YlOrRd", aspect="auto")
        ax.set_title(f"{TITLES[name]}\nbest head L{bl}H{bh}  τ={d['best_tau']:+.3f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Target block v", fontsize=9)
        ax.set_ylabel("Source block u", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Save raw B
        np.save(os.path.join(OUT_DIR, f"B_heatmap_{name.lower().replace('-','_')}_L{bl}H{bh}.npy"), B)

    fig.suptitle("Attention B Matrix (best CDL head) — Three Training Regimes", fontsize=13, fontweight="bold")
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "b_heatmap_best_heads.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {png_path}")

    # Also individual larger heatmaps
    for name in ["shuffled-L2R", "random-order", "ori-L2R"]:
        d = tau_data[name]
        bl, bh = d["best_head"]
        B = np.load(os.path.join(OUT_DIR, f"B_heatmap_{name.lower().replace('-','_')}_L{bl}H{bh}.npy"))
        fig2, ax2 = plt.subplots(figsize=(8, 7))
        im2 = ax2.imshow(B, cmap="YlOrRd", aspect="auto")
        ax2.set_title(f"{TITLES[name]}\nL{bl}H{bh}  τ={d['best_tau']:+.3f}", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Target block v"); ax2.set_ylabel("Source block u")
        fig2.colorbar(im2, ax=ax2)
        slug = name.lower().replace("-", "_")
        fig2.savefig(os.path.join(OUT_DIR, f"b_heatmap_{slug}_L{bl}H{bh}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig2)

    # Summary
    print("\nB heatmap stats (should show near-diagonal structure for random/ori, noise for shuffled):")
    for name in ["shuffled-L2R", "random-order", "ori-L2R"]:
        d = tau_data[name]
        bl, bh = d["best_head"]
        B = np.load(os.path.join(OUT_DIR, f"B_heatmap_{name.lower().replace('-','_')}_L{bl}H{bh}.npy"))
        # Mean of upper-tri vs lower-tri (directionality check)
        mask = np.triu(np.ones_like(B), k=1).astype(bool)
        upper_mean = B[mask].mean()
        lower_mean = B[mask.T].mean()
        diag1_mean = np.diag(B, k=1).mean()  # immediate forward edge
        print(f"  {name:20s} L{bl}H{bh}: upper_mean={upper_mean:.4f}  lower_mean={lower_mean:.4f}  "
              f"diag+1_mean={diag1_mean:.4f}  upper/lower={upper_mean/(lower_mean+1e-9):.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
