"""Hook L0H5 attention from existing checkpoints and plot the signal — NO TRAINING.

For each given ckpt: forward M samples (batched) with random reveal orders,
take ONLY head (layer 0, head 5), reveal->physical remap, 256->64 block-agg,
average over samples -> a single 64x64 batch-mean attention map A^(0,5).
Plot the heatmaps side by side so you can eyeball whether the L0H5 order signal
changes across checkpoints (training step / arm = the "model parameter" axis).

Each panel title also reports tau_vs_l2r (the scalar order-signal strength),
computed from the same teacher CDL pipeline that defined L0H5, so you get both
the picture and the number.

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=block_lo_arm_order_network \
    python scripts/plot_l0h5_attention_across_ckpts.py --M 200 --seed 0
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_clean_aogpt import (
    expand_model_blocks_to_token_order, N, SEQ_LEN, BLOCK_LEN,
    build_model, CleanPermutation, phys_to_model_idx_clean,
)
from training_utils import load_train_chunks
from neural_readout.teacher_labels import generate_teacher_label
from batch_readout.eval_metrics import kendall_tau_batch

LAYER, HEAD = 0, 5

# Pure random-order baseline trajectory (clean_base_random_perm) — existing on
# disk, NO training. Same arch (4L8H384) as where L0H5 was found. This is the
# clean random regime across training steps.
_CB = "block_lo_arm_order_network/probe_results/clean_base_random_perm"
DEFAULT_CKPTS = [
    ("random@0",   f"{_CB}/ckpt_step0.pt"),
    ("random@1k",  f"{_CB}/ckpt_step1000.pt"),
    ("random@5k",  f"{_CB}/ckpt_step5000.pt"),
    ("random@10k", f"{_CB}/ckpt_step10000.pt"),
    ("random@20k", f"{_CB}/ckpt_step20000.pt"),
    ("random@30k", f"{_CB}/ckpt_step30000.pt"),
    ("random@40k", f"{_CB}/ckpt_step40000.pt"),
    ("random@50k", f"{_CB}/ckpt_step50000.pt"),
    ("random@60k", f"{_CB}/ckpt_step60000.pt"),
]


def _block_agg(x):
    shp = x.shape[:-2]
    return x.reshape(*shp, N, BLOCK_LEN, N, BLOCK_LEN).mean(axis=(-3, -1))


def load_model_and_chunks_cached(path, idx_phys, M, seed, device):
    """Like extract_b._load_model_and_chunks but reuses ALREADY-tokenized idx_phys
    (so we tokenize wikitext ONCE for all ckpts instead of ~8min each)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    protocol = ckpt["clean_protocol"]
    split_indices = np.asarray(protocol["train_indices"], dtype=np.int64)
    model_args = dict(ckpt["model_args"]); model_args["block_size"] = SEQ_LEN
    clean_perm = CleanPermutation(
        block_perm_phys_to_model=torch.tensor(protocol["block_perm_phys_to_model"], dtype=torch.long),
        inv_perm_model_to_phys=torch.tensor(protocol["inv_perm_model_to_phys"], dtype=torch.long),
    )
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    model = build_model(model_args, dev, compile_model=False)
    sd = ckpt.get("model") or ckpt.get("model_state_dict")
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    model.to(dev); model.eval()

    idx_model = phys_to_model_idx_clean(idx_phys, clean_perm)
    idx_split = idx_model[split_indices]
    rng = np.random.RandomState(seed)
    ci = rng.choice(len(idx_split), size=M, replace=False).astype(np.int64); ci.sort()
    return model, idx_split[ci], clean_perm, dev


@torch.no_grad()
def l0h5_batchmean_A(model, chunks, clean_perm, device, total, batch_size, seed, gpu_batch):
    """-> (A_mean 64x64 physical, B-batch list for tau).  A = head-(0,5) attention, batch-mean."""
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    M = total // batch_size
    di = np.arange(N)
    A_sum = np.zeros((M, N, N), dtype=np.float64)   # accumulate A (for B=A^T tau) per graph
    model.eval()
    for start in range(0, total, gpu_batch):
        bs = min(gpu_batch, total - start)
        tokens = chunks[start:start + bs].to(device)
        orders = np.empty((bs, SEQ_LEN), dtype=np.int64)
        phys_tok = np.empty((bs, SEQ_LEN), dtype=np.int64)
        for j in range(bs):
            i = start + j
            gen = torch.Generator(device="cpu"); gen.manual_seed(int(seed) + int(i))
            rb = torch.randperm(N, generator=gen, device="cpu")
            to_row = expand_model_blocks_to_token_order(rb.unsqueeze(0), BLOCK_LEN)[0].numpy()
            orders[j] = to_row
            mb = to_row // BLOCK_LEN
            phys_tok[j] = inv_perm[mb] * BLOCK_LEN + (to_row % BLOCK_LEN)
        _, _, attn_list = model.forward_fn(
            tokens, torch.from_numpy(orders).to(device), return_attentions=True
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        head = attn_list[LAYER][:, HEAD].cpu().numpy()   # (bs, 257, 257)
        for j in range(bs):
            i = start + j; m = i // batch_size; pj = phys_tok[j]
            content = head[j][1:, 1:]                     # (256,256) reveal coords
            phys = np.empty_like(content)
            phys[pj[:, None], pj[None, :]] = content
            A = _block_agg(phys)                          # (64,64) physical, A[query,key]
            A_sum[m] += A
    A_batch = A_sum / batch_size                          # (M, 64, 64)
    A_mean = A_batch.mean(axis=0)                         # (64,64) overall mean for the picture
    # tau_vs_l2r via B = A^T -> teacher CDL (same as the L0H5 finding)
    B = np.swapaxes(A_batch, -1, -2).astype(np.float32)
    for m in range(M):
        np.fill_diagonal(B[m], 0.0)
    sig = np.stack([generate_teacher_label(B[m], alpha_dep=0.5)[0] for m in range(M)])
    l2r = np.tile(np.arange(N), (M, 1))
    tau = kendall_tau_batch(sig, l2r)
    return A_mean, tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=200, help="number of samples to batch-mean")
    ap.add_argument("--batch_size", type=int, default=1, help="samples per batch-mean graph (1=single big mean)")
    ap.add_argument("--gpu_batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="block_lo_arm_order_network/batch_readout/logs/l0h5_evo/l0h5_attention_grid.png")
    ap.add_argument("--ckpts", nargs="*", default=None,
                    help="optional 'label=path' overrides; default = built-in existing-ckpt set")
    args = ap.parse_args()

    if args.ckpts:
        ckpts = [tuple(s.split("=", 1)) for s in args.ckpts]
    else:
        ckpts = DEFAULT_CKPTS

    # batch_size=1 means each sample is its own graph; for one big mean picture use bs=M.
    bs = args.batch_size if args.batch_size > 1 else args.M
    total = args.M

    print("tokenizing wikitext once (shared across all ckpts) ...", flush=True)
    idx_phys = load_train_chunks(n_chunks=None)

    results = []
    for label, path in ckpts:
        if not pathlib.Path(path).exists():
            print(f"[skip] {label}: missing {path}")
            continue
        model, chunks, clean_perm, dev = load_model_and_chunks_cached(path, idx_phys, total, args.seed, args.device)
        A_mean, tau = l0h5_batchmean_A(model, chunks, clean_perm, dev, total, bs, args.seed, args.gpu_batch)
        print(f"{label:>12}  tau_vs_l2r={tau:+.3f}")
        results.append((label, A_mean, tau))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if not results:
        print("no checkpoints found"); return

    n = len(results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)
    vmax = max(np.percentile(A, 99) for _, A, _ in results)
    for k, (label, A, tau) in enumerate(results):
        ax = axes[k // cols][k % cols]
        im = ax.imshow(A, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"{label}\nL0H5  tau_vs_l2r={tau:+.3f}", fontsize=10)
        ax.set_xlabel("key block (phys)"); ax.set_ylabel("query block (phys)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("L0H5 batch-mean attention (physical coords) across checkpoints", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
