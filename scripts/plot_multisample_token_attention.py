#!/usr/bin/env python3
"""Extract multi-sample averaged token-level attention and plot.

Pipeline (matching collaborator's method):
  1. Model processes data in random block reveal order
  2. Extract token-level attention for target (layer, head) in reveal frame
  3. Remap from reveal-order coordinates back to current-frame coordinates
     (model input order after fixed data perm, before random reveal)
  4. Average over multiple probe samples with shared reveal order

Coordinate convention:
  bp = model_to_phys   = ckpt["data_permutation"]["block_perm"]
  ip = phys_to_model   = ckpt["data_permutation"]["inverse_block_perm"]
  bp[m] = p            model position m 里放的是 physical block p
  ip[p] = m            physical block p 被放到了 model position m

  x_model = x_phys[model_to_phys]            # physical data -> model input
  B_phys  = B_model[phys_to_model][:, phys_to_model]  # model-frame matrix -> physical-frame

Usage:
  python scripts/plot_multisample_token_attention.py --ckpt ... --M 200 --layer 0 --head 0 --share-reveal-order
"""

import argparse, json, os, pathlib, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "block_lo_arm_order_network"
sys.path.insert(0, str(PKG))

from clean_training_protocol import expand_model_blocks_to_token_order
from training_utils import AOGPT, AOGPTConfig, BLOCK_LEN, N, SEQ_LEN, load_train_chunks


def _load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = dict(ckpt["model_args"])
    model_args["block_size"] = SEQ_LEN
    sig = list(AOGPTConfig.__init__.__code__.co_varnames)
    valid = {k: v for k, v in model_args.items() if k in sig}
    model = AOGPT(AOGPTConfig(**valid))
    model.crop_block_size(SEQ_LEN)
    sd = dict(ckpt["model"])
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd)
    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    model.to(dev).eval()
    return ckpt, model, dev


def _load_permutations(ckpt):
    """Return (model_to_phys, phys_to_model) as 1-D LongTensors.

    model_to_phys[m] = p   -> model position m 里放 physical block p
    phys_to_model[p] = m   -> physical block p 在 model position m
    """
    model_to_phys = torch.tensor(ckpt["data_permutation"]["block_perm"], dtype=torch.long)
    phys_to_model = torch.tensor(ckpt["data_permutation"]["inverse_block_perm"], dtype=torch.long)
    return model_to_phys, phys_to_model


def _phys_chunks_to_model(chunks_phys, model_to_phys):
    """Convert physical-order chunks to model-order chunks.

    x_model[m] = x_phys[model_to_phys[m]]
    """
    token_gather = expand_model_blocks_to_token_order(model_to_phys.view(1, -1), BLOCK_LEN)[0]
    return chunks_phys[:, token_gather]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="Path to AO-GPT checkpoint")
    p.add_argument("--out-dir", default="reports/token_attention_multisample")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--M", type=int, default=200, help="Number of probe groups")
    p.add_argument("--batch-size", type=int, default=8, help="Chunks per probe group")
    p.add_argument("--fwd-batch", type=int, default=8, help="Forward batch size")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--head", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--share-reveal-order", action="store_true",
                   help="Use the SAME random reveal order for all samples (scattered spots align)")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}")
    t0 = time.time()
    ckpt, model, dev = _load_model(args.ckpt, args.device)
    model_to_phys, phys_to_model = _load_permutations(ckpt)
    print(f"  Model: {model.config.n_layer}L x {model.config.n_head}H  |  head: L{args.layer}H{args.head}")

    total = args.M * args.batch_size
    chunks_phys = load_train_chunks(n_chunks=total)
    chunks_model = _phys_chunks_to_model(chunks_phys, model_to_phys)
    print(f"  Samples: {args.M} x batch_size={args.batch_size} = {total} chunks")
    print(f"  Share reveal order: {args.share_reveal_order}")
    print(f"  Load time: {time.time() - t0:.1f}s")

    # --- Generate token orders (reveal order for each sample) ---
    token_orders = torch.empty((total, SEQ_LEN), dtype=torch.long)
    if args.share_reveal_order:
        gen = torch.Generator(device="cpu"); gen.manual_seed(int(args.seed))
        rand_blocks = torch.randperm(N, generator=gen, device="cpu")
        shared_order = expand_model_blocks_to_token_order(rand_blocks.unsqueeze(0), BLOCK_LEN)[0]
        for i in range(total):
            token_orders[i] = shared_order
        print(f"  Shared reveal blocks: {rand_blocks[:8].tolist()}...")
    else:
        for i in range(total):
            gen = torch.Generator(device="cpu"); gen.manual_seed(int(args.seed) + int(i))
            rand_blocks = torch.randperm(N, generator=gen, device="cpu")
            token_orders[i] = expand_model_blocks_to_token_order(rand_blocks.unsqueeze(0), BLOCK_LEN)[0]

    # --- Precompute remap: reveal -> model order (undo random reveal) ---
    if args.share_reveal_order:
        tok0 = token_orders[0].numpy()                      # reveal_pos -> model_pos
        reveal_to_model_shared = np.empty(SEQ_LEN, dtype=np.int64)
        reveal_to_model_shared[tok0] = np.arange(SEQ_LEN)   # model_pos -> reveal_pos
        print(f"  Precomputed reveal->model remap")

    # --- Accumulators ---
    sum_reveal = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float64)
    sum_model  = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float64)
    sum_phys   = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float64)
    count = 0

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for start in range(0, total, max(1, int(args.fwd_batch))):
            stop = min(start + max(1, int(args.fwd_batch)), total)
            tokens = chunks_model[start:stop].to(dev)
            orders = token_orders[start:stop].to(dev)

            _, _, attn_list = model.forward_fn(tokens, orders, return_attentions=True)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

            attn_batch = torch.stack(attn_list).cpu().numpy()  # (L, B, H, T+1, T+1)

            for bi in range(stop - start):
                A_raw = attn_batch[args.layer, bi, args.head, :, :].astype(np.float64)
                A_reveal = A_raw[1:, 1:]  # (T, T), strip None, keep reveal order

                sum_reveal += A_reveal

                # --- Reveal -> model order (undo random reveal) ---
                if args.share_reveal_order:
                    A_model = A_reveal[reveal_to_model_shared][:, reveal_to_model_shared]
                else:
                    tok = token_orders[start + bi].numpy()
                    inv = np.empty(SEQ_LEN, dtype=np.int64)
                    inv[tok] = np.arange(SEQ_LEN, dtype=np.int64)
                    A_model = A_reveal[inv][:, inv]

                sum_model += A_model

                # --- Model order -> physical order ---
                # B_phys[p,q] = B_model[phys_to_model[p], phys_to_model[q]]
                pm = phys_to_model.numpy()
                A_phys = A_model[pm][:, pm]
                sum_phys += A_phys

                count += 1

            elapsed = time.time() - t1
            rate = stop / elapsed if elapsed > 0 else 0
            if stop % 160 == 0 or stop == total:
                print(f"  [{stop}/{total}] {rate:.0f} samples/s", flush=True)

    A_reveal_mean = (sum_reveal / count).astype(np.float32)
    A_model_mean  = (sum_model  / count).astype(np.float32)
    A_phys_mean   = (sum_phys   / count).astype(np.float32)

    print(f"  Extraction: {time.time() - t1:.1f}s, {count} samples")

    # --- Stats ---
    def tri_stats(A, name):
        up = float(A[np.triu_indices(SEQ_LEN, k=1)].mean())
        lo = float(A[np.tril_indices(SEQ_LEN, k=-1)].mean())
        di = float(np.diag(A).mean())
        print(f"  {name}: upper={up:.6f}  lower={lo:.6f}  diag={di:.6f}  max={A.max():.4f}")
        return up, lo, di

    print()
    tri_stats(A_reveal_mean, "reveal      ")
    tri_stats(A_model_mean,  "model order ")
    tri_stats(A_phys_mean,   "physical    ")

    # --- Save ---
    np.save(out_dir / "token_attention_reveal_frame.npy", A_reveal_mean)
    np.save(out_dir / "token_attention_model_frame.npy", A_model_mean)
    np.save(out_dir / "token_attention_physical_frame.npy", A_phys_mean)

    meta = {
        "ckpt": args.ckpt,
        "iter_num": ckpt.get("iter_num"),
        "layer": args.layer, "head": args.head,
        "M": args.M, "batch_size": args.batch_size, "num_samples": count,
        "seed": args.seed,
        "share_reveal_order": args.share_reveal_order,
        "convention": {
            "model_to_phys": "ckpt['data_permutation']['block_perm']",
            "phys_to_model": "ckpt['data_permutation']['inverse_block_perm']",
            "x_model": "x_phys[model_to_phys]",
            "B_phys": "B_model[phys_to_model][:, phys_to_model]",
        },
        "shape": [SEQ_LEN, SEQ_LEN],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # --- Plot: reveal + model + physical ---
    order_mode = "shared_order" if args.share_reveal_order else "varied_order"
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")

    ax0.imshow(A_reveal_mean, cmap="viridis", aspect="equal", vmin=0, interpolation="nearest")
    ax0.set_title("reveal order", fontsize=10)
    ax0.set_xlabel("source", fontsize=9); ax0.set_ylabel("target", fontsize=9)

    ax1.imshow(A_model_mean, cmap="viridis", aspect="equal", vmin=0, interpolation="nearest")
    ax1.set_title("model order\n(permuted L2R)", fontsize=10)
    ax1.set_xlabel("source", fontsize=9)

    im = ax2.imshow(A_phys_mean, cmap="viridis", aspect="equal", vmin=0, interpolation="nearest")
    ax2.set_title("physical order\n(original L2R)", fontsize=10)
    ax2.set_xlabel("source", fontsize=9)

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.02, shrink=0.85)
    cbar.set_label("attention", fontsize=9)

    step_label = str(ckpt.get("iter_num", "?"))
    fig.suptitle(f"Token-level Attention  |  L{args.layer}H{args.head}  step={step_label}  "
                 f"M={args.M}  N={count}  {order_mode}",
                 fontsize=11, fontweight="bold")
    plt.subplots_adjust(wspace=0.3)
    png_path = out_dir / f"token_attention_L{args.layer}H{args.head}_M{args.M}_{order_mode}.png"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)

    print(f"\nSaved: {png_path}")
    print(f"Saved: {out_dir / 'token_attention_reveal_frame.npy'}")
    print(f"Saved: {out_dir / 'token_attention_model_frame.npy'}")
    print(f"Saved: {out_dir / 'token_attention_physical_frame.npy'}")
    print(f"Saved: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
