"""Across the random baseline trajectory, scan ALL (layer,head) attention heads
and track WHERE the L2R order signal lives at each training step — does it stay
in L0H5, migrate to another layer/head, or vanish? NO TRAINING.

Tokenizes wikitext ONCE, then for each ckpt: batched forward, build batch-mean
B^(l,h) = mean_x A^(l,h)(x)^T (physical-remapped, 64x64) over M samples, teacher
CDL -> sigma -> tau_vs_l2r for every (layer,head).

Outputs:
  - heads_tau_matrix.png : heatmap (32 heads x 9 steps) of tau_vs_l2r + winner marks
  - heads_tau.json       : raw table + per-step winner / per-layer best

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=block_lo_arm_order_network \
    python scripts/scan_all_heads_across_ckpts.py --M 200 --seed 0
"""
from __future__ import annotations

import argparse
import json
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

_CB = "block_lo_arm_order_network/probe_results/clean_base_random_perm"
DEFAULT_CKPTS = [
    ("0", f"{_CB}/ckpt_step0.pt"), ("1k", f"{_CB}/ckpt_step1000.pt"),
    ("5k", f"{_CB}/ckpt_step5000.pt"), ("10k", f"{_CB}/ckpt_step10000.pt"),
    ("20k", f"{_CB}/ckpt_step20000.pt"), ("30k", f"{_CB}/ckpt_step30000.pt"),
    ("40k", f"{_CB}/ckpt_step40000.pt"), ("50k", f"{_CB}/ckpt_step50000.pt"),
    ("60k", f"{_CB}/ckpt_step60000.pt"),
]


def _block_agg(x):
    shp = x.shape[:-2]
    return x.reshape(*shp, N, BLOCK_LEN, N, BLOCK_LEN).mean(axis=(-3, -1))


def load_model_and_chunks_cached(path, idx_phys, M, seed, device):
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
def all_head_tau(model, chunks, clean_perm, device, total, batch_size, seed, gpu_batch):
    """ORIGINAL L0H5 aggregation: M=total//batch_size batch-mean graphs, each over
    `batch_size` samples; tau(l,h) = mean Kendall tau(sigma_T, l2r) over the M graphs.
    Pure CDL teacher, same as diag_br1_head_layer_scan."""
    inv_perm = clean_perm.inv_perm_model_to_phys.cpu().numpy()
    di = np.arange(N)
    M = total // batch_size
    A_lh = None  # (L,H,M,N,N)
    L = H = None
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
            phys_tok[j] = inv_perm[to_row // BLOCK_LEN] * BLOCK_LEN + (to_row % BLOCK_LEN)
        _, _, attn_list = model.forward_fn(
            tokens, torch.from_numpy(orders).to(device), return_attentions=True
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        attn = torch.stack(attn_list, dim=0).cpu().numpy()  # (L, bs, H, 257, 257)
        if A_lh is None:
            L, H = attn.shape[0], attn.shape[2]
            A_lh = np.zeros((L, H, M, N, N), dtype=np.float64)
        for j in range(bs):
            i = start + j; m = i // batch_size
            content = attn[:, j, :, 1:, 1:]               # (L,H,256,256)
            pj = phys_tok[j]
            phys = np.empty_like(content)
            phys[:, :, pj[:, None], pj[None, :]] = content
            B_blocks = np.swapaxes(_block_agg(phys), -1, -2)  # (L,H,64,64), B=A^T
            B_blocks[..., di, di] = 0.0
            A_lh[:, :, m] += B_blocks
    Bb = (A_lh / batch_size).astype(np.float32)            # (L,H,M,64,64)
    Bb[..., di, di] = 0.0
    l2r = np.tile(np.arange(N), (M, 1))
    tau = np.zeros((L, H), dtype=np.float64)
    for l in range(L):
        for h in range(H):
            sig = np.stack([generate_teacher_label(Bb[l, h, m], alpha_dep=0.5)[0] for m in range(M)])
            tau[l, h] = kendall_tau_batch(sig, l2r)
    return tau, L, H


CANDIDATES = [(0, 0), (0, 5), (1, 6), (2, 3)]  # L0H0, L0H5, L1H6, L2H3


def per_seed_table(labels, taus, L, H):
    """taus: (steps,L,H). -> list of per-step dicts with best_pos/best_neg/abs_winner/candidates."""
    rows = []
    for s, label in enumerate(labels):
        t = taus[s]
        pl, ph = np.unravel_index(np.argmax(t), (L, H))         # most positive
        nl, nh = np.unravel_index(np.argmin(t), (L, H))         # most negative
        wl, wh = np.unravel_index(np.argmax(np.abs(t)), (L, H))  # |tau| winner
        rows.append({
            "step": label,
            "best_pos": {"head": f"L{pl}H{ph}", "tau": float(t[pl, ph])},
            "best_neg": {"head": f"L{nl}H{nh}", "tau": float(t[nl, nh])},
            "abs_winner": {"head": f"L{wl}H{wh}", "tau": float(t[wl, wh])},
            "candidates": {f"L{cl}H{ch}": float(t[cl, ch]) for cl, ch in CANDIDATES},
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpu_batch", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--outdir", default="block_lo_arm_order_network/batch_readout/logs/l0h5_evo")
    args = ap.parse_args()

    cache = pathlib.Path("/tmp/wikitext_idx_phys_cache.pt")
    if cache.exists():
        print(f"loading cached tokenized chunks from {cache} ...", flush=True)
        idx_phys = torch.load(cache)
    else:
        print("tokenizing wikitext once (will cache to disk) ...", flush=True)
        idx_phys = load_train_chunks(n_chunks=None)
        torch.save(idx_phys, cache)

    total = args.M * args.batch_size
    ckpts = [(lab, p) for lab, p in DEFAULT_CKPTS if pathlib.Path(p).exists()]
    labels = [lab for lab, _ in ckpts]
    L = H = None
    all_taus = {}  # seed -> (steps,L,H)

    for seed in args.seeds:
        taus = []
        print(f"\n========== SEED {seed} ==========", flush=True)
        for label, path in ckpts:
            model, chunks, clean_perm, dev = load_model_and_chunks_cached(path, idx_phys, total, seed, args.device)
            tau, L, H = all_head_tau(model, chunks, clean_perm, dev, total, args.batch_size, seed, args.gpu_batch)
            taus.append(tau)
            t = tau
            pl, ph = np.unravel_index(np.argmax(t), t.shape)
            nl, nh = np.unravel_index(np.argmin(t), t.shape)
            wl, wh = np.unravel_index(np.argmax(np.abs(t)), t.shape)
            print(f"  step {label:>4}: best+ L{pl}H{ph}={t[pl,ph]:+.3f} | best- L{nl}H{nh}={t[nl,nh]:+.3f} | "
                  f"|w| L{wl}H{wh}={t[wl,wh]:+.3f} || L0H0={t[0,0]:+.3f} L0H5={t[0,5]:+.3f} "
                  f"L1H6={t[1,6]:+.3f} L2H3={t[2,3]:+.3f}", flush=True)
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        all_taus[seed] = np.stack(taus)

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    stack = np.stack([all_taus[s] for s in args.seeds])  # (S, steps, L, H)
    mean = stack.mean(0); std = stack.std(0)             # (steps, L, H)

    # ---- JSON ----
    rec = {"steps": labels, "seeds": args.seeds, "L": L, "H": H,
           "per_seed": {str(s): per_seed_table(labels, all_taus[s], L, H) for s in args.seeds},
           "candidate_mean_std": {
               f"L{cl}H{ch}": {"mean": [float(mean[i, cl, ch]) for i in range(len(labels))],
                               "std":  [float(std[i, cl, ch]) for i in range(len(labels))]}
               for cl, ch in CANDIDATES}}
    with open(outdir / "heads_tau_3seed.json", "w") as f:
        json.dump(rec, f, indent=2)

    # ---- cross-seed stability of |tau| winner & best pos/neg head identity ----
    print("\n=== cross-seed winner-head identity per step ===")
    print(f"{'step':>5} | {'|w| winner (per seed)':<26} | {'best+ (per seed)':<22} | {'best- (per seed)':<22}")
    for i, label in enumerate(labels):
        wlist, plist, nlist = [], [], []
        for s in args.seeds:
            t = all_taus[s][i]
            wl, wh = np.unravel_index(np.argmax(np.abs(t)), t.shape); wlist.append(f"L{wl}H{wh}")
            pl, ph = np.unravel_index(np.argmax(t), t.shape); plist.append(f"L{pl}H{ph}")
            nl, nh = np.unravel_index(np.argmin(t), t.shape); nlist.append(f"L{nl}H{nh}")
        print(f"{label:>5} | {','.join(wlist):<26} | {','.join(plist):<22} | {','.join(nlist):<22}")

    print("\n=== candidate heads: 3-seed mean+-std tau_vs_l2r ===")
    print(f"{'step':>5} | " + " | ".join(f"L{cl}H{ch:>1}" + " " * 8 for cl, ch in CANDIDATES))
    for i, label in enumerate(labels):
        cells = []
        for cl, ch in CANDIDATES:
            cells.append(f"{mean[i,cl,ch]:+.2f}+-{std[i,cl,ch]:.2f}")
        print(f"{label:>5} | " + " | ".join(c.ljust(11) for c in cells))

    # ---- figure: candidate trajectories (3-seed mean+-std) ----
    steps_x = range(len(labels))
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"L0H0": "C3", "L0H5": "C0", "L1H6": "C1", "L2H3": "C2"}
    for cl, ch in CANDIDATES:
        key = f"L{cl}H{ch}"
        m = [mean[i, cl, ch] for i in steps_x]; sd = [std[i, cl, ch] for i in steps_x]
        ax.errorbar(steps_x, m, yerr=sd, marker="o", capsize=3, label=key, color=colors[key])
    # envelope: per-step max best+ and min best- across seeds (the strongest order heads)
    bestpos = [max(all_taus[s][i].max() for s in args.seeds) for i in steps_x]
    bestneg = [min(all_taus[s][i].min() for s in args.seeds) for i in steps_x]
    ax.plot(steps_x, bestpos, ls="--", color="gray", alpha=0.7, label="strongest +tau (any head)")
    ax.plot(steps_x, bestneg, ls=":", color="gray", alpha=0.7, label="strongest -tau (any head)")
    ax.axhspan(0.56, 0.64, alpha=0.10, color="C0")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(list(steps_x)); ax.set_xticklabels(labels)
    ax.set_xlabel("training step"); ax.set_ylabel("tau_vs_l2r (3-seed mean+-std)")
    ax.set_title("Order-specialized heads across training (random baseline, seeds 0/1/2)\n"
                 "candidate heads + strongest +/- envelope")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "candidate_trajectories_3seed.png", dpi=130)

    # ---- per-seed heatmaps (stacked) ----
    S = len(args.seeds)
    fig2, axes = plt.subplots(1, S, figsize=(1.0 * len(labels) * S + 2, 9), squeeze=False)
    vmax = float(np.max(np.abs(stack)))
    for si, s in enumerate(args.seeds):
        ax = axes[0][si]
        mat = all_taus[s].reshape(len(labels), L * H).T
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(range(L * H))
        ax.set_yticklabels([f"L{i//H}H{i%H}" for i in range(L * H)], fontsize=6)
        ax.set_title(f"seed {s}", fontsize=10)
        for i in range(len(labels)):
            wl, wh = np.unravel_index(np.argmax(np.abs(all_taus[s][i])), (L, H))
            ax.add_patch(plt.Rectangle((i - 0.5, wl * H + wh - 0.5), 1, 1, fill=False, edgecolor="lime", lw=1.5))
    fig2.suptitle("tau_vs_l2r per head, per seed (lime = |tau| winner)", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    fig2.savefig(outdir / "heads_tau_matrix_3seed.png", dpi=130)

    print(f"\nwrote {outdir/'candidate_trajectories_3seed.png'}, "
          f"{outdir/'heads_tau_matrix_3seed.png'}, {outdir/'heads_tau_3seed.json'}")


if __name__ == "__main__":
    main()
