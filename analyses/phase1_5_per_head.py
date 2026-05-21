#!/usr/bin/env python3
"""Phase 1.5.2 — per-(layer,head) locality on E3-control-small, CORRECT spatial
aggregation (token_to_patch_indices 16x16->8x8), n-image population-averaged.

Answers ONE yes/no: is there a stable locality head (locality_score > 0.5)? i.e. is
the global locality concentrated in a few heads (→ CEM could use head-selected B) or
distributed/emergent only after head-averaging (→ CEM uses all-head batch B)?

Bounded gate — no mechanistic 'why image attention is content-driven'.
"""
import sys
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
sys.path.insert(0, str(_REPO / "nanogpt-learned-order"))
sys.path.insert(0, str(_REPO / "scripts"))
from extract_image_attention_e2 import load_model
from diagnose_e3_control_dual_level import token_to_patch_indices, _make_manh_table

CKPT = _REPO / "nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt"
DATA = _REPO / "nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/val.bin"
OUT = _REPO / "analyses/phase1_5_20260521/per_head"
N_IMAGES = 200
M_PASSES = 3
GRID = 8
DEVICE = "cuda:0"


def pooling_matrix():
    """P (64,256): spatial mean-pool token->patch. A_block = P @ A_token @ P.T."""
    p_idx = token_to_patch_indices()
    P = np.zeros((64, 256))
    for t in range(256):
        P[p_idx[t], t] = 1.0
    P = P / P.sum(1, keepdims=True)
    return P


def block_locality(A_block):
    N = A_block.shape[0]; manh = _make_manh_table(GRID)
    A = A_block.copy(); np.fill_diagonal(A, 0.0)
    top1 = np.argmax(A, 1); amd = manh[np.arange(N), top1]
    rand_dist = float(manh[~np.eye(N, dtype=bool)].mean())
    return (float((amd <= 1).mean()),
            float(np.clip((rand_dist - amd.mean()) / (rand_dist + 1e-12), 0, 1)),
            float(amd.mean()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, margs = load_model(str(CKPT), DEVICE)
    nl, nh = margs["n_layer"], margs["n_head"]
    T = model.config.block_size
    print(f"model n_layer={nl} n_head={nh} T={T}")
    data = np.memmap(DATA, dtype=np.uint16, mode="r")
    n_imgs = min(N_IMAGES, len(data) // T)

    A_sum = torch.zeros(nl, nh, T, T, device=DEVICE)
    cnt = 0
    print(f"extracting per-(layer,head) attention, {n_imgs} imgs x M={M_PASSES} ...", flush=True)
    for i in range(n_imgs):
        tok = torch.from_numpy(data[i*T:(i+1)*T].astype(np.int64)).to(DEVICE).unsqueeze(0)
        for _ in range(M_PASSES):
            order = torch.randperm(T, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                _, _, attn_list = model.forward_fn(tok, order, return_attentions=True)
            stack = torch.stack(attn_list, 0)[:, 0]      # (nl, nh, T+1, T+1)
            content = stack[:, :, 1:, 1:]                # (nl, nh, T, T) model order
            inv = torch.argsort(order[0])
            A_sum += content[:, :, inv][:, :, :, inv]
            cnt += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_imgs}", flush=True)
    A = (A_sum / cnt).float().cpu().numpy()              # (nl, nh, T, T) physical

    P = pooling_matrix()
    rows = []
    for l in range(nl):
        for h in range(nh):
            Ab = P @ A[l, h] @ P.T
            np.fill_diagonal(Ab, 0.0)
            p1, ls, ad = block_locality(Ab)
            rows.append((l, h, p1, ls, ad))
    # all-head average per layer + global all-layer-all-head (reference)
    Ab_global = P @ A.mean(axis=(0, 1)) @ P.T; np.fill_diagonal(Ab_global, 0.0)
    g_p1, g_ls, g_ad = block_locality(Ab_global)

    with open(OUT / "per_head_locality.tsv", "w") as f:
        f.write("layer\thead\tp_nbr_le1\tlocality_score\targmax_dist\n")
        for l, h, p1, ls, ad in rows:
            f.write(f"{l}\t{h}\t{p1:.4f}\t{ls:.4f}\t{ad:.4f}\n")
        f.write(f"GLOBAL\tall\t{g_p1:.4f}\t{g_ls:.4f}\t{g_ad:.4f}\n")

    ls_all = np.array([r[3] for r in rows])
    stable = [(l, h, p1, ls) for l, h, p1, ls, ad in rows if ls > 0.5]
    print(f"\n{'layer':>6}{'head':>5}{'p_nbr<=1':>10}{'locality':>10}{'argmaxD':>9}")
    for l, h, p1, ls, ad in sorted(rows, key=lambda r: -r[3]):
        print(f"{l:>6}{h:>5}{p1:>10.3f}{ls:>10.3f}{ad:>9.3f}")
    print(f"{'GLOBAL':>6}{'all':>5}{g_p1:>10.3f}{g_ls:>10.3f}{g_ad:>9.3f}")
    print(f"\nheads with locality_score>0.5: {len(stable)}/{nl*nh}")
    print(f"per-head locality_score: max={ls_all.max():.3f} mean={ls_all.mean():.3f} "
          f"min={ls_all.min():.3f}  | global(all-head)={g_ls:.3f}")
    print("ANSWER Q2: " + (
        f"YES — {len(stable)} stable locality head(s); locality is head-concentrated "
        "=> CEM could use head-selected/weighted B"
        if stable else
        "NO stable locality head; locality is distributed/emergent after head-averaging "
        "=> CEM uses all-head batch B"))
    print(f"wrote {OUT}/per_head_locality.tsv")


if __name__ == "__main__":
    main()
