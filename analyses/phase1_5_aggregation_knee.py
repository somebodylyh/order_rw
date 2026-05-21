#!/usr/bin/env python3
"""Phase 1.5.1 — aggregation knee. Q: how many per-sample attention maps must be
averaged before block-level locality emerges? Extract per-image A_block (physical
frame, all-layer/head averaged like extract_image_attention_e2), then bootstrap the
locality of the mean over n images for n = 1..N. Find the knee.

Bounded gate experiment — answers ONE yes/no (is there a knee?), nothing mechanistic.
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
from diagnose_e3_control_dual_level import aggregate_token_to_block, _make_manh_table, _make_samequad_table

CKPT = _REPO / "nanogpt-learned-order/out/image_large/imagenet64_vq_f4_800k/seq256/patch2x2_full_baseline_l4h8e256/ckpt.pt"
DATA = _REPO / "nanogpt-learned-order/data/Imagenet64VQ_f4_800k_full_patch8x8/val.bin"
OUT = _REPO / "analyses/phase1_5_20260521"
N_IMAGES = 500
M_PASSES = 3
GRID = 8
DEVICE = "cuda:0"


def block_locality(A_block):
    """P(d<=1) and locality_score on the 8x8 grid for one 64x64 block matrix."""
    N = A_block.shape[0]
    manh = _make_manh_table(GRID)
    A = A_block.copy(); np.fill_diagonal(A, 0.0)
    top1 = np.argmax(A, axis=1)
    qs = np.arange(N)
    amd = manh[qs, top1]
    p_le1 = float((amd <= 1).mean())
    # locality_score = how much more local than random
    rand_dist = float(manh[~np.eye(N, dtype=bool)].mean())
    loc_score = float(np.clip((rand_dist - amd.mean()) / (rand_dist + 1e-12), 0, 1))
    return p_le1, loc_score, float(amd.mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, margs = load_model(str(CKPT), DEVICE)
    T = model.config.block_size
    data = np.memmap(DATA, dtype=np.uint16, mode="r")
    n_avail = len(data) // T
    n_imgs = min(N_IMAGES, n_avail)
    print(f"Extracting per-image A_block for {n_imgs} images x M={M_PASSES} ...", flush=True)

    per_img_block = np.zeros((n_imgs, 64, 64), dtype=np.float32)
    for i in range(n_imgs):
        tokens = torch.from_numpy(data[i*T:(i+1)*T].astype(np.int64)).to(DEVICE).unsqueeze(0)
        A_sum = torch.zeros(T, T, device=DEVICE)
        for _ in range(M_PASSES):
            order = torch.randperm(T, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                _, _, attn_list = model.forward_fn(tokens, order, return_attentions=True)
            attn = torch.stack(attn_list, 0).mean(dim=[0, 2])[0]   # all layer/head avg
            content = attn[1:, 1:]
            inv = torch.argsort(order[0])
            A_sum += content[inv][:, inv]
        A = (A_sum / M_PASSES).float().cpu().numpy()
        np.fill_diagonal(A, 0.0)
        per_img_block[i] = aggregate_token_to_block(A.astype(np.float32))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_imgs}", flush=True)

    np.save(OUT / "per_image_A_block.npy", per_img_block)

    ns = [n for n in [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500] if n <= n_imgs]
    R = 20  # bootstrap subsets per n
    rng = np.random.default_rng(0)
    rows = []
    for n in ns:
        p1s, lss, ads = [], [], []
        for _ in range(R):
            idx = rng.choice(n_imgs, size=n, replace=False)
            Ablk = per_img_block[idx].mean(0)
            p1, ls, ad = block_locality(Ablk)
            p1s.append(p1); lss.append(ls); ads.append(ad)
        rows.append((n, np.mean(p1s), np.std(p1s), np.mean(lss), np.std(lss), np.mean(ads)))

    # global (all n_imgs)
    p1g, lsg, adg = block_locality(per_img_block.mean(0))

    with open(OUT / "aggregation_knee.tsv", "w") as f:
        f.write("n_images\tp_nbr_le1_mean\tp_nbr_le1_std\tlocality_score_mean\tlocality_score_std\targmax_dist_mean\n")
        for n, p1m, p1s, lsm, lss, adm in rows:
            f.write(f"{n}\t{p1m:.4f}\t{p1s:.4f}\t{lsm:.4f}\t{lss:.4f}\t{adm:.4f}\n")
        f.write(f"{n_imgs}(global)\t{p1g:.4f}\t0.0000\t{lsg:.4f}\t0.0000\t{adg:.4f}\n")

    print(f"\n{'n':>8}{'P(nbr<=1)':>14}{'locality_score':>18}{'argmax_dist':>14}")
    for n, p1m, p1sd, lsm, lssd, adm in rows:
        print(f"{n:>8}{p1m:>8.3f}±{p1sd:.3f}{lsm:>11.3f}±{lssd:.3f}{adm:>14.3f}")
    print(f"{'global':>8}{p1g:>8.3f}{'':>6}{lsg:>11.3f}{'':>6}{adg:>14.3f}")

    # knee = smallest n whose mean P(nbr<=1) reaches >=80% of the global value
    target = 0.8 * p1g
    knee = next((n for n, p1m, *_ in rows if p1m >= target), None)
    print(f"\nglobal P(nbr<=1)={p1g:.3f}; 80%-of-global threshold={target:.3f}; "
          f"knee n (first to reach it) = {knee}")
    print(f"wrote {OUT}/aggregation_knee.tsv, per_image_A_block.npy")


if __name__ == "__main__":
    main()
