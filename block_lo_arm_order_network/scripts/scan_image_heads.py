"""Quick per-head signal scan on an image checkpoint.

GPU0 has ~9.5 GB free — use small M for a fast diagnostic.
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent  # block_lo_arm_order_network/
_REPO_ROOT = _REPO.parent  # order_lyu/
sys.path.insert(0, str(_REPO_ROOT / "nanogpt-learned-order"))
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO_ROOT))
from AOGPT import AOGPTConfig, AOGPT

# Reuse text-side CDL teacher
from neural_readout.teacher_labels import generate_teacher_label

# Raster order for 8x8 grid: row-major [0,1,...,63]
RASTER_ORDER = np.arange(64)
GRID = 8
N_BLOCKS = 64


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_args = ckpt.get("model_args", {})
    cfg = ckpt.get("config", {})
    keys = ["block_size", "vocab_size", "n_layer", "n_head", "n_embd",
            "dropout", "bias", "block_order_block_len", "order_impl"]
    model_args = {}
    for k in keys:
        if k in m_args:
            model_args[k] = m_args[k]
        elif k in cfg:
            model_args[k] = cfg[k]
    model = AOGPT(AOGPTConfig(**model_args))
    sd = ckpt["model"]
    if all(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:3]}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:3]}")
    model.crop_block_size(model_args["block_size"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, model_args


def extract_per_head_B_batch(model, data_tokens, tokens_per_image, n_images,
                              m_passes, device, block_len, seed=42):
    """Extract per-head batch-mean B matrices.

    Returns:
        B_lh: (L, H, 64, 64) float64 batch-mean B (zero diag)
    """
    T = model.config.block_size
    L = model.config.n_layer
    H = model.config.n_head
    assert T == tokens_per_image

    B_sum = np.zeros((L, H, N_BLOCKS, N_BLOCKS), dtype=np.float64)
    B_count = 0

    n_available = len(data_tokens) // T
    n_images = min(n_images, n_available)
    print(f"  Extracting per-head B: {n_images} images x M={m_passes}, "
          f"L={L} H={H} T={T} block_len={block_len}", flush=True)

    rng = np.random.default_rng(seed)
    t0 = time.time()

    for img_idx in range(n_images):
        start = img_idx * T
        tokens = torch.from_numpy(
            data_tokens[start:start + T].astype(np.int64)
        ).to(device).unsqueeze(0)

        for _ in range(m_passes):
            rand_order = torch.randperm(T, device=device).unsqueeze(0)

            with torch.no_grad():
                _, _, attn_list = model.forward_fn(
                    tokens, rand_order, return_attentions=True
                )

            # attn_list: list of (B, nh, T+1, T+1) per layer
            attn_stack = torch.stack(attn_list, dim=0)  # (L, B, nh, T+1, T+1)

            for layer in range(L):
                for head in range(H):
                    a = attn_stack[layer, 0, head]  # (T+1, T+1)
                    a_content = a[1:, 1:]  # (T, T) in model-pos order

                    # Remap model-pos → physical-pos
                    inv_order = torch.argsort(rand_order[0])
                    a_phys = a_content[inv_order][:, inv_order].float().cpu().numpy()

                    # Aggregate token-level → block-level
                    if block_len > 1:
                        a_phys = a_phys.reshape(N_BLOCKS, block_len, N_BLOCKS, block_len).mean(axis=(1, 3))

                    # B = A^T, zero diag
                    B = a_phys.T.copy()
                    np.fill_diagonal(B, 0.0)
                    B_sum[layer, head] += B

            B_count += 1

        if (img_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (img_idx + 1) * (n_images - img_idx - 1)
            print(f"    {img_idx + 1}/{n_images} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    B_mean = B_sum / B_count
    print(f"  Done in {time.time() - t0:.0f}s, {B_count} passes total", flush=True)
    return B_mean


def tau_vs_raster(sigma):
    """Kendall tau between sigma and raster [0,1,...,63]."""
    N = len(sigma)
    rank = np.empty(N, dtype=np.int64)
    rank[sigma] = np.arange(N)
    n = N * (N - 1) // 2
    conc = 0
    for i in range(N):
        for j in range(i + 1, N):
            d1 = rank[i] - rank[j]
            d2 = RASTER_ORDER[i] - RASTER_ORDER[j]
            if d1 * d2 > 0:
                conc += 1
            elif d1 * d2 < 0:
                conc -= 1
    return conc / n if n > 0 else 0.0


def row_concentration(B):
    """std(r) where r = out-deg - 0.5 * in-deg per node, normalized."""
    r = B.sum(axis=1) - 0.5 * B.sum(axis=0)
    r = r / (np.abs(r).mean() + 1e-12)
    return float(np.std(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--M-passes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    tokens_per_image = int(meta["tokens_per_image"])
    block_len = meta.get("block_order_block_len", 1)

    print(f"Loading model: {args.ckpt}", flush=True)
    model, model_args = load_model(args.ckpt, str(device))
    L, H = model_args["n_layer"], model_args["n_head"]
    print(f"  L={L} H={H} d={model_args['n_embd']} block_size={model_args['block_size']} "
          f"block_len={block_len}", flush=True)

    print(f"Loading data: {args.data}", flush=True)
    data = np.memmap(args.data, dtype=np.uint16, mode="r")

    B_lh = extract_per_head_B_batch(
        model, data, tokens_per_image, args.n_images, args.M_passes,
        str(device), block_len, seed=args.seed,
    )

    # --- Per-head CDL analysis ---
    print(f"\n=== Per-head tau_vs_raster & row-conc (L={L}, H={H}) ===", flush=True)
    results = []
    for layer in range(L):
        for head in range(H):
            B = B_lh[layer, head]
            sigma, _, _ = generate_teacher_label(B)
            tau = tau_vs_raster(sigma)
            rc = row_concentration(B)
            # Additional: mean pairwise tau of the teacher
            results.append({
                "layer": layer, "head": head,
                "tau_vs_raster": tau,
                "row_conc": rc,
            })

    # Sort by abs(tau)
    results.sort(key=lambda r: -abs(r["tau_vs_raster"]))

    # Print heatmap-style grid
    tau_grid = np.zeros((L, H))
    rc_grid = np.zeros((L, H))
    for r in results:
        tau_grid[r["layer"], r["head"]] = r["tau_vs_raster"]
        rc_grid[r["layer"], r["head"]] = r["row_conc"]

    lh_header = "L\\H  "
    for h in range(H):
        lh_header += "    H{}   ".format(h)
    lh_header += "    avg"
    print(lh_header)
    for layer in range(L):
        row_str = " L{}  ".format(layer)
        row_taus = []
        for head in range(H):
            t = tau_grid[layer, head]
            marker = "<<" if t < -0.3 else ("<" if t < -0.1 else (">" if t > 0.5 else ""))
            row_str += " {:>+7.3f}{}".format(t, marker)
            row_taus.append(t)
        row_str += " {:>+8.3f}".format(np.mean(row_taus))
        print(row_str)

    print(f"\n=== Top 10 heads by |tau_vs_raster| ===")
    for r in results[:10]:
        d = "PRO-RASTER" if r["tau_vs_raster"] > 0 else "ANTI-RASTER" if r["tau_vs_raster"] < -0.3 else "WEAK"
        print(f"  L{r['layer']}H{r['head']}: tau={r['tau_vs_raster']:+.4f}  "
              f"row_conc={r['row_conc']:.4f}  [{d}]")

    print("\n=== Row-concentration grid ===")
    rc_hdr = "L\\H  "
    for h in range(H):
        rc_hdr += "    H{}   ".format(h)
    print(rc_hdr)
    for layer in range(L):
        rc_row = " L{}  ".format(layer)
        for head in range(H):
            rc_row += " {:>8.4f}".format(rc_grid[layer, head])
        print(rc_row)

    # Layer stats
    print(f"\n=== Layer stats ===")
    for layer in range(L):
        taus = tau_grid[layer, :]
        print(f"  L{layer}: mean={taus.mean():+.3f} std={taus.std():.3f} "
              f"min={taus.min():+.3f}(H{np.argmin(taus)}) max={taus.max():+.3f}(H{np.argmax(taus)}) "
              f"n_neg={int(np.sum(taus < 0))} n_pos={int(np.sum(taus > 0.3))}")

    # Save
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, tau_grid=tau_grid, rc_grid=rc_grid,
                 B_lh=B_lh, results=results)
        print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
