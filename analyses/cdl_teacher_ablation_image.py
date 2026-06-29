"""Image CDL teacher ablation — locality metrics (D_manh, P(d≤1)) instead of τ_vs_raster.

Extracts per-(layer,head) attention from an image AOGPT checkpoint,
pools token→block (16×16 → 8×8 grid), picks the best head by locality,
then runs CDL teacher component ablations evaluated by:
  - D_manh: mean Manhattan distance between consecutive blocks in the rollout
  - P(d≤1): fraction of consecutive pairs at distance ≤ 1

Baselines: raster (row-major [0..63]), random shuffle.
"""
from __future__ import annotations

import sys, os, time, json
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AOGPT_DIR = os.path.join(_SCRIPT_DIR, "..", "block_lo_arm_order_network")
_REPO = os.path.join(_SCRIPT_DIR, "..")
_NANOGPT_DIR = os.path.join(_REPO, "nanogpt-learned-order")
sys.path.insert(0, _AOGPT_DIR)
sys.path.insert(0, _NANOGPT_DIR)

import torch
from AOGPT import AOGPTConfig, AOGPT

from attn_order_teacher import teacher_components, _softmax, _entropy

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

N_BLOCKS = 64
GRID = 8                    # 8×8 block grid (each block = 2×2 VQ tokens)
TOKEN_GRID = 16             # 16×16 VQ token grid
K_SEEDS = 20
TAU_T = 1.0

CKPT_PATH = os.path.join(_REPO, "probe_results_image/vq64_fixed_raster_l8h8e512/ckpt_step30000.pt")
DATA_PATH = os.path.join(_NANOGPT_DIR, "data/Imagenet64VQ_f4_800k_full_rowmajor/val.bin")
OUT_DIR = os.path.join(_SCRIPT_DIR, "cdl_teacher_ablation")
DEVICE = "cuda:0"
N_IMAGES = 200
M_PASSES = 3


# ═══════════════════════════════════════════════════════════════════
# 2D Grid / Locality
# ═══════════════════════════════════════════════════════════════════

def _make_manh_table(grid: int) -> np.ndarray:
    N = grid * grid
    rows = np.arange(N) // grid
    cols = np.arange(N) % grid
    rd = np.abs(rows[:, None] - rows[None, :])
    cd = np.abs(cols[:, None] - cols[None, :])
    return (rd + cd).astype(np.int64)


def token_to_patch_indices() -> np.ndarray:
    """Map 256 tokens → 64 block indices (2×2 token patches in 8×8 grid)."""
    PATCH_GRID = 8
    patch_idx = np.zeros(TOKEN_GRID * TOKEN_GRID, dtype=np.int64)
    for br in range(PATCH_GRID):
        for bc in range(PATCH_GRID):
            p = br * PATCH_GRID + bc
            i_tl = 2 * br * TOKEN_GRID + 2 * bc
            i_tr = i_tl + 1
            i_bl = i_tl + TOKEN_GRID
            i_br = i_bl + 1
            for tok in (i_tl, i_tr, i_bl, i_br):
                patch_idx[tok] = p
    return patch_idx


def aggregate_token_to_block(A_token: np.ndarray) -> np.ndarray:
    """Mean-pool 256×256 token attention → 64×64 block attention."""
    assert A_token.shape == (256, 256)
    p_idx = token_to_patch_indices()
    A_block = np.zeros((64, 64), dtype=np.float64)
    counts = np.zeros((64, 64), dtype=np.int64)
    for i in range(256):
        for j in range(256):
            A_block[p_idx[i], p_idx[j]] += A_token[i, j]
            counts[p_idx[i], p_idx[j]] += 1
    A_block /= np.maximum(counts, 1)
    np.fill_diagonal(A_block, 0.0)
    return A_block.astype(np.float64)


def order_locality(order, manh_table):
    """Compute D_manh and P(d≤1) for an order over 64 blocks."""
    steps = np.array([manh_table[order[t], order[t + 1]]
                      for t in range(len(order) - 1)])
    return float(steps.mean()), float((steps <= 1).mean())


# ═══════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str, device: str):
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
    model.load_state_dict(sd, strict=False)
    model.crop_block_size(model_args["block_size"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, model_args


# ═══════════════════════════════════════════════════════════════════
# Per-head extraction
# ═══════════════════════════════════════════════════════════════════

def extract_per_head_attention(model, data_tokens, n_images, m_passes, device):
    """Extract per-(layer,head) A_token, pooled to A_block (nl, nh, 64, 64)."""
    T = model.config.block_size
    nl, nh = model.config.n_layer, model.config.n_head
    n_available = len(data_tokens) // T
    n_images = min(n_images, n_available)

    A_sum = torch.zeros(nl, nh, 256, 256, device=device)
    cnt = 0

    print(f"Extracting per-head attention: {n_images} images × M={m_passes}", flush=True)
    for img_idx in range(n_images):
        start = img_idx * T
        tok = torch.from_numpy(
            data_tokens[start:start + T].astype(np.int64)
        ).to(device).unsqueeze(0)

        for _ in range(m_passes):
            rand_order = torch.randperm(T, device=device).unsqueeze(0)
            with torch.no_grad():
                _, _, attn_list = model.forward_fn(tok, rand_order, return_attentions=True)
            stack = torch.stack(attn_list, 0)[:, 0]       # (nl, nh, T+1, T+1)
            content = stack[:, :, 1:, 1:]                  # (nl, nh, T, T) model order
            inv = torch.argsort(rand_order[0])
            A_sum += content[:, :, inv][:, :, :, inv]      # physical coordinates
            cnt += 1

        if (img_idx + 1) % 50 == 0:
            print(f"  {img_idx + 1}/{n_images}", flush=True)

    A_mean = (A_sum / cnt).float().cpu().numpy()           # (nl, nh, 256, 256)
    del A_sum
    if device != "cpu":
        torch.cuda.empty_cache()

    # Pool to block-level
    print("Pooling token → block (256→64)...", flush=True)
    A_block = np.zeros((nl, nh, 64, 64), dtype=np.float64)
    for l in range(nl):
        for h in range(nh):
            A_block[l, h] = aggregate_token_to_block(A_mean[l, h])

    return A_block


# ═══════════════════════════════════════════════════════════════════
# Head screening
# ═══════════════════════════════════════════════════════════════════

def screen_heads_by_locality(A_block):
    """Rank all heads by locality (argmax D_manh). Returns sorted list."""
    nl, nh = A_block.shape[0], A_block.shape[1]
    manh_table = _make_manh_table(GRID)
    results = []
    for l in range(nl):
        for h in range(nh):
            A = A_block[l, h].copy()
            top1 = np.argmax(A, axis=1)
            d_manh = float(manh_table[np.arange(64), top1].mean())
            p_d1 = float((manh_table[np.arange(64), top1] <= 1).mean())
            results.append((l, h, d_manh, p_d1))
    # Sort by D_manh ascending (lower = more local)
    results.sort(key=lambda x: x[2])
    return results


# ═══════════════════════════════════════════════════════════════════
# Teacher rollout
# ═══════════════════════════════════════════════════════════════════

def rollout_custom(B, seeds, w_c=0.0, w_d=0.0, w_l=0.0):
    orders, ents = [], []
    for s in seeds:
        Bn = np.asarray(B, dtype=np.float64)
        N = Bn.shape[0]; rng = np.random.default_rng(s)
        S, U, last = [], list(range(N)), None; order = []
        for _ in range(N):
            if len(U) == 1:
                v = U[0]
            else:
                C, D, L, cand = teacher_components(Bn, S, U, last)
                q = w_c * C + w_d * D + w_l * L
                qq = q.copy()
                qq = (qq - qq.mean()) / (qq.std() + 1e-9)
                p = _softmax(qq, TAU_T)
                v = int(cand[rng.choice(len(cand), p=p)])
                ents.append(_entropy(p))
            order.append(v); S.append(v); U.remove(v); last = v
        orders.append(np.asarray(order, dtype=np.int64))
    return np.stack(orders), float(np.mean(ents)) if ents else 0.0


def rollout_one_shot(B, seeds, score_fn):
    scores = score_fn(np.asarray(B, dtype=np.float64))
    orders, ents = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        U = list(range(len(scores))); order = []
        for _ in range(len(scores)):
            if len(U) == 1: v = U[0]
            else:
                q = np.array([scores[u] for u in U], dtype=np.float64)
                q = (q - q.mean()) / (q.std() + 1e-9)
                p = _softmax(q, TAU_T)
                v = int(U[rng.choice(len(U), p=p)])
                ents.append(_entropy(p))
            order.append(v); U.remove(v)
        orders.append(np.asarray(order, dtype=np.int64))
    return np.stack(orders), float(np.mean(ents)) if ents else 0.0


def score_rowsum(B):  return B.sum(axis=1)
def score_colsum(B):  return B.sum(axis=0)
def score_readiness(B): return B.sum(axis=1) - B.sum(axis=0)


# ═══════════════════════════════════════════════════════════════════
# Teacher variants (same as text ablation)
# ═══════════════════════════════════════════════════════════════════

KEY_TEACHERS = [
    ("full (C−D+L)",    "rollout",  dict(w_c=1.0, w_d=-1.0, w_l=1.0)),
    ("no-S (−D+L)",     "rollout",  dict(w_c=0.0, w_d=-1.0, w_l=1.0)),
    ("no-U (C+L)",      "rollout",  dict(w_c=1.0, w_d= 0.0, w_l=1.0)),
    ("no-local (C−D)",  "rollout",  dict(w_c=1.0, w_d=-1.0, w_l=0.0)),
    ("−D only",          "rollout",  dict(w_c=0.0, w_d=-1.0, w_l=0.0)),
    ("wrong-D (C+D+L)", "rollout",  dict(w_c=1.0, w_d= 1.0, w_l=1.0)),
    ("C only",          "rollout",  dict(w_c=1.0, w_d= 0.0, w_l=0.0)),
    ("L only",          "rollout",  dict(w_c=0.0, w_d= 0.0, w_l=1.0)),
    ("rowsum ↓",        "oneshot",  score_rowsum),
    ("colsum ↓",        "oneshot",  score_colsum),
    ("readiness ↓",     "oneshot",  score_readiness),
    ("random",          "random",   None),
]

# Raster baseline: row-major [0..63] in 8×8 grid
RASTER_ORDER = np.arange(64, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manh_table = _make_manh_table(GRID)

    # Baselines
    raster_d, raster_p1 = order_locality(RASTER_ORDER, manh_table)
    rng = np.random.default_rng(42)
    rand_ds, rand_p1s = [], []
    for _ in range(100):
        o = rng.permutation(64)
        d, p1 = order_locality(o, manh_table)
        rand_ds.append(d); rand_p1s.append(p1)
    rand_d, rand_p1 = float(np.mean(rand_ds)), float(np.mean(rand_p1s))

    print(f"Baselines:")
    print(f"  raster:   D_manh={raster_d:.4f}  P(d≤1)={raster_p1:.4f}")
    print(f"  random:   D_manh={rand_d:.4f}  P(d≤1)={rand_p1:.4f}")
    print()

    # Load model
    print(f"Loading model: {CKPT_PATH}", flush=True)
    model, model_args = load_model(CKPT_PATH, DEVICE)
    nl, nh = model_args["n_layer"], model_args["n_head"]
    T = model_args["block_size"]
    print(f"  n_layer={nl} n_head={nh} n_embd={model_args['n_embd']} block_size={T}", flush=True)

    # Load data
    data = np.memmap(DATA_PATH, dtype=np.uint16, mode="r")
    print(f"  data: {DATA_PATH}, {len(data) // T} images available", flush=True)

    # Extract per-head attention
    A_block = extract_per_head_attention(model, data, N_IMAGES, M_PASSES, DEVICE)

    # Screen heads by locality
    head_ranking = screen_heads_by_locality(A_block)
    print(f"\nHead ranking by argmax D_manh (lower = more local):")
    print(f"{'rank':>5} {'L':>4} {'H':>4} {'D_manh':>9} {'P(d≤1)':>9}")
    for rank, (l, h, d, p1) in enumerate(head_ranking):
        marker = " ← BEST" if rank == 0 else ""
        print(f"{rank:>5} {l:>4} {h:>4} {d:>9.4f} {p1:>9.4f}{marker}")

    # Best head
    best_l, best_h, best_d, best_p1 = head_ranking[0]
    print(f"\nBest head: L{best_l}H{best_h} (D_manh={best_d:.4f}, P(d≤1)={best_p1:.4f})")

    # Also track L0H0 for comparison (same as text convention)
    l0h0_entry = next((x for x in head_ranking if x[0] == 0 and x[1] == 0), None)
    if l0h0_entry:
        print(f"L0H0 reference: D_manh={l0h0_entry[2]:.4f}, P(d≤1)={l0h0_entry[3]:.4f}")

    # Run ablation on top-3 heads
    top_heads = head_ranking[:3]  # best, 2nd, 3rd
    all_results = []

    for head_idx, (l, h, _, _) in enumerate(top_heads):
        label = f"L{l}H{h}"
        A = A_block[l, h].copy()
        B = A.T.copy()
        np.fill_diagonal(B, 0.0)

        print(f"\n{'='*70}")
        print(f"Ablation on {label}")
        print(f"{'='*70}")

        seeds = list(range(20000, 20000 + K_SEEDS))
        results = []

        for name, kind, params in KEY_TEACHERS:
            t0 = time.time()
            if kind == "rollout":
                orders, ent = rollout_custom(B, seeds, **params)
            elif kind == "oneshot":
                orders, ent = rollout_one_shot(B, seeds, params)
            elif kind == "random":
                orders = np.stack([np.random.default_rng(s).permutation(N_BLOCKS)
                                   for s in seeds])
                ent = np.log(N_BLOCKS)

            d_manh_vals = []
            p_d1_vals = []
            for o in orders:
                d, p1 = order_locality(o, manh_table)
                d_manh_vals.append(d)
                p_d1_vals.append(p1)

            d_mean = float(np.mean(d_manh_vals))
            d_std = float(np.std(d_manh_vals))
            p1_mean = float(np.mean(p_d1_vals))
            p1_std = float(np.std(p_d1_vals))

            # Also compute τ vs raster for reference
            from cdl_teacher_ablation import kendall_tau
            tau_vs_raster = float(np.mean([kendall_tau(o, RASTER_ORDER) for o in orders]))

            elapsed = time.time() - t0
            row = dict(name=name, D_manh=d_mean, D_manh_std=d_std,
                       P_d1=p1_mean, P_d1_std=p1_std,
                       tau_vs_raster=tau_vs_raster,
                       entropy=ent, time=elapsed)
            results.append(row)

            print(f"  {name:<22s} D_manh={d_mean:.4f}±{d_std:.4f}  "
                  f"P(d≤1)={p1_mean:.4f}  τ_vs_raster={tau_vs_raster:+.4f}  "
                  f"H={ent:.3f}  ({elapsed:.1f}s)")

        all_results.append(dict(head=label, l=int(l), h=int(h), results=results))

    # Summary table
    print(f"\n{'='*90}")
    print(f"SUMMARY: D_manh by teacher variant (lower = more local)")
    print(f"{'='*90}")
    teacher_names = [t[0] for t in KEY_TEACHERS]
    header = f"{'Teacher':<22s}"
    for hr in all_results:
        header += f"  {hr['head']:>10s}"
    print(header)
    print("-" * len(header))
    for tname in teacher_names:
        row = f"{tname:<22s}"
        for hr in all_results:
            r = next((r for r in hr["results"] if r["name"] == tname), None)
            if r:
                row += f"  {r['D_manh']:>10.4f}"
            else:
                row += f"  {'N/A':>10s}"
        print(row)

    # Δ vs raster
    print(f"\n{'='*90}")
    print(f"Δ D_manh vs raster ({raster_d:.4f}): negative = more local than raster")
    print(f"{'='*90}")
    print(header)
    print("-" * len(header))
    for tname in teacher_names:
        row = f"{tname:<22s}"
        for hr in all_results:
            r = next((r for r in hr["results"] if r["name"] == tname), None)
            if r:
                delta = r["D_manh"] - raster_d
                row += f"  {delta:>+10.4f}"
            else:
                row += f"  {'N/A':>10s}"
        print(row)

    # Baselines for reference
    print(f"\n{'raster':<22s}  {raster_d:>10.4f}")
    print(f"{'random':<22s}  {rand_d:>10.4f}")

    # Save
    out_data = {
        "ckpt": CKPT_PATH,
        "n_images": N_IMAGES, "m_passes": M_PASSES,
        "baselines": {"raster_D_manh": raster_d, "raster_P_d1": raster_p1,
                      "random_D_manh": rand_d, "random_P_d1": rand_p1},
        "head_ranking": [{"l": int(l), "h": int(h), "D_manh": d, "P_d1": p1}
                         for l, h, d, p1 in head_ranking],
        "ablations": all_results,
    }
    out_path = os.path.join(OUT_DIR, "image_teacher_ablation_locality.json")
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2, default=_json_default)
    print(f"\nSaved to {out_path}")


def _json_default(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    raise TypeError(f"not serializable: {type(obj)}")


if __name__ == "__main__":
    main()
