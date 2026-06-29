"""Aggregate 16x16 token-level attention → 8x8 patch-level, then run Graph-RW diagnostics.

Patch mapping (row-major 16x16 VQ token grid):
  block (br, bc) in 8x8 grid contains tokens:
    i     = 2*br*16 + 2*bc       (top-left)
    i+1   = 2*br*16 + 2*bc + 1   (top-right)
    i+16  = 2*br*16 + 16 + 2*bc  (bottom-left)
    i+17  = 2*br*16 + 16 + 2*bc+1 (bottom-right)
  = {i, i+1, i+16, i+17}

A_block[a, b] = mean_{i in block_a, j in block_b} A_token[i, j]

Usage:
    python block_lo_arm_order_network/diagnose_patch_aggregation.py \
        --a-path probe_results_image_large/imagenet64_vqf4_140k_l8h8e512/A_global.npy \
        --output-dir probe_results_image_large/imagenet64_vqf4_140k_l8h8e512/patch_8x8 \
        --K 500
"""

import argparse, json, os, sys, time
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from directed_graph_policy import build_directed_graph, compute_source
from order_diagnostics import _kendall_tau

# Pull in ablation machinery from the shared script
import importlib.util
_abl_path = os.path.join(_REPO, "scripts", "ablate_graph_policy_from_ckpt.py")
_spec = importlib.util.spec_from_file_location("ablate_graph_policy", _abl_path)
_abl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_abl)

TOKEN_GRID = 16
PATCH_GRID = 8
TOKENS_PER_PATCH = 4  # 2x2


def token_to_patch_indices() -> np.ndarray:
    """Return (TOKEN_GRID*TOKEN_GRID,) array mapping token_idx → patch_idx."""
    patch_idx = np.zeros(TOKEN_GRID * TOKEN_GRID, dtype=np.int64)
    for br in range(PATCH_GRID):
        for bc in range(PATCH_GRID):
            p = br * PATCH_GRID + bc
            # Tokens in this 2x2 block
            i_tl = 2 * br * TOKEN_GRID + 2 * bc
            for (dr, dc) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                token_i = i_tl + dr * TOKEN_GRID + dc
                patch_idx[token_i] = p
    return patch_idx


def aggregate_attention(A_token: np.ndarray) -> np.ndarray:
    """Aggregate A_token (256, 256) → A_block (64, 64) by 2×2 patches.

    A_block[a, b] = mean_{i in P_a, j in P_b} A_token[i, j]
    """
    assert A_token.shape == (256, 256), f"Expected (256,256), got {A_token.shape}"
    p2t = token_to_patch_indices()
    N_patch = PATCH_GRID * PATCH_GRID  # 64

    A_block = np.zeros((N_patch, N_patch), dtype=np.float64)
    counts = np.zeros((N_patch, N_patch), dtype=np.float64)

    for i in range(256):
        pi = p2t[i]
        for j in range(256):
            pj = p2t[j]
            A_block[pi, pj] += A_token[i, j]
            counts[pi, pj] += 1.0

    A_block /= counts
    return A_block


def verify_hub_token(A_token: np.ndarray) -> dict:
    """Check if hub-spoke pattern survives aggregation — identify top token sources."""
    # Out-strength per token
    out_strength = np.asarray(A_token, dtype=np.float64).sum(axis=1)  # (256,)
    top_out = np.argsort(out_strength)[::-1][:10]

    top_info = {}
    for rank, tok in enumerate(top_out[:5]):
        row, col = divmod(tok, TOKEN_GRID)
        top_info[f"top{rank+1}"] = {
            "token_idx": int(tok),
            "grid_pos": f"({row},{col})",
            "out_strength": float(out_strength[tok]),
        }
    return top_info


def run_random_raster_references(K: int, N: int, seed: int):
    """Compute random and raster reference for 8x8 grid."""
    rand_orders = _abl.random_orders_fn(K, N, seed)
    rast_orders = _abl.raster_orders(K, N, seed)
    rand_img = _abl.compute_image_metrics(rand_orders, grid=PATCH_GRID)
    rast_img = _abl.compute_image_metrics(rast_orders, grid=PATCH_GRID)
    return rand_img, rast_img


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a-path", required=True, help="Path to A_global.npy (256x256)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--K", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lam", type=float, default=_abl.DEFAULTS["lam"])
    p.add_argument("--rho", type=float, default=_abl.DEFAULTS["rho"])
    p.add_argument("--tau-start", type=float, default=_abl.DEFAULTS["tau_start"])
    p.add_argument("--tau-step", type=float, default=_abl.DEFAULTS["tau_step"])
    p.add_argument("--alpha-dep", type=float, default=_abl.DEFAULTS["alpha_dep"])
    p.add_argument("--top-k", type=int, default=_abl.DEFAULTS["top_k"])
    p.add_argument("--epsilon", type=float, default=_abl.DEFAULTS["epsilon"])
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--sweep", choices=["rho", "lambda"], default=None)
    args = p.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load and aggregate ──
    print(f"Loading A_token from {args.a_path}")
    A_token = np.load(args.a_path)
    if A_token.ndim == 3:
        print(f"  A shape: {A_token.shape}, averaging over axis 0")
        A_token = A_token.mean(axis=0).astype(np.float64)
        np.fill_diagonal(A_token, 0.0)
    else:
        A_token = np.asarray(A_token, dtype=np.float64)

    assert A_token.shape == (256, 256), f"Expected (256,256), got {A_token.shape}"
    print(f"  A_token: mean={A_token.mean():.6f}, std={A_token.std():.6f}, "
          f"max={A_token.max():.6f}")

    # Hub token check
    hub_info = verify_hub_token(A_token)
    print("  Top out-strength tokens:")
    for k, v in hub_info.items():
        print(f"    {k}: token={v['token_idx']} grid{v['grid_pos']} out={v['out_strength']:.6f}")

    # Aggregate
    print(f"\nAggregating 16x16 → 8x8 (2x2 patches)...")
    A_block = aggregate_attention(A_token)
    print(f"  A_block: shape={A_block.shape}, mean={A_block.mean():.6f}, std={A_block.std():.6f}")

    # Build B
    B_block = build_directed_graph(A_block)
    np.fill_diagonal(B_block, 0.0)
    print(f"  B_block: mean={B_block.mean():.6f}, std={B_block.std():.6f}")

    # Graph diagnostics
    asym = _abl.graph_asymmetry(B_block)
    sig = _abl.graph_readiness_signal(B_block)
    print(f"  graph_asymmetry: {asym:.4f}")
    print(f"  readiness_signal: {sig:.4f}")

    # Save A_block and B_block
    np.save(os.path.join(output_dir, "A_block_8x8.npy"), A_block.astype(np.float32))
    np.save(os.path.join(output_dir, "B_block_8x8.npy"), B_block.astype(np.float32))
    print(f"  Saved A_block_8x8.npy, B_block_8x8.npy")

    # ── Step 2: Spatial check on A_block ──
    print(f"\nSpatial check on 8x8 block-level attention:")
    A_grid = A_block.reshape(PATCH_GRID, PATCH_GRID, PATCH_GRID, PATCH_GRID)
    # Mean attention within same block vs different blocks
    diag_blocks = np.array([A_grid[r, c, r, c] for r in range(PATCH_GRID) for c in range(PATCH_GRID)])
    print(f"  intra-block (same patch) mean: {diag_blocks.mean():.6f}")

    # Adjacent blocks (d<=1 in block units)
    near_sum, near_cnt = 0.0, 0
    far_sum, far_cnt = 0.0, 0
    for r1 in range(PATCH_GRID):
        for c1 in range(PATCH_GRID):
            for r2 in range(PATCH_GRID):
                for c2 in range(PATCH_GRID):
                    a_val = A_grid[r1, c1, r2, c2]
                    d = abs(r1 - r2) + abs(c1 - c2)
                    if d <= 1:
                        near_sum += a_val
                        near_cnt += 1
                    if d >= 4:
                        far_sum += a_val
                        far_cnt += 1
    near_mean = near_sum / near_cnt if near_cnt else 0
    far_mean = far_sum / far_cnt if far_cnt else 0
    print(f"  near (d<=1 block) mean: {near_mean:.6f}")
    print(f"  far  (d>=4 block) mean: {far_mean:.6f}")
    print(f"  near/far ratio: {near_mean/far_mean if far_mean else float('inf'):.4f}")

    # ── Step 3: Run policy ablation ──
    base_params = {
        "lam": args.lam, "rho": args.rho,
        "tau_start": args.tau_start, "tau_step": args.tau_step,
        "alpha_dep": args.alpha_dep, "top_k": args.top_k,
        "epsilon": args.epsilon,
    }

    if args.sweep:
        print(f"\n{'='*60}")
        print(f"Running sweep: {args.sweep}")
        _abl.run_sweep(B_block, base_params, args.K, args.seed, "image", args.sweep, output_dir)
        print("Done.")
        return

    variants = args.variants if args.variants else _abl.ALL_VARIANTS
    print(f"\n{'='*60}")
    print(f"Running policy ablation on 8x8 patch-level attention")
    print(f"K={args.K}, variants={len(variants)}: {variants}")
    print(f"Params: lam={args.lam}, rho={args.rho}, tau_start={args.tau_start}, "
          f"tau_step={args.tau_step}, top_k={args.top_k}, eps={args.epsilon}")
    print(f"{'='*60}")

    results = _abl.run_ablation(B_block, base_params, args.K, args.seed, "image", variants)

    # References
    print("\nReferences:")
    rand_img, rast_img = run_random_raster_references(args.K, B_block.shape[0], args.seed)
    print(f"  {'random':22s}  manh={rand_img['mean_manhattan']:.3f}  "
          f"P(d≤1)={rand_img['P_d_le_1']:.4f}  same_q={rand_img['same_quadrant']:.4f}")
    print(f"  {'raster':22s}  manh={rast_img['mean_manhattan']:.3f}  "
          f"P(d≤1)={rast_img['P_d_le_1']:.4f}  same_q={rast_img['same_quadrant']:.4f}")
    results["_reference_random"] = rand_img
    results["_reference_raster"] = rast_img

    # ── Step 4: Save ──
    print(f"\nSaving results...")
    _abl.save_results(results, output_dir, "image", base_params, B_block, args.K)

    # Save hub info
    with open(os.path.join(output_dir, "hub_token_info.json"), "w") as f:
        json.dump(hub_info, f, indent=2)

    # Save aggregation metadata
    meta = {
        "aggregation": "16x16 token → 8x8 patch (2x2 VQ tokens per patch)",
        "patch_mapping": "{i, i+1, i+16, i+17} for block (br, bc) at i = 2*br*16 + 2*bc",
        "token_grid": TOKEN_GRID,
        "patch_grid": PATCH_GRID,
        "tokens_per_patch": TOKENS_PER_PATCH,
        "source_A_path": args.a_path,
        "A_token_stats": {
            "mean": float(A_token.mean()), "std": float(A_token.std()),
            "max": float(A_token.max()),
        },
        "A_block_stats": {
            "mean": float(A_block.mean()), "std": float(A_block.std()),
        },
        "hub_tokens": hub_info,
    }
    with open(os.path.join(output_dir, "aggregation_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Step 5: Plots ──
    if not args.no_plots:
        print("Generating plots...")
        _abl.make_plots(results, output_dir, "image", B_block)

    print(f"\nDone. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
