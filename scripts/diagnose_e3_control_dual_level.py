"""Dual-granularity locality diagnostic for E3-control-small (N=256 + aggregated N=64).

Mirrors the schema of
probe_results_image_large/imagenet64_vqf4_full_l8h8e512_patch2x2_fixed/diagnostics.json
so cross-experiment comparison is direct.

Token-level: native 16x16 VQ-token grid (N=256, grid=16)
Block-level: 2x2 aggregation -> 8x8 block grid (N=64, grid=8). Aggregation
    is the same 2x2 mean used in diagnose_patch_aggregation.py.

For each level we report on:
  - real A
  - per-row shuffled-columns control
  - uniform-random control
Metrics: mean_manh, median_manh, P(d<=1), P(d<=2), s_readiness, hub_score,
plus A_mean/A_std/A_max for context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _make_manh_table(grid: int) -> np.ndarray:
    N = grid * grid
    rows = np.arange(N) // grid
    cols = np.arange(N) % grid
    rd = np.abs(rows[:, None] - rows[None, :])
    cd = np.abs(cols[:, None] - cols[None, :])
    return (rd + cd).astype(np.int64)


def _make_samequad_table(grid: int) -> np.ndarray:
    N = grid * grid
    rows = np.arange(N) // grid
    cols = np.arange(N) % grid
    qrow = rows // 2
    qcol = cols // 2
    same = (qrow[:, None] == qrow[None, :]) & (qcol[:, None] == qcol[None, :])
    return same


def stats_for(A: np.ndarray, name: str, grid: int) -> dict:
    N = A.shape[0]
    assert A.shape == (N, N) and N == grid * grid

    manh_tbl = _make_manh_table(grid)
    sameq_tbl = _make_samequad_table(grid)

    A_no_diag = A.copy()
    np.fill_diagonal(A_no_diag, -np.inf)
    top1 = A_no_diag.argmax(axis=1)
    qs = np.arange(N)

    manh = manh_tbl[qs, top1]
    sameq = sameq_tbl[qs, top1]

    in_deg = A.sum(axis=0)
    a_mean_abs = float(np.abs(A).mean())
    s_readiness = float(in_deg.std() / max(a_mean_abs, 1e-12))
    hub_score = float(in_deg.max() / max(in_deg.mean(), 1e-12))

    return {
        "label": name,
        "N": int(N),
        "grid": int(grid),
        "A_mean": float(A.mean()),
        "A_std": float(A.std()),
        "A_max": float(A.max()),
        "mean_manh": float(manh.mean()),
        "median_manh": float(np.median(manh)),
        "p_dist_le1": float((manh <= 1).mean()),
        "p_dist_le2": float((manh <= 2).mean()),
        "same_quad": float(sameq.mean()),
        "s_readiness": s_readiness,
        "hub_score": hub_score,
    }


def shuffle_cols_per_row(A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(A)
    N = A.shape[1]
    for i in range(A.shape[0]):
        out[i] = A[i, rng.permutation(N)]
    return out


# ---------------------------------------------------------------------------
# 16x16 token -> 8x8 block aggregation
# (matches block_lo_arm_order_network/diagnose_patch_aggregation.py)
# ---------------------------------------------------------------------------

TOKEN_GRID = 16
PATCH_GRID = 8


def token_to_patch_indices() -> np.ndarray:
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
    """Mean over the 2x2 token group for both query and key axis."""
    assert A_token.shape == (256, 256)
    p_idx = token_to_patch_indices()  # (256,) maps token -> block 0..63
    A_block = np.zeros((64, 64), dtype=np.float64)
    counts = np.zeros((64, 64), dtype=np.int64)
    for i in range(256):
        for j in range(256):
            A_block[p_idx[i], p_idx[j]] += A_token[i, j]
            counts[p_idx[i], p_idx[j]] += 1
    A_block /= np.maximum(counts, 1)
    np.fill_diagonal(A_block, 0.0)
    return A_block.astype(np.float32)


# ---------------------------------------------------------------------------
# SUMMARY.md writer
# ---------------------------------------------------------------------------

def _row_md(r: dict) -> str:
    return (f"| {r['label']:42s} | {r['mean_manh']:.3f} | {r['p_dist_le1']:.3f} | "
            f"{r['p_dist_le2']:.3f} | {r['same_quad']:.3f} | "
            f"{r['s_readiness']:.3f} | {r['hub_score']:.3f} |")


def _format_table(rows: list[dict]) -> str:
    head = (
        "| label | mean_manh | P(d≤1) | P(d≤2) | same_q | s_readiness | hub_score |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    return head + "\n".join(_row_md(r) for r in rows)


def write_summary(token_rows: list[dict], block_rows: list[dict], outdir: Path,
                  label: str) -> None:
    real_block = block_rows[0]
    shuf_block = block_rows[1]
    rand_block = block_rows[2]
    mm = real_block["mean_manh"]
    pd1 = real_block["p_dist_le1"]

    # Compare to references
    if mm < 4.0 and pd1 > 0.15:
        verdict = ("**patch aggregation 是充分条件**：即使 scale 缩到 l4h8e256，"
                   "patch2x2 aggregation 仍恢复了 E3 那种 weak-to-moderate spatial structure。"
                   "结合 E2-large 的 no-local 结果，最终结论是 **node granularity 决定 attention 几何**，"
                   "scale 不是主因。")
    elif mm > 5.2 and pd1 < 0.08:
        verdict = ("**patch aggregation 单独不够**：scale 与 aggregation 必须同时存在才能恢复 spatial structure。"
                   "E3 的 weak-local 来自二者交互作用。")
    else:
        verdict = ("部分恢复：patch aggregation 比 E2 系列有提升但弱于 E3-baseline；"
                   "scale × aggregation 表现 super-additive。")

    md = f"""# E3-control-small: ImageNet-64 VQ-f4 patch2x2 full (l4h8e256) — RESULT

**Date:** {Path(outdir).stat().st_mtime}
**Run:** {label}
**Out:** `{outdir}`

## Purpose
Scale-controlled version of E3-baseline. Identical training pipeline, only `n_layer 8→4` and `n_embd 512→256` (~6× fewer params). Tests whether **patch2x2 aggregation alone** is enough to produce E3's weak local attention (mean_manh ≈ 3.16, P(d≤1) ≈ 0.22).

## Token-level diagnostic (16×16 native VQ grid, N=256)

{_format_table(token_rows)}

## Block-level diagnostic (8×8 = 2×2 patch aggregation, N=64)

{_format_table(block_rows)}

## Cross-experiment comparison (block-level / 8×8 grid)

| Run | Repr | Grid | Model | mean_manh | P(d≤1) | regime |
|---|---|---|---|---:|---:|---|
| E0 CIFAR continuous | continuous+MSE | 8×8 native | l4h8e256 | 2.4 | — | strong local |
| E1 ImageNet32 continuous | continuous+MSE | 8×8 native | l4h8e256 | 1.00 | 1.00 | perfect local |
| E2-small | VQ+CE | 8×8 native | l4h8e256 | 5.03 | 0.109 | uniform |
| E2-large | VQ+CE | 8×8 native | l8h8e512 | 5.50 | 0.047 | uniform |
| E3-baseline | VQ+CE | 16×16→8×8 (patch2x2) | l8h8e512 | **3.16** | **0.219** | weak local |
| **E3-control-small** (this) | VQ+CE | 16×16→8×8 (patch2x2) | **l4h8e256** | **{mm:.2f}** | **{pd1:.3f}** | — |
| shuffled-cols control | — | — | — | {shuf_block['mean_manh']:.2f} | {shuf_block['p_dist_le1']:.3f} | random |
| uniform random | — | — | — | {rand_block['mean_manh']:.2f} | {rand_block['p_dist_le1']:.3f} | random |

## Verdict

{verdict}

## Notes
- Token-level metrics use Manhattan distance on the 16×16 raster grid (so uniform-random gives mean_manh ≈ 10.67).
- Block-level metrics use 8×8 grid (uniform-random gives mean_manh ≈ 4.67–5.5).
- Aggregation = mean over the 4 tokens in each 2×2 block (matches diagnose_patch_aggregation.py).
- Diagonal is zeroed at every level.
"""
    (outdir / "SUMMARY.md").write_text(md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a-global", type=Path, required=True,
                   help="Path to A_global.npy (256x256 token-level)")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--label", type=str, default="run")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    A_token = np.load(args.a_global).astype(np.float64)
    if A_token.shape != (256, 256):
        raise ValueError(f"Expected (256, 256) token-level A, got {A_token.shape}")

    print(f"Loaded A_token shape={A_token.shape}, mean={A_token.mean():.6f}, "
          f"max={A_token.max():.6f}")

    # --- aggregate ---
    A_block = aggregate_token_to_block(A_token)
    np.save(outdir / "A_block_8x8.npy", A_block)
    print(f"Saved A_block_8x8.npy ({A_block.shape}, mean={A_block.mean():.6f})")

    rng = np.random.default_rng(args.seed)

    # --- token-level ---
    token_rows = [
        stats_for(A_token, args.label, grid=16),
        stats_for(shuffle_cols_per_row(A_token, rng), f"{args.label} [shuffled cols]", grid=16),
        stats_for(rng.random((256, 256)).astype(np.float32), "uniform random", grid=16),
    ]
    # zero diagonal on the uniform random one to match real
    A_rand_z = rng.random((256, 256)).astype(np.float32)
    np.fill_diagonal(A_rand_z, 0.0)
    token_rows[2] = stats_for(A_rand_z, "uniform random", grid=16)

    # --- block-level ---
    block_rows = [
        stats_for(A_block, args.label, grid=8),
        stats_for(shuffle_cols_per_row(A_block, rng), f"{args.label} [shuffled cols]", grid=8),
    ]
    A_rand_b = rng.random((64, 64)).astype(np.float32)
    np.fill_diagonal(A_rand_b, 0.0)
    block_rows.append(stats_for(A_rand_b, "uniform random", grid=8))

    # --- write json ---
    diagnostics = {"token_level_16x16": token_rows, "block_level_8x8": block_rows}
    (outdir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"Wrote {outdir/'diagnostics.json'}")

    # --- pretty print ---
    def _print(rows, label):
        print(f"\n{label}")
        print(f"  {'label':45s} {'mean_manh':>10s} {'P(d<=1)':>8s} {'P(d<=2)':>8s} {'s_read':>8s} {'hub':>6s}")
        for r in rows:
            print(f"  {r['label']:45s} {r['mean_manh']:10.3f} {r['p_dist_le1']:8.3f} "
                  f"{r['p_dist_le2']:8.3f} {r['s_readiness']:8.3f} {r['hub_score']:6.3f}")
    _print(token_rows, "token-level (N=256, grid=16):")
    _print(block_rows, "block-level (N=64, grid=8):")

    # --- summary ---
    write_summary(token_rows, block_rows, outdir, args.label)
    print(f"\nWrote {outdir/'SUMMARY.md'}")


if __name__ == "__main__":
    main()
