"""Image per-head locality analysis (post-extraction, loads saved B_lh).

Primary metrics (2D spatial structure):
  D_manh  = mean consecutive Manhattan distance (↓ better)
  P(d=1)  = neighbor hit rate (4-connected, ↑ better)
  P(d≤2)  = relaxed neighbor hit rate (↑ better)
  long_jumps = count of steps with d ≥ 4 (↓ better)

Auxiliary:
  tau_vs_raster = Kendall tau against raster order (row-major alignment, not primary)
  row_conc = std(out - 0.5*in) normalized

References:
  Raster:       D_manh=1.778  (but raster has jump of 7 at row boundaries)
  Hilbert-like: D_manh~1.0–1.8
  Random:       D_manh≈5.34  P(d=1)≈0.06  long_jumps≈50
"""
from __future__ import annotations
import sys, argparse, json
import numpy as np
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "neural_readout"))
from teacher_labels import generate_teacher_label

GRID = 8
N = GRID * GRID  # 64
COORDS = np.array([(r, c) for r in range(GRID) for c in range(GRID)])


def manhattan(i, j):
    return abs(COORDS[i, 0] - COORDS[j, 0]) + abs(COORDS[i, 1] - COORDS[j, 1])


def locality_metrics(sigma):
    """Compute all locality metrics for a permutation sigma of N=64 blocks.

    Returns dict with D_manh, P_d1, P_dle2, long_jumps, max_jump.
    """
    sigma = np.asarray(sigma, dtype=np.int64)
    dists = [manhattan(int(sigma[t]), int(sigma[t + 1])) for t in range(N - 1)]
    dists = np.array(dists)
    return {
        "D_manh": float(np.mean(dists)),
        "P_d1": float(np.mean(dists == 1)),
        "P_dle2": float(np.mean(dists <= 2)),
        "long_jumps": int(np.sum(dists >= 4)),
        "max_jump": int(np.max(dists)),
    }


def tau_vs_raster(sigma):
    rank = np.empty(N, dtype=np.int64)
    rank[sigma] = np.arange(N)
    raster = np.arange(N)
    n = N * (N - 1) // 2
    conc = 0
    for i in range(N):
        for j in range(i + 1, N):
            if (rank[i] - rank[j]) * (raster[i] - raster[j]) > 0:
                conc += 1
            elif (rank[i] - rank[j]) * (raster[i] - raster[j]) < 0:
                conc -= 1
    return conc / n


def row_concentration(B):
    r = B.sum(axis=1) - 0.5 * B.sum(axis=0)
    r = r / (np.abs(r).mean() + 1e-12)
    return float(np.std(r))


def B_spatial_conc(B):
    """Fraction of |B| mass on spatial 1-hop neighbors."""
    mask = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and manhattan(i, j) == 1:
                mask[i, j] = 1.0
    return float(np.abs(B * mask).sum() / max(np.abs(B).sum(), 1e-12))


def raster_metrics():
    return locality_metrics(np.arange(N))


def random_baseline(n_samples=5000, seed=42):
    rng = np.random.default_rng(seed)
    metrics = [locality_metrics(rng.permutation(N)) for _ in range(n_samples)]
    return {k: (float(np.mean([m[k] for m in metrics])),
                float(np.std([m[k] for m in metrics])))
            for k in metrics[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="path to per_head_scan.npz from scan_image_heads.py")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    B_lh = data["B_lh"]  # (L, H, 64, 64)
    L, H = B_lh.shape[:2]

    # Baselines
    rast = raster_metrics()
    rand = random_baseline(5000, 42)

    print("=" * 72)
    print("IMAGE PER-HEAD LOCALITY ANALYSIS")
    print("=" * 72)
    print(f"\n  Reference orders:")
    print(f"    Raster:     D_manh={rast['D_manh']:.3f}  P(d=1)={rast['P_d1']:.3f}  "
          f"P(d≤2)={rast['P_dle2']:.3f}  long_jumps={rast['long_jumps']}")
    print(f"    Random:     D_manh={rand['D_manh'][0]:.3f}±{rand['D_manh'][1]:.3f}  "
          f"P(d=1)={rand['P_d1'][0]:.3f}  P(d≤2)={rand['P_dle2'][0]:.3f}  "
          f"long_jumps≈{rand['long_jumps'][0]:.0f}")

    # --- Per-head analysis ---
    all_heads = []
    for layer in range(L):
        for head in range(H):
            B = B_lh[layer, head]
            sigma, _, _ = generate_teacher_label(B)
            loc = locality_metrics(sigma)
            tau = tau_vs_raster(sigma)
            rc = row_concentration(B)
            sc = B_spatial_conc(B)
            all_heads.append({
                "layer": layer, "head": head,
                **loc, "tau_vs_raster": tau, "row_conc": rc, "spatial_conc": sc,
            })

    # Sort by D_manh (best locality first)
    all_heads.sort(key=lambda h: h["D_manh"])

    # --- Print grids ---
    def grid_of(key, fmt="{:>7.3f}", label=""):
        g = np.zeros((L, H))
        for h in all_heads:
            g[h["layer"], h["head"]] = h[key]
        print(f"\n--- {label or key} ---")
        hdr = "L\\H " + "".join("    H{}   ".format(h) for h in range(H))
        print(hdr)
        for layer in range(L):
            row = " L{} ".format(layer)
            for head in range(H):
                v = g[layer, head]
                row += (fmt + " ").format(v)
            print(row)

    grid_of("D_manh", "{:>7.3f}", "D_manh (mean Manhattan distance, ↓ better)")
    grid_of("P_d1", "{:>7.3f}", "P(d=1) — 4-neighbor hit rate (↑ better)")
    grid_of("P_dle2", "{:>7.3f}", "P(d≤2) — relaxed neighbor hit (↑ better)")
    grid_of("long_jumps", "{:>7.0f}", "Long jumps (d≥4, ↓ better)")

    # --- Top heads ---
    print("\n" + "=" * 72)
    print("TOP 12 HEADS BY LOCALITY (D_manh ↑)")
    print("=" * 72)
    hdr = "{:>6}  {:>8}  {:>6}  {:>6}  {:>6}  {:>7}  {:>7}  {:>6}".format(
        "Head", "D_manh", "P(d=1)", "P(d≤2)", "long_j", "τ_raster", "row_conc", "sp_conc")
    print(hdr)
    print("-" * 72)
    for r in all_heads[:12]:
        print("{:>6}  {:>8.3f}  {:>6.3f}  {:>6.3f}  {:>6}  {:>+7.3f}  {:>7.3f}  {:>6.3f}".format(
            f"L{r['layer']}H{r['head']}", r["D_manh"], r["P_d1"], r["P_dle2"],
            r["long_jumps"], r["tau_vs_raster"], r["row_conc"], r["spatial_conc"]))

    # --- Best by different criteria ---
    print("\n--- Best head by different criteria ---")
    criteria = [
        ("D_manh (locality)", "D_manh", False),
        ("P(d=1) (neighbor hit)", "P_d1", True),
        ("|tau_vs_raster|", "tau_vs_raster", True),
        ("row_conc", "row_conc", True),
        ("spatial_conc (B 1-hop mass)", "spatial_conc", True),
    ]
    for name, key, reverse in criteria:
        if reverse:
            sorted_heads = sorted(all_heads, key=lambda h: -abs(h[key]) if 'tau' in key else -h[key])
        else:
            sorted_heads = sorted(all_heads, key=lambda h: h[key])
        best = sorted_heads[0]
        print(f"  {name:30s}: L{best['layer']}H{best['head']}  "
              f"D_manh={best['D_manh']:.3f}  P(d=1)={best['P_d1']:.3f}  "
              f"τ_raster={best['tau_vs_raster']:+.3f}")

    # --- Layer stats ---
    print("\n--- Layer means (D_manh) ---")
    for layer in range(L):
        dms = [h["D_manh"] for h in all_heads if h["layer"] == layer]
        print(f"  L{layer}: mean D_manh={np.mean(dms):.3f}  "
              f"min={min(dms):.3f}(H{np.argmin([h['D_manh'] for h in all_heads if h['layer']==layer])})  "
              f"max={max(dms):.3f}")

    # --- Correlation matrix ---
    print("\n--- Metric correlations (across 64 heads) ---")
    keys = ["D_manh", "P_d1", "P_dle2", "long_jumps", "tau_vs_raster", "row_conc", "spatial_conc"]
    n_heads = len(all_heads)
    mat = np.zeros((len(keys), len(keys)))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            vi = [h[ki] for h in all_heads]
            vj = [h[kj] for h in all_heads]
            mat[i, j] = np.corrcoef(vi, vj)[0, 1]
    print("           " + "  ".join("{:>8}".format(k[:8]) for k in keys))
    for i, ki in enumerate(keys):
        row = "  {:8}  ".format(ki[:8])
        row += "  ".join("{:>+8.3f}".format(mat[i, j]) for j in range(len(keys)))
        print(row)


if __name__ == "__main__":
    main()
