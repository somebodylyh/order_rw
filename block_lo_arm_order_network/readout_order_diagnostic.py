#!/usr/bin/env python3
"""Round-2 readout order diagnostic (L1 structural layer only — NOT training).

Implements every candidate readout that turns an attention-derived block graph B
(physical 8x8 raster frame) into a generation order, then reports structural
statistics. locality (mean_manh) is a DIAGNOSTIC, not the objective: the final
criterion is task loss / cross-order robustness / sample quality (done elsewhere).

Readouts:
  random, raster, serpentine, hilbert, v1_graph_rw, local_greedy,
  B_coverage variants: distance_only(B tie-break), balanced, B_dominant_safety

The B_coverage variants vary how much the attention graph B (vs pure spatial
distance) drives selection, so that we can later test in training whether B's
directed structure adds anything over a generic space-filling curve. We also
report each variant's divergence from the distance_only order, to confirm B
materially changes the order (not just tie-breaking).

Outputs (under --outdir):
  metrics.tsv          one row per readout with L1 stats + metadata
  orders/<name>.npy    sampled orders (K, 64)
  visit_grids.txt      8x8 visit-time grid for one sample of each readout
  metadata.json        B source, frame, params, seeds
"""

import argparse, json, sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
sys.path.insert(0, str(_REPO / "image_order"))
from directed_graph_policy import build_directed_graph, compute_source
from graph_rw_image import IMAGE_RW_PARAMS_DEFAULT, sample_image_orders_batch

G = 8
N = G * G


# ----------------------------- structural metrics -----------------------------
def order_stats(orders):
    o = np.atleast_2d(np.asarray(orders))
    r, c = o // G, o % G
    d = np.abs(np.diff(r, 1)) + np.abs(np.diff(c, 1))
    same_q = (r[:, :-1] // 4 == r[:, 1:] // 4) & (c[:, :-1] // 4 == c[:, 1:] // 4)
    return dict(
        mean_manh=float(d.mean()),
        p_d_le1=float((d <= 1).mean()),
        p_d_le2=float((d <= 2).mean()),
        same_quad=float(same_q.mean()),
    )


def visit_grid(order):
    t = np.zeros((G, G), dtype=int)
    for step, idx in enumerate(np.asarray(order)):
        t[idx // G, idx % G] = step
    return t


def order_disagreement(orders_a, orders_b):
    """Fraction of reveal positions where the two order families differ (mean over samples)."""
    a, b = np.atleast_2d(orders_a), np.atleast_2d(orders_b)
    n = min(a.shape[0], b.shape[0])
    return float((a[:n] != b[:n]).mean())


# ----------------------------- deterministic readouts -----------------------------
def raster_order():
    return np.arange(N)


def serpentine_order():
    out = []
    for r in range(G):
        cols = range(G) if r % 2 == 0 else range(G - 1, -1, -1)
        out += [r * G + c for c in cols]
    return np.array(out)


def hilbert_order():
    out = []
    for d in range(N):
        x = y = 0; t = d; s = 1
        while s < G:
            rx = 1 & (t // 2); ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x; y = s - 1 - y
                x, y = y, x
            x += s * rx; y += s * ry; t //= 4; s *= 2
        out.append(y * G + x)
    return np.array(out)


def local_greedy_order(B, start=0):
    vis = [start]; seen = {start}
    for _ in range(N - 1):
        last = vis[-1]
        nxt = max((B[last, j], j) for j in range(N) if j not in seen)[1]
        vis.append(nxt); seen.add(nxt)
    return np.array(vis)


def _rc(i):
    return i // G, i % G


def b_coverage_order(B, gamma_B, gamma_d, start=0, eps=1e-12):
    """B-guided coverage path with distance penalty.

    Per step, among unvisited v: normalize B[last,v] and manhattan distance to
    [0,1] across current candidates, then score = gamma_B*Bn - gamma_d*dn.
    gamma_B large => B materially drives selection; gamma_d provides locality.
    Deterministic argmax (for diagnostic reproducibility).
    """
    vis = [start]; seen = {start}
    for _ in range(N - 1):
        last = vis[-1]; lr, lc = _rc(last)
        cand = [j for j in range(N) if j not in seen]
        bs = np.array([B[last, j] for j in cand], dtype=np.float64)
        ds = np.array([abs(lr - j // G) + abs(lc - j % G) for j in cand], dtype=np.float64)
        bn = (bs - bs.min()) / (bs.ptp() + eps)
        dn = (ds - ds.min()) / (ds.ptp() + eps)
        score = gamma_B * bn - gamma_d * dn
        nxt = cand[int(np.argmax(score))]
        vis.append(nxt); seen.add(nxt)
    return np.array(vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-block-path",
                    default="probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy")
    ap.add_argument("--outdir", default="probe_results_image_large/grw_e3ctrlsmall/readout_diagnostic")
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.outdir); (out / "orders").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    A = np.load(args.a_block_path).astype(np.float64)
    assert A.shape == (N, N)
    B = build_directed_graph(A)
    source, out_deg, in_deg = compute_source(B, alpha_dep=0.5)
    s_readiness = float(np.std(source) / (np.mean(np.abs(B)) + 1e-12))

    K = args.n_samples
    v1_params = {**IMAGE_RW_PARAMS_DEFAULT, "top_k": 4, "epsilon_uniform": 0.0}

    readouts = {}
    readouts["random"] = np.stack([rng.permutation(N) for _ in range(K)])
    readouts["raster"] = np.tile(raster_order(), (K, 1))
    readouts["serpentine"] = np.tile(serpentine_order(), (K, 1))
    readouts["hilbert"] = np.tile(hilbert_order(), (K, 1))
    readouts["v1_graph_rw"] = sample_image_orders_batch(
        B, v1_params, K, seed_base=args.seed + 1, step=0).cpu().numpy()
    readouts["local_greedy"] = np.stack([local_greedy_order(B, st % N) for st in range(K)])
    # B_coverage variants — gamma_B : gamma_d ratio controls how much B drives selection
    readouts["Bcov_distance_only"] = np.stack([b_coverage_order(B, 0.01, 1.0, st % N) for st in range(K)])
    readouts["Bcov_balanced"] = np.stack([b_coverage_order(B, 1.0, 1.0, st % N) for st in range(K)])
    readouts["Bcov_Bdominant_safety"] = np.stack([b_coverage_order(B, 1.0, 0.2, st % N) for st in range(K)])

    # metrics + save
    rows = []
    ref = readouts["Bcov_distance_only"]
    for name, ords in readouts.items():
        np.save(out / "orders" / f"{name}.npy", ords)
        st = order_stats(ords)
        disagree = order_disagreement(ords, ref) if name.startswith("Bcov") else float("nan")
        rows.append((name, st, disagree))

    hdr = "readout\tmean_manh\tp_d_le1\tp_d_le2\tsame_quad\tdisagree_vs_distonly\n"
    with open(out / "metrics.tsv", "w") as f:
        f.write(hdr)
        for name, st, dis in rows:
            f.write(f"{name}\t{st['mean_manh']:.4f}\t{st['p_d_le1']:.4f}\t{st['p_d_le2']:.4f}\t"
                    f"{st['same_quad']:.4f}\t{dis:.4f}\n")

    with open(out / "visit_grids.txt", "w") as f:
        for name, ords in readouts.items():
            f.write(f"=== {name} (sample 0) ===\n")
            g = visit_grid(ords[0])
            for r in range(G):
                f.write("  " + " ".join(f"{g[r, c]:2d}" for c in range(G)) + "\n")
            f.write("\n")

    meta = dict(
        a_block_path=args.a_block_path,
        B_frame="physical_raster_8x8",
        note="B = A^T diag-zeroed, NOT row-normalized; orders are PHYSICAL frame (apply inverse_block_perm before feeding permuted-data model).",
        s_readiness=s_readiness, n_samples=K, seed=args.seed,
        v1_params=v1_params,
        b_coverage_variants={"distance_only": [0.01, 1.0], "balanced": [1.0, 1.0],
                             "Bdominant_safety": [1.0, 0.2]},
    )
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"s_readiness = {s_readiness:.4f}")
    print(f"{'readout':24s}{'manh':>8}{'P<=1':>8}{'P<=2':>8}{'sameQ':>8}{'disagree':>10}")
    for name, st, dis in rows:
        disp = f"{dis:.3f}" if dis == dis else "—"
        print(f"{name:24s}{st['mean_manh']:8.3f}{st['p_d_le1']:8.3f}{st['p_d_le2']:8.3f}{st['same_quad']:8.3f}{disp:>10}")
    print(f"\nwrote {out}/metrics.tsv, orders/, visit_grids.txt, metadata.json")


if __name__ == "__main__":
    main()
