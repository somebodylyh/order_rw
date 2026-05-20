#!/usr/bin/env python3
"""Structure-adaptive readout — Stage A (graph diagnostics) + Stage B (rule-based regime).

Given an attention-derived graph B (= A^T, diag-zeroed), characterize its structural
regime from graph statistics, then suggest a readout regime. The point is a UNIFIED
pipeline where modality-specific behavior emerges from graph structure, not from manually
assigning algorithms to data types:

    B -> g(B) [diagnostics] -> regime -> suggested readout

Key Stage-1 lesson encoded in the rule: readiness_strength (s = std(source)/mean|B|) is NOT
sufficient to identify text — E3-control-small image B has s=2.49 (high) yet is proximity.
So we test LOCALITY FIRST; only if locality is weak do we fall back to readiness.

locality needs the graph's native topology (image: 2D grid manhattan; text: 1D sequence
distance), passed per graph. All other metrics are topology-agnostic.
"""

import argparse, json, sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
from directed_graph_policy import build_directed_graph, compute_source


# ----------------------------- topology-agnostic metrics -----------------------------
def readiness_strength(B):
    source, _, _ = compute_source(B, alpha_dep=0.5)
    return float(np.std(source) / (np.mean(np.abs(B)) + 1e-12))


def asymmetry(B):
    return float(np.mean(np.abs(B - B.T)) / (np.mean(np.abs(B)) + 1e-12))


def row_entropy_norm(B):
    """Mean row entropy of |B| normalized to [0,1] (1 = uniform/noisy)."""
    P = np.abs(B).copy()
    np.fill_diagonal(P, 0.0)
    P = P / (P.sum(1, keepdims=True) + 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
    return float(np.mean(H) / np.log(B.shape[0] - 1))


def topk_mass(B, k):
    P = np.abs(B).copy()
    np.fill_diagonal(P, 0.0)
    P = P / (P.sum(1, keepdims=True) + 1e-12)
    part = np.sort(P, axis=1)[:, -k:].sum(1)
    return float(part.mean())


# ----------------------------- topology-aware locality -----------------------------
def _coords(N, topology, grid):
    if topology == "grid2d":
        idx = np.arange(N)
        return np.stack([idx // grid, idx % grid], 1).astype(float)  # (N,2)
    return np.arange(N).reshape(N, 1).astype(float)                   # (N,1) seq1d


def _pair_dist(coords, i, js):
    return np.abs(coords[js] - coords[i]).sum(1)


def directionality(B, topology, grid):
    """Concentration of argmax-edge directions. High (→1) = one dominant direction
    (causal/readiness, e.g. text L2R attends to i-1); low = spread over neighbors
    (spatial proximity, e.g. image 4-neighbor). This is what separates a directional
    1D readiness graph from a multi-directional 2D proximity graph."""
    N = B.shape[0]
    coords = _coords(N, topology, grid)
    bins = {}
    for i in range(N):
        j = int(np.argmax(B[i]))
        off = coords[j] - coords[i]
        key = tuple(np.sign(off).astype(int))
        if all(v == 0 for v in key):
            continue
        bins[key] = bins.get(key, 0) + 1
    if not bins:
        return 0.0
    return float(max(bins.values()) / sum(bins.values()))


def locality_metrics(B, topology, grid):
    N = B.shape[0]
    coords = _coords(N, topology, grid)
    # argmax outgoing edge distance per node
    argmax = np.array([int(np.argmax(B[i])) for i in range(N)])
    amd = np.array([_pair_dist(coords, i, [argmax[i]])[0] for i in range(N)])
    # random baseline expected distance
    all_d = np.array([_pair_dist(coords, i, [j for j in range(N) if j != i]).mean()
                      for i in range(N)])
    rand_dist = float(all_d.mean())
    # local-greedy traversal mean step distance
    vis = [0]; seen = {0}
    for _ in range(N - 1):
        last = vis[-1]
        cand = [j for j in range(N) if j not in seen]
        nxt = cand[int(np.argmax(B[last, cand]))]
        vis.append(nxt); seen.add(nxt)
    o = np.array(vis)
    steps = np.array([_pair_dist(coords, o[t], [o[t + 1]])[0] for t in range(N - 1)])
    greedy_dist = float(steps.mean())
    locality_score = float(np.clip((rand_dist - amd.mean()) / (rand_dist + 1e-12), 0, 1))
    return dict(
        argmax_dist=float(amd.mean()),
        p_nbr_le1=float((amd <= 1).mean()),
        rand_dist=rand_dist,
        local_greedy_dist=greedy_dist,
        locality_score=locality_score,
    )


# ----------------------------- Stage B: rule-based regime -----------------------------
def classify_regime(diag):
    loc = diag["locality_score"]
    ent = diag["row_entropy"]
    t1 = diag["top1_mass"]
    direc = diag["directionality"]
    # STRUCTURED FIRST: strong local edges. Then split by directionality:
    #   one dominant direction => causal/readiness (text L2R); spread => spatial proximity.
    if loc >= 0.5 and diag["p_nbr_le1"] >= 0.4:
        if direc >= 0.7:
            regime = "readiness_dominant"
            readout = "readiness-guided Graph-RW (progressive_rw_v3, high rho); directional 1D edges"
        else:
            regime = "proximity_dominant"
            readout = "coverage_readout (Bcov_balanced / Hilbert); single-step Graph-RW insufficient"
    elif ent >= 0.93 and t1 <= 0.05:
        regime = "uniform_noisy"
        readout = "no structured readout — random/mixed fallback; do not over-trust B"
    else:
        regime = "weak_mixed"
        readout = "mild structure — random-mixed continuation, validate before trusting B"
    return regime, readout


def diagnose(B, topology, grid):
    d = dict(
        readiness_strength=readiness_strength(B),
        asymmetry=asymmetry(B),
        row_entropy=row_entropy_norm(B),
        top1_mass=topk_mass(B, 1),
        top4_mass=topk_mass(B, 4),
    )
    d.update(locality_metrics(B, topology, grid))
    d["directionality"] = directionality(B, topology, grid)
    d["regime"], d["suggested_readout"] = classify_regime(d)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="probe_results_image_large/grw_e3ctrlsmall/graph_regime")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # (name, path, is_A_or_block, topology, grid)
    # A_global needs A->B and possibly aggregation; *_block / 64x64 are used directly as A then B=build_directed_graph.
    graphs = [
        ("E3_ctrl_small_image", "probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy", "grid2d", 8),
        ("E2_singletoken_vq_image", "probe_results_image/e2_imagenet32_vqf4_seq64/attention/A_global.npy", "grid2d", 8),
        ("E2_large_image", "probe_results_image/e2_large_imagenet32_vqf4_seq64_l8h8e512/attention/A_global.npy", "grid2d", 8),
        ("text_clean_method", "block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k/A_global_step50000.npy", "seq1d", 0),
    ]

    rng = np.random.default_rng(0)
    rows = {}
    for name, path, topo, grid in graphs:
        p = _REPO / path
        if not p.exists():
            print(f"[skip] {name}: {path} missing"); continue
        A = np.load(p).astype(np.float64)
        B = build_directed_graph(A)
        rows[name] = diagnose(B, topo, grid)
        if name == "E3_ctrl_small_image":
            # derived controls on the same topology
            Bsh = np.stack([B[i, rng.permutation(B.shape[0])] for i in range(B.shape[0])])
            rows["E3_shuffled_cols(ctrl)"] = diagnose(Bsh, "grid2d", 8)
            Brand = rng.random(B.shape); np.fill_diagonal(Brand, 0.0)
            rows["random_B(ctrl)"] = diagnose(Brand, "grid2d", 8)

    cols = ["readiness_strength", "asymmetry", "row_entropy", "top1_mass",
            "argmax_dist", "p_nbr_le1", "locality_score", "directionality"]
    hdr = ["graph"] + cols + ["regime", "suggested_readout"]
    with open(out / "regime_table.tsv", "w") as f:
        f.write("\t".join(hdr) + "\n")
        for name, d in rows.items():
            f.write(name + "\t" + "\t".join(f"{d[c]:.4f}" for c in cols)
                    + f"\t{d['regime']}\t{d['suggested_readout']}\n")
    with open(out / "regime_diagnostics.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"{'graph':26s}{'s_read':>8}{'entropy':>8}{'argmaxD':>8}{'P_nbr1':>8}{'locScr':>8}{'direc':>7}  regime")
    for name, d in rows.items():
        print(f"{name:26s}{d['readiness_strength']:8.3f}{d['row_entropy']:8.3f}"
              f"{d['argmax_dist']:8.3f}{d['p_nbr_le1']:8.3f}{d['locality_score']:8.3f}{d['directionality']:7.3f}  {d['regime']}")
    print(f"\nSuggested readouts:")
    for name, d in rows.items():
        print(f"  {name:26s} -> {d['suggested_readout']}")
    print(f"\nwrote {out}/regime_table.tsv, regime_diagnostics.json")


if __name__ == "__main__":
    main()
