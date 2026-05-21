#!/usr/bin/env python3
"""Multi-probe structural characterization of an ORDER set (Phase-3 Task 3.3, probes A-C).
Answers 'what structure does this order have', beyond locality:
  A. spatial: mean_manh, p_nbr_le1
  B. B-edge-following: mean B[t,t+1] (raw + ratio vs random-order baseline), top-k follow rate
  C. centrality/readiness: Spearman(visit_time, out_degree), Spearman(visit_time, source)

Order-only + supplied B; no training, no model. Preview mode runs on the saved readout
diagnostic orders to test whether Bcov follows B-edges more than distance-only / Hilbert.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
from directed_graph_policy import build_directed_graph, compute_source

G = 8


def _manh_table():
    idx = np.arange(64); r, c = idx // G, idx % G
    return np.abs(r[:, None] - r[None, :]) + np.abs(c[:, None] - c[None, :])


def probe_order_set(orders, B, source, manh, rng):
    o = np.atleast_2d(np.asarray(orders))
    K, N = o.shape
    # A. spatial
    r, c = o // G, o % G
    step_manh = np.abs(np.diff(r, 1)) + np.abs(np.diff(c, 1))
    top1 = np.argmax(B + np.eye(N) * -1e9, 1)  # not used; placeholder
    # use argmax-of-A style nbr on the order's own steps:
    p_nbr_le1 = float((step_manh <= 1).mean())
    mean_manh = float(step_manh.mean())
    # B. B-edge-following
    edge_w = np.array([[B[o[k, t], o[k, t+1]] for t in range(N-1)] for k in range(K)])
    # random-order baseline: mean off-diagonal B
    base = float(B[~np.eye(N, dtype=bool)].mean())
    mean_edge = float(edge_w.mean())
    # top-4 follow rate: is next among current node's top-4 outgoing B edges?
    top4 = {i: set(np.argsort(-B[i])[:4]) for i in range(N)}
    follow = np.array([[1.0 if o[k, t+1] in top4[o[k, t]] else 0.0 for t in range(N-1)] for k in range(K)])
    top4_follow = float(follow.mean())
    # C. centrality/readiness: visit_time vs out_degree / source
    out_deg = B.sum(1)
    sp_out, sp_src = [], []
    for k in range(K):
        vt = np.empty(N); vt[o[k]] = np.arange(N)
        sp_out.append(spearmanr(vt, out_deg).correlation)
        sp_src.append(spearmanr(vt, source).correlation)
    return dict(
        mean_manh=round(mean_manh, 3), p_nbr_le1=round(p_nbr_le1, 3),
        mean_B_edge=round(mean_edge, 6), B_edge_ratio_vs_random=round(mean_edge / (base + 1e-12), 3),
        top4_edge_follow_rate=round(top4_follow, 3),
        spearman_visit_outdeg=round(float(np.mean(sp_out)), 3),
        spearman_visit_source=round(float(np.mean(sp_src)), 3),
    )


def main():
    A = np.load(_REPO / "probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy").astype(np.float64)
    B = build_directed_graph(A)
    source, _, _ = compute_source(B, 0.5)
    manh = _manh_table()
    rng = np.random.default_rng(0)
    od = _REPO / "probe_results_image_large/grw_e3ctrlsmall/readout_diagnostic/orders"
    names = ["random", "hilbert", "Bcov_distance_only", "Bcov_balanced", "Bcov_Bdominant_safety"]
    print(f"{'order':24s}{'manh':>7}{'P<=1':>7}{'B_edge_ratio':>13}{'top4_follow':>12}{'sp(vt,outdeg)':>14}{'sp(vt,src)':>11}")
    out = {}
    for n in names:
        f = od / f"{n}.npy"
        if not f.exists():
            print(f"{n:24s} (missing)"); continue
        m = probe_order_set(np.load(f), B, source, manh, rng); out[n] = m
        print(f"{n:24s}{m['mean_manh']:>7}{m['p_nbr_le1']:>7}{m['B_edge_ratio_vs_random']:>13}"
              f"{m['top4_edge_follow_rate']:>12}{m['spearman_visit_outdeg']:>14}{m['spearman_visit_source']:>11}")
    import json
    outdir = _REPO / "probe_results_image_large/structure_adaptive/structural_probes"
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outdir / "preview_on_saved_orders.json", "w"), indent=2)
    print(f"\nKey question: does Bcov_balanced follow B-edges MORE than distance_only / hilbert?")
    if "Bcov_balanced" in out and "Bcov_distance_only" in out:
        bb, dd = out["Bcov_balanced"], out["Bcov_distance_only"]
        print(f"  B_edge_ratio: Bcov={bb['B_edge_ratio_vs_random']} vs distance_only={dd['B_edge_ratio_vs_random']} "
              f"vs hilbert={out.get('hilbert',{}).get('B_edge_ratio_vs_random','-')}")
        print(f"  top4_follow:  Bcov={bb['top4_edge_follow_rate']} vs distance_only={dd['top4_edge_follow_rate']} "
              f"vs hilbert={out.get('hilbert',{}).get('top4_edge_follow_rate','-')}")
    print(f"wrote {outdir}/preview_on_saved_orders.json")


if __name__ == "__main__":
    main()
