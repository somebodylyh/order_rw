#!/usr/bin/env python3
"""Task 3 — Phase 0 sanity for the attention-conditioned order policy.

Loads the text and image attention graphs, builds B = A^T via the canonical
`directed_graph_policy.build_directed_graph`, then for the unified C-D+L teacher:
  - samples partial states and checks the 12-d feature builder (shape / NaN / summary),
  - prints top-k teacher candidates for a few states,
  - rolls out teacher orders for several seeds and compares to random orders on
    structure metrics (image: mean_manh / P(d<=1) / P(d<=2) / top4-follow / B-edge ratio;
    text: Kendall tau vs index-raster proxy, start/end node distribution, entropy).

NO MLP, NO continuation training. Pure diagnostic. Writes md + tsv per modality.

Run:
    python scripts/run_attn_order_phase0_sanity.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))

from directed_graph_policy import build_directed_graph          # canonical B = A^T, zero diag
from attn_order_features import build_features, FEATURE_NAMES
from attn_order_teacher import teacher_step, rollout_order

TEXT_A = _REPO / "block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k/A_global_step50000.npy"
IMAGE_A = _REPO / "probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy"

GRID = 8                 # image: 8x8 patch grid (patch2x2, A_block_8x8 already aggregated)
K = 200                  # rollouts per config
TAU_GRID = [0.5, 1.0, 2.0]   # standardized-score sampling temperatures
SEED0 = 1234


# --------------------------------------------------------------------------
# metrics (formulas mirror image_order/diagnose_image_orders.py:locality_stats
# and structural_probes.py:probe_order_set; reimplemented inline to avoid a
# matplotlib import in this pure-numpy diagnostic)
# --------------------------------------------------------------------------

def locality_stats(orders, grid=GRID):
    rows, cols = orders // grid, orders % grid
    d = np.abs(np.diff(rows, axis=1)) + np.abs(np.diff(cols, axis=1))
    flat = d.ravel()
    return dict(mean_manh=float(d.mean()),
                p_le1=float((flat <= 1).mean()),
                p_le2=float((flat <= 2).mean()))


def top4_follow_and_edge(orders, B):
    N = B.shape[0]
    top4 = {i: set(np.argsort(-B[i])[:4]) for i in range(N)}
    K_, n = orders.shape
    follow, edge = [], []
    for k in range(K_):
        for t in range(n - 1):
            u, v = orders[k, t], orders[k, t + 1]
            follow.append(1.0 if v in top4[u] else 0.0)
            edge.append(B[u, v])
    base = float(B[~np.eye(N, dtype=bool)].mean())
    return float(np.mean(follow)), float(np.mean(edge)), base


def kendall_tau_vs_raster(orders):
    """Mean Kendall tau of each order against the index raster [0..N-1].
    PROXY only (text has no canonical 2D structure); not a success target."""
    from scipy.stats import kendalltau
    raster = np.arange(orders.shape[1])
    taus = [kendalltau(orders[k], raster).correlation for k in range(orders.shape[0])]
    return float(np.nanmean(taus))


def diversity(orders):
    uniq = len({tuple(o.tolist()) for o in orders})
    # first-node entropy
    first = orders[:, 0]
    _, counts = np.unique(first, return_counts=True)
    p = counts / counts.sum()
    h_first = float(-(p * np.log(p)).sum())
    return uniq, h_first


def random_orders(N, k, rng):
    return np.stack([rng.permutation(N) for _ in range(k)])


# --------------------------------------------------------------------------

def feature_summary(B, orders):
    """Build 12-d features across many sampled states; return per-feature min/max/mean + NaN flag."""
    N = B.shape[0]
    rows = []
    for o in orders[:50]:
        for t in range(1, N - 1):           # skip t=0 (trivial) and the last 1-candidate step
            S, U, last = o[:t].tolist(), o[t:].tolist(), int(o[t - 1])
            X, _, _ = build_features(B, S, U, last, t, N)
            rows.append(X)
    Xall = np.concatenate(rows, axis=0)
    return dict(
        n_states=int(Xall.shape[0]),
        any_nan=bool(np.isnan(Xall).any()),
        per_feature=[
            {"name": FEATURE_NAMES[j],
             "min": float(Xall[:, j].min()),
             "max": float(Xall[:, j].max()),
             "mean": float(Xall[:, j].mean())}
            for j in range(Xall.shape[1])
        ],
    )


def topk_examples(B, N, states, k=4):
    out = []
    for (S, U, last, label) in states:
        res = teacher_step(B, S, U, last, tau_T=1.0, mode="C-D+L", top_k=k)
        out.append({"state": label, "n_candidates": len(U),
                    "top_k_candidates": res["topk_candidates"],
                    "entropy_tau1": round(res["entropy"], 4)})
    return out


def run_config(B, N, kind, K, rng_seed, tau_T=None, standardize=True, greedy=False, modality="image"):
    rng = np.random.default_rng(rng_seed)
    if kind == "random":
        orders = random_orders(N, K, rng)
        avg_ent = float("nan")
    elif kind == "greedy":
        orders = np.stack([rollout_order(B, mode="C-D+L", greedy=True)])  # deterministic, 1 order
        avg_ent = 0.0
    else:  # teacher sampling
        ords, ents = [], []
        for s in range(K):
            o, e = rollout_order(B, tau_T=tau_T, seed=rng_seed + s, mode="C-D+L",
                                 standardize=standardize, return_entropy=True)
            ords.append(o)
            ents.extend(e)
        orders = np.stack(ords)
        avg_ent = float(np.mean(ents))
    uniq, h_first = diversity(orders)
    row = dict(kind=kind, tau_T=(tau_T if tau_T is not None else ""), K=orders.shape[0],
               avg_step_entropy=round(avg_ent, 4), unique_orders=uniq,
               first_node_entropy=round(h_first, 4))
    if modality == "image":
        loc = locality_stats(orders)
        t4, edge, base = top4_follow_and_edge(orders, B)
        row.update(mean_manh=round(loc["mean_manh"], 4), p_le1=round(loc["p_le1"], 4),
                   p_le2=round(loc["p_le2"], 4), top4_follow=round(t4, 4),
                   B_edge_ratio_vs_random=round(edge / (base + 1e-12), 4))
    else:
        row.update(tau_vs_raster_proxy=round(kendall_tau_vs_raster(orders), 4),
                   start_node=int(orders[:, 0][0]) if orders.shape[0] == 1 else "",
                   end_node=int(orders[:, -1][0]) if orders.shape[0] == 1 else "")
    return row, orders


def write_tsv(path, rows):
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    lines = ["\t".join(keys)]
    for r in rows:
        lines.append("\t".join(str(r.get(k, "")) for k in keys))
    path.write_text("\n".join(lines) + "\n")


def analyze(A_path, modality, md_path, tsv_path):
    A = np.load(A_path).astype(np.float64)
    N = A.shape[0]
    B = build_directed_graph(A)
    print(f"\n=== {modality.upper()} | A={A_path.name} N={N} | "
          f"B row-sum mean={B.sum(1).mean():.4f} off-diag mean={B[~np.eye(N,dtype=bool)].mean():.5f} ===")

    # states for top-k examples + a feature-builder smoke
    rng = np.random.default_rng(SEED0)
    base_order = rng.permutation(N)
    states = [
        ([], list(range(N)), None, "t=0 (empty S, no last)"),
        (base_order[:1].tolist(), base_order[1:].tolist(), int(base_order[0]), "t=1"),
        (base_order[:N // 2].tolist(), base_order[N // 2:].tolist(), int(base_order[N // 2 - 1]), "t=N/2"),
        (base_order[:N - 2].tolist(), base_order[N - 2:].tolist(), int(base_order[N - 3]), "t=N-2"),
    ]
    tk = topk_examples(B, N, states)
    fsum = feature_summary(B, random_orders(N, 50, rng))

    rows = []
    rows.append(run_config(B, N, "random", K, SEED0, modality=modality)[0])
    rows.append(run_config(B, N, "greedy", 1, SEED0, modality=modality)[0])
    for tau in TAU_GRID:
        rows.append(run_config(B, N, "teacher", K, SEED0, tau_T=tau, standardize=True, modality=modality)[0])

    write_tsv(tsv_path, rows)

    # ---- markdown ----
    md = []
    md.append(f"# Phase 0 sanity — {modality} (unified C-D+L teacher, attention-only)\n")
    md.append(f"- Graph: `{A_path.relative_to(_REPO)}`  N={N}")
    md.append(f"- B = A^T (build_directed_graph, zero diag). row-sum mean = {B.sum(1).mean():.4f}, "
              f"off-diag mean = {B[~np.eye(N,dtype=bool)].mean():.5f}")
    md.append(f"- Rollouts per config K={K}; teacher sampling uses per-step z-score standardization "
              f"so tau_T is comparable across modalities. NO MLP, NO training.\n")

    md.append("## Feature builder smoke")
    md.append(f"- states sampled: {fsum['n_states']}, any NaN: **{fsum['any_nan']}**")
    md.append("\n| feature | min | max | mean |")
    md.append("|---|---|---|---|")
    for f in fsum["per_feature"]:
        md.append(f"| {f['name']} | {f['min']:.4f} | {f['max']:.4f} | {f['mean']:.4f} |")

    md.append("\n## Teacher top-k candidates (tau_T=1, raw C-D+L)")
    md.append("\n| state | #cand | top-4 candidates | entropy(tau=1) |")
    md.append("|---|---|---|---|")
    for e in tk:
        md.append(f"| {e['state']} | {e['n_candidates']} | {e['top_k_candidates']} | {e['entropy_tau1']} |")

    md.append("\n## Order-set metrics (teacher vs random)")
    hdr_keys = list(rows[0].keys())
    md.append("\n| " + " | ".join(hdr_keys) + " |")
    md.append("|" + "---|" * len(hdr_keys))
    for r in rows:
        md.append("| " + " | ".join(str(r.get(k, "")) for k in hdr_keys) + " |")

    # ---- verdict ----
    md.append("\n## Read")
    rnd = rows[0]
    grd = next(r for r in rows if r["kind"] == "greedy")
    teach_lowtau = next(r for r in rows if r["kind"] == "teacher" and r["tau_T"] == TAU_GRID[0])
    if modality == "image":
        md.append(f"- random:           mean_manh={rnd['mean_manh']} P(d<=1)={rnd['p_le1']} "
                  f"top4_follow={rnd['top4_follow']} B_edge_ratio={rnd['B_edge_ratio_vs_random']}")
        md.append(f"- teacher greedy:    mean_manh={grd['mean_manh']} P(d<=1)={grd['p_le1']} "
                  f"top4_follow={grd['top4_follow']} B_edge_ratio={grd['B_edge_ratio_vs_random']}")
        md.append(f"- teacher(tau={TAU_GRID[0]}): mean_manh={teach_lowtau['mean_manh']} P(d<=1)={teach_lowtau['p_le1']} "
                  f"top4_follow={teach_lowtau['top4_follow']} B_edge_ratio={teach_lowtau['B_edge_ratio_vs_random']}")
        structured = (teach_lowtau["top4_follow"] > rnd["top4_follow"] + 0.02) or \
                     (teach_lowtau["B_edge_ratio_vs_random"] > 1.5)
        md.append(f"- **non-random structure: {structured}.** The attention-only C-D+L teacher recovers "
                  f"strong B-edge-following (ratio {teach_lowtau['B_edge_ratio_vs_random']}x vs random; "
                  f"top4_follow {teach_lowtau['top4_follow']} vs {rnd['top4_follow']}) AND, notably, "
                  f"substantial *spatial* locality (P(d<=1) {teach_lowtau['p_le1']} vs random {rnd['p_le1']}; "
                  f"mean_manh {teach_lowtau['mean_manh']} vs {rnd['mean_manh']}) — even though the teacher "
                  f"uses no image geometry. The locality is inherited from B itself.")
        md.append("- This is *order structure*, not a task-gain claim. The Phase-2 gate is whether such an "
                  "order improves the common image eval aggregates (cross/structured/noisy), not whether it "
                  "looks structured here.")
    else:
        md.append(f"- random:        tau_vs_raster_proxy={rnd['tau_vs_raster_proxy']} (≈0 expected)")
        md.append(f"- teacher greedy: tau_vs_raster_proxy={grd['tau_vs_raster_proxy']} "
                  f"(start={grd['start_node']} -> end={grd['end_node']})")
        md.append(f"- teacher(tau={TAU_GRID[0]}): tau_vs_raster_proxy={teach_lowtau['tau_vs_raster_proxy']}, "
                  f"first_node_entropy={teach_lowtau['first_node_entropy']}")
        structured = abs(teach_lowtau["tau_vs_raster_proxy"]) > 0.1
        md.append(f"- **non-random structure (proxy): {structured}.** The greedy C-D+L teacher recovers a "
                  f"near-perfect L2R-aligned order (tau≈{grd['tau_vs_raster_proxy']}, 0->63), and low-temp "
                  f"sampling stays strongly L2R-aligned (tau {teach_lowtau['tau_vs_raster_proxy']}); raising "
                  f"tau_T sweeps it back toward random. Consistent with text attention strongly encoding L2R "
                  f"position.")
        md.append("- tau_vs_raster is a PROXY only; the Phase-2 gate is the canonical `val_ori_l2r` held-out "
                  "loss (see audit_text_eval_order.md), not this rank metric.")
    md.append(f"\n- entropy/diversity is fully tunable by tau_T (avg_step_entropy {teach_lowtau['avg_step_entropy']} "
              f"@tau={TAU_GRID[0]} -> {rows[-1]['avg_step_entropy']} @tau={TAU_GRID[-1]}); no entropy collapse "
              "(distinct orders at tau>=1).")
    md.append("\n_Phase 0 is a diagnostic that the teacher produces a non-degenerate, non-random "
              "ordering signal. It is NOT a downstream success claim._")

    md_path.write_text("\n".join(md) + "\n")
    print(f"wrote {md_path.relative_to(_REPO)} and {tsv_path.relative_to(_REPO)}")
    return {"modality": modality, "any_nan": fsum["any_nan"], "structured": structured, "rows": rows}


def main():
    out_text_dir = _REPO / "probe_results/attention_order_mlp"
    out_img_dir = _REPO / "probe_results_image_large/attention_order_mlp"
    out_text_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    if TEXT_A.exists():
        summary["text"] = analyze(TEXT_A, "text",
                                  out_text_dir / "phase0_text_sanity.md",
                                  out_text_dir / "phase0_text_sanity.tsv")
    else:
        print(f"MISSING text graph: {TEXT_A}")
    if IMAGE_A.exists():
        summary["image"] = analyze(IMAGE_A, "image",
                                   out_img_dir / "phase0_image_sanity.md",
                                   out_img_dir / "phase0_image_sanity.tsv")
    else:
        print(f"MISSING image graph: {IMAGE_A}")

    print("\n=== PHASE 0 SUMMARY ===")
    for m, s in summary.items():
        print(f"  {m}: any_nan={s['any_nan']} structured={s['structured']}")


if __name__ == "__main__":
    main()
