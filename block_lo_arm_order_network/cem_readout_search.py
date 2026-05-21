#!/usr/bin/env python3
"""CEM search over unified-readout params. Pluggable fitness.
`--mode sanity` runs the three ex-ante dry-run sanity tests (Amendment 2, section G)
with CPU surrogate fitness and reports pass/fail. Frozen/continuation fitness (Phase 3)
is NOT invoked here.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "block_lo_arm_order_network"))
import unified_readout as ur
from directed_graph_policy import build_directed_graph, compute_source
from readout_order_diagnostic import order_stats


def cem(fitness, pop=16, elite=4, gens=5, seed=0, bound_override=None, log=print):
    """bound_override: {key: (lo, hi)} to clamp/disable params (e.g. gamma_d -> (0,0))."""
    rng = np.random.default_rng(seed)
    keys = ur.PARAM_KEYS
    bounds = dict(ur.PARAM_BOUNDS)
    if bound_override:
        bounds.update(bound_override)
    lo = np.array([bounds[k][0] for k in keys])
    hi = np.array([bounds[k][1] for k in keys])
    mu = (lo + hi) / 2.0
    sigma = (hi - lo) / 4.0 + 1e-6
    best = (-1e18, None); history = []
    for g in range(gens):
        cand = np.clip(rng.normal(mu, sigma, size=(pop, len(keys))), lo, hi)
        fits = np.array([fitness({k: float(v) for k, v in zip(keys, c)}) for c in cand])
        order = np.argsort(-fits); elite_idx = order[:elite]
        mu = cand[elite_idx].mean(0); sigma = cand[elite_idx].std(0) + 1e-3
        if fits[order[0]] > best[0]:
            best = (float(fits[order[0]]), {k: float(v) for k, v in zip(keys, cand[order[0]])})
        history.append(dict(gen=g, best_fit=float(fits[order[0]]), mean_fit=float(fits.mean())))
        log(f"    gen {g}: best={fits[order[0]]:.4f} mean={fits.mean():.4f}")
    return best, history


def dominant_terms(w):
    weights = {k: w[k] for k in ["beta_sup", "beta_dep", "rho", "gamma_B", "gamma_d"]}
    tot = sum(abs(v) for v in weights.values()) + 1e-9
    return {k: round(v / tot, 3) for k, v in sorted(weights.items(), key=lambda kv: -kv[1])}


def order_directionality(orders):
    """1D step-direction consistency: |mean(sign(diff))| averaged over orders."""
    o = np.atleast_2d(orders)
    d = np.sign(np.diff(o, axis=1))
    return float(np.mean(np.abs(d.mean(axis=1))))


def behavioral_metric(B, source, coords, has, w, kind, k=8, seed=1):
    """kind='manh' -> mean_manh (lower=proximity); 'direc' -> order directionality."""
    o = ur.sample_orders_batch(B, source, coords, ur.clip_params(w), k, seed, has)
    return order_stats(o)["mean_manh"] if kind == "manh" else order_directionality(o)


def term_ablation(B, source, coords, has, best_w, kind):
    """Knock out each term from best_w; report behavioral metric delta.
    Answers 'is this term NECESSARY for the behavior' (vs 'is it the biggest weight')."""
    full = behavioral_metric(B, source, coords, has, best_w, kind)
    rows = {"full_best_w": round(full, 4)}
    for term in ["beta_sup", "beta_dep", "rho", "gamma_B", "gamma_d"]:
        w = dict(best_w); w[term] = 0.0
        rows[f"{term}=0"] = round(behavioral_metric(B, source, coords, has, w, kind), 4)
    w = dict(best_w); w["fallback_mix"] = 1.0
    rows["fallback=1"] = round(behavioral_metric(B, source, coords, has, w, kind), 4)
    return rows


def _setup(a_path, topology, grid, random_B=False, seed=0):
    if random_B:
        rng = np.random.default_rng(seed)
        B = rng.random((64, 64)); np.fill_diagonal(B, 0.0)
    else:
        A = np.load(_REPO / a_path).astype(np.float64)
        B = build_directed_graph(A)
    source, _, _ = compute_source(B, 0.5)
    coords, has = ur.make_coords(B.shape[0], topology, grid)
    return B, source, coords, has


def run_sanity(outdir, pop, gens):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    E3 = "probe_results_image_large/imagenet64_vqf4_full_l4h8e256_patch2x2_control/A_block_8x8.npy"
    TEXT = "block_lo_arm_order_network/probe_results/clean_method_graph_rw_a10_to_a095_30k60k/A_global_step50000.npy"
    results = {}
    log_lines = []

    def log(m):
        print(m, flush=True); log_lines.append(m)

    # ---- Test 1: proximity on E3, FULL family, fitness = -mean_manh ----
    # PASS = BEHAVIORAL only (mean_manh<3.0). term-attribution is DESCRIPTIVE (not a gate):
    # the family is over-parameterized so dominant-term is non-identifiable.
    log("[Test 1] proximity (E3, full family, -mean_manh) — behavioral gate")
    B, src, coords, has = _setup(E3, "grid2d", 8)
    def fit1(w):
        o = ur.sample_orders_batch(B, src, coords, ur.clip_params(w), 6, 0, has)
        return -order_stats(o)["mean_manh"]
    best1, _ = cem(fit1, pop, 4, gens, log=log)
    w1 = ur.clip_params(best1[1]); mm1 = behavioral_metric(B, src, coords, has, w1, "manh")
    dt1 = dominant_terms(w1)
    pass1 = mm1 < 3.0
    abl1 = term_ablation(B, src, coords, has, w1, "manh")
    log(f"  mean_manh={mm1:.3f} -> {'PASS' if pass1 else 'FAIL'} (behavioral)")
    log(f"  [descriptive] dominant terms={dt1} fallback={w1['fallback_mix']:.3f}")
    log(f"  [ablation manh, higher=term necessary for proximity] {abl1}")
    results["test1_proximity_E3"] = dict(passed=bool(pass1), mean_manh=mm1, best_w=w1,
                                         dominant_terms_descriptive=dt1, ablation=abl1)

    # ---- Test 2: NO false structure on random B, gamma_d DISABLED ----
    log("[Test 2] no-false-structure (random B, gamma_d disabled, -mean_manh)")
    Br, srcr, coordsr, hasr = _setup(None, "grid2d", 8, random_B=True, seed=7)
    disable_gd = {"gamma_d": (0.0, 0.0)}
    def fit2(w):
        o = ur.sample_orders_batch(Br, srcr, coordsr, ur.clip_params(w), 6, 0, hasr)
        return -order_stats(o)["mean_manh"]
    best2, _ = cem(fit2, pop, 4, gens, bound_override=disable_gd, log=log)
    w2 = ur.clip_params(best2[1]); o2 = ur.sample_orders_batch(Br, srcr, coordsr, w2, 8, 1, hasr)
    mm2 = order_stats(o2)["mean_manh"]
    pass2 = mm2 > 4.5
    log(f"  random-B mean_manh={mm2:.3f} (gamma_d disabled) -> {'PASS' if pass2 else 'FAIL'}")
    # cross-check: same gamma_d-disabled search on E3 should reach < 3.5 (B-driven)
    def fit2e(w):
        o = ur.sample_orders_batch(B, src, coords, ur.clip_params(w), 6, 0, has)
        return -order_stats(o)["mean_manh"]
    best2e, _ = cem(fit2e, pop, 4, gens, bound_override=disable_gd, log=log)
    w2e = ur.clip_params(best2e[1]); mm2e = order_stats(ur.sample_orders_batch(B, src, coords, w2e, 8, 1, has))["mean_manh"]
    discr = mm2e < 3.5
    abl2 = term_ablation(Br, srcr, coordsr, hasr, w2, "manh")
    log(f"  cross-check E3 (gamma_d disabled) mean_manh={mm2e:.3f} (<3.5 expected) -> {'OK' if discr else 'WEAK'}")
    log(f"  [ablation manh on random B; should stay ~random for all] {abl2}")
    results["test2_nofalse_randomB"] = dict(passed=bool(pass2), random_B_mean_manh=mm2,
                                            E3_gd_disabled_mean_manh=mm2e, discriminates=bool(discr),
                                            ablation=abl2)

    # ---- Test 3: readiness on text, fitness = directionality ----
    log("[Test 3] readiness (text, fitness = order directionality)")
    if (_REPO / TEXT).exists():
        Bt, srct, coordst, hast = _setup(TEXT, "seq1d", 0)
        def fit3(w):
            o = ur.sample_orders_batch(Bt, srct, coordst, ur.clip_params(w), 6, 0, hast)
            return order_directionality(o)
        best3, _ = cem(fit3, pop, 4, gens, log=log)
        w3 = ur.clip_params(best3[1]); dir3 = behavioral_metric(Bt, srct, coordst, hast, w3, "direc")
        dt3 = dominant_terms(w3)
        pass3 = dir3 > 0.7  # BEHAVIORAL only; rho-dominance is descriptive (non-identifiable)
        abl3 = term_ablation(Bt, srct, coordst, hast, w3, "direc")
        log(f"  directionality={dir3:.3f} -> {'PASS' if pass3 else 'FAIL'} (behavioral)")
        log(f"  [descriptive] dominant terms={dt3}")
        log(f"  [ablation directionality, lower=term necessary for readiness order] {abl3}")
        results["test3_readiness_text"] = dict(passed=bool(pass3), directionality=dir3, best_w=w3,
                                               dominant_terms_descriptive=dt3, ablation=abl3)
    else:
        log("  text A_global missing -> SKIP (report as not-run)")
        results["test3_readiness_text"] = dict(passed=None, skipped=True)

    all_pass = all(r.get("passed") for r in results.values() if r.get("passed") is not None)
    log(f"\n[SANITY SUMMARY] " + " ".join(
        f"{k}={'PASS' if v.get('passed') else ('SKIP' if v.get('skipped') else 'FAIL')}"
        for k, v in results.items()))
    log(f"ALL PASS: {all_pass}")
    json.dump(results, open(out / "sanity_results.json", "w"), indent=2)
    (out / "sanity_log.txt").write_text("\n".join(log_lines))
    print(f"wrote {out}/sanity_results.json")
    return all_pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sanity"], default="sanity")
    p.add_argument("--outdir", default="probe_results_image_large/structure_adaptive/cem/sanity")
    p.add_argument("--pop", type=int, default=16)
    p.add_argument("--gens", type=int, default=5)
    args = p.parse_args()
    if args.mode == "sanity":
        run_sanity(args.outdir, args.pop, args.gens)


if __name__ == "__main__":
    main()
