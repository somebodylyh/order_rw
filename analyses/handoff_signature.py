"""Pillar 2 handoff-signature temporal analysis.

For each seed trajectory, identify the final order-signal carrier head, then for
each upstream layer find the head that (a) feeds the carrier most strongly by
weight-based CANDIDATE composition and (b) whose order-tau onset leads the
carrier's in training time. Nominates upstream->downstream edges for Pillar 3
causal (path-patching) verification.

Composition scores are approximate (RMSNorm/AdaLN); outputs are NOMINATED edges,
not causal handoff evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_handoff_circuit import load_trajectory


def onset_step(series_abs, steps, thr):
    idx = np.where(series_abs >= thr)[0]
    return int(steps[idx[0]]) if len(idx) else -1


def peak_step(series_abs, steps):
    return int(steps[int(series_abs.argmax())])


def best_lag(up, dn, steps, max_frac=0.25, min_overlap=20):
    """Lag (in steps) maximizing correlation of up shifted forward onto dn.
    Positive lag => up leads dn. Lag search is capped to max_frac*T indices and
    requires at least min_overlap overlapping points (avoids the degenerate
    perfect-correlation-on-2-points artifact at extreme lags)."""
    T = len(up)
    max_k = max(1, int(T * max_frac))
    best, best_c = 0, -np.inf
    for k in range(0, max_k + 1):          # up earlier by k indices
        if T - k < min_overlap:
            break
        a = up[: T - k]
        b = dn[k:]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if c > best_c:
            best_c, best = c, k
    step_gap = int(steps[1] - steps[0]) if len(steps) > 1 else 0
    return best * step_gap, best_c


def analyze_seed(seed, run_dir, method_idx=0, thr=0.7):
    tj = load_trajectory(run_dir)
    tau = tj["tau"]
    steps = np.asarray(tj["steps"])
    comp = tj["composition"]
    T, L, H, M = tau.shape
    abst = np.abs(tau[..., method_idx])  # (T,L,H)

    l_star, h_star = np.unravel_index(int(abst[-1].argmax()), (L, H))
    car_onset = onset_step(abst[:, l_star, h_star], steps, thr)
    car_peak = peak_step(abst[:, l_star, h_star], steps)
    dn_series = abst[:, l_star, h_star]

    feeders = []
    for i in range(l_star):
        key = (i, l_star)
        if key not in comp:
            continue
        cm = comp[key]  # (T,H,H,3)
        comp_into = cm[:, :, h_star, :].max(axis=2)  # (T,H) max over Q/K/V into carrier
        late_comp = comp_into[-5:].mean(axis=0)       # (H,) per upstream head
        a_star = int(late_comp.argmax())
        # is the nominated edge distinctly strong, or just average?
        # percentile of best feeder among ALL upstream heads into the carrier,
        # and among the FULL (H,H) composition matrix at the final step.
        pct_into = float((late_comp <= late_comp[a_star]).mean())          # vs feeders into carrier
        full_final = cm[-1, :, :, :].max(axis=2).ravel()                    # all H*H edges (max QKV)
        pct_all = float((full_final <= comp_into[-1, a_star]).mean())       # vs all edges in pair
        # does composition into carrier RISE over training?
        comp_early = float(comp_into[:6, a_star].mean())                    # ~steps 0..1000
        comp_late = float(late_comp[a_star])
        up_peak = peak_step(abst[:, i, a_star], steps)
        lag, xc = best_lag(abst[:, i, a_star], dn_series, steps)
        feeders.append({
            "up_layer": i, "up_head": a_star,
            "comp_late": round(comp_late, 3), "comp_early": round(comp_early, 3),
            "comp_rise": round(comp_late - comp_early, 3),
            "pct_into_carrier": round(pct_into, 2), "pct_all_edges": round(pct_all, 2),
            "up_peak": up_peak, "car_peak": car_peak,
            "peak_lead": car_peak - up_peak, "xcorr_lag": lag, "xcorr": round(xc, 3),
        })
    return {"seed": seed, "carrier": (int(l_star), int(h_star)),
            "car_onset": car_onset, "car_peak": car_peak, "feeders": feeders}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="seed:run_dir pairs")
    ap.add_argument("--thr", type=float, default=0.7)
    args = ap.parse_args()
    for spec in args.runs:
        seed, run_dir = spec.split(":", 1)
        r = analyze_seed(seed, run_dir, thr=args.thr)
        c = r["carrier"]
        print(f"\n=== seed {seed}: final carrier L{c[0]}H{c[1]} "
              f"(onset@{r['car_onset']}, peak@{r['car_peak']}) ===")
        if not r["feeders"]:
            print("  carrier is L0 — no upstream feeders to nominate.")
        for f in r["feeders"]:
            print(f"  feeder L{f['up_layer']}H{f['up_head']}: "
                  f"comp_into_carrier late={f['comp_late']} (early={f['comp_early']}, "
                  f"rise={f['comp_rise']:+}) | "
                  f"pct_among_feeders={f['pct_into_carrier']:.0%} pct_all_edges={f['pct_all_edges']:.0%} | "
                  f"peak_lead={f['peak_lead']:+} steps | xcorr_lag=+{f['xcorr_lag']} (r={f['xcorr']})")


if __name__ == "__main__":
    main()
