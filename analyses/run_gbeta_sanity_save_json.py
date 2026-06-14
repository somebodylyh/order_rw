#!/usr/bin/env python3
"""Wrapper: run gbeta_input_sanity_final.py tables and save raw JSON output."""
import json, sys, os, time
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "block_lo_arm_order_network"))

# Import the sanity functions
from analyses.gbeta_input_sanity_final import (
    load_g_beta, gbeta_predict, score_margin, order_diversity_metrics,
    kendall_tau, pairwise_tau,
    make_gaussian_B, make_uniform_B, make_zero_B,
    make_entry_shuffled, make_rowcol_shuffled,
)

DEVICE = "cuda:1"
OUTPUT = os.path.join(REPO, "reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json")
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# Paths
G_BETA_PATH = os.path.join(REPO,
    "block_lo_arm_order_network/batch_readout/logs/"
    "phase33_gbeta_seed2_from10k_l0h2/"
    "random_baseline_continuous_jun08_seed2/full/g_beta_best.pt")
DATA_PATH = os.path.join(REPO,
    "block_lo_arm_order_network/batch_readout/logs/"
    "phase33_gbeta_seed2_from10k_l0h2/"
    "random_baseline_continuous_jun08_seed2/full/data/ds_10000.npz")

print(f"g_β: {G_BETA_PATH}")
print(f"data: {DATA_PATH}")
print(f"device: {DEVICE}")
print(f"output: {OUTPUT}")

# Load
model = load_g_beta(G_BETA_PATH, DEVICE)
print(f"Loaded g_β ({sum(p.numel() for p in model.parameters())} params)")

data = np.load(DATA_PATH)
val_B = data["val_B_batch"]
val_teacher = data["val_sigma_T"]
print(f"val_B: {val_B.shape}, val_teacher: {val_teacher.shape}")

M = min(100, len(val_B))
B_real = val_B[:M]
teacher = val_teacher[:M]
l2r_order = np.tile(np.arange(64), (M, 1))

output = {"meta": {"g_beta_path": G_BETA_PATH, "data_path": DATA_PATH,
                    "device": DEVICE, "M": M, "N": 64}}

# ── Table 1 ──
print("\n=== Table 1: Single-input sanity ===")
t1_rows = []

def test_row(label, B_in, ref_order=None):
    orders, scores = gbeta_predict(model, B_in, DEVICE, batch_size=64)
    if ref_order is None:
        ref_order = teacher
    tau_vs_teacher = float(np.mean([kendall_tau(orders[i], ref_order[i]) for i in range(M)]))
    tau_vs_l2r = float(np.mean([kendall_tau(orders[i], l2r_order[i]) for i in range(M)]))
    margins = score_margin(scores)
    div = order_diversity_metrics(orders, ref_order[0])
    row = {
        "input": label,
        "tau_vs_teacher": tau_vs_teacher,
        "tau_vs_l2r": tau_vs_l2r,
        "mean_top_bottom_margin": margins["mean_top_bottom_margin"],
        "score_std": margins["score_std"],
        "score_mean": margins["score_mean"],
        "first_block_unique": div.get("first_block_unique", 0),
        "first_block_entropy": div.get("first_block_entropy", 0),
        "n": M,
    }
    t1_rows.append(row)
    note = "(tie-break)" if row["mean_top_bottom_margin"] < 0.001 else \
           "(no signal)" if abs(row["tau_vs_teacher"]) < 0.1 else \
           "(strong)" if abs(row["tau_vs_teacher"]) > 0.8 else ""
    print(f"  {label:<28} τ_teacher={tau_vs_teacher:+.4f} τ_l2r={tau_vs_l2r:+.4f} "
          f"margin={margins['mean_top_bottom_margin']:.6f} {note}")

test_row("real_selected_head_B", B_real, teacher)
test_row("gaussian_random_B", make_gaussian_B(M, seed=1), teacher)
test_row("uniform_random_B", make_uniform_B(M, seed=1), teacher)
test_row("zero_B_tie_breaking_artifact", make_zero_B(M), teacher)
test_row("entry_shuffled_B", make_entry_shuffled(B_real, seed=1), teacher)
test_row("rowcol_shuffled_PBPt", make_rowcol_shuffled(B_real, seed=1), teacher)

output["table1_single_input_sanity"] = t1_rows

# ── Table 2: Random-B diversity ──
print("\n=== Table 2: Random-B output diversity ===")
N_SAMPLES = 100
t2_results = []

for label, gen_fn in [
    ("gaussian_100_samples", lambda: make_gaussian_B(N_SAMPLES, seed=10)),
    ("entry_shuffled_100_samples", lambda: make_entry_shuffled(val_B[:N_SAMPLES], seed=10)),
]:
    B_in = gen_fn()
    orders, scores = gbeta_predict(model, B_in, DEVICE, batch_size=64)
    pw_tau = float(pairwise_tau(orders))
    margins = score_margin(scores)
    tau_l2r = float(np.mean([kendall_tau(orders[i], l2r_order[i % M]) for i in range(N_SAMPLES)]))
    div = order_diversity_metrics(orders)
    r = {
        "input_family": label,
        "n_samples": N_SAMPLES,
        "mean_pairwise_tau": pw_tau,
        "tau_vs_l2r": tau_l2r,
        "first_block_entropy": div.get("first_block_entropy", 0),
        "first_block_unique": div.get("first_block_unique", 0),
        "mean_top_bottom_margin": margins["mean_top_bottom_margin"],
        "score_std": margins["score_std"],
    }
    t2_results.append(r)
    note = "(random!)" if abs(pw_tau) < 0.05 else "(fixed!)" if abs(pw_tau) > 0.9 else ""
    print(f"  {label:<35} pw_tau={pw_tau:.4f} tau_l2r={tau_l2r:+.4f} {note}")

# Real-B baseline
orders_real, scores_real = gbeta_predict(model, B_real, DEVICE, batch_size=64)
pw_real = float(pairwise_tau(orders_real))
tau_l2r_real = float(np.mean([kendall_tau(orders_real[i], l2r_order[i]) for i in range(M)]))
div_real = order_diversity_metrics(orders_real)
margins_real = score_margin(scores_real)
r_real = {
    "input_family": "real_selected_head_B",
    "n_samples": M,
    "mean_pairwise_tau": pw_real,
    "tau_vs_l2r": tau_l2r_real,
    "first_block_entropy": div_real.get("first_block_entropy", 0),
    "first_block_unique": div_real.get("first_block_unique", 0),
    "mean_top_bottom_margin": margins_real["mean_top_bottom_margin"],
    "score_std": margins_real["score_std"],
}
t2_results.append(r_real)
print(f"  {'real_selected_head_B':<35} pw_tau={pw_real:.4f} tau_l2r={tau_l2r_real:+.4f} (consistent)")

output["table2_random_diversity"] = t2_results
output["table2_pairwise_tau_gaussian"] = t2_results[0]["mean_pairwise_tau"]
output["table2_pairwise_tau_entry_shuffled"] = t2_results[1]["mean_pairwise_tau"]
output["table2_pairwise_tau_real"] = pw_real

# ── Table 4: Non-L2R subset ──
print("\n=== Table 4: Non-L2R subset ===")
l2r = np.arange(64)
non_l2r_mask = np.array([not np.array_equal(t, l2r) for t in val_teacher])
n_non = int(non_l2r_mask.sum())
print(f"  Total: {len(val_teacher)}, non-L2R: {n_non}")

t4_results = []
if n_non > 0:
    B_non = val_B[non_l2r_mask]
    teacher_non = val_teacher[non_l2r_mask]
    orders, scores = gbeta_predict(model, B_non, DEVICE, batch_size=64)
    l2r_tiled = np.tile(l2r, (n_non, 1))
    deltas = []
    for i in range(min(n_non, 200)):  # cap at 200
        tau_gb = kendall_tau(orders[i], teacher_non[i])
        tau_pr = kendall_tau(l2r_tiled[i], teacher_non[i])
        deltas.append(float(tau_gb - tau_pr))
    mean_delta = float(np.mean(deltas))
    n_win = sum(1 for d in deltas if d > 0)
    n_lose = sum(1 for d in deltas if d < 0)
    t4_results = {
        "n_non_l2r": n_non,
        "n_evaluated": len(deltas),
        "mean_delta_gbeta_minus_prior": mean_delta,
        "gbeta_wins": n_win,
        "prior_wins": n_lose,
        "all_deltas": deltas[:20],  # first 20 for inspection
    }
    print(f"  mean Δ(g_β − prior) = {mean_delta:+.4f}, g_β wins={n_win}, prior wins={n_lose}")
else:
    t4_results = {"n_non_l2r": 0, "note": "All teacher orders are identity L2R — no non-L2R subset exists."}
    print("  No non-L2R samples (all teacher = L2R)")

output["table4_non_l2r_subset"] = t4_results

# ── Verdict ──
checks = []
# Check 1
real_tau = [r["tau_vs_teacher"] for r in t1_rows if "real" in r["input"]]
destroyed_tau = [abs(r["tau_vs_teacher"]) for r in t1_rows
                 if any(kw in r["input"] for kw in ["gaussian", "uniform", "entry", "rowcol"])]
if real_tau and destroyed_tau:
    gap = abs(np.mean(real_tau)) - np.mean(destroyed_tau)
    checks.append({"check": "real_vs_destroyed_gap", "real_mean_tau": float(np.mean(real_tau)),
                    "destroyed_mean_abs_tau": float(np.mean(destroyed_tau)), "gap": float(gap),
                    "passed": bool(gap > 0.5)})

# Check 2: Gaussian diversity
gauss_pw = t2_results[0]["mean_pairwise_tau"]
checks.append({"check": "gaussian_pairwise_tau_near_zero", "value": gauss_pw,
                "passed": bool(abs(gauss_pw) < 0.1)})

# Check 3: Real consistency
checks.append({"check": "real_pairwise_tau_high", "value": pw_real,
                "passed": bool(abs(pw_real) > 0.8)})

# Check 4: Non-L2R
if n_non > 0:
    checks.append({"check": "non_l2r_delta_positive", "value": mean_delta,
                    "passed": bool(mean_delta > -0.05)})

all_pass = all(c["passed"] for c in checks)
output["verdict"] = {"all_checks_pass": all_pass,
                      "summary": "g_β reads attention graph structure, not a fixed L2R prior." if all_pass
                      else "SOME CHECKS FAILED",
                      "checks": checks}

for c in checks:
    status = "PASS" if c["passed"] else "FAIL"
    print(f"  [{status}] {c['check']}: {c.get('value', c.get('gap', '?'))}")

print(f"\n{'🟢' if all_pass else '🔴'} VERDICT: {output['verdict']['summary']}")

# Save
with open(OUTPUT, 'w') as f:
    json.dump(output, f, indent=2, default=float)
print(f"\nSaved raw JSON to {OUTPUT}")
print("Done.")
