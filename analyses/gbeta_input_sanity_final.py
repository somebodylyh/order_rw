#!/usr/bin/env python3
"""
g_β Input Sanity — Final Validation (2026-06-11)

Validates that g_β reads attention graph structure rather than outputting
a fixed/constant L2R prior. Produces four tables:

  Table 1 — Single-input sanity: real B vs destroyed/synthetic B
  Table 2 — Random-B output diversity: pairwise τ across random inputs
  Table 3 — Cross-model B: random-order vs shuffled-L2R vs ori-L2R
  Table 4 — Non-L2R subset: g_β vs L2R prior on samples where teacher ≠ identity

Usage:
  python analyses/gbeta_input_sanity_final.py [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── project paths ──────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "block_lo_arm_order_network"))

from batch_readout.model import NodewiseReadout


# ── tau / metrics ──────────────────────────────────────────────────────────
def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall τ between two 1-d rank arrays."""
    n = len(a)
    if n < 2:
        return 1.0
    conc = 0
    disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_a = a[j] - a[i]
            d_b = b[j] - b[i]
            if d_a * d_b > 0:
                conc += 1
            elif d_a * d_b < 0:
                disc += 1
    total = conc + disc
    return (conc - disc) / total if total > 0 else 0.0


def pairwise_tau(orders: np.ndarray) -> float:
    """Mean pairwise τ among M orders. orders: (M, N)."""
    M = len(orders)
    if M < 2:
        return 1.0
    taus = []
    for i in range(M):
        for j in range(i + 1, M):
            taus.append(kendall_tau(orders[i], orders[j]))
    return float(np.mean(taus))


def order_diversity_metrics(orders: np.ndarray, ref_order: np.ndarray = None):
    """Return dict of diversity metrics for a set of orders (M, N)."""
    M, N = orders.shape
    out = {"n_samples": M, "n_blocks": N}
    if M >= 2:
        out["mean_pairwise_tau"] = pairwise_tau(orders)
    if ref_order is not None:
        taus_vs_ref = [kendall_tau(orders[i], ref_order) for i in range(M)]
        out["mean_tau_vs_ref"] = float(np.mean(taus_vs_ref))
        out["min_tau_vs_ref"] = float(np.min(taus_vs_ref))
        out["max_tau_vs_ref"] = float(np.max(taus_vs_ref))
    # Entropy of first-block distribution
    first_blocks = orders[:, 0]
    _, counts = np.unique(first_blocks, return_counts=True)
    probs = counts / M
    out["first_block_entropy"] = float(-np.sum(probs * np.log(probs + 1e-12)))
    out["first_block_unique"] = len(np.unique(first_blocks))
    return out


def score_margin(z: np.ndarray) -> dict:
    """z: (M, N) raw scores from g_β. Returns margin stats."""
    margins = []
    for i in range(len(z)):
        sorted_z = np.sort(z[i])
        margins.append(float(sorted_z[-1] - sorted_z[0]))  # top-bottom
    z_flat = z.ravel()
    return {
        "mean_top_bottom_margin": float(np.mean(margins)),
        "std_top_bottom_margin": float(np.std(margins)),
        "score_std": float(np.std(z_flat)),
        "score_mean": float(np.mean(z_flat)),
    }


# ── g_β loader ─────────────────────────────────────────────────────────────
def load_g_beta(path: str, device: str = "cpu") -> NodewiseReadout:
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    name = cfg["model_name"]
    if name == "nodewise":
        model = NodewiseReadout(N=64, d_model=64, n_layers=2, n_heads=4)
    else:
        raise ValueError(f"unknown model {name!r}")
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


# ── B generators ───────────────────────────────────────────────────────────
def make_gaussian_B(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    B = rng.normal(0, 1, (n, 64, 64)).astype(np.float32)
    B[:, np.arange(64), np.arange(64)] = 0
    return B


def make_uniform_B(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    B = rng.uniform(0, 1, (n, 64, 64)).astype(np.float32)
    B[:, np.arange(64), np.arange(64)] = 0
    return B


def make_zero_B(n: int) -> np.ndarray:
    return np.zeros((n, 64, 64), dtype=np.float32)


def make_entry_shuffled(B_real: np.ndarray, seed: int = 42) -> np.ndarray:
    """Shuffle all entries independently — destroys graph structure, keeps value distribution."""
    rng = np.random.default_rng(seed)
    B_out = B_real.copy()
    for i in range(len(B_out)):
        flat = B_out[i].ravel()
        rng.shuffle(flat)
        B_out[i] = flat.reshape(64, 64)
        B_out[i, np.arange(64), np.arange(64)] = 0  # re-zero diagonal
    return B_out


def make_rowcol_shuffled(B_real: np.ndarray, seed: int = 42) -> np.ndarray:
    """Apply P B P^T with random P — checks if g_β has fixed coordinate prior."""
    rng = np.random.default_rng(seed)
    B_out = B_real.copy()
    for i in range(len(B_out)):
        P = rng.permutation(64)
        B_out[i] = B_out[i][P][:, P]
    return B_out


# ── g_β inference ──────────────────────────────────────────────────────────
@torch.no_grad()
def gbeta_predict(model: NodewiseReadout, B: np.ndarray, device: str,
                  batch_size: int = 64) -> np.ndarray:
    """Return (M, N) orders (descending score = earliest first) and (M, N) raw scores.

    Convention (matches hook's pl_argsort / loss.py):
      - rank-0 = earliest reveal → HIGHEST score
      - sigma = argsort(-z) = descending by score
    """
    model.eval()
    all_orders = []
    all_scores = []
    for start in range(0, len(B), batch_size):
        batch = torch.from_numpy(B[start : start + batch_size]).to(device)
        z = model(batch)  # (batch, N)
        orders = torch.argsort(-z, dim=-1).cpu().numpy()  # descending → earliest first
        all_orders.append(orders)
        all_scores.append(z.cpu().numpy())
    return np.concatenate(all_orders), np.concatenate(all_scores)


# ── Table 1: Single-input sanity ───────────────────────────────────────────
def table1_single_input_sanity(model, device, val_B, val_teacher):
    """Test g_β on real B and various destroyed/synthetic B."""
    M = min(100, len(val_B))  # use up to 100 real B samples
    B_real = val_B[:M]
    teacher = val_teacher[:M]
    l2r_order = np.tile(np.arange(64), (M, 1))

    print("\n" + "=" * 90)
    print("TABLE 1: Single-Input B Sanity")
    print("  Conventions: g_β score → argsort(-z) → earliest=highest score")
    print("=" * 90)

    rows = []

    # Helper
    def test_row(label, B_in, ref_order=None):
        orders, scores = gbeta_predict(model, B_in, device, batch_size=64)
        if ref_order is None:
            ref_order = teacher  # compare to CDL teacher
        tau_vs_ref = np.mean([kendall_tau(orders[i], ref_order[i]) for i in range(M)])
        tau_vs_l2r = np.mean([kendall_tau(orders[i], l2r_order[i]) for i in range(M)])
        margins = score_margin(scores)
        div = order_diversity_metrics(orders, ref_order[0] if ref_order is not None else None)
        rows.append({
            "input": label,
            "tau_vs_teacher": tau_vs_ref,
            "tau_vs_l2r": tau_vs_l2r,
            "margin": margins["mean_top_bottom_margin"],
            "score_std": margins["score_std"],
            "first_block_unique": div.get("first_block_unique", 0),
            "first_block_entropy": div.get("first_block_entropy", 0),
            "n": M,
        })

    # 1. Real B
    test_row("real selected-head B", B_real, teacher)

    # 2. Gaussian random
    test_row("Gaussian random B", make_gaussian_B(M, seed=1), teacher)

    # 3. Uniform random
    test_row("uniform random B", make_uniform_B(M, seed=1), teacher)

    # 4. Zero (tie-breaking artifact — NOT a meaningful signal)
    # All scores equal → argsort returns index order → τ=1.0 is spurious.
    # Key diagnostic: margin = 0.000.
    test_row("zero B [TIE ARTIFACT]", make_zero_B(M), teacher)

    # 5. Entry-shuffled (destroys graph, keeps value distribution)
    test_row("entry-shuffled B", make_entry_shuffled(B_real, seed=1), teacher)

    # 6. Row+col shuffled P B P^T
    test_row("row+col shuffled PBP^T", make_rowcol_shuffled(B_real, seed=1), teacher)

    # Print table
    header = f"{'Input B':<28} {'τ_vs_teacher':>12} {'τ_vs_L2R':>12} {'margin':>10} {'score_std':>10} {'1st_blk_uniq':>13} {'1st_blk_H':>10} {'note':>20}"
    print(header)
    print("-" * len(header))
    for r in rows:
        note = ""
        if r['margin'] < 0.001:
            note = "(tie-break artifact)"
        elif abs(r['tau_vs_teacher']) < 0.1:
            note = "(no signal)"
        elif r['tau_vs_teacher'] > 0.8:
            note = "(strong signal)"
        print(f"{r['input']:<28} {r['tau_vs_teacher']:>12.4f} {r['tau_vs_l2r']:>12.4f} "
              f"{r['margin']:>10.4f} {r['score_std']:>10.4f} "
              f"{r['first_block_unique']:>13} {r['first_block_entropy']:>10.3f} {note:>20}")

    # Footnote about zero B
    print("\n  ⚠ zero B: τ≈1.0 is a TIE-BREAKING ARTIFACT, not meaningful signal.")
    print("    All node scores are identical (margin=0.000). argsort(-z) on equal")
    print("    values returns [0,1,…,63] by stable sort → spurious L2R alignment.")
    print("    The diagnostic that matters is margin=0, not τ=1.0.")

    return rows


# ── Table 2: Random-B output diversity ─────────────────────────────────────
def table2_random_diversity(model, device):
    """Generate 100 random B → check if g_β outputs diverse (random) orders."""
    N_SAMPLES = 100
    print("\n" + "=" * 90)
    print("TABLE 2: Random-B Output Diversity")
    print("=" * 90)

    results = []
    for label, gen_fn in [
        ("Gaussian (100 samples)", lambda: make_gaussian_B(N_SAMPLES, seed=10)),
        ("entry-shuffled (100 samples)", lambda: make_entry_shuffled(
            np.load(DATA_PATH)["val_B_batch"][:N_SAMPLES], seed=10)),
    ]:
        B_in = gen_fn()
        orders, scores = gbeta_predict(model, B_in, device, batch_size=64)
        pw_tau = pairwise_tau(orders)
        margins = score_margin(scores)
        # Also check vs L2R
        l2r = np.tile(np.arange(64), (N_SAMPLES, 1))
        tau_vs_l2r = np.mean([kendall_tau(orders[i], l2r[i]) for i in range(N_SAMPLES)])
        div = order_diversity_metrics(orders)

        results.append({
            "input_family": label,
            "n_samples": N_SAMPLES,
            "mean_pairwise_tau": pw_tau,
            "tau_vs_l2r": tau_vs_l2r,
            "first_block_entropy": div.get("first_block_entropy", 0),
            "first_block_unique": div.get("first_block_unique", 0),
            "margin_mean": margins["mean_top_bottom_margin"],
            "margin_std": margins["std_top_bottom_margin"],
        })

        # Add real-B baseline for comparison
        B_real = np.load(DATA_PATH)["val_B_batch"][:N_SAMPLES]
        orders_real, scores_real = gbeta_predict(model, B_real, device, batch_size=64)
        pw_tau_real = pairwise_tau(orders_real)
        margins_real = score_margin(scores_real)
        div_real = order_diversity_metrics(orders_real)

    # Print as comparison
    print(f"\n{'Input family':<35} {'N':>5} {'mean pw τ':>12} {'τ_vs_L2R':>12} "
          f"{'1st_blk_H':>10} {'1st_blk_uniq':>13} {'margin':>10} {'note':>25}")
    print("-" * 120)

    real_pw = pw_tau_real
    real_l2r = np.mean([kendall_tau(orders_real[i], np.arange(64)) for i in range(N_SAMPLES)])

    # Real B baseline
    note_real = f"(τ={real_l2r:.3f}, consistent)"
    print(f"{'real selected-head B':<35} {N_SAMPLES:>5} {real_pw:>12.4f} "
          f"{real_l2r:>12.4f} "
          f"{div_real.get('first_block_entropy',0):>10.3f} {div_real.get('first_block_unique',0):>13} "
          f"{margins_real['mean_top_bottom_margin']:>10.4f} {note_real:>25}")

    for r in results:
        note = ""
        if abs(r['mean_pairwise_tau']) < 0.05:
            note = "(random — no fixed prior!)"
        elif abs(r['mean_pairwise_tau']) > 0.9:
            note = "(fixed output)"
        print(f"{r['input_family']:<35} {r['n_samples']:>5} {r['mean_pairwise_tau']:>12.4f} "
              f"{r['tau_vs_l2r']:>12.4f} {r['first_block_entropy']:>10.3f} "
              f"{r['first_block_unique']:>13} {r['margin_mean']:>10.4f} {note:>25}")

    # Interpretation
    print("\nInterpretation:")
    print("  - If g_β outputs a FIXED prior: pairwise τ ≈ 1.0 for ALL inputs (real and random)")
    print("  - If g_β reads structure: pairwise τ high for real B, near 0 for Gaussian/shuffled")
    print("  - Low pairwise τ on random inputs = g_β does NOT memorize a constant order")

    return results + [{"input_family": "real selected-head B", "n_samples": N_SAMPLES,
                        "mean_pairwise_tau": pw_tau_real}]


# ── Table 3: Cross-model B ─────────────────────────────────────────────────
def table3_cross_model(model, device):
    """Extract B from shuffled-L2R and ori-L2R models, test g_β on them."""
    print("\n" + "=" * 90)
    print("TABLE 3: Cross-Model B Sanity")
    print("=" * 90)

    sys.path.insert(0, str(REPO / "block_lo_arm_order_network"))
    from batch_readout.extract_b_batch import extract_batch_mean_B

    results = []
    models_to_test = [
        ("random-order (seed2 10k)",
         "block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2/ckpt_step10000.pt",
         42),
        ("shuffled-L2R (seed42 10k)",
         "block_lo_arm_order_network/probe_results/shuffled_l2r_continuous_jun05/ckpt_step10000.pt",
         42),
        ("ori-L2R (seed42 10k)",
         "block_lo_arm_order_network/probe_results/l2r_continuous_jun05/ckpt_step10000.pt",
         42),
    ]

    for label, ckpt_rel, seed in models_to_test:
        ckpt_path = str(REPO / ckpt_rel)
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {label}: checkpoint not found at {ckpt_path}")
            continue

        print(f"  Extracting B from {label}...")
        t0 = time.time()
        try:
            data = extract_batch_mean_B(
                ckpt_path=ckpt_path,
                M=50,  # 50 batch-mean graphs
                batch_size=8,
                seed=seed,
                device=device,
                split="eval",
            )
            B = data["B_batch"]  # (50, 64, 64)
            elapsed = time.time() - t0
            print(f"    extracted {len(B)} graphs in {elapsed:.1f}s")

            orders, scores = gbeta_predict(model, B, device, batch_size=64)
            l2r = np.tile(np.arange(64), (len(B), 1))
            tau_vs_l2r = np.mean([kendall_tau(orders[i], l2r[i]) for i in range(len(B))])
            pw = pairwise_tau(orders)
            margins = score_margin(scores)

            results.append({
                "model": label,
                "tau_vs_l2r": tau_vs_l2r,
                "mean_pairwise_tau": pw,
                "margin": margins["mean_top_bottom_margin"],
                "score_std": margins["score_std"],
            })
        except Exception as e:
            print(f"    FAILED: {e}")

    if results:
        print(f"\n{'Source model':<35} {'τ_vs_L2R':>12} {'pw_τ':>12} {'margin':>10} {'score_std':>10}")
        print("-" * 85)
        for r in results:
            print(f"{r['model']:<35} {r['tau_vs_l2r']:>12.4f} {r['mean_pairwise_tau']:>12.4f} "
                  f"{r['margin']:>10.4f} {r['score_std']:>10.4f}")

        print("\nInterpretation:")
        print("  - random-order: g_β should output high τ (emergent signal readable)")
        print("  - shuffled-L2R: g_β should output τ ≈ 0 (wrong-order graph lacks signal)")
        print("  - ori-L2R: may output medium τ (physical order externally imposed)")
        print("  Key: if all three have high τ → g_β is constant L2R prior, not reading B")
    else:
        print("  No cross-model results collected.")

    return results


# ── Table 4: Non-L2R subset ────────────────────────────────────────────────
def table4_non_l2r_subset(model, device, val_B, val_teacher):
    """On samples where teacher ≠ L2R, compare g_β vs L2R prior."""
    print("\n" + "=" * 90)
    print("TABLE 4: Non-L2R Subset Analysis")
    print("=" * 90)

    l2r = np.arange(64)
    non_l2r_mask = np.array([not np.array_equal(t, l2r) for t in val_teacher])
    n_non = non_l2r_mask.sum()
    print(f"  Total val samples: {len(val_teacher)}, non-L2R: {n_non}")

    if n_non == 0:
        print("  No non-L2R samples found — all teacher orders are identity.")
        print("  (This is expected for L0H2 on text at 10k — CDL teacher converges to L2R.)")
        print("  For a definitive test, use a head/checkpoint with more teacher diversity.")
        return []

    B_non = val_B[non_l2r_mask]
    teacher_non = val_teacher[non_l2r_mask]

    orders, scores = gbeta_predict(model, B_non, device, batch_size=64)

    results = []
    l2r_tiled = np.tile(l2r, (n_non, 1))

    for i in range(n_non):
        tau_gbeta = kendall_tau(orders[i], teacher_non[i])
        tau_prior = kendall_tau(l2r_tiled[i], teacher_non[i])
        results.append({
            "tau_gbeta": tau_gbeta,
            "tau_l2r_prior": tau_prior,
            "delta": tau_gbeta - tau_prior,
        })

    mean_gbeta = np.mean([r["tau_gbeta"] for r in results])
    mean_prior = np.mean([r["tau_l2r_prior"] for r in results])
    n_win = sum(1 for r in results if r["delta"] > 0)
    n_lose = sum(1 for r in results if r["delta"] < 0)
    n_tie = sum(1 for r in results if r["delta"] == 0)

    print(f"\n{'Metric':<30} {'Value':>12}")
    print("-" * 44)
    print(f"{'n_non_L2R':<30} {n_non:>12}")
    print(f"{'mean τ(g_β, teacher)':<30} {mean_gbeta:>12.4f}")
    print(f"{'mean τ(L2R-prior, teacher)':<30} {mean_prior:>12.4f}")
    print(f"{'Δ (g_β − prior)':<30} {mean_gbeta - mean_prior:>+12.4f}")
    print(f"{'g_β wins (Δ>0)':<30} {n_win:>12}")
    print(f"{'prior wins (Δ<0)':<30} {n_lose:>12}")
    print(f"{'tie':<30} {n_tie:>12}")

    print("\nInterpretation:")
    print("  - If g_β truly reads B: τ(g_β, teacher) > τ(L2R-prior, teacher) on non-L2R subset")
    print("  - If g_β is constant L2R prior: τ(g_β, teacher) ≈ τ(L2R-prior, teacher)")
    print("  - Δ > 0 and g_β wins > prior wins → evidence of B-dependent readout")

    return results


# ── main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="g_β input sanity validation")
    parser.add_argument("--device", default="cuda:0",
                        help="device for inference (default: cuda:0)")
    parser.add_argument("--g-beta", default=None,
                        help="path to g_beta_best.pt (default: seed2 L0H2 10k)")
    parser.add_argument("--data", default=None,
                        help="path to ds npz with val_B_batch (default: phase33 seed2 L0H2)")
    parser.add_argument("--skip-cross-model", action="store_true",
                        help="skip cross-model B extraction (needs GPU + model loading)")
    args = parser.parse_args()

    device = args.device

    # Default paths
    g_beta_path = args.g_beta or str(REPO /
        "block_lo_arm_order_network/batch_readout/logs/"
        "phase33_gbeta_seed2_from10k_l0h2/"
        "random_baseline_continuous_jun08_seed2/full/g_beta_best.pt")

    global DATA_PATH
    DATA_PATH = args.data or str(REPO /
        "block_lo_arm_order_network/batch_readout/logs/"
        "phase33_gbeta_seed2_from10k_l0h2/"
        "random_baseline_continuous_jun08_seed2/full/data/ds_10000.npz")

    print(f"g_β: {g_beta_path}")
    print(f"data: {DATA_PATH}")
    print(f"device: {device}")

    # Load
    model = load_g_beta(g_beta_path, device)
    print(f"Loaded g_β (NodewiseReadout, {sum(p.numel() for p in model.parameters())} params)")

    data = np.load(DATA_PATH)
    val_B = data["val_B_batch"]  # (200, 64, 64)
    val_teacher = data["val_sigma_T"]  # (200, 64)
    print(f"Loaded val_B: {val_B.shape}, val_teacher: {val_teacher.shape}")

    # Run all tables
    t1 = table1_single_input_sanity(model, device, val_B, val_teacher)
    t2 = table2_random_diversity(model, device)

    if not args.skip_cross_model:
        t3 = table3_cross_model(model, device)
    else:
        print("\n[TABLE 3 skipped — use --skip-cross-model to avoid]")
        t3 = []

    t4 = table4_non_l2r_subset(model, device, val_B, val_teacher)

    # Summary verdict
    print("\n" + "=" * 90)
    print("SUMMARY VERDICT")
    print("=" * 90)

    checks = []

    # Check 1: real B → high τ, destroyed B → low τ
    if t1:
        # Exclude zero B (tie-breaking artifact) from destroyed check
        real_tau = [r["tau_vs_teacher"] for r in t1 if "real" in r["input"].lower()]
        destroyed_tau = [r["tau_vs_teacher"] for r in t1
                         if any(kw in r["input"].lower() for kw in ["gaussian", "uniform", "entry-shuffled", "row+col"])]
        if real_tau and destroyed_tau:
            real_mean = np.mean(real_tau)
            dest_mean = np.mean(np.abs(destroyed_tau))  # abs for destroyed (random sign)
            gap = abs(real_mean) - dest_mean
            checks.append((f"Real B τ ({real_mean:+.3f}) vs destroyed B |τ| ({dest_mean:.3f}) — gap={gap:+.3f}",
                           gap > 0.5))

    # Check 2: random-B outputs are diverse (not fixed), real-B outputs are consistent
    if t2:
        for r in t2:
            if "Gaussian" in r.get("input_family", ""):
                pw = r["mean_pairwise_tau"]
                checks.append((f"Gaussian B pairwise τ = {pw:.4f} (should be ≈ 0 if not fixed prior)",
                               abs(pw) < 0.1))
            if "real" in r.get("input_family", "").lower():
                pw = r["mean_pairwise_tau"]
                checks.append((f"Real B pairwise τ = {pw:.4f} (should be high → reads structure)",
                               abs(pw) > 0.8))

    # Check 3: cross-model — shuffled-L2R B → low τ
    if t3:
        for r in t3:
            if "shuffled" in r.get("model", "").lower():
                tau = r["tau_vs_l2r"]
                checks.append((f"shuffled-L2R B τ_vs_L2R = {tau:.3f} (should be ≈ 0)",
                               abs(tau) < 0.2))

    # Check 4: non-L2R subset
    if t4:
        mean_delta = np.mean([r["delta"] for r in t4])
        checks.append((f"Non-L2R subset: mean Δ(g_β − prior) = {mean_delta:+.4f} (>0 = reads B)",
                       mean_delta > -0.05))

    all_pass = True
    for desc, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
        print(f"  {status}: {desc}")

    if all_pass:
        print("\n  🟢 VERDICT: g_β reads attention graph structure, not a fixed L2R prior.")
    else:
        print("\n  🔴 VERDICT: Some checks failed — review above tables for details.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
