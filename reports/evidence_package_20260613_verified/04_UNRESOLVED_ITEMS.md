# 04 — Unresolved Items

> Items that cannot be resolved from existing data alone. No guessing — these require additional work or remain as known limitations.

---

## U1: CDL teacher seed mismatch → 🟡 IN PROGRESS (2026-06-14)

| Field | Detail |
|------|------|
| **Status** | `cdl_teacher_seed123_from10k_l0h2` **RUNNING** (step 31000/60000 as of 2026-06-14 22:12, ETA ~6.8h). Config: seed=123, head=[0,2], cdl_teacher_refresh=10, resume from `random_baseline_continuous_jun08_seed2/ckpt_step10000.pt`. |
| **Why unresolved** | Run not yet at 50k — cannot compute Recovery@50k or Step Saving@3.47 until step ≥50000 eval data exists. |
| **Blocking paper?** | 🟡 Not blocking — CDL teacher can be reported as supplementary cross-seed evidence with explicit caveat. The primary story (frozen_β recovery) is seed-matched internally. |
| **Next check** | Read eval_curve after run reaches 50k (~2026-06-15 morning). Compute matched Recovery@50k and Step Saving@3.47 vs seed-123 group baseline (3.445) and L2R (3.305). |

### U1.5: 50k Per-Head τ Signal Persistence — ⚠️ DECAYED (new, 2026-06-14)

| Field | Detail |
|------|------|
| **Finding** | Per-head CDL τ_vs_L2R at step 50k (seed-123 continuous, M=40, none_mode=b0): **all 32 heads \|τ\| < 0.06**, heavy τ = −0.013. The strong per-head signal (\|τ\|≈1.0 at 10k) decays to noise level by convergence. |
| **Source** | `random_baseline_continuous_jun08_seed2/head_scan_50k.json` |
| **Impact** | C2 ("attention contains L2R signal") must be qualified: signal is a **transient early-training phenomenon** (≤10k), not a persistent convergent property. Training acceleration (C3) remains valid via C4 transfer evidence. |
| **Blocking paper?** | 🟢 Not blocking if correctly framed. C4 indirect transfer evidence (g_β@10k→40k: 59–67% recovery) becomes the primary mechanism-persistence argument. |
| **Recommended action** | Qualify C2 in paper: "At early training (≤10k steps), random-order AOGPT develops sparse order-bearing attention heads with \|τ\|≈1.0. This signal attenuates as training converges, but the frozen g_β controller trained at 10k remains effective at later steps (20k–40k), suggesting the learned readout captures a stable B→order mapping even as raw per-head τ declines." |

---

## U2: ~~g_β sanity numbers — no primary JSON output saved~~ → RESOLVED 2026-06-14

| Field | Detail |
|------|------|
| **Resolution** | Re-ran sanity checks 2026-06-14 20:13. Raw JSON saved to `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`. |
| **Verified numbers** | Real B τ_l2r=+0.9675, Gaussian τ=-0.0034, entry-shuffled τ=+0.0165, rowcol-shuffled τ=+0.0127, Zero B τ=+1.0 margin=0.0, Gaussian pw_tau=0.0003. |
| **⚠️ Note** | Non-L2R subset only n=4 samples, Δ=-0.03 (effectively zero). The +0.11 Δ from memory cannot be reproduced — likely from a different g_β checkpoint or data split. This does NOT weaken the main sanity argument (real vs destroyed B gap is 0.97). |
| **Blocking paper?** | 🟢 Resolved. |

---

## U3: Label-free audition not end-to-end verified

| Field | Detail |
|------|------|
| **Why unresolved** | Current pipeline: head SELECTION uses cheap signals (row-concentration, S_forward); g_β TRAINING uses CDL teacher supervision; hook TRAINING uses frozen g_β. The full chain (label-free head selection → label-free g_β training → hook) has NOT been demonstrated end-to-end. |
| **What evidence is missing** | End-to-end run where: (1) head is selected purely by cheap signals, (2) g_β is trained without CDL teacher, (3) frozen hook training matches oracle-head performance. |
| **Blocking paper?** | 🟡 Not blocking if correctly framed as "head selected via cheap signals; g_β trained with teacher supervision." Cannot claim "fully label-free." |
| **Recommended action** | Either run end-to-end label-free pipeline or explicitly frame as limitation. |

---

## U4: unique_sigma_ratio definition

| Field | Detail |
|------|------|
| **Why unresolved** | The `unique_sigma_ratio` field appears in head_scan JSONs and granularity scan JSONs. Its exact definition (fraction of M samples with distinct CDL orders? unique σ / total samples?) is not documented in a primary source. |
| **What evidence is missing** | Source code comment or documentation defining the computation. |
| **Blocking paper?** | 🟢 Not blocking — the qualitative pattern (small model low diversity, large model high diversity) is clear from mean_pairwise_tau values. |
| **Recommended action** | Add definition comment in `per_head_order_scan.py`. |

---

## U5: ~~Figures 1, 3, 4 not found on disk~~ → RESOLVED

| Field | Detail |
|------|------|
| **Resolution** | All 4 core figures EXIST at `analyses/figures/`. Sizes: fig1=64KB, fig2=52KB, fig3=107KB, fig4=169KB. All dated Jun 13 22:11–22:15. |
| **Remaining risk** | Content accuracy of fig3/fig4 against corrected seed42 numbers needs visual inspection. |
| **Blocking paper?** | 🟢 Not blocking — files exist. |
| **Action** | Visual check that fig3/fig4 use correct baseline numbers. |

---

## U6: Shuffled-L2R control — "encodes no order at all" cannot be claimed

| Field | Detail |
|------|------|
| **Why unresolved** | Shuffled-L2R control shows the model does NOT recover physical L2R order (τ≈0). But we have NOT proven the model encodes "no order at all" — it could encode the shuffled order (just not physical L2R). |
| **What evidence is missing** | Per-head scan showing τ_vs_shuffled_order for the shuffled-L2R model. If τ≈1.0 vs the shuffled order, the model does encode order — just the wrong one. |
| **Blocking paper?** | 🟢 Not blocking if correctly framed: "Shuffled-L2R control does not recover physical order." |
| **Recommended action** | Compute τ_vs_shuffled_order from existing shuffled-L2R checkpoints. |

---

## U7: No seed-123 random baseline beyond 50k

| Field | Detail |
|------|------|
| **Why unresolved** | `random_baseline_continuous_jun08_seed2` and `l2r_continuous_seed123` both end at step 50000. frozen_beta runs go to 60000. No matched random baseline or L2R reference exists for step 60000 comparison. |
| **What evidence is missing** | Random baseline and L2R reference at steps 50000–60000 for seed-123. |
| **Blocking paper?** | 🟢 Not blocking — use 50k as standard comparison point. Note that 60k comparisons are not available. |
| **Recommended action** | Standardize on 50k as the comparison step. |

---

**Summary**: 3 verified limitations (U3, U4, U6—correct framing exists), 1 actionable (U1 — CDL teacher seed-123 running overnight), 1 methodology choice (U7), 2 resolved (U2, U5). No blocking issue for advisor update.
