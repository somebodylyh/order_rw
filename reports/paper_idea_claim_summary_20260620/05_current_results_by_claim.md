# Current Results by Claim

---

## Claim 1 — Legacy g_β controller accelerates canonical-order training

### Evidence
| Seed | Method | Recovery @50k | Step Saving @3.47 |
|------|--------|--------------|-------------------|
| seed-123 | frozen_beta L0H2 from10k | **86.0%** | **41.7%** |
| seed-123 | frozen_beta L0H2 from20k | **83.0%** | **35.7%** |
| seed-123 | frozen_beta L0H2 from40k | 58.7% | — |
| seed-42 | frozen_beta L0H4 from10k | **106.1%** | **39.3%** |
| seed-42 | frozen_beta L0H4 from20k | **101.5%** | **33.3%** |
| seed-42 | frozen_beta L0H4 from40k | 66.9% | — |

Conservative range: **33–42%** (frozen_beta only, from10k+from20k, both seeds).

### Source paths
- `probe_results/frozen_beta_seed2_from10k_l0h2/eval_curve.tsv`
- `probe_results/frozen_beta_seed2_from20k_l0h2/eval_curve.tsv`
- `probe_results/frozen_beta_seed2_from40k_l0h2/eval_curve.tsv`
- `probe_results/frozen_beta_random_jun05_from10k/eval_curve.tsv`
- `probe_results/frozen_beta_random_jun05_from20k/eval_curve.tsv`
- `probe_results/frozen_beta_random_jun05_from40k/eval_curve.tsv`
- `analyses/wall_clock_data.tsv`, `analyses/wall_clock_saving.py`

### Strength
**Strong** (for legacy B0 controller path)

### Caveat
- Protocol: legacy B0 controller path ([None] folded into physical block 0)
- NOT strict 65-node teacher. Strict-teacher controller is pending.
- g_β is seed-dependent and head-dependent (seed-123 uses L0H2, seed-42 uses L0H4)
- from40k is weaker (59–67%). Signal decays at convergence.
- ori-L2R is a reference, not an upper bound (seed-42 recovery = 106.1%)
- val_unstructured degrades (Δ +0.42 to +1.28)

---

## Claim 2 — g_β reads structured B, not fixed L2R prior

### Evidence
| B Input | τ vs L2R | Notes |
|---------|---------|-------|
| Real B (L0H4) | **+0.9675** | Reads real attention structure |
| Gaussian random B | −0.0034 | No signal |
| Zero B | +1.0 | Tie-breaking artifact (margin=0) |
| Entry-shuffled B | +0.0165 | Near random |
| Row+col shuffled (PBP^T) | +0.0127 | Near random |
| Pairwise Gaussian (100 samples) | 0.0003 | No consistent prior |

Real–destroyed gap = 0.907. All 4 sanity checks PASS.

### Source paths
- `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`

### Strength
**Strong**

### Caveat
- B0 legacy controller path. B1 g_β sanity not yet run.
- Zero B τ=1.0 is a known tie-breaking artifact (all edges equal, CDL picks first available = L2R).
- Non-L2R subset excluded (n=4, Δ=−0.03 in re-run, irreproducible).

---

## Claim 3 — B1 / predictor-aligned extraction preserves order signal

### Evidence
- Clean-base B1 ladder @10k: 7/32 |τ|>0.9, best L0H0 τ=1.000
- Clean-base B1 ladder @50k: 6/32 |τ|>0.9, best L0H0 τ=1.000
- Clean-base B1 ladder @60k: 6/32 |τ|>0.9, best L0H0 τ=1.000
- Continuous B1 seed=124 tracking: final L0H4 τ=0.958

### Source paths
- `batch_readout/logs/per_head_scan_b1/ckpt10000_seed0.json` (and seed1, seed2)
- `batch_readout/logs/per_head_scan_b1/ckpt50000_seed0.json`
- `batch_readout/logs/per_head_scan_b1/ckpt60000_seed0.json`
- `probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv`

### Strength
**Medium** (diagnostic only, no B1 hook acceleration)

### Caveat
- B1 result-bearing files use `none_mode=predictor` (NOT code enum `none_mode=b1`)
- B1 diagnostic does NOT prove B1 frozen-hook acceleration
- No B1 g_β sanity, no B1 seed-123 50k scan, no B1 317M scan
- B1 content-only loses [None] anchor (3/4 heads drop from τ=1.000 to τ=0.655)

---

## Claim 4 — Strict 65-node label-free discovery exists in selected early heads

### Evidence

**Collaborator ckpt @50k (M=20, full sweep)**:
- Gate distribution: 20 strong / 11 weak / 225 fail (256 total)
- Strong-pass heads (L-only, τ=1.000, first=0, phys0_rank=0, p4=4, p8=8): **L0H1, L0H2, L0H3, L0H4**
- Fail example: L0H7, τ=0.292, first=45, phys0_rank=20

**Clean-base 9-step ladder (M=8)**:
| Step | strong_total | Best head | best τ | destroyed mean |τ| |
|------|-------------|-----------|--------|-------------------|
| 0 | 0 | L0H2 C-D+L | 0.191 | 0.063 |
| 1000 | 0 | L2H4 none_edge | 0.142 | 0.069 |
| 5000 | 10 | L1H2 L | 1.000 | 0.050 |
| 10000 | 10 | L1H2 L | 1.000 | 0.050 |
| 20000 | 16 | L1H4 L | 1.000 | 0.052 |
| 30000 | 15 | L0H1 L | 1.000 | 0.053 |
| 40000 | 15 | L1H2 L | 1.000 | 0.055 |
| 50000 | 14 | L0H0 L | 1.000 | 0.057 |
| 60000 | 16 | L0H0 L | 1.000 | 0.054 |

Stable strong heads (≥5/8 steps): L1H0, L1H1, L1H2, L1H4, L0H0, L2H0, L2H4.

### Source paths
- `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/all_head_methods_strict_label_free_65.tsv`
- `reports/strict_65node_discovery_ckpt_verification_20260617/strict_65node_ckpt_sweep.tsv`

### Strength
**Strong**

### Caveat
- Head-specific (20/256 = 7.8% pass). Most heads (~88%) fail.
- Head identity drifts across runs (collaborator L0H1–H4 vs clean_base L1H0–H4).
- Extraction-frame-dependent (B1 predictor content-only is weaker).
- M=8/20 (small sample). Full M=2000+ validation pending.

---

## Claim 5 — Content-only 64-node physical adjacency exists after convention fix, but content-vs-position memory remains unresolved

### Evidence
- L0H0, without [None], true_original: d=+1 = **0.0571**, d=−1 = 0.00303, off-band ≈ 0.00033, contrast ≈ **172×**
- All-heads mean, without [None], true_original: d=+1 = 0.0099, off-band ≈ 0.0010, contrast ≈ 10×

### Source paths
- `reports/collab_extract_on_our_ckpt_L0H0/`

### Strength
**Medium**

### Caveat
- Fixed permutation (single permute_seed): cannot distinguish content adjacency vs memorized position pairs.
- Cross-permutation validation pending (E6 in 04_experiment_design.md).
- Correct physical frame uses `block_perm` (phys→model convention).

---

## Claim 6 — Strict discovery does not imply all heads or all frames work

### Evidence
- L0H7 fail: τ=0.292 (C-D+L), first=45, phys0_rank=20
- 225/256 combos fail strict gate
- B1 predictor content-only: L0H1/H2/H4 τ=0.655 (cyclic, wrong anchor) vs 65-node τ=1.000
- Best head drifts: L1H2→L1H4→L0H1→L0H0 across clean-base steps
- Collaborator vs clean_base: different strong-head sets

### Source paths
- `reports/strict_65node_discovery_ckpt_verification_20260617/05_extraction_frame_comparison.md`
- `reports/strict_65node_discovery_ckpt_verification_20260617/04_ckpt_sweep_results.md`

### Strength
**Strong** (negative result, well-documented)

### Caveat
- None. This is a boundary claim that limits other claims.

---

## Claim 7 — Destroyed controls confirm signal is not a structural artifact

### Evidence
- 65-node entry shuffle: mean |τ| = 0.050–0.069 across all steps
- Content label permutation: mean |τ| = 0.050–0.070
- Real–destroyed gap ≈ 0.95 for strong-pass heads
- L0H7 fail also near random in destroyed controls — confirms readout doesn't inflate τ
- Gaussian B g_β sanity: τ = −0.0034

### Source paths
- `reports/strict_65node_discovery_ckpt_verification_20260617/06_destroyed_controls_summary.md`
- `reports/strict_65node_discovery_ckpt_verification_20260617/destroyed_controls_by_step.tsv`
- `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`

### Strength
**Strong**

### Caveat
- Gaussian B control run only for B0 g_β sanity, not for 65-node protocol.
