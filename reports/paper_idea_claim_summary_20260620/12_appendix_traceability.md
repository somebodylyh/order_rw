# Appendix: Traceability — Source Path → Claim → Number Mapping

## Key Numbers with Source Paths

### Legacy B0 Acceleration Numbers

| Number | Value | Source |
|--------|-------|--------|
| seed-123 random baseline @50k | 3.445 | `probe_results/random_baseline_continuous_jun08_seed2/eval_curve.tsv` |
| seed-123 L2R ref @50k | 3.305 | `probe_results/l2r_continuous_seed123/eval_curve.tsv` |
| seed-42 random baseline @50k | 3.466 | `probe_results/random_baseline_continuous_jun05/eval_curve.tsv` |
| seed-42 L2R ref @50k | 3.341 | `probe_results/l2r_continuous_seed42/eval_curve.tsv` |
| seed-2 L2R ref @60k | 3.365 | `probe_results/l2r_continuous_seed2/eval_curve.tsv` |
| seed-123 fb from10k @50k | 3.325 | `probe_results/frozen_beta_seed2_from10k_l0h2/eval_curve.tsv` |
| seed-123 fb from20k @50k | 3.329 | `probe_results/frozen_beta_seed2_from20k_l0h2/eval_curve.tsv` |
| seed-123 fb from40k @50k | 3.363 | `probe_results/frozen_beta_seed2_from40k_l0h2/eval_curve.tsv` |
| seed-42 fb from10k @50k | 3.334 | `probe_results/frozen_beta_random_jun05_from10k/eval_curve.tsv` |
| seed-42 fb from20k @50k | 3.339 | `probe_results/frozen_beta_random_jun05_from20k/eval_curve.tsv` |
| seed-42 fb from40k @50k | 3.383 | `probe_results/frozen_beta_random_jun05_from40k/eval_curve.tsv` |
| seed-123 recovery from10k | 86.0% | Calculated from above |
| seed-123 step saving from10k | 41.7% | T_random=42000, T_cross=24500 |
| seed-42 recovery from10k | 106.1% | Calculated from above |
| seed-42 step saving from10k | 39.3% | T_random=42000, T_cross=25500 (est) |
| Conservative step saving range | 33–42% | Across both seeds, from10k+from20k |
| Wall-clock saving (V3, seed-123) | ~62% | `analyses/wall_clock_data.tsv`, `analyses/wall_clock_saving.py` |

### g_β Sanity Numbers

| Number | Value | Source |
|--------|-------|--------|
| Real B τ vs L2R | +0.9675 | `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json` |
| Gaussian B τ vs L2R | −0.0034 | same |
| Zero B τ vs L2R | +1.0 (margin=0) | same |
| Entry-shuffled B τ | +0.0165 | same |
| Row+col shuffled B τ | +0.0127 | same |
| Pairwise Gaussian τ | 0.0003 | same |
| Real–destroyed gap | 0.907 | same |
| Non-L2R subset n | 4 | same |
| Non-L2R Δ | −0.03125 | same (EXCLUDED from claims) |

### g_β Phase15 Reports

| g_β | τ (5k_val) | pairwise | non-L2R τ | Source |
|-----|-----------|----------|-----------|--------|
| B0 L0H2 seed2 from10k | 0.946 | 0.973 | −0.072 (n=4) | `logs/phase33_gbeta_seed2_from10k_l0h2/` |
| B1 L0H2 seed2 from10k | 0.946 | 0.973 | +0.193 (n=9) | `logs/gbeta_b1_L0H2_seed2_step10k/` |
| B1 cross 20k | 0.953 | 0.977 | +0.133 (n=18) | same |
| B1 cross 40k | 0.944 | 0.972 | +0.153 (n=24) | same |
| B0 L0H4 seed42 from10k | 0.911 | — | +0.135 | `logs/phase33_gbeta_seed42_from10k_l0h4/` |
| B0 random_perm L0H2 | 0.602 | — | (all non-L2R) | `logs/gbeta_b0_L0H2_step10k/` |

### Strict 65-Node Discovery Numbers

| Number | Value | Source |
|--------|-------|--------|
| Collaborator gate: strong | 20 | `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/all_head_methods_strict_label_free_65.tsv` |
| Collaborator gate: weak | 11 | same |
| Collaborator gate: fail | 225 | same |
| Collaborator strong heads (L) | L0H1, L0H2, L0H3, L0H4 | same |
| L0H7 fail τ (C-D+L) | 0.292 | same |
| Clean_base strong @5k | 10 | `reports/strict_65node_discovery_ckpt_verification_20260617/strict_65node_ckpt_sweep.tsv` |
| Clean_base strong @60k | 16 | same |
| Clean_base stable heads | L1H0–L1H4 (8/8 steps) | same |
| Best head drift sequence | L1H2→L1H4→L0H1→L0H0 | same |
| Destroyed mean |τ| range | 0.050–0.069 | `reports/strict_65node_discovery_ckpt_verification_20260617/destroyed_controls_by_step.tsv` |

### Extraction Frame Comparison

| Head | B1 content-only τ | 65-node τ | Source |
|------|-------------------|----------|--------|
| L0H1 | 0.655 | 1.000 | `reports/strict_65node_discovery_ckpt_verification_20260617/05_extraction_frame_comparison.md` |
| L0H2 | 0.655 | 1.000 | same |
| L0H3 | 1.000 | 1.000 | same |
| L0H4 | 0.655 | 1.000 | same |

### B1 Head Scan Numbers

| Number | Value | Source |
|--------|-------|--------|
| B1 clean_base @10k \|τ\|>0.9 | 7/32 | `batch_readout/logs/per_head_scan_b1/ckpt10000_seed0.json` |
| B1 clean_base @50k \|τ\|>0.9 | 6/32 | `batch_readout/logs/per_head_scan_b1/ckpt50000_seed0.json` |
| B1 clean_base @60k \|τ\|>0.9 | 6/32 | `batch_readout/logs/per_head_scan_b1/ckpt60000_seed0.json` |
| B1 continuous seed=124 best | L0H4 τ=0.958 | `probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv` |
| B0 seed-123 continuous @50k \|τ\|>0.9 | 0/32 (all <0.06) | `probe_results/random_baseline_continuous_jun08_seed2/head_scan_50k.json` |

### Content-Only Physical Adjacency

| Number | Value | Source |
|--------|-------|--------|
| L0H0, no [None], true_original d=+1 | 0.0571 | `reports/collab_extract_on_our_ckpt_L0H0/` |
| L0H0, no [None], true_original d=−1 | 0.00303 | same |
| L0H0, no [None], true_original off-band | ~0.00033 | same |
| L0H0 contrast (d=+1/off-band) | ~172× | same |

### 317M Scale Diagnostic

| Number | Value | Source |
|--------|-------|--------|
| 317M B0 strong @5k | 6/256 | `probe_results/large_random_baseline_16l16h1024d/head_scan_step5000.json` |
| 317M best head | L0H10 τ=+0.955 | same |

### Model Config

| Parameter | Value | Source |
|-----------|-------|--------|
| Layers | 4 | `probe_results/*/config.json` |
| Heads | 8 | same |
| d_model | 384 | same |
| Total params | ~47M | same |
| Block size | 256 tokens | same |
| Blocks | 64 × 4 tokens | same |
| Dataset | Wikitext-103 | same |
| Data mode | Continuous streaming | same |

---

## Deprecated Numbers (DO NOT USE)

| Deprecated | Correct | Why | Source |
|-----------|---------|-----|--------|
| seed42 random = 3.354 | **3.466** | 3.354 was frozen_beta run, not baseline | `evidence_package_20260613_verified/03_DEPRECATED_NUMBERS.md` |
| "28/32 heads \|τ\|>0.9" | **9/32** | Actual count from head_scan_10k.json | same |
| seed42 gap = 0.013 | **0.125** | Wrong baseline | same |
| ori-L2R "upper bound" | **ori-L2R is a reference** | Multiple methods beat L2R | same |
| Non-L2R Δ = +0.11 | **excluded (n=4, irreproducible)** | Re-run gave −0.03 | `evidence_package_20260613_verified/01_VERIFICATION_LOG.md` |

---

## Key Code Sources

| Module | File |
|--------|------|
| AO-GPT training | `block_lo_arm_order_network/train_clean_aogpt.py` |
| B65 block graph | `block_lo_arm_order_network/none_separated_block_graph.py` |
| Per-head scan | `block_lo_arm_order_network/per_head_order_scan.py` |
| Head audition | `block_lo_arm_order_network/scripts/audition_heads.py` |
| Strict LF search | `block_lo_arm_order_network/scripts/search_strict_label_free_65.py` |
| g_β training | `block_lo_arm_order_network/scripts/run_phase33_gbeta.py` |
| Hook order provider (B0) | `block_lo_arm_order_network/batch_readout/hook_order_provider.py` |
| Selected head dataset | `block_lo_arm_order_network/batch_readout/selected_head_dataset.py` |
| Pipeline (no-label) | `block_lo_arm_order_network/scripts/pipeline_no_label.sh` |
| CDL order provider | `block_lo_arm_order_network/batch_readout/cdl_order_provider.py` |

## Key Report Sources

| Report | Path |
|--------|------|
| Evidence package (verified) | `reports/evidence_package_20260613_verified/` |
| Strict 65-node discovery | `reports/strict_65node_discovery_ckpt_verification_20260617/` |
| B1 extraction summary | `reports/b1_attention_extraction_summary_20260615/` |
| Attention summary & figures | `reports/attention_summary_and_figures_20260617/` |
| Weekly summary | `reports/weekly_summary_20260617/` |
| Boss update | `reports/wednesday_boss_update_20260615/` |
| Collaborator B1 scan | `reports/collaborator_ckpt_b1_scan_20260616/` |
| Collab extract on our ckpt | `reports/collab_extract_on_our_ckpt/` |
| Collab extract L0H0 | `reports/collab_extract_on_our_ckpt_L0H0/` |

## SOURCE_MISSING

The following were referenced in the task instructions but not found in the repository:
- `reports/loss_aligned_b1_extraction_*/` — directory does not exist
- `reports/none_anchor_leak_audit_*/` — directory does not exist
- `reports/content_anchored_none_diagnostic_*/` — directory does not exist
- `memory/` — located at `~/.claude/projects/-home-admin-lyuyuhuan-order-lyu/memory/` (separate directory, accessible via memory system, not filesystem under `reports/`)
