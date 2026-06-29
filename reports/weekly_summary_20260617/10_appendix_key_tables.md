# Appendix - Key Tables and Traceability

## A. Key Source Paths

| Topic | Source path |
|---|---|
| Verified legacy tables | `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md` |
| Deprecated numbers | `reports/evidence_package_20260613_verified/03_DEPRECATED_NUMBERS.md` |
| Legacy claims lock | `reports/evidence_package_20260613_verified/05_FINAL_CLAIMS_LOCK.md` |
| g_beta sanity raw JSON | `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json` |
| B1 protocol | `reports/b1_attention_extraction_summary_20260615/01_b1_protocol_definition.md` |
| B1 signal summary | `reports/b1_attention_extraction_summary_20260615/03_b1_signal_summary.md` |
| B1 claim impact | `reports/b1_attention_extraction_summary_20260615/05_impact_on_existing_claims.md` |
| strict 65-node protocol summary | `reports/strict_65node_discovery_ckpt_verification_20260617/02_existing_result_summary.md` |
| strict clean-base ladder | `reports/strict_65node_discovery_ckpt_verification_20260617/strict_65node_ckpt_sweep.tsv` |
| strict collaborator full sweep | `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/all_head_methods_strict_label_free_65.tsv` |
| extraction-frame comparison | `reports/collaborator_ckpt_b1_scan_20260616/no_inv_label_free_readout_posthoc_M20_seed0.json` |
| destroyed controls | `reports/strict_65node_discovery_ckpt_verification_20260617/06_destroyed_controls_summary.md` |
| destroyed controls TSV | `reports/strict_65node_discovery_ckpt_verification_20260617/destroyed_controls_by_step.tsv` |
| 10k token attention reveal figure | `reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_step10000_L0H0_token_attention_model_frame.png` |
| 10k token attention original-L2R figure | `reports/collaborator_ckpt_b1_scan_20260616/figures/cleanbase_step10000_L0H0_token_attention_original_l2r.png` |

## B. Missing Requested Inputs

| Requested path/glob | Status |
|---|---|
| `memory/` | SOURCE_MISSING |
| `reports/loss_aligned_b1_extraction_*` | SOURCE_MISSING |
| `reports/none_anchor_leak_audit_*` | SOURCE_MISSING |
| `reports/content_anchored_none_diagnostic_*` | SOURCE_MISSING |

Related available sources are under `reports/collaborator_ckpt_b1_scan_20260616/`.

## C. Recovery Formula

```text
Recovery = (L_random - L_method) / (L_random - L_L2R) * 100%
```

Source: `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md`.

Example:

```text
seed42 from10k frozen_beta:
(3.466 - 3.334) / (3.466 - 3.341) * 100% = 106.1%
```

## D. Step Saving Formula

```text
Saving = (T_random - T_method) / T_random * 100%
```

For verified tables, `T_random=42000` for the seed-123 threshold crossing at `val_ori_l2r_block <= 3.47`.

Source: `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md`.

## E. Strict 65-node Graph Definition

```text
node 0       = None / BOS start node
node 1 + i   = physical content block i
block i      = x_{4i}, x_{4i+1}, x_{4i+2}, x_{4i+3}
rollout      = starts from None
first block  = selected by graph structure
```

Source:

- `reports/strict_65node_discovery_ckpt_verification_20260617/01_protocol_definition.md`
- `reports/strict_65node_discovery_ckpt_verification_20260617/02_existing_result_summary.md`

## F. Gate Definition

The strict reports use `strong_pass`, `weak_pass`, and `fail` gate labels. The practical strong-pass evidence in this package is:

```text
tau_vs_l2r = 1.000
first_block = 0
phys0_rank = 0
prefix4_overlap = 4
prefix8_overlap = 8
destroyed |tau| near random
```

Primary gate rows:

- `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/all_head_methods_strict_label_free_65.tsv`
- `reports/strict_65node_discovery_ckpt_verification_20260617/strict_65node_ckpt_sweep.tsv`

## G. Destroyed Control Definition

Entry shuffle:

```text
Shuffle allowed edge weights in B65 while keeping None as node0 and diagonal zero.
```

Content label permutation:

```text
Apply a random permutation to content node labels 1..64 while keeping None fixed.
```

Interpretation: both destroy content-specific graph structure while preserving broad graph statistics.

Source: `reports/strict_65node_discovery_ckpt_verification_20260617/06_destroyed_controls_summary.md`.

## H. inv_perm Boundary

| Stage | inv_perm allowed? | Reason |
|---|---:|---|
| Graph construction | yes | Labels may be remapped; edge structure preserved. |
| CDL rollout | no | Decisions must use graph edges only. |
| Posthoc scoring | yes | Evaluation translation only. |

Source: `reports/strict_65node_discovery_ckpt_verification_20260617/02_existing_result_summary.md`.

## I. Strict Collaborator @50k Rows

| Head | Method | tau | first | phys0_rank | p4 | p8 | destroyed |tau| | Gate | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| L0H1 | L | 1.000 | 0 | 0 | 4 | 8 | 0.0498 | strong_pass | `strict_label_free_65_search/all_head_methods_strict_label_free_65.tsv` |
| L0H2 | L | 1.000 | 0 | 0 | 4 | 8 | 0.0531 | strong_pass | same |
| L0H3 | L | 1.000 | 0 | 0 | 4 | 8 | 0.0501 | strong_pass | same |
| L0H4 | L | 1.000 | 0 | 0 | 4 | 8 | 0.0554 | strong_pass | same |
| L0H7 | C-D+L | 0.292 | 45 | 20 | 0 | 0 | 0.0825 | fail | same |

## J. Clean-base Ladder Rows

| Step | strong_total | best | tau | destroyed mean | Source |
|---:|---:|---|---:|---:|---|
| 0 | 0 | L0H2 C-D+L | 0.191 | 0.063 | `strict_65node_ckpt_sweep.tsv` |
| 1000 | 0 | L2H4 none_edge | 0.142 | 0.069 | same |
| 5000 | 10 | L1H2 L | 1.000 | 0.050 | same |
| 10000 | 10 | L1H2 L | 1.000 | 0.050 | same |
| 20000 | 16 | L1H4 L | 1.000 | 0.052 | same |
| 30000 | 15 | L0H1 L | 1.000 | 0.053 | same |
| 40000 | 15 | L1H2 L | 1.000 | 0.055 | same |
| 50000 | 14 | L0H0 L | 1.000 | 0.057 | same |
| 60000 | 16 | L0H0 L | 1.000 | 0.054 | same |

## K. Token-level Attention Coordinate Note

Two token-level figures were generated during this work:

- reveal/model order: `cleanbase_step10000_L0H0_token_attention_model_frame.png`
- original L2R labels: `cleanbase_step10000_L0H0_token_attention_original_l2r.png`

The reveal/model-frame metadata reports forbidden future region mean/max as `0.0`; original-L2R coordinates can show mass on both sides of the diagonal after remapping random reveal orders. This supports the coordinate-system boundary: causality is judged in reveal/model order, not original L2R label order.

