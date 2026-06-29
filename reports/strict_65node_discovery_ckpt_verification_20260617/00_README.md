# 00 — README: Strict 65-node Discovery Evidence Consolidation

Date: 2026-06-17

## Purpose

Consolidate existing strict 65-node None-separated block-order discovery evidence and verify stability across multiple checkpoints.

## Key Question

Does a strict 65-node label-free block graph support unanchored L2R block-order discovery, and is the result stable across checkpoints (not a single-ckpt fluke)?

## Protocol (short)

- node 0 = None / BOS start node
- node 1+i = content block i (physical)
- [None] is NOT folded into block 0
- Rollout starts from None; first content block is selected by graph edges
- inv_perm is NOT used during extraction or CDL rollout (strict label-free)
- inv_perm is applied ONLY posthoc to translate sigma_model → sigma_phys for scoring

## Deliverables

| File | Content |
|------|---------|
| `01_protocol_definition.md` | Full protocol spec, node convention, equivariance explanation |
| `02_existing_result_summary.md` | Compiled evidence from all existing scans |
| `03_ckpt_sweep_plan.md` | Multi-checkpoint verification design |
| `04_ckpt_sweep_results.md` | Sweep results across clean_base steps 0–60k + collaborator 50k |
| `05_extraction_frame_comparison.md` | B1 predictor vs loss-aligned AR frame |
| `06_destroyed_controls_summary.md` | All destroyed-control results |
| `07_claim_impact.md` | Updated claims and boundaries |
| `08_boss_update_summary.md` | 1-page Chinese summary for Wednesday |
| `09_next_steps.md` | Remaining verification gaps |

## Key Result

**Strict label-free 65-node discovery: YES, for selected early heads. Stable from 5k–60k. Head-specific, not all-head.**

## Data Sources

- `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/` — collaborator ckpt @50k, strict LF, M=20 full
- `reports/collaborator_ckpt_b1_scan_20260616/none_separated_65_head_method_search/` — collaborator ckpt @50k, oracle-remapped, M=20 full
- `reports/collaborator_ckpt_b1_scan_20260616/clean_base_none65_stability/` — clean_base steps 0–60k, oracle-remapped, M=8
- `reports/collaborator_ckpt_b1_scan_20260616/no_inv_label_free_readout_posthoc_M20_seed0.json` — B1 predictor content-only no-inv
- `reports/strict_65node_discovery_ckpt_verification_20260617/spotcheck/` — strict LF spotcheck on clean_base 10k/60k
