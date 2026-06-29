# 03 — Checkpoint Sweep Plan

Date: 2026-06-17
Status: **EXECUTED** (existing data covers the sweep; strict LF spotcheck confirms equivalence)

## P0: Clean Base Ladder (same model, same protocol, multiple checkpoints)

**Source**: `clean_base_random_perm` checkpoint ladder
**Checkpoints**: step 0, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000
**Protocol**: Oracle-remapped 65-node (proven equivalent to strict LF via spotcheck)
**Settings**: M=8, batch_size=8, methods=L/C-D+L/none_edge, control_seeds=5

**Status**: ✅ Complete. All 9 steps have full head×method scan data.

## P1: All-Head Scan at Key Steps

Selected steps: 10k, 50k, 60k (earliest strong_pass emergence + mid + late)

**Status**: ✅ Complete via stability scan.

Gate distribution at key steps:

| Step | strong_pass | weak_pass | fail | best_head | best_tau |
|------|------------|-----------|------|-----------|----------|
| 10k | 10 | 2 | 84 | L1H2 L | 1.000 |
| 50k | 14 | 1 | 81 | L0H0 L | 1.000 |
| 60k | 16 | 2 | 78 | L0H0 L | 1.000 |

## P2: Destroyed Controls

For each step in stability scan, controls are computed:
- entry_shuffled_control (shuffle all edge weights while keeping None fixed and diag zero)
- content_label_permutation_control (permute content node labels while keeping None fixed)
- 5 seeds each → 10 control samples per (step, head, method)

**Status**: ✅ Complete. All controls available in stability scan per-step JSON files.

Mean destroyed |τ| across all steps: 0.050–0.069 (always near random).

## P3: Collaborator Cross-Model Check

**Source**: `/home/admin/ych/nanogpt-learned-order/ckpt/ckpt.pt` @ iter 50000
**Protocol**: Both strict LF and oracle-remapped, M=20 full sweep (8 methods, 20 control seeds)

**Status**: ✅ Complete. Full equivalence confirmed.

## P4: Strict LF Spotcheck on Clean Base

**Steps**: 10k, 60k
**Protocol**: Strict label-free M=8 (L, C-D+L, none_edge)

**Status**: ✅ Complete. 32/32 heads match within 0.01 tau (confirmed equivalence).

## Gaps / Pending

| Item | Status | Priority |
|------|--------|----------|
| M=20 strict LF on clean_base (not just M=8 spotcheck) | Pending | Low (M=8 equivalence confirmed) |
| Collaborator ckpt at multiple steps (only single ckpt available) | Blocked | Low (no ckpt ladder from collaborator) |
| Additional seeds for clean_base | Pending | Medium (current data is single-seed training) |
| 317M model (16L) strict LF | Pending | Low (B0 results exist, 65-node not run) |
| Image model strict LF | Not started | Future |
