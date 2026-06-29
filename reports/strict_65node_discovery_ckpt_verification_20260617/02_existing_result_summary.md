# 02 — Existing Result Summary

Date: 2026-06-17

## Data Sources

| Source | Model | Steps | Protocol | M | Methods | Ctrl Seeds |
|--------|-------|-------|----------|---|---------|------------|
| `strict_label_free_65_search/` | Collaborator | 50k | strict LF | 20 | 8 | 20 |
| `none_separated_65_head_method_search/` | Collaborator | 50k | oracle-remapped | 20 | 8 | 20 |
| `clean_base_none65_stability/` | Clean base | 0–60k (9 steps) | oracle-remapped | 8 | 3 (L, C-D+L, none_edge) | 5 |
| `strict_65node_discovery_ckpt_verification_20260617/spotcheck/` | Clean base | 10k, 60k | strict LF | 8 | 3 | 5 |
| `no_inv_label_free_readout_posthoc_M20_seed0.json` | Collaborator | 50k | B1 predictor, no-inv | 20 | 1 (C-D+L) | — |

## Gate Distribution (Collaborator ckpt @50k, M=20, full sweep)

| Protocol | strong_pass | weak_pass | fail | total |
|----------|------------|-----------|------|-------|
| Strict label-free | 20 | 11 | 225 | 256 |
| Oracle-remapped | 20 | 11 | 225 | 256 |

**Result**: Identical. Confirmed permutation-equivariance.

## Strong Pass Heads (Collaborator @50k, strict label-free)

All with tau=1.000, first_block=0, phys0_rank=0, prefix@4=4, prefix@8=8:

| Head | Best Method | tau | first_block | phys0_rank | p4 | p8 | destroyed \|τ\| | Gate |
|------|------------|-----|-------------|------------|----|----|-----------------|------|
| L0H1 | L | 1.000 | 0 | 0 | 4 | 8 | 0.050 | strong |
| L0H2 | L | 1.000 | 0 | 0 | 4 | 8 | 0.053 | strong |
| L0H3 | L | 1.000 | 0 | 0 | 4 | 8 | 0.050 | strong |
| L0H4 | L | 1.000 | 0 | 0 | 4 | 8 | 0.055 | strong |

Also strong_pass with C-D+L: L0H1, L0H2, L0H3, L0H4.

**C-only note**: C-only readout also strong_pass for these heads, but NOT recommended as primary evidence — C(v) = mean attention from already-selected nodes to v, which may have low-index tie-breaking bias (earlier nodes get selected first → reinforce each other).

**none_edge note**: `none_edge` (argmax of B[0, 1:]) NEVER strong-passes. The signal is not just "None attends most to physical0" — it requires the rollout dynamics.

## Failed Head Example

| Head | Method | first_block | phys0_rank | tau | p4 overlap | first16 | Gate |
|------|--------|-------------|------------|-----|-----------|---------|------|
| L0H7 | C-D+L | 45 | 20 | 0.292 | 0 | [45,44,38,…] | fail |

L0H7 is canonically the "bad head" found in early testing. It fails under strict 65-node protocol, confirming that **discovery is head-specific, not universal**.

## Clean Base Stability Summary (9-step ladder)

| Step | strong_total | strong_L | strong_C-D+L | best_head | best_tau | destroyed \|τ\| |
|------|-------------|----------|-------------|-----------|----------|-----------------|
| 0 | 0 | 0 | 0 | L0H2 C-D+L | 0.191 | 0.063 |
| 1000 | 0 | 0 | 0 | L2H4 none_edge | 0.142 | 0.069 |
| 5000 | 10 | 5 | 5 | L1H2 L | 1.000 | 0.050 |
| 10000 | 10 | 5 | 5 | L1H2 L | 1.000 | 0.050 |
| 20000 | 16 | 8 | 8 | L1H4 L | 1.000 | 0.052 |
| 30000 | 15 | 7 | 8 | L0H1 L | 1.000 | 0.053 |
| 40000 | 15 | 8 | 7 | L1H2 L | 1.000 | 0.055 |
| 50000 | 14 | 7 | 7 | L0H0 L | 1.000 | 0.057 |
| 60000 | 16 | 8 | 8 | L0H0 L | 1.000 | 0.054 |

**Key observation**: Step 0 and 1k have ZERO strong_pass. From 5k onward, strong_pass heads appear and persist through 60k. The set of strong heads evolves (L1H2 at 5k → L0H0 at 60k: "head drift") but strong_pass counts remain stable (10–16).

## Strict LF Spotcheck (Clean Base)

Confirmed equivalence with oracle-remapped at steps 10k and 60k: 32/32 heads match within 0.01 tau.

## Extraction Frame Comparison (Preliminary)

| Head | B1 predictor + content-only (no-inv) | Loss-aligned AR + None-sep (strict LF) |
|------|-------------------------------------|---------------------------------------|
| L0H1 | tau=0.655 | tau=1.000 |
| L0H2 | tau=0.655 | tau=1.000 |
| L0H3 | tau=1.000 | tau=1.000 |
| L0H4 | tau=0.655 | tau=1.000 |

B1 predictor frame without [None] recovers cyclic order axis but loses the anchor (first physical = 6 instead of 0 for L0H1/H2/H4). Loss-aligned AR with None-separated graph retains [None] as independent BOS, enabling correct anchor selection.
