# 04 — Checkpoint Sweep Results

Date: 2026-06-17

## Clean Base Ladder: Full Sweep

**Protocol**: Oracle-remapped 65-node + strict LF full sweep (9/9 steps confirmed identical)
**Primary data**: `strict_lf_cleanbase_sweep/` (strict label-free M=8, 9 steps)
**Settings**: M=8, L/C-D+L/none_edge, 5 control seeds

### Per-Step Gate Distribution

| Step | strong_total | strong_L | strong_C-D+L | strong_none_edge | weak | fail | destroyed \|τ\| |
|------|-------------|----------|-------------|-----------------|------|------|-----------------|
| 0 | 0 | 0 | 0 | 0 | 0 | 96 | 0.063 |
| 1000 | 0 | 0 | 0 | 0 | 0 | 96 | 0.069 |
| 5000 | 10 | 5 | 5 | 0 | 2 | 84 | 0.050 |
| 10000 | 10 | 5 | 5 | 0 | 2 | 84 | 0.050 |
| 20000 | 16 | 8 | 8 | 0 | 1 | 79 | 0.052 |
| 30000 | 15 | 7 | 8 | 0 | 4 | 77 | 0.053 |
| 40000 | 15 | 8 | 7 | 0 | 1 | 80 | 0.055 |
| 50000 | 14 | 7 | 7 | 0 | 1 | 81 | 0.057 |
| 60000 | 16 | 8 | 8 | 0 | 2 | 78 | 0.054 |

### Per-Step Best Candidate

| Step | Best Head+Method | tau | first | phys0_rank | p4 | p8 | Gate |
|------|-----------------|-----|-------|------------|----|----|------|
| 0 | L0H2 C-D+L | 0.191 | 38 | 8 | 1 | — | fail |
| 1000 | L2H4 none_edge | 0.142 | 3 | 17 | 1 | — | fail |
| 5000 | L1H2 L | 1.000 | 0 | 0 | 4 | 8 | strong |
| 10000 | L1H2 L | 1.000 | 0 | 0 | 4 | 8 | strong |
| 20000 | L1H4 L | 1.000 | 0 | 0 | 4 | 8 | strong |
| 30000 | L0H1 L | 1.000 | 0 | 0 | 4 | 8 | strong |
| 40000 | L1H2 L | 1.000 | 0 | 0 | 4 | 8 | strong |
| 50000 | L0H0 L | 1.000 | 0 | 0 | 4 | 8 | strong |
| 60000 | L0H0 L | 1.000 | 0 | 0 | 4 | 8 | strong |

### Stable Strong Heads (5k–60k, appear at ≥5 steps)

| Head | Method | Steps Present | Notes |
|------|--------|--------------|-------|
| L1H0 | C-D+L | 5k–60k (8/8) | Most stable |
| L1H0 | L | 5k–60k (8/8) | Most stable |
| L1H1 | C-D+L | 5k–60k (8/8) | Most stable |
| L1H1 | L | 5k–60k (8/8) | Most stable |
| L1H2 | C-D+L | 5k–60k (8/8) | Most stable |
| L1H2 | L | 5k–60k (8/8) | Most stable |
| L1H4 | C-D+L | 5k–60k (8/8) | Most stable |
| L1H4 | L | 5k–60k (8/8) | Most stable |
| L0H0 | L | 20k–60k (5/8) | Emerges late |
| L0H0 | C-D+L | 20k–60k (6/8) | Emerges late |
| L2H0 | C-D+L | 20k–60k (5/8) | Emerges late |
| L2H0 | L | 20k–60k (4/8) | Emerges late |
| L2H4 | C-D+L | 20k–60k (4/8) | Emerges late |

**Note**: L0H1–L0H4 (strong in collaborator) are NOT consistently strong in clean_base. The clean_base's stable strong heads are L1H0–L1H4 and (from 20k) L0H0/L2H0/L2H4. This is cross-training-run head drift — expected for random-order AO-GPT training.

### Head Drift Analysis

The "best head" (highest-scoring strong_pass) shifts across checkpoints:
```
5k:  L1H2 → 10k: L1H2 → 20k: L1H4 → 30k: L0H1 → 40k: L1H2 → 50k: L0H0 → 60k: L0H0
```

This confirms the earlier finding (BR-1 status, 2026-05-29): **head identity is not fixed across training runs or even across checkpoints within a single run**. The discovery phenomenon is real but the specific head index drifts.

## Collaborator ckpt @50k (M=20, Full Sweep)

Gate distribution: 20 strong / 11 weak / 225 fail (identical strict LF vs oracle-remapped).

Strong pass heads (all with tau=1.000): L0H1, L0H2, L0H3, L0H4 (L and C-D+L methods).

## Cross-Model Comparison

| Model | Best Heads | Protocol | M |
|-------|-----------|----------|---|
| Clean base 5k–60k | L1H0–L1H4 (stable), L0H0 (late) | oracle-remapped | 8 |
| Collaborator 50k | L0H1–L0H4 | strict LF + oracle-remapped | 20 |

Different training runs produce different strong heads. But strong_pass heads exist in BOTH models from 5k onward. The phenomenon is robust to training seed/run, while head identity is not.

## Key Conclusions

1. **Persistence**: strong_pass heads exist from 5k to 60k (stable count: 10–16).
2. **Emergence**: steps 0 and 1k have ZERO strong_pass. Signal requires training.
3. **Head drift**: specific strong head indices shift across checkpoints (L1H2→L1H4→L0H1→L0H0).
4. **Cross-model**: both clean_base and collaborator models have strong_pass heads, but with different head indices.
5. **Destroyed controls**: always near random (|τ| ≤ 0.07), confirming the signal depends on real attention structure.
6. **none_edge NEVER passes**: the signal requires CDL rollout dynamics, not just None→content edge sorting.
