# Boss Summary — One-Page Executive Overview

## One-Line Result

> **Order-agnostic ≠ order-free**: AO-GPT attention heads internally encode recoverable L2R block order. We read it out with a strict 65-node protocol and use it to accelerate training by 33–42%.

---

## Core Idea (Two Layers)

| Layer | Status | Key Finding |
|-------|--------|-------------|
| **Mechanism** | ✅ Verified | Sparse early attention heads recover physical L2R under strict 65-node label-free protocol |
| **Acceleration** | 🟡 B0 verified / Strict pending | Legacy g_β controller saves 33–42% steps; strict-teacher hook is next priority |

---

## Top Evidence

### 1. Strict 65-node label-free discovery
- 4 strong-pass heads (L0H1–L0H4, τ=1.000, first=0, phys0_rank=0) on collaborator ckpt @50k
- 10–16 strong heads stable 5k–60k on clean-base ladder
- Gate distribution: 20 strong / 11 weak / 225 fail (7.8% pass) — sparse, not dense
- Destroyed controls: |τ| ≤ 0.07, real–destroyed gap ≈ 0.95
- Source: `reports/strict_65node_discovery_ckpt_verification_20260617/`

### 2. Legacy B0 controller acceleration
- seed-123: Recovery 86% (from10k), 83% (from20k). Step saving 42%.
- seed-42: Recovery 106% (from10k), 102% (from20k). Step saving 39%.
- Conservative range: **33–42%** step saving across 2 seeds, from10k+from20k.
- Wall-clock: ~62% saving including g_β training overhead.
- Source: `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md`

### 3. g_β sanity
- Real B τ=+0.9675 vs L2R; Gaussian B τ=−0.0034; pairwise Gaussian τ=0.0003.
- g_β reads structured B, not a fixed L2R prior.
- Source: `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`

### 4. Protocol rigor
- inv_perm boundary strictly enforced (graph construction: OK; CDL rollout: forbidden; posthoc scoring: OK)
- Extraction frame comparison: loss-aligned AR + None-sep >> B1 predictor content-only for anchored discovery
- Permutation convention clarified (nanogpt vs block_lo_arm_order_network)
- Deprecated numbers corrected (seed42 baseline = 3.466, not 3.354)

---

## Can Claim

1. Strict 65-node label-free block-level L2R discovery in selected early heads
2. Order-agnostic ≠ order-free — internal order structure exists
3. Legacy B0 g_β controller accelerates training 33–42%
4. g_β reads structured attention, not fixed L2R prior
5. Extraction frame matters for discovery protocol
6. Destroyed controls confirm non-artifact

## Cannot Claim

1. All heads discover L2R (only 7.8% pass)
2. Strict-teacher controller hook is done (pending)
3. Content-only proves content-driven adjacency (fixed perm = position memory risk)
4. 317M acceleration established (only diagnostic at 5k)
5. Image/multimodal side solved
6. Fully label-free end-to-end chain verified (audition uses supervised train, g_β uses CDL teacher)
7. Same g_β transfers across seeds/heads

---

## Next Priority Experiments

| Priority | Experiment | Why |
|----------|-----------|-----|
| **P0** | Strict 65-node teacher → g_β → hook smoke | Closes mechanism-to-acceleration chain |
| **P0** | Cross-permutation adjacency validation | Separates content vs position memory |
| **P1** | Label-free head audition | Removes last oracle dependency |
| **P1** | Third seed (seed=123 with B1) | Statistical completeness |
| **P2** | 317M strict 65-node diagnostic | Scale generalization |
| **P2** | Unified figure generation | Paper-ready visuals |

---

## Paper Venue Readiness

- **Mechanism paper** (ICLR / ICML workshop level): Ready with current evidence + P0 closure experiments.
- **Full method paper** (NeurIPS / ICML): Needs strict-teacher hook closure, multi-scale validation, label-free audition.
- **Current state**: Strong mechanism story, B0 controller evidence, pending strict-teacher controller chain.
