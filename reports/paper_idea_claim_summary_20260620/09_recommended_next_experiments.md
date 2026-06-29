# Recommended Next Experiments

## P0 — Close Mechanism-to-Acceleration Chain

### P0a: Strict 65-node teacher → g_β distillation → hook smoke

**Goal**: Close the loop from strict 65-node label-free discovery to g_β controller acceleration.

**Steps**:
1. Select strong head(s) from strict 65-node discovery (e.g., L0H1–L0H4 from collaborator ckpt, or L1H0–L1H4 from clean_base). Use existing discovery data — no new head search needed.
2. Generate strict 65-node CDL teacher orders for selected head(s). Use M=2000 for g_β training data.
3. Train g_β on strict 65-node teacher (nodewise + pairwise, same architecture as legacy g_β).
4. Sanity check: real B vs destroyed B (Gaussian, entry-shuffled).
5. Run short hook from 10k or 20k baseline (e.g., 20k steps, 5k warmup).
6. Compare to: (a) legacy B0 hook, (b) random baseline, (c) L2R reference.
7. Gate: τ ≥ 0.6 AND pairwise ≥ 0.8 AND model_order < unstructured − 0.3.

**Success criteria**: model_order improves over random baseline, and ideally reaches within 0.1 NLL of legacy B0 hook performance.

**Estimated time**: ~6–8 hours (CDL teacher ~2h + g_β ~1h + hook ~3.5h).

---

### P0b: Cross-permutation validation for content-only adjacency

**Goal**: Distinguish content-driven physical adjacency from memorized position pairs.

**Steps**:
1. Use a different permute_seed (or different ckpt with different block_perm).
2. Recompute d=+1 band attention in correct physical frame (using the other ckpt's block_perm).
3. If d=+1 band follows physical adjacency (not fixed training permutation), this supports content-driven adjacency.
4. If d=+1 band stays at the positions memorized from training permutation, this supports position memory.

**Success criteria**: d=+1 physical > d=+1 position-memorized by ≥ 3× margin.

**Estimated time**: ~1 hour (compute only, no training).

---

## P1 — Strengthen Existing Evidence

### P1a: Label-free head audition (no oracle τ)

**Goal**: Remove the last oracle dependency in head selection.

**Approach**: Use the existing `audition_heads.py` pipeline (small supervised train, 100 steps, select by advantage over random baseline) to select heads. Then verify with strict 65-node readout that the audition-selected head is a strong-pass head.

**Success criteria**: Audition-selected head achieves strict 65-node τ ≥ 0.9 AND first=0 AND phys0_rank=0.

**Estimated time**: ~2 hours per checkpoint.

---

### P1b: Third seed — seed=123 with B1 protocol

**Goal**: Statistical completeness — add a third matched seed group.

**Steps**:
1. Run B1 head scan on seed-123 continuous baseline at 10k, 20k, 50k.
2. If strong head found, run full pipeline (CDL → g_β → frozen_beta).
3. Add to recovery/step-saving table.

**Estimated time**: ~10 hours for full pipeline.

---

### P1c: Increase M for strict discovery gate statistics

**Goal**: More reliable gate distribution with M=200+ (current: M=8/20).

**Steps**: Re-run strict 65-node sweep on clean_base at 2–3 representative steps (e.g., 10k, 30k, 60k) with M=200.

**Estimated time**: ~2 hours per step.

---

## P2 — Scale and Polish

### P2a: 317M strict 65-node diagnostic

**Goal**: Verify that the strict 65-node discovery phenomenon generalizes to larger models.

**Steps**: Run strict 65-node full sweep on 317M model at step 5000 (where B0 shows 6/256 |τ|>0.9). Gate distribution and strong-pass head identification.

**Estimated time**: ~4 hours (large model inference).

---

### P2b: Unified figure generation

**Goal**: Produce paper-ready figures with verified numbers.

**Figures to generate/regenerate**:
1. Strict 65-node graph schematic (node layout, rollout arrows)
2. L0H1–L0H4 strong-pass B65 heatmap panel
3. Gate distribution bar chart (20/11/225)
4. Clean-base ladder strong_pass count vs step
5. Updated catchup curves with correct seed42 baseline
6. g_β sanity bar chart (real vs destroyed)
7. Extraction frame comparison panel

**Estimated time**: ~4 hours.

---

### P2c: CDL teacher ablation (C-only, D-only, L-only) at scale

**Goal**: Systematic comparison of readout methods. Current evidence is from M=8/20.

**Estimated time**: ~3 hours.

---

## Priority Summary

| Priority | Experiment | Time | Blocks What |
|----------|-----------|------|------------|
| **P0a** | Strict-teacher g_β hook | 6–8h | Full method paper |
| **P0b** | Cross-permutation adjacency | 1h | Content-vs-position claim |
| P1a | Label-free audition | 2h | "Fully label-free" claim |
| P1b | Third seed | 10h | Statistical completeness |
| P1c | M=200 strict gate | 2h | Gate distribution confidence |
| P2a | 317M strict diagnostic | 4h | Scale generalization |
| P2b | Paper figures | 4h | Paper readiness |
| P2c | CDL ablation at scale | 3h | Method comparison |
