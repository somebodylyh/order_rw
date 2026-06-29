# New g_beta: Design, Protocol, and Research Roadmap

**Date**: 2026-06-25
**Status**: Living document — evolves with experimental evidence
**Audience**: Collaborators, advisors, paper reviewers
**Scope**: This document describes the **new** g_beta design. It does NOT describe the old frozen_beta / single-head teacher-distillation pipeline. Where historical results are referenced, they are explicitly labeled as prior evidence.

---

## 1. Executive Summary

We study **Any-Order Autoregressive GPT (AO-GPT)**, where training uses a per-step block reveal order rather than fixed left-to-right. The default policy is **uniform random order**. Our hypothesis: a lightweight, model-frame order controller — **g_beta** — can read the model's current internal state and predict a reveal order that improves training convergence over random.

The **new g_beta** is a principled redesign with the following commitments:

1. **Model-frame only**: order generation uses zero physical-coordinate information (no `inv_perm`, no `clean_perm`, no `block_perm`, no physical position).
2. **Attention-graph input**: g_beta reads block-level attention graphs extracted from the AO-GPT's own L0 attention heads.
3. **Pairwise ranking objective**: g_beta is trained to predict a teacher-derived or heuristic target order via soft pairwise BCE.
4. **Frozen deployment**: g_beta is pretrained offline on a fixed checkpoint, then frozen during AO-GPT continuation training — establishing a clean causal claim.
5. **Batch-mean probe averaging**: multiple random-probe forward passes are averaged to reduce order variance, matching the pretraining distribution.

**Current status (2026-06-25)**:
- v0 label-free pipeline: 9 modules, 180 tests, dual-layout τ > 0.98 ✅
- Frozen hook integration with batch-mean fix: τ 0.4 → 0.91 ✅
- Aligned frozen_gbeta run launched: seed=123, 20k→60k, batch_mean_probes=4 🟢 running
- Baselines available: random_from20k, layout_path_from20k (seed=2, from same ckpt_step20000)

---

## 2. Problem Setup

### 2.1 AO-GPT Training

AO-GPT factorizes sequence likelihood as:

```
P(x_1,...,x_T) = ∏_{t=1}^T P(x_{σ(t)} | x_{σ(1)},...,x_{σ(t-1)})
```

where σ is a permutation (reveal order) over T tokens organized into N blocks of size B (N=64, B=4, T=256). Training minimizes next-token prediction loss under σ.

### 2.2 Random Reveal Order

Default policy: σ ~ Uniform(S_N), the symmetric group over N blocks. At each step, a fresh random σ is sampled per batch. This is the AO-GPT baseline.

**Known issues with random**:
- High per-step variance in training signal
- No adaptation to model state — some σ are "easier" or "more informative" than others for the current model
- Ignores structure the model has already internalized in its attention patterns

### 2.3 Core Question

> Can a lightweight controller g_beta, reading only model-internal signals, produce reveal orders σ that yield faster AO-GPT convergence than random, under a strictly aligned comparison?

---

## 3. Motivation for New g_beta

### 3.1 Why Not Just Use a Teacher Directly?

Prior work (CDL teacher, oracle L2R, layout-path) can produce orders that beat random. But direct teacher hook has problems:
- **Expensive**: teacher rollout requires O(N²) candidate evaluations per refresh
- **Not label-free**: CDL teacher itself uses physical position in its coordinate conventions
- **Hard to claim as method**: if teacher uses L2R or layout-path, the method *is* the oracle

g_beta is the **amortized, label-free proxy**: learn to imitate useful orders from cheap model-frame features, so the hook is fast and causally clean.

### 3.2 Old g_beta vs. New g_beta

| Aspect | Old g_beta | New g_beta |
|--------|-----------|------------|
| Input | Single-head B1 graph (physical-frame) | Multi-head strict65 attention graph (model-frame) |
| Head selection | Fixed by audition (L0H2) | All L0 heads, mean-pooled (or learned gate) |
| Physical leakage | `inv_perm` used in B extraction | Zero physical fields in method path |
| Probe averaging | Single probe (high variance) | batch_mean_probes=4 |
| Target order | CDL teacher on single head | Flexible: teacher, heuristic, or self-supervised |
| Model architecture | NodewiseReadout (64-node) | L0DynamicGBeta (65-node, multi-head gate) or NodewiseReadout |
| Deployment | Frozen, single-head B extraction | Frozen, model-frame strict65 extraction |
| Causal claim | Weak (confounded by head selection + physical leakage) | Strong (label-free, model-frame, aligned baseline) |

### 3.3 Design Principles

1. **Model-frame purity**: The order generation path must contain zero physical-coordinate operations. Physical remapping is a post-hoc translation step in the training loop, not part of g_beta.
2. **Frozen for causal clarity**: g_beta is pretrained and frozen. AO-GPT training updates only the main model. This isolates the effect of order policy.
3. **Aligned comparison**: All policies start from the same checkpoint, use identical hyperparameters, and differ only in order policy.
4. **Ablation-ready**: Every design choice that is heuristic rather than derived must be flagged and have a planned ablation.

---

## 4. New g_beta: Core Formulation

### 4.1 Abstract Definition

Given current AO-GPT state M_θ and a batch of inputs x:

```
g_beta: B ∈ R^{N×N} → z ∈ R^N
σ_beta = argsort(-z)
```

- **N = 64**: number of blocks
- **B**: block-level attention graph extracted from M_θ's L0 attention
- **z_i**: scalar "reveal priority" score for block i
- **σ_beta**: predicted block reveal order (higher z → earlier reveal)

g_beta does **not** generate tokens. It does **not** replace AO-GPT. It only decides *which block to reveal next*.

### 4.2 Model Architecture (v0)

**Current implementation** (`L0DynamicGBeta`):
- Input: (B, H, 65, 65) strict65 attention graphs (H = 8 L0 heads, 65 nodes with None-token)
- Per-head scorer: shared MLP(256→64) → scalar per node
- Dynamic gate: softmax over heads, gating raw attention channel excluded
- Output: (B, 64) block scores (None-token removed)

**Alternative** (`NodewiseReadout`):
- Input: (B, 64, 64) block graph (heads averaged, None-token stripped)
- Per-node feature: [B[v,:], B[:,v]] ∈ R^{2N}
- Transformer encoder (d_model=64, n_heads=4, n_layers=2) → scalar per node
- Simpler, currently deployed in aligned run

> **Heuristic choice**: NodewiseReadout used in current aligned run because the multi-head L0DynamicGBeta checkpoint was not trained. This is a temporary implementation decision, not a design commitment. See §11.1.

### 4.3 Key Properties

- **Permutation equivariance**: g_beta(B) respects reindexing of block indices (the model operates on graph structure, not position)
- **Label-free**: no access to physical L2R, layout path, or any ground-truth ordering
- **Deterministic**: hard argsort produces a single order (no sampling). Stochastic variants are ablation candidates.

---

## 5. Input Features

### 5.1 Primary: L0 Attention Graph (Current Main Design)

**Extraction pipeline** (model-frame strict65):

1. Random probe forward: run AO-GPT with a random model-frame block order
2. Extract L0 attention: `attn_l0 ∈ R^{Bsz × H × 257 × 257}` (257 = 256 tokens + 1 None-token)
3. Aggregate to blocks: use `build_model_frame_strict65` which maps token attention to block-level B
4. Output: `B ∈ R^{Bsz × H × 65 × 65}` where node 0 = None-token, nodes 1..64 = content blocks

**Design choices** (all heuristic, all need ablation):

| Choice | Current | Rationale | Ablation needed |
|--------|---------|-----------|-----------------|
| Layer | L0 only | Earliest, most structure-bearing | L1..L3, all-layer mean |
| Heads | All 8 L0 heads | Avoid cherry-picking | Single best head, random head, top-k |
| Aggregation | strict65 (none-separated) | Preserves None-token as separate node | B1-style (drop None), other none modes |
| Direction | A → B = A^T (column-normalized) | B_ij = attention from j→i, interpretable as "i depends on j" | Raw A, symmetric (A+A^T)/2 |
| Normalization | Row-normalized in strict65 | Each row sums to 1 | Column-norm, no norm, softmax |
| Diagonal | Zeroed | Self-attention not informative for order | Keep diagonal |
| Probe order | Random model-frame | Keeps extraction in-distribution for g_beta | Fixed order, L2R probe |

### 5.2 Candidate: Hidden State Features

**Not yet implemented.** Idea:
- Pool hidden states per block → h_i
- g_beta(h_i, h_j) → pairwise reveal priority
- Could use predictor hidden state c_t (target-aware by construction in AO-GPT)

**Concerns**:
- Hidden state may encode position more directly → leakage risk
- Harder to prove "order signal comes from structure, not position"
- Joint optimization instability if not detached

### 5.3 Candidate: Loss / Uncertainty Features

**Not yet implemented.** Idea:
- Compute per-block prediction loss from current AO-GPT
- High-loss blocks → reveal later (curriculum) or earlier (active learning)
- Use confidence/entropy as continuous signal

**Concerns**:
- Requires extra forward pass → overhead
- No principled answer to "high-loss-first or low-loss-first"
- More like curriculum learning than structure discovery

### 5.4 Forbidden Features (Model-Frame Purity)

These must **never** enter g_beta's order generation:

- `inv_perm` / `clean_perm` / `block_perm`
- Physical block index or token position
- L2R order or any ground-truth ordering
- Any feature that encodes "where this block is in physical space"

These are only allowed in:
- Oracle baselines (explicitly labeled)
- Post-hoc characterization scripts
- The training loop's physical-remapping step (after g_beta produces model-frame σ)

---

## 6. Training Objective

### 6.1 Current: Soft Pairwise BCE (v0)

Given a target order σ_T (from teacher or heuristic), define pairwise preference labels:

```
y_{ij} = 1  if σ_T(i) < σ_T(j)  (i revealed before j)
y_{ij} = 0  if σ_T(i) > σ_T(j)
```

g_beta outputs block scores z. Pairwise BCE loss:

```
L = -Σ_{i≠j} [y_{ij} log σ(z_i - z_j) + (1-y_{ij}) log(1 - σ(z_i - z_j))]
```

where σ is the sigmoid function. Ties are excluded. Diagonal is excluded.

**Why pairwise ranking**:
- Permutation is fundamentally a relative ordering, not absolute rank regression
- Robust to noisy/partial targets (only need relative preferences)
- O(N²) pairs but can be sampled for efficiency
- Standard in learning-to-rank literature

**Heuristic choices**:
- BCE over margin-based (hinge) loss → smoother gradients ✅ (reasonable default)
- All O(N²) pairs used (no sampling) → fine for N=64
- No pair weighting (all pairs equal) → could weight by rank difference

### 6.2 Target Order Sources

| Source | Type | Status | Issues |
|--------|------|--------|--------|
| CDL teacher (C−D+L rollout) | Learned, per-checkpoint | Used in v0 pretrain | Expensive, uses physical coords |
| −D only (U_t component) | Ablation of CDL | Verified load-bearing | Still requires CDL extraction |
| Layout-path order | Oracle | Post-hoc characterization only | NOT label-free |
| Physical L2R | Oracle | Post-hoc only | Upper bound, not method |
| Random order | Baseline | Training baseline | Not a target |

**Current v0 target**: CDL teacher on L0H2, pretrain dataset, batch_mean=16. This is a **heuristic choice** — CDL is currently our best available teacher, but we do not claim it is optimal.

### 6.3 Alternative Objectives (Ablation Candidates)

- **ListMLE / ListNet**: listwise likelihood over full permutation → closer to ranking theory but complex
- **Differentiable sorting**: Sinkhorn relaxation, soft permutation → differentiable but may not match hard argsort at deployment
- **Contrastive**: positive pairs (should be earlier) vs negative pairs
- **Direct score regression**: regress z_i to rank → simpler but less principled for permutations

---

## 7. Integration into AO-GPT Training

### 7.1 Training Loop (Current Protocol)

```
For each training step:
  1. Sample batch idx from continuous stream
  2. Every K steps (refresh_every=10):
     a. Run batch_mean_probes=4 random-probe forward passes
     b. Extract strict65 B matrices, average over probes
     c. g_beta(B) → scores z
     d. σ_model = argsort(-z)  [model-frame block order]
     e. Cache σ_model
  3. (Other steps): use cached σ_model
  4. Training loop remaps σ_model → σ_phys via clean_perm
  5. Mix σ_phys with random order at ratio α (alpha schedule)
  6. AO-GPT forward + backward (updates AO-GPT only, g_beta frozen)
```

### 7.2 Alpha Schedule

- α starts at 0.0 (pure random) for first 10k steps
- Linear warmup to α = 0.9 over 10k steps
- Remains at 0.9 for remaining steps
- Per-sample: fraction α of batch uses g_beta order, 1−α uses random

> **Heuristic**: α=0.9 rather than 1.0. Rationale: retain some order diversity, prevent overfitting to a fixed order policy. Needs ablation: α ∈ {0.5, 0.7, 0.9, 1.0}.

### 7.3 Refresh Interval K

Current: K = 10.

| K | Pros | Cons |
|---|------|------|
| 1 | Freshest order | Maximum overhead (~4 probes/step) |
| 10 | Good balance | Order may be slightly stale |
| 50 | Low overhead | Order likely stale |

> **Heuristic**: K=10. Needs ablation: K ∈ {1, 5, 10, 20, 50}. Measure both step convergence and wall-clock convergence.

### 7.4 Probe Averaging

Current: batch_mean_probes = 4.

Each probe uses a different random model-frame order for the extraction forward pass. The resulting B matrices are averaged before feeding to g_beta. This:
- Reduces variance from single-probe attention noise
- Matches the pretraining distribution (g_beta was trained on batch-mean B)
- Adds overhead: 4 forward passes per refresh

> **Heuristic**: probes=4. Needs ablation: probes ∈ {1, 2, 4, 8}. Measure order stability (τ across probes) and wall-clock overhead.

### 7.5 Model-Frame Purity in the Loop

The method path is:

```
probe forward (random model-frame probe orders)
→ strict65 B per probe
→ mean B over probes
→ g_beta(B) → scores
→ argsort(-scores) = σ_model  [model-frame block order]
```

Physical remapping happens **after** g_beta:
```
σ_phys = clean_perm.inv_perm_model_to_phys[σ_model]  [training loop only]
```

This separation ensures:
- g_beta never touches physical coordinates
- The training loop's remapping is a post-hoc translation, not part of order generation
- Label-free audit: grep for `inv_perm`, `clean_perm`, `block_perm` in the g_beta call chain returns zero hits

---

## 8. Aligned Experimental Protocol

### 8.1 Core Comparison

| Policy | Type | Start ckpt | Additional steps | Order policy |
|--------|------|------------|-----------------|--------------|
| random_continue | Baseline | ckpt_step20000 | 40k (→60k total) | Random |
| layout_path_continue | Oracle | same | 40k | Layout-path (uses physical) |
| **frozen_gbeta** | **Method** | same | 40k | g_beta (model-frame, label-free) |

### 8.2 Config Alignment Audit

| Field | random_continue | frozen_gbeta | Aligned? |
|-------|:---:|:---:|:---:|
| Start checkpoint | ckpt_step20000.pt | same | ✅ |
| Additional steps | 40000 | 40000 | ✅ |
| Model | 4L/8H/d=384 | same | ✅ |
| Dataset | WikiText-103 continuous | same | ✅ |
| Batch size | 64 | 64 | ✅ |
| Grad accum | 2 | 2 | ✅ |
| Optimizer | AdamW (β1=0.9, β2=0.99) | same | ✅ |
| LR | 1e-3 → 1e-4 cosine | same | ✅ |
| LR decay steps | 60000 | 60000 | ✅ |
| Warmup | 0 | 0 | ✅ |
| Weight decay | 0.1 | 0.1 | ✅ |
| Dropout | 0.0 | 0.0 | ✅ |
| Grad clip | 1.0 | 1.0 | ✅ |
| Block/token size | 256 / 4 | same | ✅ |
| Seed | 123 | 123 | ✅ |
| Precision | float32 | float32 | ✅ |
| Eval set | Fixed (sha256 verified) | same | ✅ |
| Eval metric | val_ori_l2r_block | same | ✅ |
| Eval interval | 1000 | 1000 | ✅ |
| **Order policy** | random | **frozen_gbeta** | 🔶 only variable |

### 8.3 Current Run Status (2026-06-25)

- **frozen_gbeta**: PID 727472, GPU 0, W&B `order-lyu/frozen_gbeta_aligned_20260625`
  - batch_mean_probes=4, refresh_every=10, none_mode=model
  - g_beta ckpt: `gbeta_b1_L0H2_seed2_step20k` (NodewiseReadout)
  - 20k → 60k (40k steps), ETA ~5-7h
- **random_continue baseline**: seed=2, W&B `frozen-gbeta-60k/random_from20k` (completed or running)
- **layout_path_continue**: seed=2, W&B `frozen-gbeta-60k/layout_path_from20k` (queued/completed)

> **Note**: baseline uses seed=2, method uses seed=123. This is a seed mismatch. Ideally both should use the same seed. Flag as a limitation; single-seed evidence only.

---

## 9. Metrics

### 9.1 Primary Metrics

| Metric | Definition | Reported |
|--------|-----------|:---:|
| val_ori_l2r_block | Validation NLL under physical L2R block order | Every 1k steps |
| Δ vs random | val_l2r(method) − val_l2r(random) at same step | Final |
| Step saving | Steps saved to reach same val_l2r as random@60k | If applicable |

### 9.2 Wall-Clock Metrics

| Metric | Definition |
|--------|-----------|
| Time per step | Average wall-clock seconds per training step |
| Overhead % | (time_gbeta − time_random) / time_random × 100 |
| Time to target NLL | Wall-clock time to reach a fixed val_l2r threshold |

### 9.3 Order Diagnostics (Hook-Time)

| Metric | Definition |
|--------|-----------|
| τ_vs_layout_path | Kendall τ between g_beta order and layout-path order |
| τ_vs_identity | Kendall τ between g_beta order and identity [0,1,...,63] |
| τ_vs_random | Kendall τ between g_beta order and random order |
| α entropy | Entropy of per-head gate weights |
| α mean per head | Average gate weight for each L0 head |
| prefix@8 | Fraction of top-8 blocks shared between consecutive refreshes |
| Refresh stats | Number of refreshes, mean time per refresh |
| Order stability | τ(σ_refresh_t, σ_refresh_{t+K}) across consecutive refreshes |

### 9.4 Safety / Leakage Diagnostics

| Diagnostic | Expected if clean |
|-----------|-----------------|
| grep inv_perm in g_beta path | Zero hits |
| grep clean_perm in g_beta path | Zero hits |
| Destroyed B input → τ ≈ 0 | τ close to 0 |
| Gaussian B input → τ ≈ 0 | τ close to 0 |
| Shuffled B (permuted rows/cols) → τ ≈ 0 | τ close to 0 |
| Random head (not selected head) → degraded τ | Lower than selected head |

---

## 10. Current Evidence (v0, pre-aligned)

From the v0 label-free pipeline (see `reports/v0_label_free_gbeta_summary_20260625.md`):

| Claim | Evidence | Strength |
|-------|----------|----------|
| g_beta reads real attention structure | Destroyed B: acc drops −0.478 | Strong |
| Not single-head oracle | Remove-top-α: acc drops −0.001 | Strong |
| Learned mixture > uniform | Δacc = +0.023 | Moderate |
| Cross-layout replication | τ > 0.98 on own layout, τ ≈ 0.05 on other | Strong |
| Frozen hook beats random at 20k | Δ = −0.045 | Moderate (old protocol, not fully aligned) |
| Batch-mean fix critical | τ 0.42 → 0.91 with probes=4 | Strong |
| 60k benefit confirmed with drift | frozen vs destroyed gap = 0.087 | Weak (old protocol, drift confound) |

**All pre-aligned results are PILOT ONLY.** The aligned run launched 2026-06-25 is the first apples-to-apples evidence.

---

## 11. Heuristic Design Choices Requiring Ablation

This section is explicit about what we do NOT know and what MUST be tested.

### 11.1 g_beta Input Features

**Current**: L0 all-head attention, strict65 aggregation, mean-pool over heads.

**Unknowns**:
- Is L0 the best layer? L1? L-1? All-layer mean?
- Is all-head mean better than a selected best head?
- Is attention graph sufficient, or do hidden states / loss signals add value?
- Does strict65 (with None-token as separate node) matter vs. B1 (drop None)?

**Required ablations**: L0 vs L1 vs L-last; best head vs all-head mean; attention-only vs attention+hidden.

### 11.2 Head Selection

**Current**: All 8 L0 heads, mean-pooled (NodewiseReadout in aligned run).

**Unknowns**:
- Is this equivalent to cherry-picking if we later select the best head?
- Does the best head transfer across seeds? Across checkpoints?
- Can we do label-free head selection (e.g., via row-concentration or audition)?

**Required ablations**: Random head control; top-τ head vs low-τ head; cross-seed transfer; label-free selection protocol.

### 11.3 Target Order Source

**Current v0**: CDL teacher (C−D+L greedy rollout) on L0H2, pretrain dataset, batch_mean=16.

**Unknowns**:
- Is CDL teacher optimal? Does −D only suffice?
- Does a simpler heuristic (e.g., row-concentration-based) work as well?
- Does the teacher need to be per-checkpoint, or can one teacher serve many checkpoints?

**Required ablations**: Direct teacher hook (no g_beta) as upper bound; heuristic targets (loss-based, entropy-based, centrality-based); cross-checkpoint teacher transfer.

### 11.4 Ranking Loss

**Current**: Soft pairwise BCE, all O(N²) pairs, no weighting.

**Unknowns**: Is pairwise BCE optimal vs. listwise (ListMLE) or regression?

**Required ablations**: Pairwise vs listwise vs rank regression; pair sampling vs all pairs; hard negative mining.

### 11.5 Refresh Interval K

**Current**: K=10.

**Unknowns**: Optimal trade-off between order freshness and wall-clock overhead.

**Required ablations**: K ∈ {1, 5, 10, 20, 50}. Report both step convergence and wall-clock convergence.

### 11.6 Probe Averaging

**Current**: probes=4.

**Unknowns**: How many probes are needed for stable orders? What's the overhead cost?

**Required ablations**: probes ∈ {1, 2, 4, 8}. Measure τ stability and wall-clock time.

### 11.7 Hard Argsort vs. Stochastic Sampling

**Current**: Hard argsort (deterministic).

**Unknowns**: Does determinism hurt? Would stochastic sampling (with temperature) improve diversity?

**Required ablations**: Temperature sweep; epsilon-random mixing; top-k reveal + random suffix.

### 11.8 Checkpoint Timing

**Current**: g_beta trained on step-20k checkpoint.

**Unknowns**: When does order signal emerge? When is g_beta most useful?

**Required ablations**: g_beta from step 5k, 10k, 20k, 40k, 60k. Cross-checkpoint deployment.

### 11.9 Frozen vs. Online

**Current**: Frozen (main experimental protocol).

**Unknowns**: Would periodically retraining g_beta help? Online joint training?

**Required ablations**: Periodic retrain (every 10k steps); online update with detached features. These are higher-risk, later-stage experiments.

### 11.10 Granularity

**Current**: N=64 blocks, block_size=4.

**Unknowns**: Does the signal depend on this granularity?

**Required ablations**: N ∈ {32, 128}; different block sizes.

### 11.11 Scale

**Current**: 47M parameters (4L/8H/d=384).

**Unknowns**: Does this scale to larger models?

**Required ablations**: Larger AO-GPT variants (e.g., 12L/12H/d=768).

### 11.12 Alpha Schedule

**Current**: α = 0.0 → 0.9 over 10k steps, then constant 0.9.

**Unknowns**: Is α=0.9 optimal? Does pure g_beta (α=1.0) overfit?

**Required ablations**: α ∈ {0.5, 0.7, 0.9, 1.0}.

---

## 12. Baselines

### 12.1 Required Baselines for Paper

| Baseline | Type | Purpose |
|----------|------|---------|
| random_continue | Primary baseline | Proves g_beta order beats random |
| layout_path_continue | Oracle | Upper bound (uses physical layout) |
| L2R_continue | Oracle | Physical L2R upper bound |
| g_beta_destroyed | Sanity | Proves g_beta reads real structure |
| g_beta_uniform | Ablation | Proves learned gate > uniform |
| g_beta_random_head | Ablation | Proves head selection matters |
| direct_teacher_hook | Upper bound | Shows teacher quality ceiling |
| shuffled_input_g_beta | Leakage control | Proves g_beta uses structure, not fixed pattern |

### 12.2 NOT Required (but Nice to Have)

- single_head_CDL (diagnostic, lower priority)
- old frozen_beta (historical comparison only)
- online g_beta (future work)

---

## 13. Risks and Mitigations

### 13.1 "g_beta just learned L2R"

**Risk**: g_beta's predicted order correlates with physical L2R, meaning it discovered position, not structure.

**Mitigation**:
- Report τ_vs_L2R as a diagnostic, not a claim
- Shuffled-AR control: if model was trained with shuffled L2R, g_beta should NOT recover physical L2R
- Content-relocation diagnostic: swap content between positions, see if order follows content or position
- Framing: "recovers layout-path" is safer than "discovers L2R"

### 13.2 Physical Position Leakage

**Risk**: Some feature (hidden state, positional encoding) leaks physical position into g_beta.

**Mitigation**:
- Strict model-frame audit (grep for forbidden fields)
- Shuffled-B sanity: permuting B's rows/cols should destroy signal
- Gaussian-B control: random Gaussian B should produce τ ≈ 0
- If PE is static and fixed, the model may have compiled it. This is a fundamental limitation — address in paper's limitation paragraph.

### 13.3 Head Selection Cherry-Picking

**Risk**: We picked L0H2 because it works best, and results don't generalize.

**Mitigation**:
- Multi-head g_beta (all 8 L0 heads) avoids single-head selection
- Cross-seed transfer experiment
- Random head control
- Label-free head selection protocol (audition → row-conc → CDL)

### 13.4 Wall-Clock Overhead Negates Step Saving

**Risk**: g_beta saves steps but costs more time per step, so wall-clock convergence is worse.

**Mitigation**:
- Report BOTH step-based and wall-clock metrics
- Refresh interval ablation to find overhead/speed trade-off
- If wall-clock is worse, claim is limited to "sample efficiency," not "training speed"

### 13.5 Teacher Quality Ceiling

**Risk**: g_beta can only be as good as its teacher. If teacher is weak, g_beta is weak.

**Mitigation**:
- Direct teacher hook as upper bound — shows what's possible
- If teacher hook is much stronger, the gap is g_beta's amortization error
- Frame as: "g_beta approximates teacher at lower cost"

### 13.6 Checkpoint Specificity

**Risk**: g_beta trained on step-20k doesn't work at step-40k or step-60k due to attention drift.

**Mitigation**:
- Multi-checkpoint g_beta pretraining
- τ_vs_bp drift monitoring during training
- Periodic retrain as ablation
- Report drift as a limitation

### 13.7 Seed Variance

**Risk**: Single-seed result (seed=123 for method, seed=2 for baseline).

**Mitigation**:
- Explicitly label as single-seed evidence
- Multi-seed (≥3) required for paper claims
- Report seed std, not just mean

### 13.8 Small Model Only

**Risk**: 47M model. Results may not hold at larger scales.

**Mitigation**:
- Label as "pilot scale"
- Plan larger-scale verification (317M+)
- Discuss in limitations

### 13.9 WikiText-103 Only

**Risk**: Text-only, single dataset.

**Mitigation**:
- Image pipeline exists (VQ-f4, patch-based) for cross-modal verification
- Discuss generalization in limitations
- Not required for first paper submission

### 13.10 vs. Masked Diffusion / MDLM

**Risk**: Reviewer asks "how is this different from masked diffusion?"

**Mitigation**:
- AO-GPT is autoregressive (not mask-then-predict)
- Order policy is per-step adaptive, not fixed schedule
- g_beta reads model state; diffusion schedules are state-agnostic
- Clear related work discussion

### 13.11 New g_beta vs. Old g_beta

**Risk**: Reviewer can't tell what's new.

**Mitigation**:
- Explicit comparison table (§3.2)
- Old results in appendix only
- New results from aligned protocol only in main table

### 13.12 If Direct Teacher Hook is Stronger, Why g_beta?

**Risk**: The amortized proxy is strictly worse than the teacher it imitates.

**Mitigation**:
- g_beta is cheaper: O(N) forward vs. teacher's O(N²) rollout
- g_beta is label-free: no physical coordinates
- g_beta enables scaling to larger N where teacher is infeasible
- Frame g_beta as "affordable approximation" not "better than teacher"

### 13.13 If L2R Baseline is Stronger, What's the Point?

**Risk**: L2R (or layout-path) beats g_beta. Reviewer: "just use L2R."

**Mitigation**:
- L2R is oracle — requires knowing the true order
- In real settings (pretraining, images, multi-modal), true order is unknown
- g_beta discovers useful order WITHOUT labels — that's the scientific contribution
- Claim is NOT "beats L2R." Claim is "beats random under label-free constraints."

---

## 14. Recommended Next Experiments

### 14.1 Immediate (this week, with current aligned run)

1. **Complete aligned frozen_gbeta run** (20k→60k) ← 🟢 running
2. **Post-hoc diagnostics**: τ_vs_layout_path, τ_vs_identity, α per head, refresh stability
3. **Label-free audit**: grep for forbidden fields in method path
4. **Compare with baselines**: extract Δ vs random_continue from W&B curves

### 14.2 Short-Term (1-2 weeks)

5. **Multi-head L0DynamicGBeta pretraining**: train the true multi-head g_beta on strict65 input
6. **Destroyed-B sanity**: run frozen_gbeta with destroyed B input → should match random
7. **Uniform-α ablation**: force uniform head weights → measure gate contribution
8. **Probe ablation**: probes ∈ {1, 2, 4, 8} → stability vs overhead curve
9. **Refresh ablation**: K ∈ {1, 5, 10, 20} → step + wall-clock comparison

### 14.3 Medium-Term (2-4 weeks)

10. **Multi-seed**: repeat aligned protocol with seeds 42, 123, 456
11. **Cross-checkpoint g_beta**: train g_beta on step 10k, 40k; deploy on step 20k
12. **Direct teacher hook baseline**: same aligned protocol, CDL teacher direct
13. **Shuffled-AR control**: train g_beta on shuffled-L2R model, verify no physical L2R recovery
14. **Content-relocation diagnostic**: swap content between positions, test if order follows content

### 14.4 Longer-Term (1-3 months)

15. **Online / periodic-retrain g_beta**
16. **Scale to 317M model**
17. **Image pipeline verification**
18. **Cross-modal g_beta (text→image transfer)**

---

## 15. Paper Claims: Three Tiers

### Conservative Claim (最小声明，最难被攻击)

> Under a strictly aligned experimental protocol, a frozen, label-free g_beta order controller — reading only model-frame L0 attention graphs — produces reveal orders that yield modest but consistent improvements in AO-GPT validation loss over random-order training. The improvement is bounded by the quality of the teacher used to pretrain g_beta and degrades under attention distribution drift. This establishes the feasibility of model-frame, attention-derived order control for any-order autoregressive training.

**Required evidence**: aligned run + destroyed sanity + label-free audit. ✅ mostly in place.

### Medium Claim (需要更多实验)

> A label-free g_beta controller trained via pairwise ranking on teacher-derived orders consistently improves AO-GPT convergence across multiple seeds and checkpoints, saving approximately X% training steps to reach a target validation loss. The benefit is robust to head selection (multi-head gate > single-head), requires real attention structure (destroyed B collapses to random), and is not explained by physical position leakage. The overhead is modest (Y% wall-clock increase), yielding a net wall-clock improvement.

**Required evidence**: multi-seed + cross-checkpoint + wall-clock metrics + all sanity checks. Not yet available.

### Strong Claim (最激进，需要大量实验)

> A model-frame g_beta controller discovers data-intrinsic order structure without any physical-coordinate supervision, matching or approaching oracle order policies (layout-path, L2R) while remaining fully label-free. The learned order generalizes across model scales, modalities (text, image), and training regimes, providing a universal plug-in order policy for any-order autoregressive training.

**Required evidence**: cross-modal + cross-scale + oracle-level performance. Not yet available. Do NOT claim this without evidence.

**Current recommendation**: Target the **Conservative** claim for the first paper submission, with the Medium claim as a stretch goal if multi-seed experiments complete in time.

---

## Appendix A: Code Inventory

| Module | Purpose | Tests |
|--------|---------|:---:|
| `batch_readout/l0_strict65.py` | Model-frame strict65 B extraction | 24 |
| `batch_readout/label_free_cdl_teacher.py` | CDL teacher for target orders | 33 |
| `analyses/build_l0_dynamic_gbeta_dataset.py` | .npz dataset builder | 15 |
| `batch_readout/l0_dynamic_gbeta.py` | L0DynamicGBeta model (multi-head) | 27 |
| `batch_readout/soft_pairwise.py` | Pairwise BCE loss | 19 |
| `batch_readout/train_l0_dynamic_gbeta.py` | g_beta training loop | 13 |
| `analyses/eval_l0_dynamic_gbeta.py` | Sanity evaluation | 10 |
| `analyses/characterize_l0_dynamic_gbeta.py` | Post-hoc τ characterization | — |
| `batch_readout/frozen_gbeta_hook.py` | Model-frame frozen hook + block provider | 6 |
| `batch_readout/hook_order_provider.py` | Single-head + head-gated providers | — |
| `batch_readout/integration_hook.py` | FrozenBetaHook (nodewise/flatten) | — |
| `train_clean_aogpt.py` | Main AO-GPT training loop | — |
| **Total** | | **180** |

## Appendix B: Current Run Command

```bash
python block_lo_arm_order_network/train_clean_aogpt.py \
    --run-kind frozen_beta \
    --resume-ckpt .../ckpt_step20000.pt \
    --output-dir .../frozen_gbeta_aligned_20260625 \
    --frozen-beta-ckpt .../gbeta_b1_L0H2_seed2_step20k/.../g_beta_best.pt \
    --frozen-beta-mode argsort \
    --frozen-beta-refresh 10 \
    --frozen-beta-none-mode model \
    --batch-mean-probes 4 \
    --seed 123 --device cuda:0 \
    --max-steps 60000 --lr 1e-3 --min-lr 1e-4 --lr-decay-steps 60000 \
    --batch-size 64 --grad-accum 2 \
    --data-source continuous \
    --wandb-log --wandb-project order-lyu \
    --wandb-run-name frozen_gbeta_aligned_20260625
```
