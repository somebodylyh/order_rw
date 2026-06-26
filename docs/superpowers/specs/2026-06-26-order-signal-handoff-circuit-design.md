# Order-Signal Handoff Circuit — Online Training-Dynamics Design

Date: 2026-06-26
Status: design (spec). Implementation plan to follow via writing-plans.

## Research Question

The order-bearing signal (attention that recovers physical L2R from a None/BOS
start) emerges on different layers/heads across seeds. We want to understand the
**training dynamics** of that emergence and disentangle:

1. **Time vs seed.** Within one seed, does the carrier *move* across layers as
   training proceeds (e.g. L0 early → L1 later)? Or does each seed *lock onto* a
   layer early and stay, so the across-seed spread is seed variance, not motion?
2. **Inheritance / handoff.** When the carrier appears to move, is the later-layer
   signal *inherited* from the earlier layer through a **cross-layer handoff
   circuit** (early layer writes an order representation into the residual stream;
   a later-layer head reads it via composition and refines it), or does the later
   layer emerge independently?
3. **Mechanism.** If a handoff exists, *how* does it form during training and
   *why* (what is written / read).

### What existing data already says (motivating, not conclusive)

Across existing 5k/10k random-baseline checkpoints (multiseed carrier scans):

- Strong-pass carrier heads are mostly L0/L1; head identity drifts across seeds.
- In *aggregate* the 5k→10k shift is L1→L0 (signal concentrating into L0), the
  opposite of a naive "L0→L1" story.
- The "L0→L1" narrative is currently driven mainly by one seed (seed123_new,
  L1-only at 5k → L0+L1 at 10k).
- This rests on only 2 steps and binary strong-pass counts — too coarse. We need
  a dense, continuous, in-training measurement.

## Scope Of This Spec

This spec covers the **online recording phase** (Pillars ① + ②): train from
scratch and record, in-training, both the per-(layer,head) order signal and the
candidate cross-layer handoff (composition) signal. Causal verification and the
"why" (Pillars ③ path patching, ④ training-dynamics inheritance, ⑤ representation
content) are **out of scope here** and get their own spec after the map exists.

### Run configuration (locked with user)

- **3 seeds**, each trained **from step 0 to 10,000**.
- `run_kind=baseline`, `data_source=continuous` (matches the clean random-baseline
  trajectories already analysed).
- **Record every 200 steps** (`--eval-interval 200` → 50 time points to 10k).
- Online recording of: per-(layer,head) order τ **and** L0×L1 composition scores.

## Five Pillars (full research arc; only ①② implemented now)

- **① Online flow tracking** — per-(layer,head) order τ vs step, in training, on a
  fixed eval set. *(this spec)*
- **② Handoff circuit identification** — weight-based K/Q/V composition between
  early-layer carrier heads (OV) and later-layer heads (QK/OV), tracked over
  training. *(this spec)*
- **③ Causal handoff verification** — path patching of the single
  `L_i_carrier → L_j_carrier` edge. *(later spec)*
- **④ Training-dynamics inheritance** — freeze/graft early carrier layer, test
  whether later carrier still emerges and whether its onset depends on the
  earlier one. *(later spec)*
- **⑤ Why** — what the early head writes (position/content/partial-order) and what
  the later head composes on top. *(later spec)*

## Existing Infrastructure

### `block_lo_arm_order_network/attention_trajectory.py`
- `AttentionTrajectoryLogger` with fixed eval samples (`_set_fixed_samples`,
  `_probe_orders`) so trajectories are comparable across seeds/steps.
- `log_snapshot()` is called from training at each eval step.
- **Limitation 1:** it extracts only `attn_list[0]` (L0). Must extend to all layers.
- **Limitation 2:** it currently computes locality/entropy summaries, **not order
  τ_vs_l2r**. The order signal must be added by wiring in
  `none_separated_block_graph`.

### `block_lo_arm_order_network/train_clean_aogpt.py`
- Flags: `--attn-trajectory`, `--attn-trajectory-samples` (default 8),
  `--attn-trajectory-heatmap-interval`, `--eval-interval` (default 1000), `--seed`.
- `run_eval_and_save(global_step, ...)` runs at every `--eval-interval` step
  (line ~2012) and already calls the trajectory snapshot.

### Model (`model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm.py`)
- `CausalSelfAttention`: `c_attn` (n_embd → 3·n_embd, split into q,k,v),
  `c_proj` (n_embd → n_embd); per-head `q_norm`/`k_norm` (RMSNorm); `Block` applies
  AdaLN modulation and a residual gate `x = x + gate_msa * attn_y`.
- Per-head weight slices are recoverable: `W_Q,W_K,W_V` from thirds of
  `c_attn.weight`, `W_O` from the head-column block of `c_proj.weight`.

## Design

### A. All-layer order-signal extraction (Pillar ①)

In `log_snapshot`, replace the L0-only extraction with a loop over all layers.
For each layer ℓ:
- `attn_l = attn_list[ℓ].cpu().numpy()` → `build_model_frame_strict65(attn_l, probe_orders)`
  → batch-mean B per head (reuse `_batch_mean_B`).
- Per head h and method m ∈ {`C-D+L`, `L`}: `build_none_separated_B` →
  `rollout_by_method` → `discovery_metrics`, record **signed `tau_vs_l2r`**,
  `phys0_rank`, `prefix8_overlap`, gate status, plus the existing locality summary.

Output a per-snapshot record keyed by `(layer, head, method)` with the signed τ
and gate fields. This is the core flow signal — kept at head granularity (no
premature best-head collapse); best-head-per-layer is a derived view.

### B. Composition score over training (Pillar ②)

New pure module `attn_composition.py` computing weight-based composition between
an upstream head (layer i, head a) and a downstream head (layer j>i, head b):
- `W_OV(i,a) = W_O(i,a) @ W_V(i,a)` (writes into residual).
- Downstream reads: `W_QK(j,b) = W_Q(j,b)^T W_K(j,b)`; also `W_OV(j,b)`.
- **Q-composition** `‖W_QK(j,b)^T · W_OV(i,a)‖_F / (‖W_QK(j,b)‖_F ‖W_OV(i,a)‖_F)`,
  and analogously **K-composition** and **V-composition** (Transformer-Circuits
  definitions).
- Record the full upstream×downstream composition matrix for adjacent layer pairs
  (minimum L0→L1; extensible to all i<j), each snapshot.

**Approximation caveat (documented, not hidden):** per-head RMSNorm on q/k and
AdaLN modulation mean these are *approximate* composition scores, not exact under
this architecture. They are a screen for candidate handoff edges; the causal
ground truth is path patching in Pillar ③. The spec deliberately pairs the
weight-based score with that later causal cross-check rather than over-claiming.

### C. Training run

- Add/confirm a launcher that runs `train_clean_aogpt` with
  `run_kind=baseline data_source=continuous --attn-trajectory --eval-interval 200
  --max-steps 10000` for 3 seeds.
- **Seed-label integrity:** record the true `--seed` and read it back from the
  saved ckpt; never trust directory names (existing trajectories had ambiguous
  seed2/seed123 labels).

### Outputs

Per seed:
- `tau[layer, head, method, step]` (signed τ + gate fields), JSONL/npz.
- `composition[layer_pair][up_head, down_head, {Q,K,V}, step]`, npz.
- Flow heatmaps (`layer×step` best-head τ) and per-head τ-vs-step curves.
- Composition-vs-step curves for top candidate edges.

## Hypotheses & Decision Criteria

- **H_temporal (carrier moves):** within a seed, early-layer head τ rises then
  plateaus/decays while a later-layer head τ rises later — an intra-seed crossover.
- **H_seed (carrier locked):** each seed's carrier layer is fixed early and stable;
  across-seed spread is variance, not motion.
- **Handoff signature:** a later-layer τ rise is preceded/accompanied by a rising
  composition score on a specific upstream→downstream edge, with the upstream head
  τ leading in time. This nominates the edge for Pillar ③ causal testing.

## Risks

- **Composition approximation** under RMSNorm/AdaLN (above) — mitigated by Pillar ③.
- **Overhead:** all-layer strict65 + composition every 200 steps. Keep
  `--attn-trajectory-samples` modest (8) and confirm per-snapshot wall-clock is a
  small fraction of a 200-step interval before committing all 3 seeds.
- **Random-order nondeterminism:** the random-baseline objective is degenerate and
  same-config runs diverge (compile/no-compile ablation). 3 seeds is a floor, not a
  guarantee of a clean cross-seed law; report per-seed first, aggregate second.
- **Seed-label integrity** (above).

## Testing (TDD)

Pure functions are unit-tested with synthetic inputs before any training:
- all-layer extraction returns one B per (layer, head) with correct shapes;
- per-(layer,head) τ matches `search_none_separated_65_heads` on a fixed ckpt for
  at least L0 (regression anchor against existing reports);
- composition score: known synthetic W matrices give hand-computed Q/K/V scores;
  permutation/scaling invariances hold.

## Out Of Scope

Pillars ③④⑤ (path patching, freeze/graft inheritance, representation-content
probing); image modality; beating L2R; any g_beta/hook training. Those follow once
the online emergence + handoff map exists.
