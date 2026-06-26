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
  opposite of a naive "L0→L1" story — **but this is based on binary strong-pass
  counts over sparse (2-step) checkpoints and must not be treated as the working
  hypothesis.**
- The "L0→L1" narrative is currently driven mainly by one seed (seed123_new,
  L1-only at 5k → L0+L1 at 10k).

**The primary question is therefore kept open-ended:** *does the carrier location
move within a seed across training, or is it seed-locked from early on?* The dense,
continuous, in-training measurement below is designed to answer that without
presupposing a direction.

## Scope Of This Spec

This spec covers the **online recording phase** (Pillars ① + ②): train from
scratch and record, in-training, both the per-(layer,head) order signal and the
candidate cross-layer handoff (composition) signal. Causal verification and the
"why" (Pillars ③ path patching, ④ training-dynamics inheritance, ⑤ representation
content) are **out of scope here** and get their own spec after the map exists.

### Locked configuration

```
seeds              = [2, 42, 123]          # case-study seeds (see caveat)
max_steps          = 10000
record_steps       = [0, 200, 400, ..., 10000]   # 51 points incl. step 0
eval_interval      = 200
data_source        = continuous
run_kind           = baseline
order_policy       = random                # uniform random reveal order
attn_trajectory    = true
attn_traj_samples  = 8
bs_mean            = 16 (final)            # fallback: dense 8 + 1000-step anchor 16
composition_pairs  = all i<j by default; fallback adjacent-only under overhead
composition_types  = Q, K, V
layers             = 0,1,2,3
heads              = 0..7
methods            = C-D+L, L
ckpt_save_steps    = [0, 1000, 2000, ..., 10000]  # needed for Pillars ③④⑤
overhead_budget    = target ≤5%, hard cap ≤10% of training wall-clock
```

**Seed caveat (important):** these are *from-scratch* training seeds (values 2,
42, 123), **not** the historical checkpoints that carried those lineage labels.
Same seed *value*, a new trajectory under current code/data — there is no
guarantee they reproduce the specific historical seed2/seed42/seed123 phenomena.
We use them to map *representative trajectory modes* as per-seed case studies, not
to estimate population-level statistics. Read the true `--seed` back from each
saved ckpt; never trust directory names.

**Record at step 0** before any optimizer update (init-artifact control), then
every 200 steps through 10k. Checkpoints are saved at least every 1000 steps so
later pillars (path patching, freeze/graft, probing) have the trajectory to work
on; trajectory summaries alone are insufficient for ③④⑤.

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
  → batch-mean B per head over **`bs_mean` probes** (reuse `_batch_mean_B`).
  `bs_mean=16` for the final map; under overhead use dense `bs_mean=8` with a
  `bs_mean=16` anchor every 1000 steps. **Never default to `bs_mean=4`** (it
  decays/misjudges the old L0 signal).
- Per head h and method m ∈ {`C-D+L`, `L`}: `build_none_separated_B` →
  `rollout_by_method` → `discovery_metrics`, record **signed `tau_vs_l2r`**,
  `phys0_rank`, `prefix8_overlap`, gate status, plus the existing locality summary.

Output is structured in three levels to avoid selected-head bias:

- **Raw:** `tau[layer, head, method, step]` (signed) + gate fields, full tensor.
  No premature best-head collapse.
- **Derived:** `best_head_per_layer`, `best_head_global`,
  `strong_pass_count_per_layer`, `signed_consensus_per_layer` — all per
  `(seed, step, method)`.
- **Visualization:** `layer×step` heatmaps of **both `max_abs_tau` and
  `max_signed_tau`** (keep both: anti-L2R heads reach τ=−1.0; signed-best alone
  would miss strong reversed structure, abs alone would conflate forward/reverse),
  plus `strong_pass_count` and per-head τ-vs-step curves.

### B. Candidate composition score over training (Pillar ②)

New pure module `attn_composition.py` computing weight-based **candidate
composition** (a.k.a. composition compatibility) between an upstream head
(layer i, head a) and a downstream head (layer j>i, head b).

**Shape contract (activation-space, to avoid transpose errors).** PyTorch
`Linear.weight` is `[out, in]`; convert every head's slices into this convention
before computing anything:

```
x_resid : [d_model]
q = x_resid @ W_Q     # [d_head]      W_Q, W_K, W_V : [d_model, d_head]
k = x_resid @ W_K     # [d_head]      W_O           : [d_head, d_model]
v = x_resid @ W_V     # [d_head]
head_out = v @ W_O    # [d_model]
W_OV = W_V @ W_O      # [d_model, d_model]   (upstream write)
W_QK = W_Q @ W_K.T    # [d_model, d_model]   (downstream read)
```

Scores (Transformer-Circuits definitions), each in [0,1] via Frobenius norms:
- **Q-composition** `‖W_QK(j,b) · W_OV(i,a)‖_F / (‖W_QK(j,b)‖_F · ‖W_OV(i,a)‖_F)`
- **K-composition** `‖W_QK(j,b).T · W_OV(i,a)‖_F / (...)`
- **V-composition** `‖W_OV(j,b) · W_OV(i,a)‖_F / (...)`

Record the full upstream×downstream composition matrix per `{Q,K,V}` for **all
i<j layer pairs by default** (4 layers → 6 pairs × 8×8 head pairs × 3 scores;
cheap relative to attention extraction). **Fallback under overhead: adjacent pairs
only** (`L0→L1, L1→L2, L2→L3`) — but `L1→L2` must stay in even the fallback,
because prior scans found L2 carriers in the old lineage. Composition is the
*cheapest* part of a snapshot; cut it last.

**Naming + caveat (documented, not hidden):** these are *candidate* composition
scores, **not** causal handoff evidence. Per-head RMSNorm on q/k and AdaLN
modulation make them *approximate* under this architecture. A high score only
nominates an upstream→downstream edge whose temporal alignment with τ dynamics is
to be tested by path patching in Pillar ③ — the weight-based score is
deliberately paired with that later causal cross-check rather than over-claimed.

### C. Training run

- Launcher runs `train_clean_aogpt` per seed with `run_kind=baseline
  order_policy=random data_source=continuous --attn-trajectory --eval-interval 200
  --max-steps 10000`, saving ckpts every 1000 steps. (Confirm the exact flag names
  for `order_policy`/save-steps in the plan; the intent is uniform-random reveal
  order, not any structured policy.)
- **Overhead calibration gate (before launching all 3 seeds):** run seed 2 to 400
  steps, record snapshot wall-clock at step 200 and 400, extrapolate to 10k. If
  snapshot overhead > **10%** (hard cap; target ≤ **5%**), degrade in this order:
  (1) lower heatmap/PNG interval; (2) save raw B only every 1000 steps (keep τ
  summary every 200); (3) `bs_mean` dense 16→8 (keep 1000-step anchor at 16);
  (4) `attn_traj_samples` 8→4. **Do not cut composition** — it is the cheapest
  part. The expensive part is all-layer strict65 + none-separated rollout.
- **Seed-label integrity:** record the true `--seed` and read it back from the
  saved ckpt; never trust directory names (existing trajectories had ambiguous
  seed2/seed123 labels).

### Outputs

Per seed:
- `tau[layer, head, method, step]` (signed τ + gate fields), JSONL/npz — raw +
  derived views (§A).
- `composition[layer_pair][up_head, down_head, {Q,K,V}, step]`, npz — all i<j.
- Flow heatmaps (`layer×step` max_abs_tau **and** max_signed_tau) + per-head
  τ-vs-step curves; composition-vs-step curves for top candidate edges.
- Checkpoints at `[0, 1000, ..., 10000]` retained for Pillars ③④⑤.

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
- **Overhead:** all-layer strict65 + none-separated rollout every 200 steps is the
  expensive part (composition is cheap). Gated by the calibration step in §C
  (target ≤5%, hard cap ≤10%, with an explicit degradation ladder).
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
