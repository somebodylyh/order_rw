# Hidden Graph Diagnostic — Design

- **Date**: 2026-05-26
- **Status**: design approved, ready for implementation plan
- **Positioning**: **frozen diagnostic, NO training.** Decides whether to later train a
  two-graph order policy. A null result closes the hidden-graph route *for that substrate*;
  it is not a claim that the method is impossible at other scales.
- **Supersedes the framing of**: `2026-05-25-hidden-residual-order-diagnostic-design.md`
  (hidden-as-residual-scalar). This line reframes hidden state as a **relational graph**,
  not a per-candidate residual regression target.

## 1. Motivation & framing

The attention-order MLP policy currently reads a **single** graph — the attention graph
`B_A = Aᵀ` (one global matrix). It is sample-invariant:

```
s(v) = f_A( φ(B_A, v, S_t, U_t, last) )      # current: attention-only
```

The hidden-residual line asked hidden state to predict a per-sample attention-order
residual scalar `r_x(v) = s_x(v) − s_G(v)`. That is a supervised scalar regression and is
easy to come out negative, because hidden state need not encode information as a per-candidate
residual. Hidden state is a vector space; it more naturally encodes **relations** (which
blocks are representationally similar / co-segment / are near in context). Relations are a
**graph**, not a scalar.

So we reframe: let hidden state induce its own block-level graph and let the order policy
read **two graphs**.

- `B_A = Aᵀ` — attention dependency / global order prior (the established, working signal).
- `B_H = sim(H, H)` — hidden representation / context-similarity graph (the new signal).

**Endpoint architecture (out of scope for this spec — gated by this diagnostic):**

```
two-graph MLP policy:   s(v) = f_A( φ(B_A, v) ) + γ · f_H( φ(B_H_resid, v) )
```

i.e. attention is the main graph, the (position-residualized) hidden graph is a
context-dependent auxiliary branch. We do **not** train this here. This spec only decides
whether `B_H_resid` is (a) non-artifact and (b) complementary to `B_A` enough to be worth
training that policy.

## 2. Scope & non-goals

**In scope (frozen):** build `B_A / B_H / B_pos`, position control & residualization, graph
structure metrics, C-D+L readout orders, graph-level and score-level mixes, and frozen
teacher-forced NLL-under-order of every generated order.

**Out of scope / deferred to v2 (do NOT build):**
- training any policy (the two-graph MLP endpoint);
- learned hidden graphs: bilinear `hᵀWh`, query-key `q·k`, state-conditioned `g(h_{S_t},v)`;
- Graph-RW readout (C-D+L `rollout_order` is the only readout here);
- short continuation / fine-tune smoke;
- confidence / readiness signals; hidden-residual scalar regression (closed).

## 3. Substrate & checkpoints

Run **both** modalities. Text is the main line; image is the contrast (text `B_A` has
collapsed to ~L2R — reconfirmed live: alt α=1.0 run teacher τ_vs_L2R≈0.97 @10k — so on text
"complementary to `B_A`" ≈ "≠ L2R", whereas image `B_A` carries real locality structure).

| modality | ckpt (frozen) | `B_A` source | N blocks |
|---|---|---|---|
| **text — primary** | `block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step30000.pt` | `clean_base_random_perm/A_global_eval.npy` → `build_directed_graph` | per meta (`block_order_block_len`) |
| **text — contrast** | `probe_results/attention_order_mlp/alt_from0_mlp_finetune/ckpt_step30000.pt` | its `A_global*` | same |
| **image — primary** | `probe_results_image/vq64_alt_from0_mlp_patch2x2_l8h8e512/ckpt_step30000.pt` | `…/A_global_step30000.npy` | 64 |
| **image — contrast** | `probe_results_image/vq64_fixed_random_l8h8e512/ckpt_step30000.pt` (produced by the running fixed baseline) | extract via `extract_image_attention*` | 64 |

Two ckpts per modality (random-trained vs order-curriculum-trained) answer whether `B_H`
structure depends on the training curriculum.

**Execution staging** (the image contrast ckpt is still training — do not let it block):
- **Stage A** (all ckpts ready now): text primary `clean_base_random_perm`, text contrast
  `alt_from0_mlp_finetune`, image primary `vq64_alt_…_l8h8e512`.
- **Stage B** (after `vq64_fixed_random_l8h8e512@30k` finishes): image contrast `fixed_random`.

The diagnostic and its verdict are computed per-ckpt, so Stage A produces complete results
on its own; Stage B is appended when the ckpt lands.

## 4. The three graphs (N×N, diag = 0)

- **`B_A`** = `directed_graph_policy.build_directed_graph(A_global)` = `Aᵀ` with diagonal
  zeroed (directed). Existing.
- **`B_H`** = `cos(h_u, h_v)` (symmetric). `H` from
  `hidden_residual_hidden.extract_oracle_hidden(model, …, mode="original")` →
  `(n, N, E)` per-sample block hidden in **physical** frame (block hidden = mean over the
  block's token hiddens, full context). Compute **per-sample** `B_H ∈ (n,N,N)` and the
  **global mean** `B̄_H ∈ (N,N)`. Keep raw. (Stage-1 = oracle; Stage-2 causal in §7.)
- **`B_pos`** (position/distance baseline):
  - text: block-index graph `B_pos(u,v) = exp(−|u−v| / τ_pos)`;
  - image: Manhattan graph `B_pos(u,v) = exp(−d_manh(u,v) / τ_pos)` (distances reuse the
    coverage infra's Manhattan table).
  - `τ_pos` fixed in the plan; diag = 0.

## 5. Normalization & residualization (locked)

Two distinct normalization uses — keep them separate:

1. **For corr / residualization / mixing** — off-diagonal **z-score**:
   `B_z = (B − mean_offdiag(B)) / std_offdiag(B)`, diag = 0.
2. **Position residualization** — linear regression of `B_H` off-diagonal entries on `B_pos`
   off-diagonal entries (both z-scored): `B_H_resid = B_H_z − fit(B_pos_z)`. Computed on the
   global mean graph for the headline; per-sample residualization optional/secondary.
3. **For C-D+L readout input** — C-D+L coverage semantics assume non-negative edge mass, but
   z-scored / residualized graphs have negative edges. Apply **shift-to-nonnegative** (NOT
   clip — clipping discards negative-edge ordering):
   `B_readout = B_z − min_offdiag(B_z)`, diag = 0.
   Additionally `rollout_order(..., standardize=True)` is available to make `tau_T`
   comparable across graphs of different magnitude (per-step score z-score).

## 6. Diagnostics

Compute for each graph in `{B_A, B_H_raw, B_H_resid, B_pos}` unless noted.

**(a) Structure:** sharpness (peak/mean, row entropy, top-k mass), top1/top4 edge identity,
spectral gap / cluster modularity (if cheap).

**(b) Correlations** (off-diag, Pearson **and** Spearman): `corr(B_H, B_pos)`,
`corr(B_H, B_A)`, `corr(B_A, B_pos)`, and `corr(B_H_resid, B_A)`,
`corr(B_H_resid, B_pos)` (the latter should be ≈0 by construction — sanity check).

**(c) Readout orders** via `attn_order_teacher.rollout_order` (C-D+L, greedy + sampled),
on the shift-to-nonnegative graph. Order metrics: τ_vs_L2R, step-entropy, unique orders
(over samples / seeds), displacement, top4-follow / locality, mean C-D+L score.

**(d) Graph-level mix** — `B_mix = λ·B_A_z + (1−λ)·B_H_z`, λ∈{0,0.25,0.5,0.75,1}; readout
order + structure per λ. (Answers: can the two graphs fuse into one structured graph?)

**(e) Score-level two-branch mix** (the load-bearing one — frozen proxy of the endpoint
policy):
```
s(v) = s_CDL(B_A, v) + γ · s_CDL(B_H_resid, v),   γ ∈ {0, 0.25, 0.5, 1, 2}
```
`γ=0` is the A-only control. Generate the order by sequential argmax/sample on the summed
per-step scores (compute `s_CDL` on the shift-to-nonnegative graphs at each step). The two
branches' per-step scores are on different scales, so **z-score each branch's per-step score
vector before the γ-weighted sum** (otherwise γ is not interpretable); this is the per-step
standardize already in `rollout_order(standardize=True)`.

**(f) Frozen NLL-under-order** — for every generated order
(`order_A`, `order_H_raw`, `order_H_resid`, `order_mix(λ)`, `order_scoremix(γ)`), compute the
frozen model's teacher-forced token-avg NLL on a fixed val subset, reusing
`train_clean_aogpt.py::evaluate_orders` / `eval_model_orders` (text) and the image
`evaluate_*orders` analog. **Boundary (write verbatim in code/docs):**
> Frozen NLL-under-order is **diagnostic only, not a training target**. It ranks complete
> generated orders (one forward per order), the same metric family as
> `val_random / val_raster / val_model_order`. It is **not** a per-candidate NLL teacher and
> does not supervise residuals.
> **Checkpoint-local:** each ckpt is evaluated with orders generated from *its own*
> `B_A / B_H / B_pos`. Never score one ckpt's order with another ckpt's model unless the row
> is explicitly labelled a cross-check (order and model state must match or NLL is meaningless).

## 7. Stage gating (oracle-first → causal)

**Stage 1 — oracle** (`mode="original"`): run all of §6. If, for a modality, `B_H_resid`
shows **no** structure (≈ random after position removal) → **stop that modality**: report the
strong null ("even mature full-context hidden graph carries no position-independent order
structure"); do not run causal.

**Stage 2 — causal** (gated; only if Stage-1 oracle `B_H_resid` has structure):
`extract_causal_hidden(mode="predictor")` at `t ∈ {0,16,32,48}`. Compare causal vs oracle:
`corr(B_H_causal, B_H_oracle)` (raw + resid), `order_H_causal vs order_H_oracle`. Answers:
how much of the oracle upper bound survives at decision time.

## 8. Decision criteria (anti-fooling)

**"Structure" / "non-random" is judged against a null**: shuffle each graph's off-diagonal
entries (row-wise permutation preserving the value multiset) to build a matched-random
baseline, and require the metric (sharpness / spectral gap / readout-order non-triviality) to
exceed the shuffled null by a margin reported explicitly. **"Better / no-worse" NLL is judged
against a noise floor**: estimate the NLL spread from ≥5 rollout seeds (for sampled orders)
and treat differences within that spread as ties (cf. prior single-seed noise-floor pitfalls).

"Positive" (bucket #4 → consider training the two-graph policy) requires **all** of:
1. `B_H_resid` is **non-artifact**: structurally non-random (vs shuffled null) AND
   `corr(B_H_resid, B_pos)≈0`;
2. **not a `B_A` clone**: `corr(B_H_resid, B_A)` not near 1;
3. **score-level A+H_resid lowers frozen NLL vs A-only** (`γ=0`);
4. **γ-sweep stability**: the improvement is not a single-point fluke — at least two adjacent
   γ in {0.25,0.5,1} are **no worse** than A-only and at least one is **clearly better**
   (a lone `γ=2` blip does not count);
5. **control guard**: A+H_resid beats a matched control where `B_H_resid` is replaced by a
   matched-random residual graph or by `B_pos` itself — i.e. the gain is specific to the
   hidden graph, not to "adding any second signal". The matched-random residual is built by
   **row-wise / off-diagonal permutation of `B_H_resid`** (after residualization, before
   readout preprocessing), preserving the off-diagonal value multiset and diag = 0, then run
   through the **same** shift-to-nonnegative + per-step standardization as `B_H_resid`. This
   controls for "a second signal of equal strength", not a differently-scaled noise matrix.

Interpretation buckets (per modality):
- raw `B_H` already structureless → route closed (strong null);
- raw structured, `B_H_resid` empty → position artifact → closed;
- `B_H_resid` structured but ≈ `B_A` → redundant, no new info;
- `B_H_resid` structured, complementary, passes §8.1–8.5 → **win** → Stage-2 causal, then
  later two-graph policy training (separate spec).

## 9. Code units & reuse

New, each independently testable:
- `hidden_graph.py` — `H → B_H` (cosine; per-sample + global mean).
- `position_graph.py` — `B_pos` for text / image.
- `graph_normalize.py` — off-diag z-score; shift-to-nonnegative; `B_pos`-residualize; corr utils.
- `graph_structure_metrics.py` — sharpness / entropy / top-k / spectral.
- `hidden_graph_diagnostic.py` (driver) — load frozen ckpt + data → build `B_A/B_H/B_pos` →
  normalize/residualize → corr + structure → C-D+L readout orders → graph-mix(λ) + score-mix(γ)
  → frozen NLL-under-order → write report (JSON + MD).

Reuse (do not reimplement): `extract_oracle_hidden` / `extract_causal_hidden`
(`hidden_residual_hidden.py`), `build_directed_graph` (`directed_graph_policy.py`),
`teacher_scores` / `teacher_step` / `rollout_order` (`attn_order_teacher.py`),
`evaluate_orders` / `eval_model_orders` (`train_clean_aogpt.py`, text) and the image
evaluate-under-order analog.

Unit tests for the pure functions: cosine graph correctness, off-diag z-score & shift-to-
nonnegative (order-preserving), `B_pos`-residualization (resid ⟂ `B_pos`), `B_pos`
construction, mix scale-equality.

## 10. Why this is the right next step

It replaces a brittle supervised scalar regression (`h → r_x`) with a falsifiable structural
question (`H → B_H → order`, controlled for position). It produces a clean per-modality
verdict and, if positive, a concrete next artifact (the two-graph MLP policy). It also aligns
with the latent-structure framing of DiLaDiff/LoMDM: a hidden-state relational graph as the
context-dependent substrate, combined with an explicit attention-derived order prior.
