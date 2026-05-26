# Causal-Hidden Order Probe — Design Spec

**Date:** 2026-05-26
**Status:** Design approved (brainstorming), pending writing-plans.
**Type:** Frozen, falsifiable diagnostic. **No training.**
**Branch:** attn-order-alternating. **Python:** `/home/admin/anaconda3/envs/X1/bin/python` (torch 2.4.1, numpy 1.22.3, pytest 8.3.5).

---

## §0 Context & relation to prior lines

Two static-hidden routes are already CLEAN NEGATIVE:
1. **hidden as scalar residual predictor** (`s = s_G + Δs_H(v,x)`) → null (`hidden_residual_diag_line`).
2. **hidden as static symmetric cosine graph** (`B_H = cos(h_u,h_v)`) → null across text primary/contrast + image primary (`hidden_graph_diag_line`, Stage-A).

Both probe **static, parameter-free hidden geometry**. The decisive failures: `B_H` is dominated by
position (corr 0.62–0.76 with `B_pos`); residualized `B_H_resid` carries no structure above a
shuffled null. Two scoping facts emerged: (a) **cosine is symmetric**, but order needs direction —
the C-D+L readout injects all direction, the edge weights carry none; (b) **a single-step attention
logit `q(h_i)·k(h_j)` IS the model's own asymmetric hidden transition**, and `B_A` is its collapsed
(mean-over-samples/heads/layers) form.

**This probe tests the one open hidden direction:** does **per-step CAUSAL hidden** — the model's
evolving predictor state conditioned on the revealed prefix `S_t` — carry order signal **beyond what
the static `B_A` C-D+L order already captures**? This is *not* static hidden geometry; it is the
dynamic, context-dependent signal the two negatives did not touch.

**Outcome routing (decided by Level-1 only; see §10):**
- **Case A (WIN):** per-step causal hidden stably beats A-only → justifies designing a *trained*
  context-dependent controller `s(v) = f(φ(B_A, v, S_t), ψ(H_θ(x, S_t), v))` (out of scope here).
- **Case B (NULL):** `B_A` already absorbs the order signal at this scale → stop hidden; pivot to
  attention ablation / α=1 / scale-up / image substrate.
- **Case C (shared NULL but per-sample signal):** signal is sample-specific, not compressible into a
  single shared curriculum order → future work is a *per-sample* controller, not a shared order.

---

## §1 Question & scope

> On a frozen model, can an order built from **per-step causal hidden** (dynamic, context-dependent,
> parameter-free) achieve lower **frozen teacher-forced NLL-under-order** than the static `B_A` C-D+L
> order — stably, beating position and shuffled-hidden controls, without degenerating to a trivial
> L2R/raster position order?

**Frozen. No training. No new learned parameters in Level 1.** All "scores" are param-free functions
of the frozen model's own activations/weights plus the attention graph `B_A`. The probe reuses the
Stage-A machinery (`graph_normalize`, `graph_order` score-mix shape, `graph_structure_metrics`,
`hidden_graph_modelio` NLL, the `run_hidden_graph_diag` report/verdict pattern).

---

## §2 Substrates & frozen checkpoints (reuse Stage-A's exact ckpts)

| role | modality | ckpt |
|---|---|---|
| primary | text | `block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step30000.pt` (+ `A_global_eval.npy`) |
| contrast (optional) | text | `probe_results/attention_order_mlp/alt_from0_mlp_finetune/ckpt_step30000.pt` |
| primary | image | `probe_results_image/vq64_alt_from0_mlp_patch2x2_l8h8e512/{ckpt_step30000.pt, A_global_step30000.npy}` |

Conventions: `N_BLOCKS=64`. text `block_len = int(model.block_order_block_len)` (=4); image
`block_len = int(meta["block_order_block_len"])` (patch2x2 → 64 patches, seq 256; do **not** hardcode
— read it from meta, exactly as the Stage-A image run did). Graphs/orders scored in PHYSICAL block
frame (`PHYS_FRAME`). **Text first** (smoke + main), then image.

---

## §3 Core mechanics — shared greedy rollout + per-step predictor hidden

- **One shared greedy order** (not per-sample), to align with static `B_A` C-D+L (also one order) and
  with the NLL-under-order eval (one order applied to all eval samples). Hidden is aggregated (mean)
  over `n_roll` rollout samples to score candidates.
- **One forward per step.** At step `t`, prefix `S_t` (length `t`) is fixed. Run
  `forward_fn(idx, token_order=[S_t + arbitrary completion of U_t], return_hidden=True,
  hidden_return_mode="predictor")`; read the predictor hidden at reveal-position `t`,
  `c_t = out[2][:, t, :]` → `(n_roll, E)`. By the §4 PRE-GATE, `c_t` depends only on `S_t`, so the
  completion is irrelevant. Aggregate over `n_roll`: `c_t_bar = mean_n(c_t)`. Total: N=64 forwards per
  rolled order.
- **The §4 PRE-GATE is a precondition of this mechanic.** If predictor hidden is not
  completion-invariant, "one forward + arbitrary completion" is invalid and the probe aborts.
- Greedy is deterministic given the score; no seed needed for the order itself. Sampling-based seeds
  are only used for the per-sample audit (§7) and the shuffled-hidden control (§6).

---

## §4 PRE-GATE — causal invariance of predictor hidden (HARD; run first)

**Claim under test:** `c_t` (predictor hidden at reveal-position `t`) depends only on `S_t`, not on
how positions `t+1..N` are filled.

**Procedure:**
- `model.eval()`, dropout off, **fp32** (isolate architectural leakage from bf16 rounding).
- For `t ∈ {0, 1, 4, 16, 32, 48, 63}`, and for **3–5 distinct `S_t` patterns** per `t` (e.g. L2R
  prefix, random prefix, reverse prefix), build ≥2 completions of the remaining `U_t`
  (L2R / random / reverse). Run the forward for each completion, read `c_t`.
- Compare across completions: **primary criterion `cosine(c_t^a, c_t^b) > 0.99999`**; also report
  `max_abs_diff` (expected near bit-identical under correct causal masking).
- Run for **both text and image** primary ckpts.

**Gate:** if any (t, S_t) fails the cosine criterion → **ABORT** the probe, write
`PREGATE_FAILED.md` ("future completion leaks into predictor hidden; dynamic-hidden probe invalid"),
do not run Levels 0–2. This is a non-negotiable correctness gate.

---

## §5 Dynamic scores (param-free) + poolings + candidate embedding

For candidate `v ∈ U_t` at step `t`, with context vector `p_t` (§5.3):

### §5.1 cos-emb (REQUIRED, primary)
```
s_H^cos(v) = cos( p_t , e_v )
```
geometric similarity between the current context and the candidate's content embedding.

### §5.2 model-attn (SECONDARY — only if a clean hook exists)
```
s_H^attn(v) = aggregate_{ℓ,h∈top4var} [ q_{ℓ,h}(p_t)^⊤ k_{ℓ,h}(e_v) / sqrt(d_h) ]
```
the model's own learned query–key transition (un-collapsed `B_A`), using the model's own
`W_q, W_k`. **Aggregation rules (to keep the口径 interpretable):**
- aggregate **pre-softmax logits** (our downstream pipeline is softmax-free: z-score + γ-mix), over
  the **same top-4 variance heads** used to build `A_global` in `extract_A_matrices`;
- **also** record a post-softmax-matched variant for comparison (note: `A_global` is built from
  post-softmax attention averaged over top-4 heads, so the logit and post-softmax variants will not
  be identical — this is expected and documented);
- save **per-head** scores as diagnostics before aggregation.
- If exposing `W_q/W_k` requires heavy intrusion or the layer/head semantics are unclear, **do not
  force it**; cos-emb alone answers the core question. (Model has a `force_manual_attention` flag in
  `nanogpt-learned-order/AOGPT.py` that should make q·k recomputation tractable.)

### §5.3 Pooling of context (`p_t`)
- **summary mode `p_t = c_t` (REQUIRED, primary):** the predictor hidden at step `t` — the model's
  own summary of `S_t`. Interface-native (one forward, §3).
- **last mode `p_t = h_last`** and **mean-revealed mode `p_t = mean_{u∈S_t} h_u`**: add **only if the
  interface cleanly provides them** (these need revealed-block original-mode hiddens, which is closer
  to the static-oracle extraction we are avoiding). Do NOT mislabel `c_t` as `mean(h_{S_t})`.

### §5.4 Candidate embedding `e_v` (write it down; watch position leakage)
- **Version A — content-aware (Phase 1, run first):** `e_v = mean token embedding of candidate
  block` = mean over the block's `block_len` input **token embeddings** (`wte`), **token-only**, no
  positional embedding. If the implementation naturally adds positional embedding, that is a separate
  **`token+pos`** variant and must be **explicitly flagged** (positional leakage would make `s_H^cos`
  a disguised `B_pos`).
- **Version B — content-free (only if Version A WINS):** `e_v = block-id / positional / learned block
  query` (no candidate content). Answers whether dynamic order signal survives without seeing
  unrevealed candidate content.
- **Report caveat (mandatory):** Version A is a **content-aware frozen diagnostic, NOT a deployable
  generation-time controller** (it sees unrevealed candidate content). A positive Version-A result
  motivates, but does not by itself establish, a generation-time controller.

---

## §6 Position residualization + γ-mix (reuse Stage-A `graph_normalize`/`graph_order`)

Per step, the dynamic score vector over candidates is position-residualized and z-scored, then mixed
with the static C-D+L score on `B_A`:
```
s_H^resid = s_H − proj_{B_pos}(s_H)      # remove position component (text idx-dist / image manhattan)
s(v)      = z( s_CDL(B_A, v) ) + γ · z( s_H^resid(v) ),   γ ∈ {0, 0.25, 0.5, 1, 2}
```
- `γ=0` ≡ A-only ≡ the static `B_A` C-D+L order (identity to Level-0 baseline; an invariant to test).
- `s_CDL(B_A, v)` reuses `attn_order_teacher.teacher_scores(..., mode="C-D+L")`; the score-mix rollout
  mirrors `graph_order.score_mix_order` but with the **per-step dynamic** `s_H` injected (the new
  rollout cannot reuse `score_mix_order` verbatim because `s_H` requires a model forward each step).
- Position residualization uses `B_pos` from `position_graph.{text_position_graph,image_manhattan_graph}`
  and `graph_normalize.{offdiag_zscore, residualize}` semantics applied at the **per-step candidate-
  vector** level (residualize the candidate score vector against the corresponding `B_pos` row slice).
- **Controls:** (i) **position-control** — γ-mix with `s_H` replaced by the pure `B_pos` score;
  (ii) **shuffled-hidden / matched-random residual** — `s_H^resid` replaced by a value-multiset-matched
  random permutation per step (reuse `graph_normalize.matched_random_residual` analogue at vector
  level, seeded).

---

## §7 Comparison matrix — Levels 0/1/2 + per-sample audit

All orders are evaluated by the same **frozen NLL-under-order** (§9).

**Level 0 — baselines:** `random` (lower bound) · `static B_A C-D+L` (= A-only = γ0) · `current MLP
rollout` (source_start, via `sample_orders_batched_mlp`) · `L2R` (text) / `raster` (image).

**Level 1 — param-free dynamic (content-aware Version A first):** for each (pooling ∈ {summary[, last,
mean-revealed]}) × (score ∈ {cos-emb[, model-attn]}) × (γ ∈ {0,.25,.5,1,2}): roll the shared greedy
order (§3, §5, §6), eval NLL. Plus the two §6 controls. **This is the load-bearing level; the WIN gate
(§10) is computed ONLY from Level 1.**

**Level 2 — oracle upper bound (DIAGNOSTIC ONLY; NOT in WIN gate):** per-step greedy choosing the
candidate whose block tokens have the lowest true teacher-forced next-block NLL given `S_t` (reveal
each candidate, measure, pick min). Expensive (~O(N²/2·n) forwards) → small `n`. **Report must state:**
"Level-2 oracle is a diagnostic upper bound only — not a feasible order policy, not a teacher for
training, and not evidence that hidden itself is sufficient." It answers only: does a better
candidate-choice space exist for this frozen model? The controller decision rests on Level 1.

**Per-sample audit (DIAGNOSTIC; NOT in main verdict):** for 16–32 samples, run **per-sample** greedy
rollout (cos-emb summary, best Level-1 γ); report order **diversity** (pairwise Kendall-τ across sample
orders, entropy) and NLL on that small subset. If shared loses but per-sample shows high diversity +
partial NLL improvement → Case C ("dynamic hidden is sample-specific, not shareable").

---

## §8 Content-aware vs content-free (sequencing)

Run **Version A (content-aware)** first — simplest, and the most permissive test (if even content-aware
cannot beat A-only, the line is dead). Only if Version A WINs, run **Version B (content-free)** to
check whether the signal survives without peeking at unrevealed content (the deployable-controller
condition).

---

## §9 Evaluation metric & noise floor

- **Metric:** frozen teacher-forced token-avg **NLL-under-order** (reuse `hidden_graph_modelio.
  nll_under_order_text` / `nll_under_order_image`); text orders go phys→model via
  `physical_blocks_to_model_blocks` before NLL (image orders consumed in phys frame directly).
- **Noise floor:** bootstrap over the eval set (and/or a small set of rollout-aggregation seeds /
  `n_roll` resamples) to get an NLL std `σ`. Used by the WIN gate's `2σ` margin.
- `n_roll` (rollout hidden aggregation) small; `n_eval` (NLL) a separate ~256-sample set.

---

## §10 Hardened WIN gate + decision matrix

**WIN (per modality) requires ALL of:**
1. best `γ>0` NLL is **clearly lower** than `γ=0` (A-only);
2. the improvement exceeds eval noise — at least **2σ** or a preset absolute margin;
3. **not a single-γ fluke** — adjacent γ values do not crash (stability, as in Stage-A);
4. **beats the position-control** (§6 i);
5. **beats the shuffled-hidden / matched-random residual control** (§6 ii);
6. the winning **order is not degenerate** — not a trivial L2R/raster position order (check via
   Kendall-τ vs L2R/raster and the order-diversity metrics; a "win" that is just the position order
   does not count).

Else **NULL**. (Computed from **Level 1 only**; Level 2 and the per-sample audit never enter the gate.)

**Decision matrix (image quantified):**
| text | image | decision |
|---|---|---|
| WIN | WIN | **Case A** — design trained context-dependent controller (both modalities). |
| WIN | NULL | **Case A (text-only)** — proceed with text controller; note modality-specific; image NULL does **not** block it. |
| WIN | worse-than-A-only | **Case A (text-only)** + **flag modality-specific risk** for image. |
| NULL | (any) | **Case B** — `B_A` already absorbs order signal at this scale; pivot away from hidden. |
| NULL-shared + per-sample signal | — | **Case C** — pursue per-sample controller, not shared curriculum. |

Rationale: text order signal is directional/L2R; image is geometric/locality — they differ by nature,
so image is a **bonus**, not a necessary condition for a text controller.

---

## §11 What this probe does and does NOT show (framing guard)

- It tests whether **per-step causal hidden** adds order signal **beyond static `B_A`** — the one
  hidden direction not yet falsified.
- It does **not** (in Phase 1) establish a generation-time controller: Version A is content-aware
  (sees unrevealed candidate content). Version B addresses this only if A wins.
- The **Level-2 oracle** and **content-aware `e_v`** must **not** be allowed to smuggle in a
  "peek at real candidate content / real loss" order as if it were the result. **The only thing that
  decides the controller go/no-go is whether the Level-1 param-free causal-hidden dynamic score
  stably beats A-only under the §10 gate.**

---

## §12 Code structure & TDD

**Reuse (no changes):** `graph_normalize` (offdiag_zscore, residualize, matched_random, offdiag_corr),
`graph_order` (cdl_order; score_mix shape as reference), `graph_structure_metrics`,
`position_graph`, `hidden_graph_modelio` (nll_under_order_{text,image}), and the
`run_hidden_graph_diag` report/verdict pattern.

**New:**
- `block_lo_arm_order_network/causal_hidden_rollout.py`:
  - `causal_invariance_check(model, ...)` → PRE-GATE report (§4).
  - `step_context_hidden(model, idx, S_t, U_t, ...)` → `c_t_bar` via one forward (§3).
  - `candidate_embedding(model, block_ids, mode="content_token"|"content_free", block_len)` → `e_v` (§5.4).
  - `dynamic_score_cos(p_t, E_cand)` and `dynamic_score_attn(model, p_t, E_cand, top_heads)` (§5.1/§5.2).
  - `causal_score_mix_rollout(model, idx, B_A, B_pos, gamma, pooling, score_kind, e_v_mode, ...)` →
    phys-frame order via the per-step γ-mix greedy rollout (§3/§6).
  - per-sample audit + Level-2 oracle helpers (§7).
- `scripts/run_causal_hidden_probe.py`: driver mirroring `run_hidden_graph_diag` — load ckpt → PRE-GATE
  (abort on fail) → Level-0 baselines → Level-1 γ-sweep (+ controls) → Level-2 oracle → per-sample
  audit → NLL eval → §10 verdict + decision → JSON/MD report. Text smoke on small `--n-roll/--n-eval`
  first.

**TDD:** tiny `BlockAOGPT` (N=4, BL=1/2, n_layer=1, n_embd=8) smoke tests (mirror T6) for: invariance
check (construct a model and confirm completion-invariance holds, or a deterministic toy where it must),
context-hidden shape, candidate-embedding modes, the two dynamic scores (shapes + a known-direction
case), and the γ=0 ≡ A-only invariant of the rollout. CLI smoke (mirror T7) for the driver. Tests need
`block_lo_arm_order_network`, `nanogpt-learned-order`, **and `scripts/`** on `sys.path` (the T6 gotcha).

---

## §13 Self-review — open implementation flags (resolve at the named step; do not assume)

1. **§4/§3** — predictor-hidden completion-invariance: must PASS the PRE-GATE before the rollout
   mechanic is valid. Confirm `forward_fn(..., hidden_return_mode="predictor")` returns per-reveal-
   position predictor hidden indexable at `t` (verified for fixed orders in the residual line's
   `extract_causal_hidden`).
2. **§5.2** — `W_q/W_k` exposure & top-4-variance-head selection口径 vs `extract_A_matrices`; logits
   vs post-softmax. Resolve when implementing; cos-emb is the fallback.
3. **§5.4** — confirm `e_v` is token-only (no positional embedding leaking in); flag if not.
4. **§6** — per-step vector-level residualization/z-score reuse of `graph_normalize` (which operates on
   N×N graphs) — adapt to candidate-vector slices; confirm the `γ=0 ≡ A-only` invariant holds exactly.
5. **§9** — text phys→model order frame conversion before NLL (the T7 smoke gate).
