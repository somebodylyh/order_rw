# Hidden-Residual Order Diagnostic (Phase 1) — Design

- **Date**: 2026-05-25
- **Status**: design approved, ready for implementation plan
- **Positioning**: **preliminary** pipeline-validation + initial signal probe. **No training.**
  A null result is a *weak-model preliminary null*, NOT a final refutation.

## 1. Motivation & framing

The attention-only MLP order policy (`attn_order_mlp_policy`) reads a **global** graph
`B_θ` (one matrix for the dataset) and is therefore sample-invariant. The next step toward a
**context-dependent** order policy ("Attn Transformer in Transformer") is to let the policy
also use **hidden state**. Target decomposition:

```
score(v, x) = s_G(v) + Δs_H(v, x)
```

- `s_G(v)` — global attention-order prior (C-D+L score from global graph `B_G`).
- `Δs_H(v, x)` — sample/context-dependent residual the hidden state should explain.

The residual's supervision target is the **per-sample attention-order deviation**:

```
r_x(v | S_t) = s_x(v | S_t) − s_G(v | S_t)
```

where `s_x` uses a **per-sample** graph `B_x` and `s_G` uses the **global** graph `B_G`,
both scored by the same C-D+L teacher at the same partial state `S_t`.

**Why this target (not alternatives):**
- NOT the global C-D+L teacher: that is a deterministic function of `B_G`, which the 12-d
  attention features already span → hidden would be mathematically redundant → forced `Δs_H≈0`.
  We would wrongly conclude "hidden is useless" when the test was rigged.
- NOT realized task NLL first: expensive (per-candidate forward eval), high single-sample
  noise, and slides back into confidence/readiness — which is explicitly NOT the main line.
- `r_x` is per-sample, still attention-derived (not confidence), and is exactly the
  "global prior + hidden context adjustment" math.

## 2. Substrate

**Modality: TEXT** (image/sudoku are fallbacks if text residual is weak).

| role | checkpoint | arch | nature |
|---|---|---|---|
| **primary** | `block_lo_arm_order_network/probe_results/clean_base_random_perm/ckpt_step30000.pt` | 4L / 8H / 384, block_size 256, vocab 50304, block_len 4 | clean random-perm trained, **not** MLP-sculpted |
| **secondary** | `probe_results/attention_order_mlp/alt_from0_mlp_finetune/ckpt_step30000.pt` | same arch | MLP-order sculpted |

Same architecture, same 30k step → apples-to-apples. `N = 64` blocks, each block = 4 tokens.

**Why primary = clean random ckpt (not the alternating/MLP ckpt):**
1. Cleaner substrate — measures what a generic random-order AOGPT naturally forms.
2. Avoids circularity — must not use MLP-order-trained hidden to "prove" MLP-order residual.
3. Closer to the eventual LoMDM / context-dependent entry point (generic model state).

**Secondary comparison logic:**
- random has no residual, alt has → order curriculum *sculpts* hidden/order residual.
- random already has residual → generic model already carries context-dependent signal.
- both none → text residual is weak → pivot to image/sudoku.

## 3. Reused components (no new model surgery)

- `train_clean_aogpt.extract_A_matrices(model, idx_chunks, clean_perm, device, n_chunks)` —
  returns per-chunk `A` (N×N). Global `B_G` = mean over chunks; per-sample `B_x` = per-chunk
  (no averaging). Both via `build_directed_graph`.
- `attn_order_teacher.teacher_scores(B, S_t, U_t, last, mode="C-D+L")` → per-candidate scores
  `s(v|S_t)`; `teacher_components` → C/D/L parts; `rollout_order` → canonical order.
- `attn_order_features.build_features(...)` → 12-d `φ_B(v)` (`NUM_FEATURES=12`).
- `mlp_residual_policy.MLPResidualPolicy` — residual head (zero-init last layer + γ knob).
- `AOGPT_block.forward(..., return_hidden=True, hidden_return_mode=...)`:
  - `"original"` → per-position hidden in physical frame → **oracle `h_v`** = mean of block v's
    4 token hiddens (uses v's true content; not available at generation → upper bound).
  - `"predictor"` → predictor hidden at rank t → **causal `h_{S_t}`** = state after revealing
    `S_t` (generation-time available).
- `clean_perm` (physical↔model frame), `load_train_chunks` (text data).

## 4. Budget & noise control

- **Default**: `n = 256` chunks, `M = 4` extraction passes per chunk (random AO orders).
- `M = 4` → split-pass noise floor uses **2 vs 2** halves (symmetric; M=3 would be 1-vs-2,
  unstable). If signal is borderline, rerun with `n = 512` and/or `M = 6` (3 vs 3).
- **Split-pass noise floor**: `B_x^{(a)}` from passes 1..M/2, `B_x^{(b)}` from M/2+1..M.
  The pass-to-pass variation of `r_x` is the noise floor; require signal > floor.

## 5. Pre-gate: does per-sample `B_x` even vary? (runs BEFORE any residual learning)

Compute and report:
1. `mean_x ||B_x − B_G||_F`
2. `corr(B_x, B_G)` (mean over samples)
3. `Var_x(s_x(v))` (across-sample variance of per-candidate score)
4. residual target `r_x(v) = s_x(v) − s_G(v)` — **norm** and **SNR** (vs split-pass floor)
5. whether `r_x` is clearly non-zero above the noise floor

**Stop rule**: if `B_x ≈ B_G` or `r_x ≈ 0` (within noise floor) → **stop**, record weak-model
preliminary null. Do not over-interpret (this ckpt has high ppl). Reserve a stronger/later
ckpt, or pivot to image/sudoku.

## 6. Phase 1a — representation probe (no ordering yet)

For oracle `h_v` and causal `h_{S_t}` **separately**, fit linear + small-MLP probes for a
**focused** set (do not spread thin):

1. **position/id sanity** — block position / block identity (expected easy; sanity that
   hidden carries basic structure).
2. **sample / coarse content** — sample identity or coarse content cluster.
3. **residual sign / quantile** — predict `sign(r_x(v))` or its top/bottom quartile. **This is
   the most aligned with the downstream goal** and bridges to 1b.

**Controls**: shuffled-hidden (break sample↔hidden pairing) + position-only baseline.
**Stop rule**: if hidden decodes none of these above controls → stop.

## 7. Phase 1b — per-sample residual-deviation probe

Target `r_x(v | S_t)`. Reference partial states `S_t` come from the **canonical global-B
C-D+L rollout** (`rollout_order(B_G, mode="C-D+L")`), probed at `t ∈ {0, 16, 32, 48}` (context
grows). Same `S_t` used for `s_x`, `s_G`, and hidden extraction.

**Baselines** (must beat the strongest to claim hidden value):
- **B0**: predict mean residual only.
- **B1**: global `φ_{B_G}(v)` → `r_x` (expected R²≈0: global features are sample-invariant,
  cannot explain per-sample deviation — confirms the target is "what global can't give").
- **B2**: position/id embedding → `r_x`. **Hidden must beat B2** to show sample/context value.

**Hidden heads** (each measured as R² *gain* over its matched non-hidden baseline, so the
gain is attributable to the hidden term):
- **oracle**: `Δs_H(v) = MLP([φ_{B_G}(v), h_v])` — base features identical to **B1**, so the
  lift over B1 is attributable to `h_v`. → **R²_oracle = ceiling**.
- **causal** (must include candidate↔state interaction, since `h_{S_t}` is shared across
  candidates at a step):
  ```
  Δs_H(v, t) = MLP([ emb(v), pos(v), h_{S_t}, emb(v) ⊙ (W_h · h_{S_t}) ])
  ```
  plain concat alone makes per-candidate residuals structurally hard → the interaction term
  is required. → **R²_causal**.

**Headline metrics**: `R²_oracle` (ceiling), `R²_causal / R²_oracle` (fraction of ceiling the
deployable causal version reaches).

## 8. Order-effect metrics (beyond R²)

A residual can be predictable yet too small to matter. Compare orders from `s_G(v)` vs
`s_G(v) + Δs_H(v, x)`:
- `tau_vs_global` (Kendall τ to the global order)
- mean displacement
- top-k changed ratio
- argmax (first pick) changed rate
- residual magnitude / score-margin ratio (is `|Δs_H|` comparable to the gap between
  adjacent candidates' `s_G`?)

If R² > 0 but order is essentially unchanged → record "signal present, not yet order-relevant".

## 9. Decision flow

```
pre-gate (§5) ──fail──> weak-model null, stop
   │ pass
1a probes (§6) ──no signal──> stop
   │ signal
1b oracle+causal (§7,§8) ──> report R²_oracle, R²_causal/oracle, order-effects
```
Run primary ckpt first; then secondary for the sculpt comparison (§2).

## 10. Cost / feasibility

256 chunks × 4 passes ≈ 1k forward passes of a 4L/384 model on 256-len seqs, plus tiny
sklearn / small-MLP probe fits. **CPU minutes, GPU seconds — fully GPU-free.**

## 11. Out of scope (Phase 1)

- No policy training, no co-training, no model fine-tuning.
- No confidence/readiness objective (explicitly avoided as a main line).
- No realized-NLL-first objective (deferred to later validation).
- Image/sudoku substrates (fallback if text residual is weak).

## 12. Implementation guards (hard asserts — anti-regression)

Both stem from past coordinate/aggregation bugs (image E3 line). The plan must enforce them
with explicit asserts, not just comments.

1. **Frame consistency for `B_x` vs `B_G`.** Both graphs MUST be scored by the C-D+L teacher
   in the **same physical block frame**. `extract_A_matrices` emits model-frame attention;
   remap via `clean_perm` to physical block frame **before** `build_directed_graph` /
   `teacher_scores`, for *both* `B_x` and `B_G`. Assert frame tag equality before computing
   `r_x = s_x − s_G`. A frame mismatch silently corrupts the residual target.

2. **Fixed global `S_t` across all samples.** At each probed `t ∈ {0,16,32,48}`, every sample
   MUST use the *same* `S_t` taken from the single global-B C-D+L canonical rollout
   (`rollout_order(B_G, mode="C-D+L")`). Do NOT use a per-sample rollout. Assert that the `S_t`
   passed into `s_x(·|S_t)` and `s_G(·|S_t)` is identical across samples at a given `t`;
   otherwise `r_x` would absorb state differences instead of pure per-sample graph deviation.

## 13. Outputs

Write to `block_lo_arm_order_network/probe_results/hidden_residual_diag/<ckpt_tag>/`:
- `gate.json` (§5 metrics + pass/fail)
- `phase1a.json` (probe scores vs controls)
- `phase1b.json` (R²_oracle, R²_causal per t, ratios, baselines)
- `order_effects.json` (§8)
- `SUMMARY.md` (decision + caveats)
