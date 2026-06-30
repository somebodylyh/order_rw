# P6 / Version B — Online Co-adaptive H-Residual Controller — Design

> Date: 2026-06-30
> Branch: `p5-direct-nll-routing` (continue) → P6 work
> Predecessors:
> - `2026-06-29-p5-attention-scaffolded-utility-controller-design.md` (frozen, pairwise)
> - `2026-06-30-p5-direct-nll-soft-routing-design.md` (Version A, frozen routing)

## 0. Core question

P5 (frozen) + Version A (frozen, cleaner objective) both show: **frozen ckpt +
post-hoc H residual ≈ null**. Neither tested whether wiring the controller into
the *training loop* changes this. P6 asks:

> When the reveal-order controller is wired into AO-GPT training, does hidden
> state H become a useful **sample-specific** utility signal — lowering val NLL
> above a B-only controller — through co-adaptation?

Claim is **not** "H already contains recoverable order info." Claim is "H **may
become** useful when model + controller train in the loop." A negative result
does **not** prove the model lacks content capacity — only that, under this
controller/readout/training setup, H residual yields no utility gain.

## 1. Mechanism (per batch)

1. Forward current model → extract B (multi-head attention scaffold) and H
   (block-level hidden states, see §9).
2. Controller: `z = g_B(B) + α · g_H(H)`.
3. Candidate pool `Σ = {σ_k}` (§5) → priority vectors `y_k`, normalized logits
   `a_k = cosine(z, y_k)/τ` (§4), `p_k = softmax(a_k)`.
4. For each σ_k: `L_k = L_AO(model; x, σ_k)` (teacher-forced NLL under current
   model).
5. `L_route = Σ_k p_k · L_k` (+ optional entropy reg, §12).
6. Backward → update controller (always) and AO-GPT (in B2).

Gradients: `∂L_route/∂controller` via `p_k` only (L_k is constant w.r.t.
controller — order is not differentiated); `∂L_route/∂model = Σ_k p_k ∂L_k/∂model`
when the model is trainable.

## 2. No argsort(z) (locked from Version A)

Version A proved direct-NLL routing learns a candidate **router**, not a general
**ranker** (free-argsort regret ~0.17-0.20 vs pairwise ~0.13). Therefore P6 uses
**candidate soft-routing** as the training path, never `σ = argsort(-z)`. Eval
uses `σ_pred = σ_{argmax_k p_k}`. argsort(z) is at most a side diagnostic.

## 3. Logit scale (locked from Version A)

Version A: unnormalized `z·y_k` lets ‖z‖ grow → softmax saturates (entropy→0,
max_p→1) even at τ=1.0, τ loses control. P6 uses **cosine logits**:
`a_k = (z/‖z‖) · (y_k/‖y_k‖) / τ`. Monitor entropy(p), max_k p_k, ‖z_B‖,
‖α δ_H‖, residual ratio ‖α δ_H‖/‖z_B‖, selection distribution every eval step.

## 4. Candidate pool (§ fixed protocol, dedup)

K = 6 (smoke) → 8 (full). Fixed protocol, byte-identical orders deduped
(reuse Version A's `priority_matrix` dedup — fixes phys==local):
`σ_B_multihead`, `σ_CDL_B`, `σ_phys` (must include — near-optimal per A),
`σ_reverse_phys`, `σ_random_1`, `σ_noisy_B_1` (+ `σ_random_2`, `σ_noisy_B_2`
for full). noisy_B matters because a useful H residual most likely makes *small*
corrections near the B scaffold, not a wholesale reorder.

## 5. Staging (do NOT jump to joint)

- **B0 — frozen-model online sanity (engineering only).** Model frozen, train
  controller only; re-extract B/H + recompute L_k per batch on fresh data.
  **RED LINE: B0 has no co-adaptation → H is still frozen-posthoc → B0 will
  reproduce Version A's null by construction. B0 validates the online code path
  and optimizer, it is NOT a scientific test of H.** Gate to B2: L_route
  decreases, selected NLL ≤ σ_B, entropy does not instantly collapse.
- **B2 — joint online co-adaptation (the experiment).** Model + controller
  trainable, continue from early ckpt (§7).

(B0 and the earlier "B1 frozen-trunk" collapse into one — both are frozen-model
debug.)

## 6. Eval protocol (resolves the curriculum confound — IMPORTANT)

In joint training each arm co-adapts a **different** AO-GPT, and the controller's
`p_k` shapes **which order-mixture the model trains on**. So "B+H val NLL <
B-only" alone is ambiguous: it could be H content, or H merely changed the
training-order curriculum. We disambiguate with two val metrics + the controls.

- **Primary — fixed-order val NLL (model quality):** evaluate every arm's model
  on a held-out val set under a **single fixed canonical order (physical/L2R)**.
  Isolates model quality from routing differences → apples-to-apples across arms.
- **Secondary — hard-selected val NLL (end-to-end system):** val NLL under each
  arm's own `σ_{argmax_k p_k}`. Measures controller+model together.
- Plus soft expected NLL `Σ p_k L_k` (routing-objective diagnostic).

A B+H gain that appears in hard-selected but NOT in fixed-order val NLL means
"H improved the curriculum/selection," not "the model got better" — report it
honestly as such, do not over-claim content.

## 7. Start point

Early ckpt, **not** mature (10k/20k multi-head B headroom already ~0.006).
Primary: **seed123 @ 5k → 20k**. If resources allow: 2k → 20k. Bet (not
guaranteed): at 5k the scaffold is not yet near-optimal, leaving room H might
fill in the loop.

## 8. Experiment groups (≥4; same pool / seed / data order, only H pathway differs)

1. **B-only online** — `z = g_B(B)` (main baseline).
2. **B+H online** — `z = g_B(B) + α g_H(H)` (main experiment).
3. **B+shuffled-H online** — H shuffled across texts within batch (`B_i + H_j`,
   j≠i); training-time shuffle. Isolates sample-specific content from capacity.
4. **B+zero-H / mean-H online** — capacity control.
Optional: random/AO baseline (no controller).

## 9. H extraction

P5 already swept L0-L3 / contexts (frozen all-null), so P6 fixes a simple config
rather than re-searching: **H = block-level hidden states at L1** (L0 too close
to embedding/position; L2/L3 carry stronger frozen confound; L1 = compromise),
extraction context = `σ_B_multihead`. Config-support `h_layer ∈ {L0,L1,L2}`,
main run L1.

## 10. H stop-gradient

- **B2a — H detached** (`g_H(H.detach())`): controller loss does not directly
  reshape AO-GPT hidden states; the model is only shaped via L_k token-prediction
  loss. Cleaner, safer, prevents controller-code injection. **First version.**
- **B2b — H not detached**: controller loss flows back into H → model can learn
  to write utility signal into H. Stronger co-adaptation, but shortcut/injection
  risk and harder to interpret.

Default B2a. If detached shows zero H gain and the core hypothesis is "make the
model write info into H," run B2b. (B2b is ultimately required if the bet is
specifically about H being written to.)

## 11. Regularization (light, first version)

- Entropy reg `L = L_route − β H(p)`, β=0.001 (prevent early collapse; off if
  entropy healthy with cosine logits).
- Residual-norm: **monitor** ‖α δ_H‖/‖z_B‖ first; add `λ‖α δ_H‖²` only if ratio
  > 1.
- B-only anchor: `α_init ≈ 0.01` so B+H ≈ B-only at start (don't wreck training).
g_B trainable, initialized from A/P5 B-only controller; g_H trainable.

## 12. Compute

B2 backprops the model through all K weighted forwards
(`Σ p_k ∂L_k/∂model`) → ~K× train cost (K=6 → 6×) + K× activation memory. Smoke
uses B0 (forward-only) first; B2 smoke uses small K + short steps; consider grad
accumulation. Needs GPU (unlike Version A).

## 13. Metrics & success

Per arm, over training: train/val NLL curves (fixed-order primary + hard-selected
secondary), AUC/avg-NLL-over-steps, step-saving-to-target, final val NLL.
Controller: hard-selected NLL, candidate regret `L_sel − min_k L_k`, oracle gap,
entropy, selection distribution, residual ratio, H-shuffle drop.

**Success (Outcome A):** B+H fixed-order val NLL < B-only **AND** B+shuffled-H
loses the gain **AND** B+zero/mean-H does not match B+H **AND** residual ratio
bounded. → "H becomes useful only in controller-in-the-loop training."

## 14. Outcomes

- **A** B+H>B-only, shuffle falls back → positive (target).
- **B** B+H≈B-only, shuffle no effect → online co-adaptation still does not make
  block-level H useful; reveal-order utility stays attention-scaffold dominated.
  Closes the H line.
- **C** B+H worse → H destabilizes routing; check residual ratio/entropy.
- **D** B+H and shuffled-H both improve → capacity/regularization, not
  sample-specific H. Cannot claim content utility.

## 15. Red lines

1. B is **not** a repeat of P5 frozen readout — it tests controller-in-the-loop
   co-adaptation.
2. **No hard argsort(z)** as the main controller (Version A: router not ranker).
3. H gain must be confirmed by shuffled-H control, else not sample-specific.
4. A negative B does **not** prove the model lacks content capacity — only that
   H residual gives no utility gain under this setup.
5. B0 null is meaningless (no co-adaptation by construction).
6. Curriculum confound: B+H>B-only must show in **fixed-order** val NLL (not only
   hard-selected) before claiming model improvement.

## 16. Task breakdown

1. **Online candidate pool** — per-batch `candidate_orders_for_batch(...)` w/ dedup
   (reuse A's `candidate_orders`/`priority_matrix`).
2. **Normalized logits** — `candidate_logits(z, Y, mode="cosine")`; test scale
   invariant to ‖z‖.
3. **Online routing loss** — `routing_loss(model, idx, B, H, sigmas, controller,
   detach_h)` → (loss_route, L_k, p_k, entropy, selected); test grad sign.
4. **Controller modules** — reuse `BOnlyController` / `ScaffoldedController`;
   modes B-only / B+H / shuffled-H / zero-H / mean-H.
5. **detach_h config** (default True).
6. **Training-loop integration** — wire routing loss into AO-GPT continuation;
   arms B-only / B+H / B+shuffled-H (+ zero/mean).
7. **Eval loop** — fixed-order val NLL (primary) + hard-selected (secondary) +
   soft expected; regret, entropy, selection dist, residual ratio, shuffle drop.
8. **Logging/plots** — val-NLL curves, ΔNLL vs B-only, entropy, residual ratio,
   selection histogram.
9. **Smoke runner** — `run_p6_smoke.py --seed 123 --start_ckpt 5k --steps 2000`.
10. **README** — `P6_online_coadaptive_controller_README.md` (motivation, why not
    argsort, relation to P5/A, controls, can/cannot, results).

## 17. Execution order

1. Spec (this doc).
2. Implement online routing loss + eval; validate with **frozen model** (B0
   code-path sanity — remember §5 red line: a B0 null is expected, not a result).
3. seed123 5k, 2k-step B2 smoke: B-only vs B+H vs shuffled-H. Watch entropy /
   residual ratio / fixed-order val NLL. If entropy collapses, τ=1.0 or β up.
4. If smoke stable → 5k→20k full (add zero/mean-H, random baseline).
5. If B+H shows stable fixed-order gain with shuffle fallback → expand to 2k /
   seed2 / seed42.

## 18. Smoke config

seed=123, start=5k, ~2k continuation steps, K=6 (σ_B_multihead, σ_CDL_B, σ_phys,
σ_reverse, σ_random_1, σ_noisy_B_1), cosine logits τ=0.3 (→1.0 if collapse),
entropy monitoring, arms B-only / B+H / B+shuffled-H, H=L1, detach_h=True.
zero/mean-H + random baseline in the second round.
