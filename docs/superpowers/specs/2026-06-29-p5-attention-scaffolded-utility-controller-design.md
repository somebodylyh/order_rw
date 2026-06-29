# P5 — Attention-Scaffolded Utility Controller

**Date:** 2026-06-29 · **Branch:** `attn-order-alternating`
**Subtitle:** *Does a hidden-state residual on top of an attention-derived order scaffold lower
downstream teacher-forced AO NLL — i.e. does H carry sample-specific ordering utility beyond
attention topology?*

**Foundations (read first):**
- `analyses/final_mechanism_synthesis_README.md` — line A is **attention-only**; the signal is a
  position-implemented, layout-bound global order scaffold (not content recovery).
- `analyses/layout_ood_README.md` — under unseen layouts the attention signal is pure **lookup**.
- Memory `attention-only-vs-hidden-state-framing-20260629` — line A vs line B split.

---

## 0. Core question, claim, and the red line

Under the **fixed training layout** (data shuffled exactly as before — *unchanged*), with the
controller input containing **both** the attention graph `B` and block-level hidden states `H`:

> **Core question:** Given the attention-derived order scaffold `B`, does the hidden state `H`
> provide a *sample-specific utility correction* that makes the reveal order better — measured by
> **downstream teacher-forced AO NLL** — than the B-only scaffold?

**Core controller:** `z_i = g_B(B)_i + α · g_H(H)_i` (per-block scores → reveal order).

**Core claim (what P5 can establish):** *H improves downstream reveal-order utility beyond the
attention scaffold.* **NOT** *H recovers physical order* (that is line A, and is layout-bound).

### The red line (why P5 must use a utility target)

> **In fixed-layout training, neither physical-rank supervision nor CDL(B) supervision can force H
> to be used: physical rank is slot-deterministic (the target is constant across texts → content
> carries zero information about it), while CDL(B) is B-deterministic (H is redundant given B in the
> input). Therefore P5 uses downstream AO NLL as the utility signal.**

This is the load-bearing reason the target is *not* "order recovery". Downstream NLL is
sample-dependent and is **not** a deterministic function of `B`, so `H` has genuine non-redundant
room to contribute (content cues about which blocks to reveal first for this text).

---

## 1. Measurement: downstream teacher-forced AO NLL (the utility signal)

For a reveal order σ (a permutation of the 64 model blocks → token order) and a text sample, the
utility is the **negative teacher-forced AO NLL** of the model under that order:

```
loss = model.forward_fn(idx_sample, token_order_of(σ))[1]   # (logits, loss) -> loss
U(σ; sample) = - loss
```

This reuses `train_clean_aogpt.order_loss` exactly. One forward per (sample, σ). Lower NLL = better
utility. The model is frozen during all of P5 (we train only the controller, never the AO-GPT).

---

## 2. Utility-headroom gate (kill gate — runs BEFORE any controller training)

If the attention scaffold σ_B is already near-optimal within the candidate pool, `H` has nothing to
learn. Measured on **held-out texts**, over the candidate pool {σ_k}:

```
Headroom_abs = NLL(σ_B) - min_k NLL(σ_k)          # absolute
Headroom_rel = Headroom_abs / |NLL(σ_B)|          # relative
```

**Gate:** proceed only if `mean Headroom_abs > 0` with a bootstrap CI that does **not** cross 0. No
hard threshold (e.g. 0.01) at first — use statistical significance + effect size. If the gate fails,
**stop**: under fixed layout the attention scaffold is already sufficient and there is no utility
room for H. (This is a publishable negative: "attention scaffold is utility-sufficient".)

---

## 3. Candidate orders (diverse, scaffold-centred but not scaffold-only)

Per sample, generate a candidate pool (8–16 orders) — diverse enough that the best-NLL order can
differ from σ_B, but anchored near the scaffold for residual learning:

- `σ_B_current` — current attention scaffold / existing g_β order
- `σ_CDL_B` — CDL(B) rollout
- `σ_phys` — physical L2R; `σ_reverse_phys` — reversed
- `σ_random_1..4` — random orders
- `σ_local_neighbor` — local-adjacency heuristic
- `σ_noisy_B_1..K` — `z_B` + small random perturbation (explores the residual neighbourhood of the
  scaffold — the space `g_H` is meant to navigate)

The pool must not be only σ_B-derived, or the headroom is artificially capped.

---

## 4. Utility teacher: soft preference (not hard argmin)

Candidate NLLs can be close; hard argmin is noisy. Build a **soft pairwise** teacher per sample:

```
w_k  = softmax(-NLL_k / T)                          # temperature-weighted candidate weights
P_ij = Σ_k w_k · 1[i before j in σ_k]               # soft "block i revealed before block j"
```

Controller predicts `P_hat_ij = sigmoid(z_i - z_j)`. Loss is confidence-weighted BCE:

```
c_ij = |P_ij - 0.5|                                 # emphasise pairs the teacher is sure about
L    = Σ_{i,j} c_ij · BCE(P_hat_ij, P_ij)
```

This admits content-dependent orders and never forces physical order.

---

## 5. Architecture & the three training stages (frozen B-only baseline)

```
z_B,i = g_B(B)_i                     # B-only controller (per-block score)
δ_H,i = g_H(H)_i                     # hidden-state residual (per-block score), g_H = MLP(pool(H_i))
z_i   = z_B,i + α · δ_H,i            # α = softplus(a), init α ≈ 0.01 (small, learnable)
```

**Stages (fixed so B+H gains are attributable to H):**
1. **Train `g_B`** (B-only) on the **same** soft-utility teacher (§4).
2. **Freeze `g_B`.**
3. **Train `g_H`** (the residual) with `z = g_B(B) + α·g_H(H)`, α small/learnable.

Because the B-only baseline is trained on the *same* utility teacher, any B+H improvement is
attributable to the H residual — not to a target mismatch. Report `||α·δ_H|| / ||z_B||` to confirm
the residual stays a correction, not the trunk.

`B` features = the selected-head attention graph (reuse the existing g_β B65 / selected-head
features). `H_i` = **block-level mean-pooled hidden state** of model-block i, extracted by hooking
the block residual (`path_patch_handoff.capture_block_input`) and pooling token positions per block
via the reveal/block mapping (`per_head_order_scan` block labels). Layer of H is a config (try
L0/L1/L2/L3).

---

## 6. Data split (TEXT-level, fixed layout)

Split on **texts**, disjoint, same fixed layout, same candidate-generation protocol:
`train 70% / val 15% / test 15%` (or 384/64/64 at M=512). **Report only held-out-text metrics.**
Splitting on candidate order or reveal seed is forbidden (would let the controller memorise
sample-level utility patterns).

---

## 7. Controls & primary metrics

**Controls (the shuffled-H is decisive):**
- **B-only** — `z = g_B(B)` (frozen baseline).
- **B+H** — `z = g_B(B) + α·g_H(H)`.
- **B+shuffled-H** — `B` from text i, **H from a different text j** (block structure intact, but
  mismatched to B/text). Text-level shuffle (§detail D). If the B+H gain disappears here, H used
  **sample-specific content**, not extra capacity.
- **B+zero-H / B+mean-H** — capacity controls (H replaced by 0 / batch-mean).

**Primary metrics (held-out texts):**
- `ΔNLL = NLL(B+H order) − NLL(B-only order)` — want **< 0**.
- `Utility regret = NLL(pred order) − min_k NLL(candidate order)` — want B+H regret < B-only regret.
- `H-shuffle drop = NLL(B+shuffled-H) − NLL(B+H)` — want the shuffle to **lose most/all** of the gain.

**Success requires all three:**
1. `ΔNLL(B+H − B-only) < 0` on held-out texts;
2. `B+shuffled-H` loses most/all of the gain;
3. `B+zero-H` / `B+mean-H` do **not** match B+H.

**τ is a side analysis only** ("does the utility-improving order stay physical-order aligned?") —
never a P5 success/failure criterion.

---

## 8. Sequencing: Phase 0 (one ckpt) → Phase 1 (g_β-ckpt sweep)

**Phase 0 — method validation on ONE checkpoint** (preferably the seed123 strong ckpt):
headroom gate → utility teacher → train B-only → freeze → train B+H residual → controls. Decide
whether the paradigm works (success criteria §7). Do NOT scan ckpts here.

**Phase 1 — only if Phase 0 is positive — the co-adaptation curve:**
scan `pre-gbeta / gbeta+2k / gbeta+5k / gbeta+10k` checkpoints; for each, run B-only / B+H /
B+shuffled-H; plot `ΔNLL(B+H − B-only)` vs g_β-training step. The Phase-1 question is distinct:
*"Does controller-in-the-loop training make the H residual increasingly useful?"* — want ΔNLL to
trend more negative with g_β step while B+shuffled-H stays flat. **Framing:** this is **not**
original attention-only emergence; it is the **controller-in-the-loop utility residual** (a
co-adaptation effect). Do not merge Phase 0/1 conclusions.

---

## 9. Can / cannot claim

**Can:** H provides sample-specific utility corrections on the attention scaffold that lower
downstream AO NLL (if §7 met); the gain is content-driven (shuffled-H falls back) and not capacity
(zero/mean-H controls); under co-adaptation the residual grows (Phase 1).

**Cannot:** that H recovers physical order (line A, layout-bound); that the corrections generalise
across layouts (the setting is fixed-layout by construction — content-vs-position is not separable
here, and P5 does not claim it is); anything from τ.

---

## 10. Feasibility (verified)

- **NLL-under-order:** `model.forward_fn(idx, token_order) → (logits, loss)`; reuse `order_loss`. ✓
- **H extraction:** hook block residual via `path_patch_handoff.capture_block_input`, mean-pool per
  model-block using the reveal/block mapping. Modest plumbing, machinery exists. ✓
- **Candidate generation / readout:** reuse `none_separated_block_graph` rollouts and the existing
  g_β / selected-head B features. ✓
- Compute: CPU-feasible for Phase 0 smoke (frozen model, ~512 texts × ~12 orders forwards). No
  AO-GPT training. Phase 1 needs the g_β-running ckpts (already produced by the hook line).

---

## 11. Recommended task split (~11, for writing-plans)

1. **H extraction** — `block_hidden_states(model, idx, token_order, layer) -> (n_block, d)` via
   residual hook + per-block mean-pool; test shapes + determinism.
2. **Candidate-order pool** — `candidate_orders(sample, scaffold, K)` (σ_B/CDL/phys/rev/random/local/
   noisy-B); test diversity + includes σ_B.
3. **Order NLL utility** — `order_nll(model, idx, sigma, clean_perm) -> float` (wrap `order_loss`);
   `utility_pool(model, idx, sigmas) -> [nll]`; test matches order_loss.
4. **Utility-headroom gate** — `headroom(nll_pool, idx_B) -> {abs, rel}` + bootstrap CI; test gate
   logic on synthetic pools (positive/zero headroom).
5. **Soft pairwise teacher** — `soft_pref(sigmas, nlls, T) -> P_ij`; weighted-BCE loss
   `pairwise_loss(z, P)`; test against hand-computed small case.
6. **Controllers** — `g_B(Bfeat)`, `g_H(H)`, `ScaffoldedController` (`z = z_B + softplus(a)·δ_H`);
   test residual ratio reporting + α init.
7. **Stage-1/2 train B-only + freeze** — train `g_B` on soft teacher (held-out-text split); test it
   learns (val pairwise acc > chance) and freezes.
8. **Stage-3 train residual** — train `g_H` with frozen `g_B`; controls B+H / B+shuffled-H (text
   shuffle) / B+zero-H / B+mean-H; test shuffle wiring (B text i, H text j).
9. **Metrics** — `delta_nll`, `utility_regret`, `h_shuffle_drop`, residual-norm-ratio on held-out;
   `classify_p5(...)` (success = all three of §7); test confirm + null cases.
10. **Phase-0 driver + plot** — single-ckpt runner writing `runs/p5/seed123/phase0.json` + a metrics
    plot (ΔNLL / regret / shuffle-drop bars); test writes outputs.
11. **README + memory** — `analyses/p5_utility_controller_README.md` (red line, headroom gate,
    success criteria, can/cannot, Phase-0 result); MEMORY pointer. (Phase 1 ckpt-sweep is a
    follow-on plan, gated on Phase 0.)

---

### Sentences to preserve verbatim in code/README
1. "P5 tests whether hidden states provide sample-specific utility corrections on top of an
   attention-derived order scaffold, using downstream teacher-forced AO NLL as the supervision
   signal."
2. "In fixed-layout training, neither physical-rank supervision nor CDL(B) supervision can force H
   to be used; therefore P5 uses downstream AO NLL as the utility signal."
3. "The core claim is that H improves downstream reveal-order utility beyond the attention scaffold,
   NOT that H recovers physical order."
