# P5 Direct-NLL Soft-Routing (Version A) — Design

> Date: 2026-06-30
> Branch: `p5-direct-nll-routing`
> Predecessor: `2026-06-29-p5-attention-scaffolded-utility-controller-design.md`
> Phase-0 verdict (frozen, pairwise teacher): `no_gain` — H residual provides no
> post-hoc utility across 11 configs (shuffle_drop ≈ 0).

## Positioning (read first)

This is **Version A**: a *frozen* differentiable-routing sanity check + a B-only
**objective-function** comparison. It is **NOT** a new scientific claim about H.

We replace the offline **pairwise-teacher imitation** objective with a direct
**NLL-weighted candidate-routing** objective on the *same* frozen scaffold (same
ckpt, same H, same candidate pool, same pre-computed candidate NLLs `L_k`).
Information content is unchanged → we **expect H to remain null**. A is worth
doing because:

- **A1** validate the soft-routing gradient chain `z → a_k → p_k → Σ p_k L_k`
  trains stably (the mechanism Version B needs).
- **A2** test whether direct-NLL routing is a better-conditioned objective than
  pairwise imitation for **B-only** utility.
- **A3** confirm (or refute) that frozen H is still null under the cleaner loss.
- **A4** produce reusable loss/controller code for Version B.

Do **not** over-interpret A's H result. If H turns positive it is a bonus and
must still pass the shuffle-H control.

The real scientific bet — **Version B: online co-adaptation from an early ckpt**
— is captured as a forward-looking goal at the end and gets its **own spec**.

## Objective

Per text, reuse the existing dataset (`build_dataset`): `B_feat`, `H`, candidate
orders `σ_k` (the `cands` pool), and pre-computed teacher-forced AO NLL per
candidate `L_k = nll_by_label[k]`. The frozen model is **not** in the gradient
path — `L_k` are constants.

```
y_k(i) = 1 - rank_{σ_k}(i) / (N - 1)      # reveal priority of block i in σ_k
z      = g_B(B_feat)                        # B-only
       | g_B(B_feat) + α · g_H(H)           # B+H (and its controls)
a_k    = Σ_i z_i · y_k(i)                   # candidate logit = z · y_k
p_k    = softmax(a_k / τ)
L      = Σ_k p_k · L_k                      # differentiable in z via p_k
```

Gradient: `∂L/∂a_k = p_k · (L_k − E_p[L])`. Low-NLL candidates get pushed up,
high-NLL pushed down. No REINFORCE, no argsort-through-loss.

`g_B` / `g_H` reuse the existing `BOnlyController` / `ScaffoldedController`
(α via softplus, g_B frozen when training the residual).

## Components

- `candidate_priority(sigma, N)` → `y` vector, `y[i] = 1 − rank_σ(i)/(N−1)`.
- `priority_matrix(cands, N)` → `Y` of shape `(K, N)` for a sample's pool
  (stable label ordering shared with `L_k`).
- `direct_nll_routing_loss(z, Y, L_k, tau)` → scalar
  `Σ_k softmax((z·yk)/τ)_k · L_k`; also returns `p` for monitoring.
- `train_b_only_routing(samples, tau, epochs, lr)` → trained `g_B`.
- `train_residual_routing(samples, g_B, h_dim, tau, epochs, lr, h_mode)` →
  trained `ScaffoldedController` (reuses `_apply_h_mode` for real / shuffle /
  zero / mean).
- Eval (both modes, see below) + monitoring stats.
- `run_routing_A(ckpt, ...)` driver: build dataset (reuse), τ-sweep, train all
  arms, dump JSON + a small comparison table.

These slot into `analyses/p5_utility_controller.py` alongside the existing
pairwise pipeline (do not remove pairwise code — A2 compares against it).

## Experiment matrix

**Main** (headroom present, can distinguish the two objectives):
- ckpt `runs/handoff_overnight/seed123/ckpt_step10000.pt`, single-head **L0H1**,
  M=64, same `build_dataset` config as Phase-0.
- τ sweep `{0.03, 0.1, 0.3, 1.0}`.
- Arms: `pairwise B-only` (existing baseline), `direct B-only`,
  `direct B+H`, `direct B+shuffled-H`, `direct B+zero-H`, `direct B+mean-H`.

**Control** (near-zero headroom ≈ 0.006): same ckpt, multi-head **H[1,2,3,4]
mean** scaffold. Purpose: when the attention scaffold is already near-optimal,
does direct routing **degrade gracefully into stably selecting σ_B**, rather
than picking random/noisy candidates? This is a sanity/control, **not** the A2
primary judgment.

## Evaluation (both modes — fairness matters)

Direct-NLL routing has two valid interpretations: a **router** over a finite
order pool, and a **score-based ranker** through z. We report both.

- **Primary — pool-selected NLL**: `σ_pool = σ_{argmax_k p_k}`,
  `NLL_pool = L_{argmax}`, and regret vs pool-best. This is the main metric for
  the routing objective.
- **Diagnostic — free-argsort NLL**: `σ_free = argsort(−z)`,
  `NLL_free = order_nll(σ_free)`, and regret. Tests whether z generalizes
  beyond the candidate pool.

Interpretation:
- `NLL_pool` good, `NLL_free` bad → direct loss learned a candidate router, not
  a general block ranker. Does not block Version B, but implies **B should keep
  candidate soft-routing rather than a hard argsort(z) controller**.
- both good → direct-NLL also yields a sortable z (stronger).

The pairwise B-only baseline is evaluated under **both** modes too, so the
comparison is apples-to-apples per eval mode.

Also report, per τ and arm: `entropy(p)`, `max_k p_k`, selected-candidate
distribution, best-candidate distribution.

## Success criteria (H-independent)

A succeeds — regardless of H — if:
- direct B-only routing loss decreases stably;
- direct B-only pool-selected NLL is **≥ (no worse than)** pairwise B-only;
- candidate selection does **not** collapse to all-random/all-noisy;
- τ is controllable (entropy/max-p behave sensibly across the sweep).

H interpretation (kept light):
- If H stays null: *"Frozen direct-NLL routing confirms the prior static
  conclusion — H provides no stable post-hoc utility even under a cleaner
  direct-NLL objective. This does not answer online co-adaptation."*
- If H turns positive: only credible if `B+H` beats `B-only` **and**
  `B+shuffled-H` falls back **and** `B+zero/mean-H` does not match `B+H`.
  Otherwise it is capacity/noise.

## Risks / watchpoints

1. **Temperature collapse** — too-small τ → instant one-hot; monitor
   entropy/max-p, that is why we sweep τ.
2. **Trivial selection** — if routing always picks σ_B, that is *fine* for the
   objective but signals ID headroom is insufficient → points Version B toward
   early ckpt / OOD layout, not toward a different loss.
3. **Sparse z shaping** — direct routing constrains z only through K candidate
   alignments vs pairwise's dense N² pairs; this is exactly why `NLL_free` is a
   required diagnostic.

## Out of scope (Version B — separate spec)

`min_{model, controller} E_x [ Σ_k p_θ(k|B,H) · L_AO(model; x, σ_k) ]` where
`L_k` is recomputed each step because the model changes. Staged as **B0**
(controller-only online, AO-GPT frozen within refresh interval) → **B1** (joint
co-adaptation). Start from an **early ckpt (2k/5k)** — mature ckpts have
near-zero headroom and would likely reproduce the frozen null. Four arms
(B-only / B+H / B+shuffled-H / B+zero-H), metrics = val-NLL curve / step-saving
/ training-curve AUC / hard-selected NLL / eval-time shuffle drop. Only then can
we answer *"does H become useful only when trained in the loop?"*

## Testing

TDD per existing `block_lo_arm_order_network/tests/test_p5_*.py` style:
- `candidate_priority` / `priority_matrix`: correct rank→priority, σ_B priority
  monotone, identity-order sanity.
- `direct_nll_routing_loss`: matches manual `Σ softmax·L`; gradient sign — a
  candidate with below-mean L_k gets its logit pushed up after one step;
  τ→0 approaches hard-min selection; permutation-invariance over candidate order.
- training: B-only routing reduces expected NLL on a toy 2-candidate set;
  zero-H residual == B-only.

## Outputs

- Code: `analyses/p5_utility_controller.py` (new routing functions + driver).
- Results: `runs/p5/seed123/routing_A/` — `main_L0H1.json`,
  `control_multihead.json`, per-τ entries, comparison table.
- Tests: `block_lo_arm_order_network/tests/test_p5_routing.py`.
