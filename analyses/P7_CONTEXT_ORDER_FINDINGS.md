# Context-Dependent Reveal-Order in Text — Findings (2026-07-01)

Paper-ready paired result from the H / P5 / P6 / P7 line. Branch
`p5-direct-nll-routing`. Companion: `analyses/P6_H_CONTEXT_ORDER_SUMMARY.md`.

## The two questions (previously conflated — now separated)

1. Does hidden state **H** carry a context-dependent reveal-order signal beyond
   attention **B**?
2. Does context-dependent reveal-order structure **exist** in text at all, and is
   it **learnable** from available features?

## Result 1 — H signal: CLOSED (negative)

H is a **redundant physical-position carrier**, not a content signal:
raw H ≈80% shared position variance; trained readout z_H↔z_B per-text corr 0.935;
position-profiles corr 0.999; position-profile τ_vs_phys 0.86. Null across frozen
probes (P5), direct-NLL routing (A), online co-adaptation (B2, 3 seeds),
CDL-trained ckpts, and residualized-H modulation (9-run replication: real−shuffle
mean −0.0013, 44% negative = noise). Across CDL continuation (5k→60k), H stays
physical; the CDL teacher order itself converges to physical.

> H is not empty, but its readable component is a redundant physical-position
> carrier; the residual beyond B/position is not a stable sample-specific signal.

## Result 2 — context-dependent order: EXISTS but is SEARCH-ONLY (not learnable)

Frame-controlled (physical-block `order_nll`, baseline = TRUE physical L2R):

- **Exists (oracle headroom).** Per-sample hill-climbed order beats true L2R by
  **+0.19 nat** (all samples), is sample-specific (cross-sample transfer −0.13,
  specificity +0.33), and best orders are per-sample different (pairwise τ 0.09).
- **Genuine, not an easy-first artifact.** hill-climb best +0.187 vs L2R, while
  dynamic greedy easy-first is **−0.070 (worse than L2R)**, static easy −0.180,
  dyn-hard −0.235; best−dyn_easy = −0.257, τ(best, dyn_easy) = −0.11. The good
  orders are **non-myopic**.
- **Not learnable from available features (M=40, held-out).** No per-block feature
  predicts σ* well enough to beat L2R:

  | feature | τ_to_oracle | pred order vs L2R |
  |---|---|---|
  | B (attention) | +0.34 | +0.18 (worse) |
  | H (hidden) | +0.24 | +0.19 (worse) |
  | H⊥ (residual) | +0.06 | +0.18 (worse) |
  | content embedding | **−0.02** | +0.17 (worse) |
  | one-step difficulty | **−0.03** | +0.14 (worse) |
  | content+onestep | −0.01 | +0.16 (worse) |

  Content embedding and local difficulty have ~zero correlation with σ*. B/H carry
  only a weak position component (τ 0.24–0.34); the content-specific deviation that
  makes σ* beat L2R is in **none** of the features.

> Context-dependent reveal-order structure genuinely exists in text (per-sample
> orders beat true L2R by ~0.19 nat, non-myopic, not easy-first), but it is **not a
> learnable function of available per-block features**. The headroom is real but
> **search-only**; a cheap feed-forward controller cannot exploit it.

## Why prior nulls were not the whole story

All routing/H experiments (P5/A/B2) used a **weak, L2R-centric candidate pool** —
the good per-sample orders were never in the pool, so routing/H could not reach
them. B2's null = "can't beat L2R among L2R-ish candidates," not "no
context-dependent order." An earlier "headroom ≈ 0" claim was a weak-pool +
model-frame-arange (no inv_perm) **frame bug**, since corrected. Consistent with
the reranker history: oracle Δ huge, learned-MLP Δ ≈ 0.

## Final experiment (P7 loss-only) — in progress

Since σ* is not supervised-learnable from current features, the last check: can
**AO-NLL policy gradient exploration** break through the probe null? Train the
reveal MLP directly from AO-GPT NLL as a **Plackett-Luce reveal policy**
(EMA-baseline PG, frozen AO-GPT, train/test split), arms B / B+H / B+H⊥ / random /
L2R / oracle. **Caveat (locked): loss provides a training signal, not input
information.** If B/H inputs carry only position, PG can explore but cannot
generalize a content-dependent policy. Expected outcome A (B/H policy still
cannot beat L2R held-out) would fully close the line:

> Oracle context-dependent order exists, but a cheap controller cannot learn it
> from B/H even with AO-loss exploration — the limitation is the absence of
> exploitable input signal, not the teacher/optimizer.

Outcome C (B+H⊥ stably beats B held-out, multi-seed) would reopen the H line.

## Code / results
- `analyses/p6_oracle_headroom.py` — frame-correct oracle headroom + transfer
- `analyses/p6_easy_first_control.py` — easy-first artifact control
- `analyses/p7_predictability_probe.py` — feature→σ* learnability probe
- `analyses/p7_reveal_policy.py` — P7 loss-only PL policy-gradient (final exp)
- results under `runs/p6/**`, `runs/p7/**`
