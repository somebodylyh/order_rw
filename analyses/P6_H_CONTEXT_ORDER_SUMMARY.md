# H / Context-Dependent Order — Line Summary (2026-06-30)

Branch `p5-direct-nll-routing`. Two outcomes: (1) the **H signal** for
context-dependent order is closed (H = redundant position carrier); (2) but
**context-dependent order itself is NOT closed for text** — a frame-controlled
oracle test found large sample-specific headroom over true L2R (earlier "headroom
≈ 0" was a weak-pool + frame-bug error, now corrected below).

## Question

Across AO-GPT, can per-block hidden state **H** carry **sample-specific** order
information that improves reveal-order utility **beyond the attention scaffold B**
(which encodes a global physical/L2R order)? I.e. a context-dependent order.

## What we ran (all on text, seed123 unless noted)

| # | Experiment | Setup | Verdict |
|---|---|---|---|
| P5 | frozen post-hoc H probe | pairwise teacher, 11 configs (5 layers × ctx) | **null** (ΔNLL>0, shuffle_drop≈0) |
| A | frozen **direct-NLL soft-routing** | cosine-free, dual eval | routing works; **H null**; direct = router not ranker |
| B2 | **online co-adaptation** | model+controller trained in loop, 5k→, continuous stream, lr=1e-5/bs=8 | **H null**, 3 seeds (123/2/42); ΔNLL within ±0.0024 noise |
| CDL sanity | H vs CDL order on CDL-trained ckpt | cdl_from10k_L1H7 20k, batch-mean CDL | pool diverse (headroom exists) but H = B selection, both pick phys |
| what-H | position vs content decomposition | cdl 20k | raw H **80% position**, z_H 86%; pos-profile τ_vs_phys 0.86; **z_H↔z_B corr 0.935, pos-profiles corr 0.999** |
| H-residual | z=z_B+λ·g_H(H⊥), penalize B/pos copy | cdl 20k, single | first real<shuffle (−0.043) hint |
| **replication** | H-residual across 9 runs | seed 123/2/124 × ckpts | **NOISE**: real−shuffle mean −0.0013, 44% neg, std 0.016 |
| H-order evo | what order H represents per ckpt | cdl_from10k 5k→60k | H~phys flat 0.85–0.95; CDL converges to phys (cdl~phys 0.04→1.0) |

## Findings

1. **H is a redundant physical-position carrier.** ~80% of raw H (86% of the
   trained readout) is a shared-across-text position profile that **is** the
   physical block order (τ 0.86) and is **0.999 identical** to attention's
   position profile (per-text readouts corr 0.935). H re-encodes what B already
   has.
2. **No stable sample-specific content signal.** After removing the position copy
   and penalizing B/position redundancy, the residual's real-vs-shuffle gap is
   noise across 9 seed/ckpt runs (mean ≈ 0, sign ≈ coin flip). The single-run
   −0.043 was sampling noise (B extraction has unseeded randperm).
3. **Online co-adaptation doesn't help.** Training H into the loop (B2, 3 seeds)
   leaves H null — H does not "become useful when trained in the loop."
4. **H stays physical even under CDL training.** Across 5k→60k of CDL-teacher
   continuation, H keeps representing the physical order (τ~0.85–0.95). It is the
   **CDL teacher order that converges onto physical** (cdl~phys 0.04→1.0), not H
   drifting to CDL. Physical order is the stable attractor.

## Why the H signal fails — and a CORRECTION on "headroom"

The H readout fails because the **candidate pools used everywhere (P5/A/B2) were
weak and L2R-centric** (only good candidate = phys; rest random/noisy-B). The
good per-sample orders were never in the pool, so routing/H could never reach
them. B2's null = "can't beat L2R among L2R-ish candidates," NOT "no
context-dependent order exists."

**CORRECTION (supersedes an earlier wrong claim that text has ~0 headroom).** That
earlier claim was measured (a) against the weak pool and (b) with a FRAME BUG
(model-frame `arange` ≠ physical L2R; no inv_perm applied). A frame-controlled
oracle test (our ckpt, validated `order_nll(phys-block order, clean_perm)`,
baseline = TRUE physical L2R = arange, M=8, hill-climb 800 steps) shows:

| metric | value |
|---|---|
| per-sample best vs TRUE L2R (own_gain) | **+0.203** (8/8, 0.17–0.24) |
| transfer (best_j applied to sample i) | **−0.131** (hurts others) |
| specificity (own − transfer) | **+0.335** |
| τ(best, true phys L2R) | 0.33 |
| τ between best orders | 0.09 (per-sample different) |

→ **Real, context-dependent reveal-order headroom EXISTS on text, even over true
L2R**, and is sample-specific (transfer HURTS → rules out a generic better fixed
order). Consistent with the reranker history (oracle Δ huge, MLP Δ ≈ 0): headroom
is large; **learnability/reachability is the bottleneck**, not existence.

## Verdict (two layers)

1. **H as a signal is closed:** attention/hidden geometry carries position, not
   content; H is a 0.999-redundant copy of attention. Confirmed across frozen,
   online co-adapt (3 seeds), CDL-trained, and H-residual replication (noise).
2. **Context-dependent order itself is NOT closed for text:** there is large,
   sample-specific oracle headroom over true L2R. The open problems are (i) is it
   meaningful vs a teacher-forced-NLL artifact, and (ii) is it learnable.

## Easy-first control — DONE: genuine non-myopic structure (NOT artifact)

`analyses/p6_easy_first_control.py`, seed123 10k, M=4, hill-climb 600. Gain vs L2R:

| order | gain vs L2R |
|---|---|
| hill-climb best | **+0.187** |
| dynamic greedy easy-first | **−0.070** (worse than L2R) |
| static easy-first | −0.180 |
| dynamic greedy hard-first | −0.235 |

best − dyn_easy = **−0.257**; τ(best, dyn_easy) = **−0.11**. Easy-first (static &
dynamic) cannot even beat L2R; hill-climb best beats it by 0.26 nat and the orders
are anti-correlated. → **The +0.19 headroom is genuine, non-myopic, sample-specific
order structure, NOT an easy-first teacher-forced-NLL artifact.**

## Learnability probe — DONE: σ* is NOT supervised-learnable (search-only)

`p7_predictability_probe.py`, seed123 10k, M=40, held-out test=12. No per-block
feature predicts σ* well enough to beat L2R (all pred orders +0.14–0.19 WORSE than
L2R): B τ*=0.34, H 0.24, H⊥ 0.06, **content −0.02, one-step −0.03**. Content
embedding and local difficulty have ~zero correlation with σ*; B/H carry only a
weak position component. → **context-dependent order exists but is search-only; the
content-specific deviation that beats L2R is in none of the available features.**

Full paired result: `analyses/P7_CONTEXT_ORDER_FINDINGS.md`.

## Final experiment — P7 loss-only (in progress)

Last check: can AO-NLL policy-gradient *exploration* break the probe null? Train
the reveal MLP directly from AO-GPT NLL as a Plackett-Luce policy (EMA-baseline PG,
frozen AO-GPT, train/test split), arms B/B+H/B+H⊥/random vs L2R/oracle.
**Locked caveat: loss gives a training signal, not input information** — if B/H
carry only position, PG explores but cannot generalize a content-dependent policy.
`analyses/p7_reveal_policy.py`.

## Code / artifacts (branch p5-direct-nll-routing)

- `analyses/p5_utility_controller.py` — P5 frozen + Version A routing
- `analyses/p6_online_controller.py` — B0/B2 online co-adapt + HOnlyController
- `analyses/p6_cdl_sanity.py`, `p6_cdl_sanity_v2.py`, `p6_what_h_encodes.py`
- `analyses/p6_h_residual_modulation.py` + `scripts/run_p6_hres_replication.py`
- `analyses/p6_h_order_evolution.py`
- specs: `docs/superpowers/specs/2026-06-30-p5-direct-nll-soft-routing-design.md`,
  `...p6-online-coadaptive-controller-design.md`, `...p6-h-residual-modulation-design.md`
- results: `runs/p6/**`, `runs/p5/seed123/routing_A/**`
