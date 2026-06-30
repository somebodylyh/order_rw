# H / Context-Dependent Order — Line Summary (2026-06-30)

Branch `p5-direct-nll-routing`. Closes the "does hidden state H provide a
context-dependent (sample-specific) reveal-order signal beyond attention?" line.

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

## Root cause — why context-dependence has no traction ON TEXT

Not a method failure: **text has ~no context-dependent order headroom.** Physical
/ L2R order is near-optimal for ~94% of texts (per-sample CDL headroom 0.02–0.1
nat, in the noise floor). Even a perfect context-dependent controller has ≈0 to
gain. We were searching for a signal (step 2) without confirming the substrate
has per-sample headroom (step 1) — and on text step 1 ≈ 0.

## Verdict

**Context-dependent / H-conditioned order is closed for text.** Two reasons, both
necessary: (a) the readable order signal in attention/hidden geometry is position,
not content; (b) text offers no per-sample order headroom for content to exploit.

This does NOT prove context-dependent order is impossible in general — only that
text is the wrong substrate and H/B geometry is a position carrier.

## If pursued further (different substrate)

Context-dependence could exist where the per-sample-optimal order genuinely varies
(graph / molecule / code AST / structured docs; possibly images). **Precondition
test first (cheap):** per-sample oracle-best-order NLL vs best-fixed-order NLL —
is the gap large AND per-sample-varying? Only chase a predicting signal if that
headroom exists.

## Code / artifacts (branch p5-direct-nll-routing)

- `analyses/p5_utility_controller.py` — P5 frozen + Version A routing
- `analyses/p6_online_controller.py` — B0/B2 online co-adapt + HOnlyController
- `analyses/p6_cdl_sanity.py`, `p6_cdl_sanity_v2.py`, `p6_what_h_encodes.py`
- `analyses/p6_h_residual_modulation.py` + `scripts/run_p6_hres_replication.py`
- `analyses/p6_h_order_evolution.py`
- specs: `docs/superpowers/specs/2026-06-30-p5-direct-nll-soft-routing-design.md`,
  `...p6-online-coadaptive-controller-design.md`, `...p6-h-residual-modulation-design.md`
- results: `runs/p6/**`, `runs/p5/seed123/routing_A/**`
