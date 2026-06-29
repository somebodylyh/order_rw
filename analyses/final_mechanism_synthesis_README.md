# Final Mechanism Synthesis — Physical-Order Signal

**Date:** 2026-06-29  
**Scope:** canonical correction → emergence → P2 source decomposition → P3′ causal boundary

## Final claim

> AOGPT attention develops a global physical-order-aligned signal in L0. Under the fixed-layout
> protocol, this signal is primarily explained by a learned slot→physical map, with smaller
> content modulation. Causal probes show the order signal lives in the position/QK path, but do
> not confirm a clean carrier-specific modular B+ mechanism.

The final causal verdict is **B+ partial causal support, not a clean confirmation**. P3′ robustly
shows the order signal (τ) is position-path causal in every seed; it does **not** establish a
carrier-specific fixed-base-map locus (the position ablation is global — null heads collapse too)
and finds a content residual in only one head of one seed. The per-seed classifier marks seed123
`B+ confirmed`, but that label is overridden at the synthesis level by the selectivity caveat.

## Evidence chain

### 1. Model-frame correction

The old ③/A/⑤ identity-reveal, model-frame readout primarily measured an L1 slot/reveal
scaffold. It cannot support a physical-order or content-bound recovery claim. Same-head
output ablation was also non-diagnostic because the readout is taken from the head's upstream
attention map.

### 2. Canonical phenomenon

Under strict-65, random reveal, physical-frame scoring, a real global physical-order signal
emerges in L0: it is near the destroyed floor at initialization and reaches canonical τ≈1
around step 2k–3k across seeds. This establishes the phenomenon and its layer localization;
it does not identify the signal source.

### 3. P2: correlational source decomposition

Held-out B65 analysis explains the L0 carrier with a dominant fixed slot→physical table
(raw slot-only R² 0.85–1.00) plus a smaller between-text residual above the reveal-split noise
floor (content/floor ratio 2.2–19.4). OOD relayout collapses τ from about 1.0 to 0.13. Thus P2
supports **correlational B+** under the fixed-layout protocol, not pure content-bound recovery.

### 4. P3′: causal probe

P3′ validates the intervention geometry: same-head OV/output readout is degenerate, position/QK
perturbations strongly reduce τ in every seed (0.5–1.0 → ≈0.06), and multi-head carriers behave as
weakly coupled parallel copies. The order signal is therefore **robustly position-path causal**.
But the base-map half is not carrier-specific: the position ablation (zeroing wpe/wtpe) is a global
intervention that collapses the fixed-map R² of **null** heads too, so "R² collapse" cannot be
localised to the carrier. P3′-C exceeds its noise floor on only one seed123 head. The per-seed
classifier marks seed123 `B+ confirmed` (4/4 base-collapse + H2 content) and seed2/42
`mixed/departure`, but the **selectivity caveat overrides** a clean confirmation. Net: **B+ partial
causal support, not a clean carrier-specific confirmation.**

## Combined effect-size table

P3′ Δ values are clean minus intervened; positive values indicate collapse.

P3′-B values are post-fix: τ from the position ablation (not the A0 self-QK patch), and ΔR² from
the **clean-predict** base-map R² (`r2_none − r2_abl`, clean fixed-map table predicting the ablated
B), not the old self-referential `slot_only_r2(B_abl)`.

| seed | L0 carriers | P2 raw R² | P2 content/floor | P3′-B Δτ | P3′-B ΔR² (clean-pred) | joint B heads | P3′-C ratio | D: QK / OV-degenerate | classifier |
|------|-------------|------------|------------------|------------|--------------------------|---------------|--------------|------------------------|------------|
| 2 | {2,3,4,5} | 0.960–0.995 | 7.17–19.37 | 0.74–0.91 | 0.12–0.55 | 2/4 | 0.416–0.988 (0/4) | 4/4 · 4/4 | mixed/departure |
| 42 | {2} | 0.869 | 2.249 | 0.77 | 0.51 | 1/1 | 0.124 (0/1) | 0/1 · 1/1 | mixed/departure |
| 123 | {1,2,3,4} | 0.848–0.996 | 2.76–19.22 | 0.42–0.94 | 0.34–0.90 | 4/4 | 0.216–1.465 (1/4) | 4/4 · 4/4 | B+ confirmed* |

\*seed123 meets the mechanical classifier bar (majority base-collapse + ≥1 content-driven head),
but the **net synthesis verdict is partial support, not confirmation**: the base-map R² collapse is
non-selective (null heads collapse under the same global position ablation — see
`p3prime_causal_README.md`), so it does not localise a carrier-specific fixed base map.

## Claim boundary

**Supported:**

- a real, training-emergent L0 global physical-order signal;
- dominant fixed-layout slot→physical predictability with smaller content modulation in P2;
- QK/position sensitivity, OV/output degeneracy, and parallel-copy carrier redundancy in P3′.

**Not supported:**

- content-bound recovery under the fixed-layout protocol;
- a seed-uniform causal content residual;
- a clean modular causal B+ mechanism;
- a head-to-head handoff chain.

## Next decision

Do not tune P3′ thresholds toward confirmation. An optional bounded power check may rerun only
P3′-C on selected high-SNR seed2/seed123 heads at `M=32` or `M=64`; it should be reported as
an effect-size sensitivity analysis. A stronger content-recovery claim requires P4 multi-layout
training so that fixed slot lookup is unavailable.
