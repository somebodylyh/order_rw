# Canonical 65-node Re-analysis — Results (supersedes the model-frame ③/A/⑤)

**Date:** 2026-06-28 · **Branch:** `attn-order-alternating` · **Compute:** no new training (CPU)
**Code:** `analyses/canonical_reanalysis.py`, `analyses/plot_canonical_reanalysis.py`
**Outputs:** `runs/canonical_reanalysis/seed{2,42,123}/{strict65_sweep.tsv,sweep.json,emergence.png}`,
`all_results.json`
**Readout:** the project's sealed **strict 65-node None-separated** protocol
(`reports/strict_65node_discovery_ckpt_verification_20260617/`): random reveal, loss-aligned
AR frame, posthoc-inv physical scoring, method **C-D+L**, gate strong/weak/fail, destroyed
controls. Run via `search_none_separated_65_heads`-equivalent functions on the
`runs/handoff_overnight` checkpoints, K=3 sampling seeds.

> **Supersession note (2026-06-29):** use this report for the existence, emergence timing, and
> L0 localization of the canonical signal. Its original `content-bound` source interpretation
> is superseded by P2 and P3′: under fixed layout the signal is dominated by a learned
> slot→physical map with smaller content modulation, and the causal verdict is mixed/departure.

## Why this re-analysis exists

The ③/A/⑤ arc read order with `build_model_frame_strict65` **but scored τ vs `arange` in the
model frame under identity reveal** — skipping the posthoc-inv physical scoring and using
identity instead of random reveal. That made step-0 τ≈0.77 a tautology (identity reveal +
causal mask ⇒ ascending) and identified an **L1** "carrier". Under the canonical readout the
picture is different and correct.

## Headline (corrected)

| seed | carrier (canonical, L0) | emergence (best \|τ\|: step0 → step10k) | crossing | content-perm collapse |
|------|--------------------------|------------------------------------------|----------|------------------------|
| 2   | L0 {2,3,4,5} | 0.16 → **1.00** | step **1000→2000** | 1.00 → **0.08** |
| 42  | L0 {2}       | 0.13 → **1.00** | step **1000→2000** | 1.00 → **0.08** |
| 123 | L0 {1,2,3,4} | 0.15 → **1.00** | step **2000→3000** | 1.00 → **0.08** |

1. **The order signal genuinely emerges** — best |τ| sits at the destroyed floor (~0.05–0.16,
   `n_strong=0`) at step 0–1000, then jumps to **1.0** at step ~2000 (seed123 ~3000), and a
   **redundant L0 carrier set** grows (`n_strong` 0 → 2 → 4–7). This matches
   `analyses/figures/head_signal_emergence.png` (transition ~1800) and Spec A's ~1400–2000
   window. **Spec A's *timing* was right; its *carrier identity* (L1) was a model-frame artifact.**
2. **Carrier is L0** in all 3 seeds (L0H2–5 / L0H2 / L0H1–4), not L1. The model-frame ③/A/⑤
   L1 carrier was a tautological-readout artifact.
3. **The canonical signal is perturbation-sensitive, but the control is not source-identifying.**
   Under `content_label_permutation_control` τ collapses 1.00 → 0.08 (entry-shuffled → 0.04).
   P2 later showed that this collapse is compatible with a dominant fixed-layout slot→physical
   map, so it cannot by itself establish content-bound recovery.

## Per-component corrections

- **A (emergence):** real, single rise floor→1.0 at step ~1000–2000; a redundant L0 carrier set
  forms (not a single head). Coarse (1000-step) timing; the exact crossing sits in the 1000→2000
  interval — fine timing would need a GPU re-train with online canonical logging (deferred).
- **③ (handoff/load-bearing):** the carrier is **single-layer L0** → no L0→L1 handoff to test.
  Mean-ablating the L0 carrier heads' `c_proj` output does **not** drop the canonical strong-pass
  count (seed2 5→5, seed42 4→5, seed123 4→4; same as null). **Methodological caveat:** the 65-node
  τ is read from each head's *attention map* (q·kᵀ), which `c_proj`-output ablation does not change
  — so this ablation cannot test a head's own readout (same limitation as Pillar ③). The finding
  is that order is an L0 *attention-pattern* phenomenon, read in a single layer, with a redundant
  head set; a causal test would need QK-path / input ablation, not output ablation.
- **⑤ (binding):** the content-permutation collapse establishes sensitivity of the canonical
  readout, not a content-bound mechanism. P2 is the source-disambiguation result.

## Caveats

- Per-seed, K=3 sampling seeds; destroyed floor ~0.05–0.07 anchors "absent". 3 training seeds is
  a floor.
- Coarse 1000-step ckpt resolution (emergence crossing inside 1000→2000).
- C2 output-ablation is degenerate for an attention-map readout (above); not a causal load-bearing
  test.

## Net

The global physical-order-aligned signal is absent at init and emerges around step 2000 into a
redundant L0 carrier set. The model-frame ③/A/⑤ result was a tautological identity-reveal readout.
Mechanism attribution is deferred to P2/P3′: P2 finds fixed-map-dominant correlational B+, while
P3′ gives mixed/departure causal evidence rather than a content-bound or modular B+ confirmation.
