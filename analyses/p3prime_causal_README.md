# P3′ — Causal Verification of the L0 Global Physical-Order Carrier (B+)

**Date:** 2026-06-29 · **Branch:** `attn-order-alternating` · **Compute:** CPU
**Code:** `analyses/p3prime_causal_verify.py`, `analyses/plot_p3prime_causal.py`
**Spec:** `docs/superpowers/specs/2026-06-29-p3prime-causal-verification-design.md`
**Outputs:** `runs/p3prime_causal/seed{2,42,123}/{p3prime.json,p3prime.csv,p3prime_metrics.png}`

This report uses the session’s verification-scale run (`M=2`, `n_reveals=2`) to keep the
batch bounded in the current environment.

## What this verifies

Causal validation of the P2 **B+** decomposition of the **L0 global physical-order carrier**:
a dominant fixed slot→physical base map plus a smaller content-driven residual.
**P3′ does not attempt to convert B+ into C** — that needs multi-layout training (P4).

## Red lines

- Output/OV/`c_proj` ablation read on the same head's τ is degenerate; P3′ uses Q/K, QK-score,
  and input-to-QK interventions as primary causal tests.
- Self-QK patch read on the same head (A0) is not treated as load-bearing evidence — calibration only.
- P3′-B verdict = joint collapse of τ and slot-only R²; P3′-C residual is primary, τ secondary.

## Results

| seed | carrier | B base-map | C residual/floor | D: OV degenerate / QK changes | verdict |
|------|---------|------------|------------------|-------------------------------|---------|
| 2    | L0{2,3,4,5} | intact, mixed_tau_only, intact, intact | all heads below floor (`content_driven=false`) | OV degenerate on all carrier heads; QK changes = false | mixed/departure |
| 42   | L0{2} | mixed_r2_only | `content_driven=false` | OV degenerate; QK changes = false | mixed/departure |
| 123  | L0{1,2,3,4} | intact, intact, mixed_r2_only, mixed_r2_only | all heads below floor (`content_driven=false`) | OV degenerate on all carrier heads; QK changes = false | mixed/departure |

## Notes

- The seed-level verdict is `mixed/departure` for all three verification-scale seeds above.
- The carrier remains real and stable under canonical readout, but this session did not show the
  stronger P3′ load-bearing collapse signature required for a clean `B+ confirmed` call.
- `A0/A1` are calibration and redundancy checks, not load-bearing evidence.
- This is a single-layout causal check; it does not replace the multi-layout content test deferred to P4.
