# P6 H-Residual Modulation — Design

> Date: 2026-06-30. Branch: `p5-direct-nll-routing`.
> Follows the "what does H encode" finding: raw H ≈80% shared position profile,
> trained z_H ≈86.5% position, z_H pos-profile corr 0.999 with z_B's, z_H↔z_B
> per-text corr 0.935, pos-profile τ_vs_phys 0.855. → H-only is a redundant copy
> of the attention/physical-position signal.

## Goal

Stop using H as a standalone order predictor. Instead remove the position copy
and test whether the REMAINING per-text residual (the ~13–20% content) can
usefully MODULATE the B-readout. Recommended boxed form:

```
H_i^⊥ = H_i − H̄_pos(i)          # H̄_pos estimated on TRAIN only, per block
z_i   = z_B(i) + λ · g_H(H_i^⊥)   # λ learnable scalar, init 0 (model opens H if useful)
```

g_B frozen (the established B/CDL backbone), g_H reads only the residual.

## Loss (pairwise-teacher frame, reuse P5)

```
L = PairwiseBCE(z, P_teacher)
  + α_corr · corr(z_Hres, z_B.detach())²     # don't re-copy B
  + α_pos  · corr(z_Hres, phys_rank)²         # don't re-copy physical position
  + α_l2   · ‖z_Hres‖²  (optional)
```

P_teacher = P5 `soft_pref` over the candidate pool (existing). λ init 0 so the
arm starts == B-only; if H_res is useless λ stays ~0.

## Arms (ablation)

- `b_only` (baseline), `b_plus_rawH` (no residual/penalty), `b_plus_Hres`
  (residual, no penalty), `b_plus_Hres_pen` (residual + corr/pos penalties).
- H controls per arm: real / zero / shuffle (across-text) H_res.

## Diagnostics (must report)

1. `lambda`, `norm_ratio = ‖λ z_Hres‖ / ‖z_B‖` (is H opened?)
2. `corr(z_Hres, z_B)` — must be << raw 0.935.
3. `corr(z_Hres, phys_rank)` — must be << raw 0.855.
4. `real_vs_zero_delta`, `real_vs_shuffle_delta` (downstream order NLL): does
   sample-specific H_res content matter, not capacity?
5. `downstream order NLL` (argsort(z) → order_nll) vs b_only — the real test.
6. pairwise val acc, τ to teacher.

## Success criteria

- λ does NOT collapse to 0, norm_ratio non-trivial.
- corr(z_Hres, z_B) and corr(z_Hres, phys) clearly below raw H's 0.935 / 0.855.
- `b_plus_Hres_pen` downstream NLL < `b_only` AND real beats shuffle/zero.
- Otherwise: H_res is decoration, not a useful signal → H line closes for utility.

## Ckpt / config

CDL from10k seed123 step20000, B head L1H7 (matches CDL teacher), H at L1,
h_context sigma_B, M=64. CPU (frozen model, pairwise teacher precomputed).

## Out of scope (mentioned, deferred)

FiLM (γ,β from H), LoRA-gate hypernetwork, per-pair confidence weighting,
inner-loop meta-update. Start with the boxed additive residual; escalate only if
it shows a non-zero, non-redundant, useful signal.
