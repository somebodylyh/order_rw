# 09 — Next Steps

Date: 2026-06-17

## Immediate (This Week)

### 1. Update CLAIMS_LOCK
- Add strict 65-node claim to `05_FINAL_CLAIMS_LOCK.md`.
- Distinguish from old B0 anchored-controller claim.
- Mark Tier-1 items as VERIFIED, Tier-2/3 as PENDING.

### 2. Paper Figure
- Generate a "model-frame → posthoc physical" comparison figure for L0H3 (best case) and L0H7 (failure case).
- Show: raw model-frame B65 heatmap → rollout sigma_model → posthoc sigma_phys → tau vs L2R.
- This replaces/upgrades the old oracle-remapped figure.

### 3. Update Main Table
- If the main paper table currently reports B0/B1 anchored results, add a row or footnote for strict 65-node label-free results.
- Key numbers: tau=1.000 for L0H1–L0H4 (collaborator), destroyed |τ| ≈ 0.05.

### 4. Wednesday Presentation
- Use `08_boss_update_summary.md` as slide notes.
- Key slide: "Strict 65-node Protocol: Before vs After" comparison.
- Key slide: "Gate Distribution: 20/256 pass, L0H7 fails" as evidence of specificity.

## Short-Term (Next 1–2 Weeks)

### 5. Multi-Seed Training Verification
- **Gap**: All current evidence comes from single-seed training runs (clean_base seed42, collaborator seed?).
- **Plan**: Run strict 65-node scan on at least one additional training seed's checkpoints.
- **Priority**: Medium. Cross-run evidence already exists (clean_base ≠ collaborator), but same-protocol multi-seed would strengthen the claim.

### 6. M=20 Strict LF on Clean Base
- Current clean_base stability uses M=8 (lightweight). Spotcheck confirmed equivalence at M=8.
- Run M=20 strict LF at 2–3 key steps (10k, 30k, 60k) for statistical robustness.
- **Priority**: Low. M=8 equivalence already confirmed; M=20 would reduce sampling noise.

### 7. Continuous Stream Model Verification
- Current evidence is on fixed-chunk models. The continuous-stream ori-L2R run (seed=124) may have different head dynamics.
- Run strict 65-node scan on continuous model checkpoints.
- **Priority**: Low-Medium. Important for claiming "not specific to fixed-chunk training."

## Medium-Term (2–4 Weeks)

### 8. 317M Model (16L) 65-Node Scan
- B0 results exist (6/256 heads |τ|>0.9). No 65-node results.
- **Priority**: Low. Scale evidence exists for B0; 65-node would be confirmatory.

### 9. Image Model 65-Node Scan
- Image side uses different metrics (D_manh, not tau). The 65-node protocol may need adaptation.
- **Priority**: Future. Text side should solidify first.

## NOT To Do Now

- ❌ Do NOT start new training runs.
- ❌ Do NOT modify the frozen-hook to use 65-node teacher (separate engineering decision).
- ❌ Do NOT run exhaustive all-method all-head sweeps on every checkpoint (existing data sufficient).
- ❌ Do NOT change the main paper story from "step savings" to "discovery mechanism" — the 65-node result is supporting mechanism evidence, not the main claim.
