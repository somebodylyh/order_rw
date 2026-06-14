# CLAIMS LOCK — 2026-06-13 (updated 2026-06-14)

> **2026-06-14 update**: Verification pass complete. g_β sanity raw JSON saved (U2 closed). Non-L2R +0.11 claim removed (n=4, not reproducible). CDL teacher seed-123 matched runs launched (U1 pending, ETA ~8h overnight). Full verified package at `reports/evidence_package_20260613_verified/`.

> Text-side evidence consolidation.  
> Final numbers flagged [TO-VERIFY] require cross-check before external use.  
> All claims below are seed2/seed42 on Wikitext-103, 4L/8H/384d, **unless stated otherwise**.  
> ori-L2R is **reference**, NOT upper bound.

---

## CAN CLAIM (solid, multi-line evidence)

### 1. g_β reads structured B, not a constant L2R prior

| B source | g_β τ_vs_L2R | Verdict |
|----------|:--:|------|
| Real B (random-order model) | **+0.9675** | strong readout |
| Gaussian B (μ=0, σ matched) | −0.0034 | no signal |
| Entry-shuffled B | +0.0165 | no signal |
| Row+col shuffled B | +0.0127 | no signal |
| Zero B | 1.0 | tie-breaking artifact (margin=0) |
| Gaussian family pairwise | 0.0003 | no fixed prior |

→ **Cannot be reduced to "just memorizing the L2R order."** Primary evidence: real B τ=0.97 vs destroyed B τ≈0. Zero B τ=1.0 is CDL tie-breaking (margin=0 — all edges indistinguishable).

⚠️ Non-L2R subset analysis excluded from main claim (n=4, statistically inconclusive). The destroyed/randomized B controls alone suffice.

Ref: `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json` (verified 2026-06-14)

### 2. Random-order AOGPT attention contains physical-order-readable signal

| Diagnostic | Result |
|------------|--------|
| CDL τ_vs_L2R @ step10k | ≈1.0 (multiple heads) |
| Per-head τ_vs_L2R | individual heads reach ±1.0 |
| Shuffled-L2R control | τ falls to ~0 (wrong order destroys signal) |
| 317M scale diagnostic | sparse order-bearing heads exist (6/256) |

→ **Attention graphs encode the data's intrinsic structure even when input order is randomized.**

Ref: `probe_results/clean_base_random_perm/head_scan_10k.json`

### 3. Frozen g_β accelerates training across 2 seeds and multiple resume points

| Seed | Head | from10k Recovery | from20k | from40k |
|------|------|:--:|:--:|:--:|
| seed2 | L0H2 | **160%** | **155%** | 109% |
| seed42 | L0H4 | **105%** | **101%** | 66% |

\* Recovery = fraction of random→ori-L2R gap closed. >100% = beats L2R.
\*\* Random baselines: seed2=3.445, seed42=3.466 (both continuous).
\*\*\* L2R references: seed2=3.370, seed42=3.340 (both continuous).

Step saving @3.47:
- frozen_β from10k: **39–42%** (beats L2R at 38%)
- frozen_β from20k: **33–36%**
- CDL teacher from10k: **48%**

→ **Training acceleration is replicated, not single-seed artifact. Frozen_β from10k beats ori-L2R in both recovery and step saving.**

Ref: `probe_results/frozen_beta_seed2_from*_l0h2/`, `probe_results/frozen_beta_seed42_from*_l0h4/`

### 4. g_β@10k transfers to later checkpoints (same seed)

Single g_β trained at step 10k works when plugged into the model at steps 20k, 40k — the attention graph structure is stable enough for the readout to remain effective.

Ref: frozen_beta multi-start runs all use the same g_β@10k

### 5. Phenomenon survives granularity and scale diagnostics

| Ablation | Result |
|----------|--------|
| 32/64/128 block aggregation | τ > 0.96 all three |
| 32/64/128 training granularity | per-head τ = ±1.0 all three |
| 317M (16L/16H/1024d) | |τ| > 0.9 heads exist (6/256), signal present |

→ **Not a 64-block artifact. Not a 47M toy artifact.**

Ref: `daily-summary-20260613.md`

---

## CANNOT CLAIM YET (needs more evidence or careful qualification)

### 1. One g_β transfers across seeds or heads

Current evidence: g_β is trained per-seed, per-head. We have NOT demonstrated that a g_β trained on seed2/L0H2 works for seed42/L0H4.

**Correct framing**: "Pipeline works with seed-dependent selected heads."

### 2. ori-L2R is an upper bound

ori-L2R is a **reference point**, not a proven upper bound. The model could in principle discover orders better than L2R for its own loss.

**Correct framing**: "Recovery = fraction of random→ori-L2R reference gap closed."

### 3. Shuffled-L2R encodes no order at all

Shuffled-L2R control shows the model does NOT recover the *physical* order when trained on a fixed wrong (shuffled) order. This could mean:
- The model encodes the shuffled order (just not physical L2R)
- The model encodes no order

We have evidence the model does NOT encode physical order under shuffled training, but we have NOT proven "no order at all."

**Correct framing**: "Shuffled-L2R control does not recover physical order."

### 4. Fully label-free head selection

Audition provides a label-free head selector, but we have NOT yet shown that:
- label-free selection ≈ oracle τ-based selection
- the pipeline is end-to-end label-free

**Correct framing**: "Head selected via cheap diagnostic signals; oracle τ used for post-hoc validation."

### 5. Image side solved

Image results are diagnostic only (D_manh, raster comparison). No frozen hook training on image models yet.

### 6. 317M frozen hook acceleration

Large model shows τ > 0.9 diagnostically but no hook training has been run.

---

## NEED BOSS DECISION

### 1. Paper scope
- **Option A**: Text-focused paper (AAAI-style), image as supporting/limitation section
- **Option B**: Multimodal paper (text + image), requires image hook experiments

### 2. Compute allocation
- **Option A**: Third seed (statistical completeness, cheaper)
- **Option B**: 317M frozen hook (scale demonstration, expensive, higher risk)
- **Option C**: Both (need compute budget approval)

### 3. Method positioning
- **Option A**: "Attention-derived order controller for faster AO-GPT training" (systems/optimization framing)
- **Option B**: "Emergent data-structure signal in attention graphs" (understanding/analysis framing)

### 4. g_β → w\* automatic mapping (Stage 3)
- Whether to invest in learning a direct g_β → order-weight mapping before paper submission

---

## EVIDENCE STATUS BY SECTION

| Section | Status | Key gap if any |
|---------|:--:|------|
| Mechanism (g_β reads B) | 🟢 SOLID | — |
| Training acceleration | 🟢 SOLID | exact recovery% [TO-VERIFY] |
| Protocol / confounds | 🟢 SOLID | — |
| Granularity robustness | 🟢 SOLID | — |
| Scale robustness (317M) | 🟡 DIAGNOSTIC ONLY | no hook training |
| Label-free selection | 🟡 AUDITION EXISTS | not end-to-end verified |
| Cross-seed g_β | 🔴 NOT CLAIMED | — |
| Image | 🔴 DIAGNOSTIC ONLY | — |

---

## NEXT ACTIONS (post-boss)

1. If boss says "converge on text paper": write paper, finalize tables, generate report figures
2. If boss says "add image": redesign image teacher, run image hook
3. If boss says "strengthen statistics": add third seed
4. If boss says "demonstrate scale": run 317M hook

**Do NOT start any of these without explicit direction.**
