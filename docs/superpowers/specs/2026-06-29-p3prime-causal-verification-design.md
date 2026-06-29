# P3′ — Causal Verification of the L0 Global Physical-Order Signal

**Date:** 2026-06-29 · **Branch:** `attn-order-alternating` · **Compute:** no new training (CPU; frozen ckpts)
**Subtitle:** *Causally separating the fixed slot→physical base map from the content-modulation residual.*

**Foundations (read first):**
- `analyses/canonical_order_README.md` — the canonical physical readout + four sanities (the L0 carrier emerges, is real, above the destroyed floor, not an inv_perm artifact).
- `analyses/physical_signal_source_README.md` — P2 verdict **B+** (dominant fixed-layout slot→physical map + smaller content-modulation residual), 3 seeds.
- Memory `physical-order-signal-20260629.md` — framing/naming/readout LOCK.

---

## 0. What P3′ is (and is not)

P3′ is the **causal** follow-up to the **correlational** P2 result. P2 established, by held-out
predictability and variance decomposition, that the L0 global physical-order carrier's attention
is *dominated by a fixed slot→physical map with a measurable content residual* (**B+**). P3′ asks
whether that decomposition holds **causally**: when we intervene on the carrier's computation, does
the base map collapse where P2 says it lives (the positional / QK-geometry path), and does the
content residual move where P2 says it lives (content at fixed slots)?

> **P3′ does not attempt to convert B+ into C. It causally validates the B+ decomposition found in
> P2: a dominant fixed slot→physical base map plus a smaller content-dependent residual.**

Turning B+ into C — proving the recovery *could* be content-driven if the model had seen more than
one layout — requires **multi-layout training** and is explicitly **out of scope** (deferred to P4).

### Locked naming
The signal is the **L0 global physical-order carrier** (a.k.a. *global physical-order signal*).
Never "content-bound carrier" (disproven by P2) and never a bare "order carrier" (re-confuses the
slot-scaffold vs physical-order distinction).

---

## 1. Non-negotiable measurement rule

**Primary metric (the only one that scores a verdict):** the canonical physical readout —
strict 65-node None-separated graph, **random reveal order**, σ_model → posthoc-inv → scored vs
physical L2R, method **C-D+L**, with destroyed / entry-shuffled floor anchors. Code path:
`analyses/canonical_reanalysis.canonical_scan` (and the sealed
`scripts/search_none_separated_65_heads.py` for cross-checks). **Forbidden:** model-frame
identity-reveal τ (a slot-order scaffold tautology — the ③/A/⑤ confound).

**Intervention rule (the red line that shapes every test in P3′):**

> **Because the canonical readout is a function of the head's own attention map, interventions on
> the head output / OV path are degenerate for same-head τ. P3′ therefore uses Q/K, QK-score, and
> input-to-QK interventions as primary causal tests.**

Concretely:
- **Allowed primary interventions:** the *input* to a carrier head's Q/K (residual decomposed into
  position vs content components), the head's Q/K vectors, the pre-softmax QK scores.
- **Forbidden as load-bearing evidence:** anything on the head's **output** path — OV, `c_proj`,
  mean-ablating the head's contribution — *read on the same head's τ*. Such an ablation cannot
  change `softmax(QKᵀ)`, so same-head τ is unmoved; it is degenerate, not evidence.
- **Also not load-bearing (the A0 caveat below):** directly mean-patching the *same* head's Q/K
  and then reading *that same head's* τ. Since canonical τ is a direct function of that head's QK
  attention map, its collapse is mechanically guaranteed — calibration, not causality.

---

## 2. Targets (carriers per seed, from P2)

| seed | L0 carrier set | character |
|------|----------------|-----------|
| 2    | {2, 3, 4, 5}   | multi-head redundant, early (~step 2000) |
| 42   | {2}            | single-head, late (~step 5000) |
| 123  | {1, 2, 3, 4}   | multi-head redundant (~step 3000–6000) |

Read-step: the converged ckpt used in P2 (the `runs/handoff_overnight` / emergence lineage at the
late step where τ≈1.0). Controls reuse the **L1 slot-order scaffold** heads and a **null-head** set
(heads with |τ|≈0 at the same layer) for selectivity.

---

## 3. P3′-A — Calibration and carrier-redundancy diagnostics (NOT load-bearing)

> **Self-QK patching of the same head being read out is not treated as load-bearing evidence,
> because canonical τ is a direct function of that head's QK attention map. It is used only as
> calibration / floor validation. Non-tautological evidence comes from input-to-QK interventions on
> position / content components (P3′-B / P3′-C) and from cross-head / aggregate redundancy
> diagnostics (A1).**

(中文红线，进 README：对被读出的同一个 head 做 Q/K patch 再观察其自身 τ 下降，**不**作为承重因果证据，因为
canonical τ 本身就是该 head QK attention map 的函数；该实验只用于校准 patch 是否生效和 floor。真正非平凡证据
来自 position/content 的 input-to-QK 干预，以及跨头 / aggregate redundancy 诊断。)

### A0 — Self-QK patch calibration
- **Purpose:** technical calibration only — confirm the patch hook actually reaches the canonical
  attention readout, and establish the τ-collapse floor for B/C interventions.
- **Intervention:** mean-patch the Q/K (and, as a variant, the pre-softmax QK scores) of the *same*
  carrier head being read out.
- **Expected:** `τ_self` collapses toward the destroyed floor.
- **Interpretation:** confirms readout sensitivity to the intended QK intervention. **Does not**
  establish load-bearing causality — the metric is a direct function of the intervened QK.

### A1 — Redundancy / non-self diagnostics
- **Purpose:** test whether the physical-order signal is **redundantly** represented across carrier
  heads (the genuinely non-tautological part of A).
- **Interventions (two distinct kinds — keep them separate in reporting):**
  1. **Head-local patch h → read h′ (≠ h):** patch one carrier head's Q/K, read a *different*
     carrier head's τ. Because L0 heads read the same residual **in parallel**, a head-local QK
     replacement on h generally does **not** alter h′'s Q/K. A null effect here is the *expected*
     signature of **parallel independent copies** — it means redundancy is "many parallel carriers",
     not head-to-head dependence. (A non-null effect would itself be a finding.)
  2. **Aggregate cluster leave-k-out:** form the cluster readout `B_cluster = mean over carrier
     heads`, then read the cluster / aggregate physical τ under leave-one-out, leave-two-out, … of
     the carrier set. This tests the **contribution of multi-head redundancy to the aggregate
     physical signal** and is *not* a self-readout tautology.
- **Interpretation:**
  - multi-head seeds (2, 123): if aggregate τ degrades **gradually** under leave-k-out → redundant
    parallel carriers; if head-local patch h→h′ is null → parallel independent copies.
  - single-head seed (42): its only carrier (L0H2) dominates; removing it collapses the aggregate.
    This contrasts the single-head vs multi-head redundancy structure across seeds.

---

## 4. P3′-B — The fixed base-map causal path (load-bearing core #1)

**Claim under test:** the dominant fixed slot→physical map (the P2 "B base", raw R² 0.85–0.99)
is causally implemented through the **positional / slot-geometry** input to the carrier's Q/K.

- **Intervention:** decompose the residual that feeds the carrier head's Q/K into a **position
  component** and a **content component**, then **ablate / corrupt the position-path component**
  (zero or shuffle the positional / slot contribution) while leaving content fixed. Reuse the
  position-path machinery from ⑤ (`test_pp_pe_ablation.py` — wpe/wtpe zeroing) and the per-head
  QK-recompute-from-residual (`analyses/path_patch_handoff._l1_qkv_from_residual`, generalized
  from L1 to arbitrary layer; see §7 Task 1).
- **Readout:** canonical physical τ of the carrier, and the slot-only R² recomputed on the
  intervened B65 (does the held-out fixed-map predictability drop?).
- **Expected (B+ signature):** the **bulk** of physical τ collapses when the position path is
  removed — the fixed map lives there. The slot-only fixed-map component should lose most of its
  predictive structure.
- **Selectivity:** the same intervention on a **null head** (control 1) and on the **L1 slot
  scaffold** (P3′-D) should not reproduce the carrier's collapse pattern.

---

## 5. P3′-C — The content-residual causal path (load-bearing core #2)

**Claim under test:** the smaller content modulation (the P2 "+", content/floor ratio 2–19) is
causally **content-driven** — not measurement noise.

- **Intervention:** hold the layout and reveal orders **fixed**, and corrupt **content at fixed
  slots** — block-swap / cross-text block replacement / random-token chunk at chosen slots
  (the P2 corruptions in `analyses/physical_signal_source.py`: `block_swap_chunk`,
  `cross_sample_replace`, `random_token_chunk`) — then recompute the carrier's Q/K from the
  corrupted residual.
- **Readout:** the change in the carrier's B65 **residual component** (the cross-text variance above
  the within-text reveal-split sampling-noise floor from P2), and the carrier τ change.
- **Expected (B+ signature):** a **small, head-specific** change — measurably **above** the
  within-text sampling-noise floor, but **much smaller** than the position-path collapse of P3′-B.
  This is the causal counterpart of P2's content-variance / noise-floor ratio.
- **Negative control:** content-invariant synthetic replacement (P2's `synthetic_content_invariant`)
  must move the residual ≈ 0; content-randomized must bound the upper end.

---

## 6. P3′-D — Locus / direction (which path carries the signal)

**Claim under test:** the canonical signal lives in the **QK attention geometry**, not the OV /
output path; and the base-map intervention is **selective** to the L0 carrier.

- **Pre-QK vs post-output:** show that an **input-to-QK** intervention *changes* the carrier's
  attention map (and τ), while an **output / OV / `c_proj`** intervention on the same head does
  **not** change the same head's attention map — demonstrating, on this model, the degeneracy that
  §1 asserts. (This is the empirical justification of the red line, not a load-bearing order claim.)
- **Layer selectivity:** run the P3′-B position-path intervention on **L0 carrier vs L1 slot
  scaffold vs null heads**. Expect the carrier's physical-τ collapse to be specific to L0; the L1
  scaffold's *model-frame* behavior is a different (confounded) phenomenon and must not be scored on
  the canonical metric as if it were the carrier.

---

## 7. Controls (all five mandatory, every seed)

1. **Null-head intervention** — every P3′-B/C intervention repeated on |τ|≈0 heads at the same
   layer; expect no carrier-like collapse (selectivity).
2. **Destroyed / entry-shuffled floor** — the canonical destroyed-control floor (≈0.05–0.13)
   anchors "collapsed" vs "intact".
3. **Randomized-content anchor** — content-randomized corruption upper-bounds the content-residual
   change (P3′-C); synthetic content-invariant lower-bounds it ≈ 0.
4. **Redundancy ladder** — A1 leave-k-out + head-local h→h′, reported per seed (parallel-copy vs
   single-head structure).
5. **Seed consistency** — the full B+ signature must replicate across seeds **2, 42, 123** (with
   42 reported separately as the single-head / weakest-residual case, as in P2).

---

## 8. Expected B+ signature (the verdict shape)

P3′ **confirms B+** iff, consistently across seeds:
- **Position-path ablation (P3′-B) collapses the bulk** of physical τ and the slot-only fixed-map
  R² → the base map is causally positional / QK-geometric.
- **Content corruption at fixed slots (P3′-C) moves a small, head-specific residual** above the
  sampling-noise floor but far below the P3′-B collapse → the "+" is causally content-driven yet
  secondary.
- **Cross-layout non-generalization persists** (consistent with B, inherited from P2's relayout
  collapse 1.0→0.13; not re-litigated, cited as boundary).
- Interventions are **selective** to the L0 carrier (null heads / L1 scaffold do not reproduce it).

Any departure (e.g. position-path ablation leaves τ intact, or content corruption dominates) is a
reportable update to the B+ picture, **not** silently fit to B+.

---

## 9. Can / cannot claim

**P3′ CAN claim:**
- the fixed slot→physical base map is **causally** carried by the positional / QK-geometry input to
  the L0 carrier;
- the content residual is **causally content-driven** (above the sampling-noise floor), not
  measurement noise;
- the signal lives in **QK attention geometry**, not the OV/output path;
- the intervention effects are **selective** to the L0 carrier and **consistent across seeds**.

**P3′ CANNOT claim:**
- that the recovery is content-driven recovery **C** (B+ → C needs **multi-layout training**);
- anything from same-head **output-ablation** or same-head **self-QK patch** as load-bearing
  causality (degenerate / calibration only);
- generalization beyond the single trained layout (the core P2 limitation stands).

A decisive strengthening of the content side remains **multi-layout training + re-running P2/P3′**
(deferred to P4).

---

## 10. Reuse / new code

- **Reuse:** `analyses/path_patch_handoff.py` (`_l1_qkv_from_residual`, `capture_block_input`,
  `mean_ablation_prehook`, `_causal_softmax`), `test_pp_pe_ablation.py` (position-path zeroing),
  `analyses/physical_signal_source.py` (`block_swap_chunk`, `cross_sample_replace`,
  `random_token_chunk`, `valid_edge_mask`, `row_normalize_l1`, `slot_only_r2`,
  `within_text_noise_floor`), `analyses/canonical_reanalysis.canonical_scan` (canonical τ).
- **New (single focused module):** `analyses/p3prime_causal_verify.py` — the L0-generalized
  QK-recompute, the position/content residual split, the A0/A1/B/C/D drivers, and the verdict
  classifier. Plot + README mirror P2's layout.

---

## 11. Recommended task split (~12, for writing-plans)

1. **Generalize `qkv_from_residual` to arbitrary layer** — parametrize `_l1_qkv_from_residual`
   (and `l1_attn_from_residual`) over `layer`; unit-test bit-identical to current L1 path and a new
   L0 path; export `qkv_from_residual(model, layer, x_resid, cond)`.
2. **Residual position/content split** — function returning (position-component, content-component)
   of a carrier head's input residual, reusing the pe-ablation decomposition; test that
   position-zeroed + content-zeroed ≈ original (additive sanity).
3. **A0 self-QK patch calibration** — `selfqk_calibration(ckpt, layer, head)` → τ_self before/after;
   test collapse-toward-floor on a synthetic head.
4. **A1 head-local h→h′** — `head_local_crosshead(ckpt, layer, carriers)` → matrix of patch-h /
   read-h′ τ deltas; test parallel-copy null on independent-head synthetic.
5. **A1 aggregate leave-k-out** — `cluster_leave_k_out(ckpt, layer, carriers)` → aggregate τ vs k;
   test graceful-degradation on redundant synthetic, full-collapse on single-head.
6. **P3′-B position-path ablation** — `basemap_position_ablation(ckpt, layer, carriers)` → τ and
   slot-only R² before/after; test bulk-collapse on a position-driven synthetic.
7. **P3′-C content corruption** — `content_residual_causal(ckpt, layer, carriers)` → residual-change
   vs noise-floor, with synthetic-invariant (≈0) and randomized (upper-bound) anchors; reuse P2
   corruptions; test invariant→0, randomized→large.
8. **P3′-D locus** — `locus_qk_vs_ov(ckpt, layer, head)` → input-to-QK changes attention,
   OV/`c_proj` does not (same-head); test the degeneracy assertion holds.
9. **Controls wiring** — null-head + L1-scaffold variants threaded through B/C/D; destroyed-floor
   anchor; test null-head shows no carrier-like collapse.
10. **Verdict classifier** — `classify_p3prime(...)` → confirms/updates B+ from the §8 signature;
    unit-test the confirm case and at least one departure case.
11. **Per-seed driver + plot** — `run_seed(seed, ...)` over seeds {2,42,123}; `plot_p3prime` mirrors
    `plot_physical_signal_source`; outputs to `runs/p3prime_causal/seed{2,42,123}/`.
12. **README + memory** — `analyses/p3prime_causal_README.md` (verdict, red-line sentences EN+中文,
    can/cannot, multi-layout deferral); update `MEMORY.md` pointer.

---

### Critical sentences to preserve verbatim in code/README
1. "Because the canonical readout is a function of the head's own attention map, interventions on
   the head output/OV path are degenerate for same-head τ. P3′ therefore uses Q/K, QK-score, and
   input-to-QK interventions as primary causal tests."
2. "P3′ does not attempt to convert B+ into C. It causally validates the B+ decomposition found in
   P2: a dominant fixed slot→physical base map plus a smaller content-dependent residual."
3. "Self-QK patching of the same head being read out is not treated as load-bearing evidence,
   because canonical τ is a direct function of that head's QK attention map. It is used only as
   calibration/floor validation."
