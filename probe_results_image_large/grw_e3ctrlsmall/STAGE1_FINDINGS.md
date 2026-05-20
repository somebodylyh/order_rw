# Stage 1 (E3-control-small, v1 Graph-RW) — SEALED conclusion block

**Date sealed:** 2026-05-20. This is the report/paper-ready conclusion for the v1 Graph-RW
continuation stage. Round-2 (higher-level readout) is a separate stage; do not mix its data
with the v1 diagnostic, the Bcov diagnostic, and the raster-specialization findings below.

## Overall verdict

> E3-control-small's attention graph itself contains strong spatial-locality structure, but
> the default progressive Graph-RW v1/v3 readout does not convert that structure into local
> traversal orders; consequently Graph-RW continuation is close to random at both the loss
> level and the attention-drift level. Raster demonstrates that a genuinely local order can
> substantially change training behavior and attention structure, but mainly as
> single-order specialization rather than cross-order generalization.

## Four findings

**Finding 1 — Local B exists.** The E3-control-small block-level attention graph is highly
local: pre-continuation mean_manh ≈ 1.156, P(d≤1) ≈ 0.984. The VQ 2×2 patch-block
representation restored spatial proximity in attention.

**Finding 2 — Default Graph-RW readout fails to use it.** v1/v3 progressive_rw sampled
orders are spatially near-random: v1 graph_rw mean_manh ≈ 4.95 vs random 5.33, raster 1.78
(P(d≤1) 0.15). Root cause: the `support` term (Σ over revealed nodes of B[·,v]), shared by
v1/v2/v3, is a global popularity/coverage signal that dominates the local term; v3's lam/rho
do not touch it. Therefore cont_graph_rw ≈ cont_random on loss (Δcross −0.0001) is the
expected consequence of a non-local readout — NOT a refutation of local B.

**Finding 3 — Raster proves a genuinely local order has a strong effect.** Raster
continuation: val_raster −0.082 vs random, but +~0.04 on every non-raster eval. A truly
local order strongly changes the model, but as order *specialization*, not generalization.

**Finding 4 — Attention drift confirms the mechanism.** Post-train re-extraction (fresh
baseline reproduces 1.156/0.984, frame-consistent): random/graph_rw/shuffled_B/eps015 leave
locality essentially unchanged (1.156/0.984, relFrob ~0.073, rank ρ ~0.94); only raster
moves the graph — sharpening it to perfect 4-neighbor locality (mean_manh 1.000, P(d≤1)
1.000, same_quad 1.000; largest drift relFrob 0.087). So v1 graph_rw ≈ random at the
attention level too, while raster genuinely reshapes the graph.

## Supporting tables
- 5-arm endpoint / Δ-vs-random / aggregate: `SUMMARY.md` Tables 1–3.
- Attention drift: `SUMMARY.md` Task 6 section.
- Readout L1 diagnostic: `readout_diagnostic/metrics.tsv`.

## Paper framing (modality-dependent readout)
Attention-derived graph structure is modality-dependent and the readout must operate in the
matching regime. Text graphs are readiness/dependency-dominant and well matched by
support/readiness Graph-RW; image patch-block graphs are proximity-dominant, and the default
support-based single-step Graph-RW produces near-random spatial traversals — motivating
higher-level / coverage-path / learned readouts for proximity graphs. Not "Graph-RW works
for all modalities," and not "image local B is useless."
