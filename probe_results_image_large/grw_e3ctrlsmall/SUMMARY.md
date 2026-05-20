# Image VQ Graph-RW Continuation on E3-control-small — 5-arm RESULT

**Date:** 2026-05-20
**Source ckpt:** E3-control-small l4h8e256 patch2x2 best-val (iter 16250, val=7.463).
**Continuation:** 5000 steps/arm, eff-batch 256, lr 1e-4, α=0.9 (warmup 1000), B fixed,
float32 eval, eval_interval=500. Physical block orders remapped to model frame via
`inverse_block_perm` (baseline trained with permute_data=True).

> **Read this first — interpretation口径.** This run is a **readout-policy diagnostic**,
> NOT a verdict on local-B curriculum utility. The `cont_graph_rw` arm uses
> `progressive_rw` **v1** (not v3, not the adaptive rule). Independently measured, the v1
> Graph-RW orders are spatially **near-random** (mean_manh≈4.95 vs random 5.33, raster
> 1.78; P(d≤1)=0.15). So the "graph_rw" arm did not actually train on spatially-local
> orders. Any result of the form "graph_rw ≈ random" therefore means **the v1 readout
> failed to convert B's locality into a local order**, NOT that local attention structure
> is useless. locality is one structural diagnostic; the final criterion remains task loss
> / cross-order robustness / sample quality.

## Table 1 — endpoint loss @5000 (lower=better)

| arm | train | random | raster | rw_top4 | rw_eps015 | rw_topk8 |
|---|--:|--:|--:|--:|--:|--:|
| random | 7.2470 | 7.2426 | 7.2272 | 7.2455 | 7.2476 | 7.2483 |
| graph_rw | 7.2770 | 7.2482 | 7.2248 | 7.2417 | 7.2501 | 7.2460 |
| raster | 7.1841 | 7.2839 | **7.1451** | 7.2840 | 7.2846 | 7.2857 |
| shuffled_B | 7.2695 | 7.2464 | 7.2347 | 7.2470 | 7.2498 | 7.2511 |
| eps015 | 7.2764 | 7.2454 | 7.2277 | 7.2458 | 7.2478 | 7.2488 |

## Table 2 — Δ vs cont_random (negative = better than random baseline)

| arm | random | raster | rw_top4 | rw_eps015 | rw_topk8 |
|---|--:|--:|--:|--:|--:|
| graph_rw | +0.0056 | −0.0024 | −0.0038 | +0.0025 | −0.0023 |
| raster | +0.0413 | **−0.0821** | +0.0385 | +0.0370 | +0.0374 |
| shuffled_B | +0.0038 | +0.0075 | +0.0015 | +0.0022 | +0.0028 |
| eps015 | +0.0028 | +0.0005 | +0.0003 | +0.0002 | +0.0005 |

## Table 3 — aggregate (structured={raster,rw_top4,rw_topk8}, noisy={random,rw_eps015})

| arm | cross_avg | struct_avg | noisy_avg | Δcross | Δstruct | Δnoisy |
|---|--:|--:|--:|--:|--:|--:|
| random | 7.2422 | 7.2403 | 7.2451 | +0.0000 | +0.0000 | +0.0000 |
| graph_rw | 7.2422 | 7.2375 | 7.2492 | −0.0001 | −0.0028 | +0.0040 |
| raster | 7.2567 | 7.2383 | 7.2843 | +0.0144 | −0.0021 | +0.0391 |
| shuffled_B | 7.2458 | 7.2443 | 7.2481 | +0.0036 | +0.0039 | +0.0030 |
| eps015 | 7.2431 | 7.2408 | 7.2466 | +0.0009 | +0.0004 | +0.0015 |

## Findings

1. **graph_rw ≈ random on cross-order (Δcross = −0.0001).** A faint structured
   specialization (Δstruct = −0.0028) is offset by a noisy-order penalty (Δnoisy = +0.0040).
   Net: no meaningful benefit. All magnitudes are at noise level (~0.003 nat).

2. **Controls confirm no real-B effect at the loss level.** shuffled_B (Δcross +0.0036)
   and eps015 (Δcross +0.0009) are also ≈ random. graph_rw is only ~0.007 better than
   shuffled_B on struct_avg — a faint hint that real B does *something*, but within noise,
   not conclusive. This is the **expected** outcome given Finding 0 (the readout produces
   near-random spatial orders, so the real-B and shuffled-B arms train on similarly
   non-local orders).

3. **raster is a pure single-order specialist.** It crushes its own eval order
   (val_raster Δ=−0.082) but pays everywhere else (Δnoisy +0.039, val_rw_top4 +0.039),
   so its struct_avg is only ≈ random. This proves a *genuinely* local order has a strong
   training effect — but toward **specialization, not generalization**.

## Verdict (decision matrix)

Outcome **(b) "structure is policy-redundant"** at face value (graph_rw ≈ random, raster
shows a gap) — **but corrected by the readout diagnostic**: graph_rw ≈ random because its
v1 readout never produced spatially-structured orders (manh≈4.95). The proper conclusion:

> The default progressive Graph-RW (v1/v3) readout does not convert E3-control-small's
> highly local attention graph (P(d≤1)=0.984) into local traversal orders; its sampled
> orders are spatially near-random, which explains the absence of a curriculum effect and
> the failure to separate from the shuffled-B / noisy controls. raster demonstrates that a
> genuinely local order produces a strong (specialization) effect. Whether a genuinely
> local, B-*driven* order generalizes rather than merely specializes is the Round-2 question.

## Round-2 (designed, not yet run)

readout diagnostic (L1, CPU) already done — see `readout_diagnostic/metrics.tsv`:
random 5.35 / v1_graph_rw 4.95 / local_greedy 3.31 / raster 1.78 / serpentine 1.00 /
hilbert 1.00 / Bcov_distance_only 1.13 / **Bcov_balanced 2.41 (94% different from
distance-only → B materially drives)** / Bcov_Bdominant 3.06.

Round-2 continuation matrix: `random / v1_graph_rw (failed-readout control) / hilbert
(generic-locality upper) / Bcov_balanced (B-driven main method) / raster (upper ref)`.
Core training comparison: **Bcov_balanced vs hilbert** — does B's directed structure add
anything over a generic space-filling curve? locality is a diagnostic, not the objective;
decide by val loss / cross-order robustness / sample quality.

## Pending
- Task 6 (post-train A re-extraction + dual-level drift metric on each ckpt_final.pt) —
  GPU ~25 min, not yet run. Records how each arm's attention graph drifted from the
  pre-continuation baseline (descriptive, no hard pass/fail).
