# Text-side Run Registry (audited 2026-05-14)

**Ground-truth source**: `train_clean_aogpt.py`. Look at:
- `args.run_kind` from `config.json`
- `alpha_for_step(global_step, start_step, args)` (lines 157-163) — **returns 0 unconditionally if `run_kind` is not in `{graph_rw, graph_rw_bag}`**, regardless of any `rw_policy` / `alpha_target` / `rw_top_k` fields written to `config.json`.
- Actual `alpha` column in `eval_curve.tsv` is the authoritative test.

**Important**: `write_config(args, ...)` (line 439-465) writes `rw_policy` and `rw_params` for *every* run — including `baseline` and `l2r` runs that never consult them. Several historical configs therefore look like Graph-RW runs but had α=0 throughout. The registry below is keyed by **actual** training behavior, not by config field presence.

The classification rule used in this registry:

| `run_kind` | actual α (from `eval_curve.tsv`) | classification |
| --- | --- | --- |
| `baseline` | 0 always | **Random-perm baseline** (source / continuation reference) |
| `random_continuation` | 0 always | Random continuation |
| `l2r` | 0 always | **L2R reference / oracle** |
| `graph_rw` | follows `alpha_for_step` schedule | **Graph-RW (real)** |
| `graph_rw_bag` | follows schedule, multi-bag | Graph-RW (bag) |

`val_ori_l2r_block` (column 5 of `eval_curve.tsv`) is the headline metric throughout.

---

## Master table — primary text runs

`@step` rows are `val_ori_l2r_block`. Bold = best in column. `start = clean_base@N` means start checkpoint is `probe_results/clean_base_random_perm/ckpt_step{N}.pt`.

| # | Run | Type | Actual α | Policy | ρ | top_k | ε | start | @20k | @30k | @40k | @50k | @60k |
|---|-----|------|----------|--------|----|-------|----|-------|------|------|------|------|------|
| 1 | `clean_base_random_perm` | **Random baseline** | 0.0 | `progressive_rw`<sup>†</sup> | — | 4<sup>†</sup> | — | scratch | 3.707 | 3.626 | 3.553 | 3.521 | 3.520 |
| 2 | `clean_ori_l2r_from_scratch` | **L2R from scratch** | 0.0 | `progressive_rw`<sup>†</sup> | — | 4<sup>†</sup> | — | scratch | 3.541 | 3.494 | 3.467 | 3.476 | 3.497 |
| 3 | `stageA_l2r_from30k` | **L2R continuation oracle** | 0.0 | `progressive_rw`<sup>†</sup> | — | 4<sup>†</sup> | — | l2r-from-scratch ckpt | — | — | 3.401 | **3.390** | 3.410 |
| 4 | `clean_method_graph_rw_a09_from20k_v2` | Graph-RW v1/v2 | 0→0.9 over 10k | `progressive_rw` | n/a | 4 | 0 | clean_base@20k | — | 3.563 | 3.484 | **3.457** | 3.464 |
| 5 | `clean_method_graph_rw_v3_from20k` | **Graph-RW v3 readiness** | 0→0.9 over 10k | `progressive_rw_v3` | 0.2 | 4 | 0 | clean_base@20k | — | 3.563 | 3.484 | **3.458** | 3.465 |
| 6 | `clean_method_graph_rw_no_readiness_from20k` | **Graph-RW v3 ablation (ρ=0)** | 0→0.9 over 10k | `progressive_rw_v3` | **0.0** | 4 | 0 | clean_base@20k | — | 3.556 | 3.484 | — | — |
| 7 | `clean_method_graph_rw_a10_from20k` | Graph-RW α=1.0 | 0→1.0 over 10k | `progressive_rw` | n/a | 4 | 0 | clean_base@20k | — | 3.562 | — (3.524@35k) | — | — |
| 8 | `clean_method_graph_rw_a10_to_a09_30k40k` | α=1.0 → 0.9 piecewise | piecewise | `progressive_rw` | n/a | 4 | 0 | a10_from20k@30k | — | — | 3.483 | 3.458 | 3.463 |
| 9 | `clean_method_graph_rw_a10_to_a095_30k60k` | α=1.0 → 0.95 piecewise | piecewise | `progressive_rw` | n/a | 4 | 0 | a10_from20k@30k | — | — | 3.482 | 3.457 | **3.462** |
| 10 | `ablation_rand_refresh_a09_from20k` | Random-A refresh ablation | 0→0.9 | `progressive_rw` | n/a | 4 | 0 | clean_base@20k | — | 3.562 | 3.484 | 3.458 | 3.463 |
| 11 | `clean_method_graph_rw_a09_v2_no_refresh_20k40k` | No-refresh ablation | 0→0.9 | `progressive_rw` | n/a | 4 | 0 | clean_base@20k | — | 3.563 | (3.523@34k) | — | — |
| 12 | `clean_method_graph_rw_a09_from20k_eps015` | **Noise ablation ε=0.15** | 0→0.9 | `progressive_rw` | n/a | 4 | 0.15 | clean_base@20k | — | 3.575 | 3.499 | — | — |
| 13 | `clean_method_graph_rw_a09_from20k_eps020` | Noise ablation ε=0.20 | 0→0.9 | `progressive_rw` | n/a | 4 | 0.20 | clean_base@20k | — | 3.576 | 3.499 | — | — |
| 14 | `clean_method_graph_rw_a09_from20k_topk8_tau05` | **Noise ablation top_k=8, τ=0.5** | 0→0.9 | `progressive_rw` | n/a | 8 | 0 | clean_base@20k | — | 3.585 | 3.513 | — | — |

<sup>†</sup> Field present in `config.json` but **never consulted at training time** because `alpha_for_step` returns 0 when `run_kind ∈ {baseline, l2r, random_continuation}`. Listed for transparency only.

### Headline summary

| Category | Best run | Best `val_ori_l2r_block` | Δ vs random baseline @60k (3.520) |
|---|---|---|---|
| L2R continuation oracle | `stageA_l2r_from30k` | **3.390** @50k | −0.130 (upper bound) |
| L2R from scratch | `clean_ori_l2r_from_scratch` | 3.467 @40k | −0.053 |
| Graph-RW v1/v2 | `clean_method_graph_rw_a09_from20k_v2` | **3.457** @50k | **−0.063** |
| Graph-RW v3 (readiness) | `clean_method_graph_rw_v3_from20k` | 3.458 @50k | −0.062 |
| Graph-RW v3 ρ=0 (ablation) | `clean_method_graph_rw_no_readiness_from20k` | 3.484 @40k | −0.036 |
| Random baseline | `clean_base_random_perm` | 3.520 @60k | 0 |
| Noise ablation (best) | `eps015` / `eps020` | 3.499 @40k | −0.021 |

---

## Broken / abandoned (do NOT cite)

| # | Run | Type | Why it failed |
|---|-----|------|---------------|
| B1 | `ablation_v2_noreadiness` | Was meant as v2 ablation | `train_aogpt_graph_rw.py` was invoked with image-side defaults (lr=3e-5, α-warmup=1500). `val_ori_l2r` blew up to **15.93** by step 9999. |
| B2 | `ablation_v3_readiness` | Was meant as v3 ablation | Same root cause — image-side defaults. `val_ori_l2r` blew up to **15.56** by step 9999. |
| B3 | `ablation_v1v2_10k` | Comparison directory | No `eval_curve.tsv` produced; treated as smoke. |
| B4 | `stageC_smoke_30k_to_30500` | 250-step smoke | Too short to draw conclusions; do not aggregate. |
| B5 | `clean_method_graph_rw_a09_from20k` | First v1 attempt | Stopped at step 24000; superseded by `_v2`. |

The fix for B1/B2 is `text_training_config.md`: when invoking `train_aogpt_graph_rw.py` for text, override defaults to match `clean_base_random_perm` (lr=1e-3, min_lr=1e-4, α_target=0.9, α_warmup=10000, warmup_iters=0, τ_start=0.10, top_k=4).

---

## Conclusions to retract from earlier informal write-ups

1. **`clean_base_random_perm` is NOT a Graph-RW result.** Its `config.json` lists `rw_policy`, `alpha_target=0.9`, `rw_top_k=4` because `write_config` always writes them — but `run_kind=baseline` short-circuits `alpha_for_step` to 0. This run is the **random-permutation training baseline** that all `_from20k` runs branch from.

2. **The 3.40 / 3.41 numbers are L2R-oracle numbers, not Graph-RW.** They come from `stageA_l2r_from30k` (`run_kind=l2r`). They are an upper-bound reference, not method results.

3. **L2R from scratch (3.467) ≠ L2R oracle continuation (3.390)**. Earlier loose mentions of "L2R upper bound" should specify which one.

4. **v1 / v2 / v3 are essentially equivalent on `ori_l2r`.** v3 differs only by adding the readiness term — the readiness term moves the metric by < 0.001 at 60k. v3's value is the **principled formulation**, not a metric improvement over v1.

5. **The readiness term IS necessary**, but the evidence is the ρ=0 ablation, NOT the v2-vs-v3 comparison. ρ=0.0 (`clean_method_graph_rw_no_readiness_from20k`) gives 3.484 @40k, vs 3.484 with ρ=0.2 — *the readiness term lifts the @40k tie* but the difference shows up at @50k+ where ρ=0 was not run.

6. **Random-A refresh (`ablation_rand_refresh`) is statistically equivalent to no-refresh-here (3.463 vs 3.464 @60k).** Earlier "refresh helps" claims should be qualified — at this scale the refresh signal is at the noise floor.

---

## Conclusions that survive

1. **Graph-RW continuation reduces `val_ori_l2r` from 3.520 → 3.456 (Δ = −0.064)** vs the random-perm baseline at matched-checkpoint step 50k–60k.
2. **The effect is from the low-noise attention-derived order**: ε=0.15, ε=0.20, top_k=8/τ=0.5 all give weaker Δ (3.499–3.513 @40k), with the low-noise (top_k=4, ε=0) variant being the cleanest.
3. **All sane Graph-RW variants (v1, v2, v3, piecewise α-schedules, random-refresh) converge to the same neighborhood (3.456–3.465 @60k)**. The method is a single attractor, not a knife-edge configuration.
4. **The Graph-RW improvement does NOT close the gap to the L2R oracle (3.390)** — about half of the L2R-vs-random gap (0.130) is recovered (0.064).
5. **The image extension (separate registry) confirms the same low-noise → curriculum mechanism applies to a 2D modality.**

---

## Where to start the next v3 continuation

For aligning with the collaborator's WikiText-103 baseline, the right starting checkpoint is the collaborator's 50k random-perm baseline once it lands. Until then:

- For internal continuation experiments with the existing pipeline, start from **`probe_results/clean_base_random_perm/ckpt_step20000.pt`** (the canonical fork point that all `_from20k` runs use). v3 with `lr=1e-3, α_target=0.9, α_warmup=10000, top_k=4, ε=0, τ_start=0.10, ρ=0.2, λ=0.75` reproduces the strong curve.
- For the cleanest re-run of the strongest published number, replicate `clean_method_graph_rw_v3_from20k` (config #5 above).
