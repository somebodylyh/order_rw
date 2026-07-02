# Repo Cleanup Report — order_lyu

**Generated:** 2026-07-02
**Repo:** `/home/admin/lyuyuhuan/order_lyu` (branch `p5-direct-nll-routing`, HEAD `d487f8a`)
**Principle:** inventory + classify + *minimal move*. **No deletions, no logic
refactor, no file moves executed.** This report + `cleanup_plan.sh` (dry-run) are
proposals for human review.

Goal: three layers — **mainline (reproducible main experiments)** + **diagnostics
(traceable analysis)** + **history (archived, off the main line)** — so you and
collaborators share ONE baseline.

---

## 0. Scope (what this cleanup touches)

**IN scope** (this project's tracked code):
- `analyses/` — 84 tracked `.py`
- `scripts/` — 75 tracked `.py` + ~90 `.sh`
- `block_lo_arm_order_network/` — 358 tracked `.py` (the AO-GPT lib + order/CDL/gβ + eval + tests + many one-off analysis scripts)
- `docs/` — specs/plans (already organized under `docs/superpowers/`)
- `configs` — `block_lo_arm_order_network/configs/{text,image,image_large}/`

**OUT of scope** (separate nested `.git` repos, or gitignored artifacts — do NOT reorganize):
- Nested repos: `AO-GPT-MDM/` (gitignored), `nanogpt-learned-order/` (own `.git`), the `.worktrees/*`
- Other people's / other-line projects: `chenhe_nanogpt_learned_order/`, `by/`, `cv/` (gitignored), `image_order/`
- Output/artifact dirs (gitignored or untracked): `probe_results*/`, `outputs/`, `wandb/`, `logs/`, most of `runs/` (82 tracked JSONs are result records), `figures/`
- Binaries already gitignored: `*.pt/*.bin/*.npy/*.npz/*.pth/*.safetensors`

---

## 1. MAINLINE (keep in import path — supports shared baseline + paper core)

### Entrypoint + core (block_lo_arm_order_network/)
The single production entrypoint is **`train_clean_aogpt.py`**; its import closure
is the mainline core:
- `train_clean_aogpt.py` — training entry (run-kinds: baseline, l2r, random_continuation, graph_rw, frozen_beta, cdl_teacher, direct_policy; + the decoupled OrderHead PG path)
- `clean_training_protocol.py` — **frame conversion + data protocol** (`CleanPermutation`, `physical_blocks_to_model_token_order`, `model_blocks_to_physical_blocks`, block/reveal, permutation). **Appears to be the single source** of frame conversion (verify no `analyses/` duplicate in the exhaustive pass).
- `directed_graph_policy.py` — graph-RW order policy (`build_directed_graph`, `sample_order`)
- `attn_order_mlp_policy.py` — MLP/position order policies
- `training_utils.py` — data loading, `SEQ_LEN`, chunk utils
- `model.py` / the AO-GPT model modules — backbone
- `per_head_order_scan.py` — attention→A extraction primitives used by providers

### Order / controller (block_lo_arm_order_network/batch_readout/)
- `model.py` (`FlattenReadout`, `NodewiseReadout`) — gβ readout архитектures
- `l0_dynamic_gbeta.py` (`L0DynamicGBeta`) — multi-head gβ
- `integration_hook.py` (`FrozenBetaHook`, `_build_from_config`) — gβ loader
- `hook_order_provider.py` (`HookOrderProvider`, head-gated providers, extraction)
- `frozen_gbeta_hook.py` (`FrozenGBetaModelFrameBlockProvider`, extraction)
- `cdl_order_provider.py` — CDL teacher order provider
- `direct_order_provider.py` — non-learned L0 policies
- `label_free_cdl_teacher.py` — **CDL teacher** (rollout, soft pairwise)
- `soft_pairwise.py` — pairwise BCE loss + accuracy
- `train_l0_dynamic_gbeta.py` — L0DynamicGBeta CDL trainer
- `train_offline.py`, `dataset_batch.py`, `pl_sampling.py`, `eval_metrics.py`, `loss.py`, `order_distillation_losses.py` — readout training/eval infra
- `orderhead_pg.py` — **(new, Phase-1/2)** PG helpers (grad scorer, batch/GRPO advantage)

### Controller / producer (analyses/)
- `order_head_module.py` — embedded OrderHead wrapper (V3)
- `v3_group_credit.py`, `v3_group_trainer.py` — group-credit PG infra
- `p5_utility_controller.py`, `p7_gbeta_policy.py` — controller + PL policy (`sample_pl`, `GBETA_CKPT`, HEAD)
- `gbeta_cdl_pretrain.py` — **(new)** multi-head CDL pretrain orchestrator
- `uniform_label_free_v1.py` — label-free head selection (Stage A) + readout training
- `build_l0_dynamic_gbeta_dataset.py`, `add_consensus_order_to_gbeta_dataset.py` — CDL dataset builders

### Producer script (scripts/)
- `train_nodewise_gbeta.py` — **(new)** single-head NodewiseReadout CDL producer (canonical gβ)

### Configs (MAINLINE — shared baselines exist as Python config modules)
`block_lo_arm_order_network/configs/text/`:
- `wikitext103_seq256_block64_l2r_reference.py` ← L2R baseline
- `wikitext103_seq256_block64_random_base.py` ← random-order baseline
- `wikitext103_seq256_block64_v3_readiness.py`, `..._v3_no_readiness.py` ← controller variants

**FINDING:** the shared *text* baselines you listed (`random_warmup_l2r_keepopt`,
`random_warmup_l2r_resetopt`) do **not** exist as configs — those runs are launched
via **CLI args** (e.g. the `--resume-ckpt ... --run-kind l2r ...` chain), not config
files. Recommendation: add `configs/baselines/` (or extend `configs/text/`) with
explicit budget-matched baseline configs so collaborators stop hand-writing CLI
args. This is a **NEW artifact to create**, not a move. (Highest-value cleanup for
"shared baseline".)

### Docs (already organized)
`docs/superpowers/specs/*` + `docs/superpowers/plans/*` — keep. The decoupled-init
+ GRPO Phase-2 specs/plan live here.

---

## 2. DIAGNOSTIC (not on the training path; produce paper figures/metrics — move to `scripts/diagnostics/` or keep, but tag)

High-confidence by pattern (needs a one-line open to confirm each is still used for a paper figure/table):
- `analyses/plot_*.py` (**19 files**) — all figure generators (e.g. `plot_cdl_gbeta_*`, `plot_b_heatmaps`, `plot_B_diagnostics`, `plot_old_gbeta_*`, `plot_attn_diagnostic_heatmap`)
- `analyses/{validate_*,eval_*,scan_*,diag_*,summarize_*}.py` — validators / metric dumps (`validate_unsupervised_selector`, `validate_selector_cross_ckpt`, `eval_l0_dynamic_gbeta`, `scan_seed1_10k_heads`, `diagnose_readout_collapse`, `head_stability_seed2`)
- `analyses/{physical_signal_source,canonical_reanalysis,characterize_l0_dynamic_gbeta}.py` — analysis
- `analyses/p3prime_*`, `p6_*` (8), `p7_*` (image/policy scans) — phase analyses (some are the "findings" scripts behind memory notes; keep as diagnostics)
- `scripts/{plot_*,summarize_*,inspect_*,verify_*,vis_order_animation}.py` — figures / summaries / inspection

**Distinguishing DIAGNOSTIC vs HISTORY for the `.py` long tail requires opening
each** (is it referenced by a current report/figure, or a one-off from a closed
line?). This report does NOT move any `.py` — see §5.

---

## 3. HISTORY (archived; off the main line — safe to move: nothing imports `.sh`)

**`scripts/*.sh` — dated / one-off launch scripts** (~55). Nothing imports them;
they are records of past experiment launches. High-confidence archive:
- `overnight_*`, `queue_*` (all) — one-off queued runs
- `run_*_2026*` (dated: `run_frozen_gbeta_aligned_20260625`, `run_gbeta_loss_comparison_20260625`, `run_compile_5k_ablation_20260626`, `run_from10k_gbeta_attention_maps_20260623`, `overnight_from40k_orientation_20260622`, `run_direct_policy_aligned_20260625`, `overnight_20260625_from5k_aligned_suite`, `run_overnight_20260610_seed2_multistart`, `run_overnight_20260611_cdl_multistart`)
- `run_nr1_*` (NR-1 line, closed), `run_br1_*` (BR-1 line), `run_ct8_all` (CT8 line), `run_round2_*`, `run_e2_*`/`run_e3*`/`run_image*` (image line launches), `run_per_head_order_scan_ladder*`, `run_handoff_circuit_trajectory`, `run_head_gated_*`, `post_grw_diagnose`, `check_large_model_progress`, `_br1_wait_for_gpu`

**Keep as LAUNCH (a few reusable templates)** — confirm before archiving:
- `run_frozen_gbeta_aligned_20260625.sh` (the canonical frozen_beta protocol — arguably a template)
- `run_l2r_baselines_s2_s42_60k.sh`, `run_baselines_full_chain_60k.sh` (baseline launchers)

**`.py` history candidates** (closed lines — CONFIRM before moving, they may still be imported):
- image line: `scripts/train_vq*.py`, `scripts/train_imagenet*.py`, `analyses/cdl_teacher_ablation_image*.py`
- superseded gβ: `analyses/plot_old_gbeta_*` (old gβ comparison)

---

## 4. UNKNOWN (needs per-file confirmation — use the exhaustive Codex pass)

The `block_lo_arm_order_network/` `.py` long tail (358 files, incl. many
`*_scan`, `*_diag`, `test_*`, and one-off analysis modules) and the `analyses/`
`.py` that aren't in §1 cannot be safely auto-classified without opening each and
checking whether a current entrypoint/config/report references it. **Recommended:
run the Codex prompt (your section 五) for an exhaustive per-file reference check**
before moving any `.py`. `tests/` and `block_lo_arm_order_network/tests/` stay put
(they gate the mainline).

---

## 5. Recommended target structure (phased, minimal-move-first)

The idealized `src/order/{l2r,random,cdl,gbeta,manual_one_shot,frame_conversion}.py`
split is a **large refactor** — order logic currently lives inside
`train_clean_aogpt.py` + `clean_training_protocol.py` + `batch_readout/*providers`.
Doing it now conflicts with "minimal moves / don't break口径". **Defer** it to a
dedicated refactor (with a shared `build_order(policy, B, step, config)` interface
proposed, not yet implemented).

**Phase 1 (safe, this plan) — archive `.sh` history + scaffold dirs:**
```
history/
  README.md
  launch_scripts/        # dated one-off *.sh
scripts/
  diagnostics/           # (dir created; py moved later after per-file confirm)
```

**Phase 2 (after Codex per-file audit) — move DIAGNOSTIC `.py`** into
`scripts/diagnostics/`, HISTORY `.py` into `history/` (with import-reference fixes).

**Phase 3 (dedicated refactor) — `src/order/` unified interface** + a
`configs/baselines/` set of budget-matched configs.

---

## 6. Frame-conversion / order / config checks (as requested)

- **Frame conversion**: single mainline source = `clean_training_protocol.py`
  (`CleanPermutation`, `physical_blocks_to_model_token_order`,
  `model_blocks_to_physical_blocks`). `analyses/order_head_module.py` reuses it.
  ⚠ Exhaustive pass should confirm no divergent copy in `analyses/`.
- **Order policies**: scattered (not one dir) — `l2r`/`random` inline in
  `train_clean_aogpt.py`; `graph_rw` in `directed_graph_policy.py`; `mlp`/`position`
  in `attn_order_mlp_policy.py`; `cdl`/`gβ`/`direct` in `batch_readout/*provider`.
  Unifying under `src/order/` = Phase-3 refactor.
- **Baseline configs**: `configs/text/` has L2R + random + v3 variants and they
  share the seq256/block64 protocol; but the warmup→L2R (keepopt/resetopt) baselines
  are CLI-only. Recommend materializing them as configs (NEW).

---

## 7. Danger points

- Don't move any `.py` before confirming it isn't imported by `train_clean_aogpt.py`
  or a config module — silent `ImportError` at run time.
- `runs/` has 82 tracked JSON result records — keep (provenance), don't archive as code.
- The branch is far ahead of `master` (per prior review, ~325 commits) with
  uncommitted `analyses/v3_group_trainer.py` + `runs/v3_sweep/`. Cleanup should
  happen on a dedicated branch, not mixed into the feature branch.
- Nested repos (`AO-GPT-MDM`, `nanogpt-learned-order`) — never `git mv` across them.
