# Project Memory

## 2026-06-29: P3′ causal verification

- [P3′ Causal Verification 2026-06-29](analyses/p3prime_causal_README.md) — L0 global physical-order carrier: P3′-B joint τ+R², P3′-C residual>floor, D says OV degenerate and QK path matters; A is calibration/redundancy only; not B+→C (needs P4).

## 2026-06-21: seed123 CDL/g_beta closed-loop pipeline

- Do not trust directory names or `args.seed` alone for CDL/g_beta text runs. The effective layout/protocol comes from the resumed checkpoint's `clean_protocol`.
- For the main text closed-loop comparison, group by effective protocol/layout. The current clean main group is protocol seed123 with `block_perm_first16[:8] = [12, 2, 1, 15, 43, 22, 57, 10]`.
- Relocation diagnostic stands: recovered order is position-bound, not content-driven. Do not claim the model discovered original content L2R. The safe framing is layout/position-template recovery, with posthoc remapping producing L2R-looking order under fixed layout.
- `val_ori_l2r_block` is the main comparable metric. Be cautious with `val_cdl_order`; the training eval path may reuse cached CDL order state. Prefer posthoc forced-refresh diagnostics for teacher-order claims.

### Clean seed123 closed-loop components

- Random baseline 60k: `block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2_ext60k`
- L2R seed123 reference: `block_lo_arm_order_network/probe_results/l2r_continuous_seed123`
- g_beta seed123 from10/20/40 are already available:
  - `frozen_beta_seed2_from10k_l0h2`
  - `frozen_beta_seed2_from20k_l0h2`
  - `frozen_beta_seed2_from40k_l0h2`
- Strict CDL seed123 from10 is available:
  - `cdl_teacher_seed123_from10k_l0h2`
- Strict CDL seed123 from20/from40 are the required missing positive runs:
  - `cdl_teacher_seed123_from20k_l0h2`
  - `cdl_teacher_seed123_from40k_l0h2`
- Negative control added for tonight:
  - `cdl_teacher_seed123_from20k_l0h2_reverse`

### Scripts

- Full overnight pipeline:
  - `scripts/queue_seed123_closed_loop_overnight.sh`
- L2R seed123 50k->60k:
  - `scripts/queue_extend_l2r_seed123_50k60k.sh`
- CDL seed123 missing from20/from40:
  - `scripts/queue_cdl_seed123_missing_20k40k.sh`
- CDL reverse negative control from20:
  - `scripts/queue_cdl_seed123_reverse_from20k.sh`
- Matched plotting:
  - `analyses/plot_cdl_gbeta_matched.py`
  - Outputs: `cdl_matched_by_seed.png`, `gbeta_matched_by_seed.png`, `cdl_gbeta_matched_by_seed.png`, `cdl_gbeta_matched_by_seed.json`

### Verification already run

- `bash -n` passed for the queue scripts.
- `python3 -m py_compile analyses/plot_cdl_gbeta_matched.py analyses/plot_cdl_gbeta_comprehensive.py` passed.
- `python3 -m pytest tests/test_plot_cdl_gbeta_matched.py` passed: 4 tests.
- `SMOKE_TEST=1 bash scripts/queue_seed123_closed_loop_overnight.sh` passed end-to-end:
  - L2R smoke: 50000->50001.
  - CDL positive smoke: 20000->20001 and 40000->40001.
  - CDL reverse smoke: 20000->20001 with `rev=True`.
  - Plot regeneration succeeded.

### Live run launched

- Formal overnight run launched with `setsid` because plain `nohup` did not keep the job alive from this environment.
- Launch command used:

```bash
setsid bash -lc 'cd /home/admin/lyuyuhuan/order_lyu; exec bash scripts/queue_seed123_closed_loop_overnight.sh' > /tmp/seed123_closed_loop_overnight.launch2.log 2>&1 < /dev/null &
```

- Initial PID: `450190`.
- Main log:
  - `block_lo_arm_order_network/probe_results/logs/seed123_closed_loop_overnight_20260621_233216.log`
- Launch log:
  - `/tmp/seed123_closed_loop_overnight.launch2.log`
- At last check, the formal run was in the L2R seed123 extension stage on GPU1, around step `50700/60000`, with LR `1.00e-04`.

### Claim boundary

- If tonight finishes, the result supports a single-seed matched closed-loop comparison for protocol seed123: random, L2R, g_beta from10/20/40, CDL from10/20/40, and CDL reverse-from20 negative control.
- It does not support multi-seed robustness.
- Old runs named `cdl_teacher_seed2_*` and `cdl_teacher_*seed124*` may inherit protocol seed123 or seed42. Treat them as inherited-protocol historical curves, not strict seed2/seed124 evidence.

## 2026-06-22: shuffled-AR control status

- We previously ran a shuffled-AR / shuffled-L2R control, but the old result is the weaker version of the control.
- Usable run:
  - `block_lo_arm_order_network/probe_results/shuffled_l2r_continuous_jun05`
  - `run_kind=shuffled_l2r`, seed42/layout, trained to 50k.
  - Local convention: `block_perm[physical] = model`, `inv_perm[model] = physical`.
- Non-usable run:
  - `block_lo_arm_order_network/probe_results/shuffled_l2r_continuous_jun07` only has step0, so do not use it as shuffled-AR evidence.
- Old diagnostic output:
  - `analyses/attention_diagnostic_20260609/cdl_tau_diagnostic.json`
  - `shuffled-L2R best_tau_vs_physical_L2R = 0.0397`
  - `random-order best_tau = 0.4901`
  - `ori-L2R best_tau = 0.3730`
- What the old result proves:
  - shuffled-AR does not recover the physical L2R / semantic-neighbor physical order.
- What the old result does not prove:
  - It does not prove shuffled-AR has no order at all.
  - It does not prove whether shuffled-AR recovers the imposed model-position/identity order.
- Current mechanism framing needs the stronger check:
  - Compare shuffled-AR `sigma_model` against identity model order `0,1,...,63`.
  - Compare shuffled-AR `sigma_model` against the semantic-neighbor model path `block_perm[np.arange(N)]`.
  - Compare posthoc `sigma_phys = inv_perm[sigma_model]` against both physical L2R and imposed shuffled physical order `inv_perm[np.arange(N)]`.
- New diagnostic scripts added:
  - `analyses/diag_shuffled_ar_model_frame_order.py`
  - `scripts/queue_shuffled_ar_model_frame_diag.sh`
- New diagnostic queued:
  - PID `499150`
  - Launch log: `/tmp/shuffled_ar_model_frame_diag.launch.log`
  - Output directory: `reports/shuffled_ar_model_frame_diag/`
  - Expected files: `summary.json`, `shuffled_ar_model_frame_order.tsv`
- Until this diagnostic finishes, only write the conservative claim:
  - "shuffled-AR does not recover physical L2R."
  - Do not write "shuffled-AR has no coherent order" unless the model-frame diagnostic confirms it.

## 2026-06-22: g_beta mainline and W&B logging requirement

- Treat the new head-gated g_beta work as smoke-only for now. Do not spend full-grid/60k budget on head-gated unless explicitly requested later.
- Continue the main training/evidence line with the existing single-head/frozen g_beta pipeline.
- Existing single-head g_beta distillation is already strong:
  - `gbeta_b1_L0H2_seed2_step20k`: Kendall tau `0.9653`, pairwise acc `0.9827`.
  - `gbeta_b1_L0H2_seed2_step10k`: Kendall tau `0.9456`, pairwise acc `0.9728`.
  - `gbeta_b1_L0H2_seed2_step40k`: Kendall tau `0.9412`, pairwise acc `0.9716`.
  - `gbeta_b1_L0H0_from40k_seed124`: Kendall tau `0.9939`, pairwise acc `0.9988`.
- Existing 60k feedback results show the useful comparison:
  - random baseline 60k: `val_ori_l2r_block = 3.4996`.
  - frozen g_beta B1 seed2 from20k: `val_ori_l2r_block = 3.3322`.
  - frozen g_beta B1 seed2 from40k: `val_ori_l2r_block = 3.3659`.
  - frozen g_beta from20k fixseed: `val_ori_l2r_block = 3.3523`.
  - frozen g_beta from20k seed124: `val_ori_l2r_block = 3.3560`.
  - frozen g_beta from40k seed124: `val_ori_l2r_block = 4.2090`; treat as a failure/instability case needing diagnosis.
  - CDL teacher from20k fixseed: `val_ori_l2r_block = 3.3469`.
  - CDL teacher from20k seed124: `val_ori_l2r_block = 3.3385`.
  - CDL teacher from40k seed124: `val_ori_l2r_block = 3.3692`.
- Current conclusion:
  - from20k frozen g_beta and CDL teacher both beat random baseline by a clear margin.
  - g_beta and CDL teacher are close; neither should be claimed as universally better without matched curve/seed analysis.
  - from40k seed124 is the key failure case to explain before making robustness claims.
- Follow-up diagnostics on 2026-06-22:
  - `reports/gbeta_feedback_final_summary/model_frame_head_drift/` shows the bad g_beta run still has a strong best semantic head (`L2H3`, tau `0.2302`), so failure is not simply disappearance of order signal.
  - `reports/gbeta_feedback_final_summary/controller_order_diag/controller_order.tsv` directly replays frozen g_beta controller orders.
  - The bad `frozen_beta_from40k_seed124` run uses `L0H0` and its emitted controller order has `tau_controller_model_vs_identity = -0.2609`; this is a plausible direct failure mechanism.
  - Important correction: `from40k seed124` was selected by small-train audition, but the selected row was `L0H0,rev` (`final_val_l2r=3.4504`, `total_advantage=0.1607`) with `L0H2,fwd` nearly tied (`final_val_l2r=3.4503`, `total_advantage=0.1576`). The full failed run appears to deploy `L0H0` without an explicit reverse orientation, so describe it as an audition-to-deployment/orientation mismatch plus horizon sensitivity, not as "small-train did not select it."
  - `train_clean_aogpt.py` now has `--frozen-beta-rev`; use it for the `from40k seed124 L0H0,rev` control, together with `--wandb-log` for any non-throwaway run.
  - Smoke run completed for `from40k seed124 L0H0,rev`: `block_lo_arm_order_network/probe_results/smoke_frozen_beta_from40k_seed124_l0h0_rev_20260622_cpu_nowandb/`.
    It resumed `random_baseline_continuous_jun08_seed2/ckpt_step40000.pt`, used `gbeta_b1_L0H0_from40k_seed124/.../g_beta_best.pt`, logged `rev=True`, ran CPU 40000->40001, and wrote `eval_curve.tsv` plus `ckpt_step40001.pt`.
    This is only a path smoke: eval was deliberately tiny (`stream_eval_windows=8`, `batch_size=2`) and should not be compared to formal 60k results.
  - Attempted W&B offline smoke failed in the current sandbox because W&B tried to write `/home/admin/.cache/wandb` and open local sockets; the no-W&B smoke completed. For a formal run, launch in the normal training environment with W&B enabled, or rerun with permissions that allow W&B's cache/socket usage.
  - Formal from40k orientation run launched on 2026-06-22:
    - tmux `from40k_orientation_20260622` runs `from40k_seed124_l0h0_rev` on `cuda:1`.
    - Output: `block_lo_arm_order_network/probe_results/from40k_orientation_controls_20260622_overnight_main_tmux/from40k_seed124_l0h0_rev/`.
    - W&B online run: `from40k_seed124_l0h0_rev`, run id `abu5vtxm`, URL `https://wandb.ai/1113488238-dalian-university-of-technology/order-lyu/runs/abu5vtxm`.
    - Training log confirms `head=L0H0 none_mode=b1 mode=argsort rev=True refresh_every=10`.
    - Early evals through 44k: `val_ori_l2r_block` improved from `3.5220` at 40500 to `3.4478` at 44000; wait for 60k before drawing final conclusions.
  - Remaining overnight controls are queued in tmux `from40k_orientation_remaining_20260622` with `RUN_SET=remaining`; it waits for GPU1 to free, then runs `from40k_seed124_l0h2_fwd` followed by `from40k_seed124_l0h0_fwd`.
  - Good from20k gains should be described as controller/curriculum usefulness under protocol, not as a naive "controller equals physical L2R" story.
- Next work should prioritize:
  - final result table across random/L2R/frozen g_beta/CDL/reverse controls;
  - curve overlays from `eval_curve.tsv`;
  - model-frame head-drift diagnostics for good and bad g_beta feedback runs;
  - boss-facing summary with claim boundaries.
- W&B requirement for future formal training:
  - Before launching any new long training run, make sure training loss and eval metrics are logged to Weights & Biases.
  - At minimum log: `train_loss`, `val_train_objective`, `val_ori_l2r_block`, `val_model_order`, `val_unstructured_order`, `val_rw_order`, `val_beta_order`, `val_cdl_order`, `alpha`, and `lr`.
  - Include run metadata: run kind, resume checkpoint, output dir, seed, permute seed, controller/g_beta checkpoint, selected head, none mode, refresh interval, data source, and git status/commit when available.
  - Do not rely only on `eval_curve.tsv` for future overnight/formal runs; keep TSV as local backup, but W&B should be the primary live monitoring surface.
  - `train_clean_aogpt.py` now has explicit W&B flags:
    - `--wandb-log`
    - `--wandb-project order-lyu`
    - `--wandb-run-name <name>`
    - `--wandb-tags ...`
    - `--wandb-mode online|offline|disabled`
  - Future formal runs should include `--wandb-log --wandb-project order-lyu --wandb-run-name <descriptive-name>` unless deliberately running a throwaway smoke.
- Raw all-head attention map tracking added on 2026-06-23 for studying how attention maps change under g_beta feedback without doing top-head selection or metric aggregation during training.
  - User preference for this line of experiments: attention tracking should be raw and per-head. Do not run heavy all-head signal scans, top-head selection, tau/CDL/diversity summaries, or aggregation during training unless explicitly requested. Save the concrete per-head map data first; offline processing/top-head analysis comes later.
  - `train_clean_aogpt.py` flags:
    - `--track-head-maps`
    - `--track-head-map-interval 200`
    - `--track-head-map-samples <n>`
    - `--track-head-map-dtype float32|float16`
    - extraction mode is controlled by existing `--track-head-none-mode`, usually `b1`.
  - Each snapshot is saved under `<out_dir>/head_maps_raw/head_maps_stepXXXXXX.npz`.
  - NPZ fields:
    - `head_maps`: shape `(sample, layer, head, block_query, block_key)`.
    - `probe_idx`: fixed indices into `idx_eval_model`.
    - `probe_orders`: fixed token probe orders.
    - `meta_json`: shape/dtype/mode/probe metadata.
  - The tracker deliberately does not compute CDL labels, tau, diversity, row concentration, top heads, or batch means. It fixes the same eval windows and probe orders across snapshots so later differences are primarily model/attention changes.
  - CPU smoke passed: `block_lo_arm_order_network/probe_results/smoke_from10k_headmaps_cpu_20260623/head_maps_raw/head_maps_step010000.npz`, with shape `(1, 4, 8, 64, 64)` and dtype `float32`.
  - Formal from10k repeat queued in tmux `from10k_headmaps_20260623`:
    - Script: `scripts/run_from10k_gbeta_attention_maps_20260623.sh`.
    - Resumes `block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2/ckpt_step10000.pt`.
    - Uses `block_lo_arm_order_network/batch_readout/logs/gbeta_b1_L0H2_seed2_step10k/random_baseline_continuous_jun08_seed2/full/g_beta_best.pt`.
    - Repeats prior `frozen_beta_b1_seed2_from10000_l0h2` setup with W&B online, and only adds raw head-map dumps every 200 steps.
    - Output target for this launch: `block_lo_arm_order_network/probe_results/frozen_beta_b1_seed2_from10000_l0h2_headmaps_20260623_0110/`.
    - As of 2026-06-23 01:09, it is waiting for GPU1 memory to fall below 2000 MiB before starting because the from40k remaining control is still using GPU1.
