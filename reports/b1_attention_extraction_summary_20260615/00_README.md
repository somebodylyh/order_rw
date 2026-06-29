# B1 Attention Extraction Summary

Date: 2026-06-15

Scope: read-only summary of the latest B1 attention extraction update and its relation to legacy B0 / canonical extraction. No new training was launched and no training code was modified.

## Executive status

- Final positioning: B1 is the collaborator-aligned attention diagnostic protocol, not the current frozen-hook acceleration protocol.
- The latest result-bearing B1 files use `none_mode=predictor`, not the code enum `none_mode=b1`; in this package this is called the B1/predictor-aligned diagnostic convention.
- In local scripts, this is described as the B1 predictor-aligned ladder: `attn[:-1, :-1]`, followed by batch-mean `B = A.T`, diagonal zeroed.
- The code also contains `_attn_to_A_block_b1_vec`, which means predictor frame plus physical remap. I did not find a result file produced with `none_mode=b1`.
- Strong order signal survives the B1/predictor-aligned diagnostic. The clean-base B1 ladder has best head `L0H0 tau=1.000000` at 10k, 50k, and 60k. The latest continuous B1 head tracking has final best head `L0H4 tau=0.957589`.
- Frozen g_beta and hook acceleration evidence remains legacy B0: the selected-head dataset and hook provider explicitly use `none_mode="b0"` / `_attn_to_A_block_b0_vec`.

Recommended one-sentence framing:

> B1 strengthens the mechanism story, but does not replace the B0 acceleration evidence.

## B1-related file inventory

| File | Type | B1-related content | Key numbers | Trust level | Notes |
|---|---|---|---|---|---|
| `block_lo_arm_order_network/per_head_order_scan.py` | Code | Defines `_attn_to_A_block_predictor_vec`, `_attn_to_A_block_b1_vec`, `_attn_to_A_block_b0_vec`, `_batch_mean_B`, and scan JSON schema. | N=64, block_len=4 from imports; `B = A.T`; diag zeroed. | HIGH | Primary protocol source. |
| `block_lo_arm_order_network/train_clean_aogpt.py` | Code | In-loop head trackers select `track_head_none_mode` from `b0`, `b1`, `predictor`; all-head tracker writes `head_signal_all.tsv`. | Latest run config used `track_head_m=4`, `track_head_interval=10`, `track_head_none_mode=predictor`. | HIGH | Primary source for latest head tracking semantics. |
| `scripts/run_per_head_order_scan_ladder_b1.sh` | Script | B1 ladder wrapper using `--none-mode predictor`. | Steps `0,1000,5000,10000,20000,30000,40000,50000,60000`; default seeds `0 1 2`; M=100, batch_size=32. | HIGH | The script comment calls this B1 predictor-aligned. |
| `scripts/run_per_head_order_scan_ladder_b1_fast.py` | Script | Fast B1 ladder; reuses chunks per seed; writes `batch_readout/logs/per_head_scan_b1`. | Output directory `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1`; `none_mode="predictor"`. | HIGH | Primary producer for B1 ladder JSONs. |
| `scripts/queue_random_b1_headscan_after_gpu1.sh` | Script/log launcher | Latest queued continuous baseline with all-head B1 tracking. | Seed=124; max_steps=60000; `--track-all-heads`; `--track-head-m 4`; `--track-head-none-mode predictor`. | HIGH | Primary launcher for latest continuous B1 run. |
| `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/config.json` | Config JSON | Latest continuous B1 headscan config. | seed=124; device=cuda:1; max_steps=60000; 4L/8H/384d; `track_head_none_mode=predictor`. | HIGH | Machine-generated config. |
| `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv` | TSV result | All-head B1/predictor tracking every 10 steps. | Dedup rows=192032; steps=6001; final best `L0H4 tau=0.957589`, pair=0.930060; final strong heads: `|tau|>0.9` = 1, `|tau|>0.7` = 9. | HIGH | Primary latest B1 result. |
| `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/eval_curve.tsv` | TSV result | Training/eval curve for latest B1-tracked baseline. | Final step=60000; `val_ori_l2r_block=3.476874`; best eval at 59000 with `val_ori_l2r_block=3.472651`. | HIGH | Training metric source, not a controller result. |
| `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/queued_launch.log` | Log | Confirms completion and repeated top head. | Exit rc=0; final saved checkpoint at 60000; final log top `L0H4 tau=+0.9576`. | HIGH | Machine-generated runtime log. |
| `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt*_seed*.json` | JSON results | B1/predictor ladder across clean-base checkpoints and scan seeds. | 27 JSONs. At 10k, each seed has `|tau|>0.9`=7, best `L0H0 tau=1.000000`; at 50k/60k, `|tau|>0.9`=6. | HIGH | Primary B1 ladder result. |
| `analyses/plot_attn_map_b1.py` | Analysis script | Plots B1 predictor frame no-remap and predictor frame + physical remap side by side. | Default head `L0H2`, default ckpt seed-123 continuous step 30000. | MEDIUM | Useful for defining raw-vs-physical B1 visualization. |
| `analyses/figures/attn_map_b1/b1_L0_H2_step30000_model_vs_phys.png` | Figure | B1 map visualization. | Visual only; no scalar result used. | MEDIUM | Figure generated from analysis script. |
| `block_lo_arm_order_network/batch_readout/selected_head_dataset.py` | Code | Selected-head g_beta dataset uses B0 canonical extraction by default. | Default `none_mode="b0"`; returns batch-mean `B = A.T`. | HIGH | Shows existing g_beta datasets are B0 legacy. |
| `block_lo_arm_order_network/batch_readout/hook_order_provider.py` | Code | Frozen hook online extraction uses B0. | Imports `_attn_to_A_block_b0_vec`; docstring says hook matches `none_mode='b0'`. | HIGH | Shows existing hook acceleration is not B1. |
| `block_lo_arm_order_network/scripts/run_phase33_gbeta.py` | Code/script | g_beta pretrain driver states B0 canonical extraction. | `none_mode="b0"`; phase reports under B0. | HIGH | B0 controller source. |
| `block_lo_arm_order_network/probe_results/clean_base_random_perm/head_scan_10k.json` | Legacy B0 JSON | B0 canonical per-head scan. | 32 heads; `|tau|>0.9`=9; best `L0H0 tau=1.000000`; heavy tau=0.876002. | HIGH | B0 comparison source. |
| `block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2/head_scan_50k.json` | Legacy B0 JSON | Seed-123 continuous B0 50k scan. | 32 heads; `|tau|>0.9`=0; best abs `L1H7 tau=-0.050199`; heavy tau=-0.013343. | HIGH | B0 late-step diagnostic. |
| `block_lo_arm_order_network/probe_results/large_random_baseline_16l16h1024d/head_scan_step5000.json` | Legacy B0 JSON | 317M B0 scale diagnostic. | 256 heads; `|tau|>0.9`=6; best `L0H10 tau=0.955357`; heavy tau=0.036855. | HIGH | No B1 317M scan found. |
| `reports/evidence_package_20260613_verified/05_FINAL_CLAIMS_LOCK.md` | Verified summary | Existing claims lock before B1 update. | Claims 1-5 and allowed/forbidden claims. | LOW | Human summary, but paths point to primary sources. |
| `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md` | Verified summary | B0 tables and frozen-hook claims. | Recovery, head counts, 317M, g_beta sanity. | LOW | Use only as secondary narrative; primary paths repeated in this report. |

## Generated files in this summary package

- `01_b1_protocol_definition.md`
- `02_b0_vs_b1_diff.md`
- `03_b1_signal_summary.md`
- `04_b1_vs_b0_result_comparison.md`
- `05_impact_on_existing_claims.md`
- `06_paper_method_text.md`
- `07_boss_update_summary.md`
- `08_unresolved_questions.md`
