# 05 - Impact on Existing Claims

Source claims: `reports/evidence_package_20260613_verified/05_FINAL_CLAIMS_LOCK.md`.

## Claim impact table

| Claim | Does B1 affect it? | B1 evidence | Status after B1 | Needed action |
|---|---|---|---|---|
| 1. Sparse order-bearing heads exist and survive B1/predictor-aligned extraction. | Yes, directly. | Clean-base B1/predictor has best `L0H0 tau=1.000000` at 10k/50k/60k; latest continuous B1 seed=124 final best `L0H4 tau=0.957589`, `|tau|>0.9`=1. Sources: `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt10000_seed*.json`, `ckpt50000_seed*.json`, `ckpt60000_seed*.json`; `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv`. | Strengthened. B1/predictor diagnostics show the signal is not a B0 extraction artifact. | Update claim text exactly to include "survives B1/predictor-aligned extraction." |
| 2. Legacy B0 g_beta reads structured selected-head B, not a fixed L2R prior. | Not directly strengthened. | No B1 g_beta sanity found. Existing sanity is B0: real selected-head B `tau_vs_l2r=0.967500`; gaussian `tau_vs_l2r=-0.003353`; entry-shuffled `0.016488`; row/col shuffled `0.012748`; gaussian pairwise tau=0.000313. Source: `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`. | Keep as B0 legacy controller evidence. Add that B1 independently confirms strong upstream order-bearing structure. | Do not claim g_beta reads B1 graphs unless a B1 g_beta sanity run exists. |
| 3. Legacy B0 frozen g_beta accelerates training across two matched seed groups. | Indirect only. | No B1 frozen-hook run found. Hook code uses B0: `block_lo_arm_order_network/batch_readout/hook_order_provider.py`; selected dataset uses B0: `block_lo_arm_order_network/batch_readout/selected_head_dataset.py`. | Unchanged. Recovery remains 86%-106%; step saving 33%-42%, under B0 legacy hook path. | No immediate rerun needed. B1 hook rerun only if claiming B1-controller acceleration. |
| 4. g_beta@10k transfers to later checkpoints within the same seed/head under the legacy hook path. | Indirect only. | Phase reports are B0: seed-123 report has `none_mode="b0"` and PASS; seed-42 report has `none_mode="b0"` and PASS. Sources: `block_lo_arm_order_network/batch_readout/logs/phase33_gbeta_seed2_from10k_l0h2/random_baseline_continuous_jun08_seed2/full/phase15_report.json`; `block_lo_arm_order_network/batch_readout/logs/phase33_gbeta_seed42_from10k_l0h4/random_baseline_continuous_jun05/full/phase15_report.json`. | Remains B0 legacy. B1 does not change the historical transfer result. | State "legacy B0 controller transfer" unless B1 controller transfer is rerun. |
| 5. Signal is robust across granularity and scale diagnostics, but 317M and B1 hook acceleration remain untested. | Partially. | B1 317M and B1 granularity scans not found. Existing 317M is B0: best `L0H10 tau=0.955357`, `|tau|>0.9`=6. Source: `block_lo_arm_order_network/probe_results/large_random_baseline_16l16h1024d/head_scan_step5000.json`. | Keep granularity and 317M as B0 legacy diagnostics; B1 strengthens small-model mechanism diagnostics. | Optional B1 317M/B1 granularity/B1 hook reruns only if needed for paper scope. |

## Q1: Does B1 replace B0 as the main method?

Answer: Partial, with this final positioning:

> B1 is the collaborator-aligned attention diagnostic protocol, not the current frozen-hook acceleration protocol.

Recommended status:

- Diagnostics should use the B1/predictor-aligned extraction wording.
- Existing frozen g_beta datasets, g_beta sanity checks, and hook acceleration runs remain B0 legacy evidence.
- Do not present B1 diagnostics and B0 hook acceleration as if they used one identical extraction path.

Practical wording:

> B1/predictor-aligned extraction is the main post-alignment attention diagnostic convention. Existing frozen g_beta acceleration results were obtained with the legacy B0 selected-head extraction, and should be labeled as such until a B1 hook rerun exists.

## Q2: Do existing frozen g_beta training results need to be rerun?

Answer: No for the existing B0 claim; partial if claiming B1-controller acceleration.

Exact cautious answer:

> Existing training-acceleration results remain valid under B0, while B1 demonstrates that the order signal is not an extraction artifact. A full B1 hook rerun would be needed to claim B1-controller acceleration.

Why:

- `block_lo_arm_order_network/batch_readout/hook_order_provider.py` imports `_attn_to_A_block_b0_vec` and states that online hook extraction mirrors `none_mode='b0'`.
- `block_lo_arm_order_network/batch_readout/selected_head_dataset.py` defaults to `none_mode="b0"`.
- `block_lo_arm_order_network/scripts/run_phase33_gbeta.py` builds g_beta datasets with `none_mode="b0"`.

## Which old B0 results should be legacy / appendix?

Move to legacy / appendix if the paper's main method section says all attention diagnostics use B1:

- B0 per-head scans: `block_lo_arm_order_network/probe_results/clean_base_random_perm/head_scan_10k.json`, `shuffle_gran_32/head_scan_10k.json`, `shuffle_gran_128/head_scan_10k.json`, `random_baseline_continuous_jun08_seed2/head_scan_50k.json`.
- B0 317M diagnostic: `block_lo_arm_order_network/probe_results/large_random_baseline_16l16h1024d/head_scan_step5000.json`.
- B0 selected-head g_beta sanity: `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`.
- B0 frozen-hook acceleration curves: `block_lo_arm_order_network/probe_results/frozen_beta_*` eval curves.

Do not call these wrong. Call them:

> legacy B0 convention results, retained as historical controller and robustness evidence.

## Recommended final five-claim wording

1. Sparse order-bearing heads exist and survive B1/predictor-aligned extraction.
   - B1 clean-base ladder keeps `tau=1.000000` through 60k; latest continuous B1 tracking ends with `L0H4 tau=0.957589`.
2. Legacy B0 g_beta reads structured selected-head B, not a fixed L2R prior.
   - Real B `tau_vs_l2r=0.967500`; gaussian `-0.003353`; destroyed controls near 0; zero-B is a margin=0 tie-breaking artifact.
3. Legacy B0 frozen g_beta accelerates training across two matched seed groups.
   - Recovery 86%-106%; conservative step saving 33%-42%.
4. g_beta@10k transfers to later checkpoints within the same seed/head under the legacy B0 hook path.
5. The signal is robust across granularity and scale diagnostics, but 317M B1 and B1 hook acceleration remain untested.
