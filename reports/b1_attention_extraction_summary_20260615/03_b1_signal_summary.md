# 03 - B1 Signal Summary

## Bottom line

B1/predictor-aligned diagnostics still show strong order-bearing signal.

Two B1 evidence streams exist:

1. Clean-base B1 ladder: `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt*_seed*.json`.
2. Latest continuous B1 all-head tracking: `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv`.

They are separate experiments and should not be merged into one seed table.

## Table 1: B1 per-head scan summary

| Group | Checkpoint step | Model | Seed | Head | tau vs L2R | Orientation | Source | Verified? |
|---|---:|---|---:|---|---:|---|---|---|
| clean_base random-perm | 10000 | 4L/8H/384d, chunk-based, seed=42 training config | scan seed 0 | L0H0 | 1.000000 | B1/predictor, `B=A.T` | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt10000_seed0.json` | Yes |
| clean_base random-perm | 10000 | same | scan seed 1 | L0H0 | 1.000000 | B1/predictor, `B=A.T` | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt10000_seed1.json` | Yes |
| clean_base random-perm | 10000 | same | scan seed 2 | L0H0 | 1.000000 | B1/predictor, `B=A.T` | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt10000_seed2.json` | Yes |
| clean_base random-perm | 50000 | same | scan seed 0 | L0H0 | 1.000000 | B1/predictor, `B=A.T` | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt50000_seed0.json` | Yes |
| clean_base random-perm | 50000 | same | scan seed 1 | L0H0 | 1.000000 | B1/predictor, `B=A.T` | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt50000_seed1.json` | Yes |
| clean_base random-perm | 50000 | same | scan seed 2 | L0H0 | 1.000000 | B1/predictor, `B=A.T` | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt50000_seed2.json` | Yes |
| clean_base random-perm | 60000 | same | scan seed 0 | L0H0 | 1.000000 | B1/predictor, `B=A.T` | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt60000_seed0.json` | Yes |
| clean_base random-perm | 60000 | same | scan seed 1 | L0H0 | 1.000000 | B1/predictor, `B=A.T` | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt60000_seed1.json` | Yes |
| clean_base random-perm | 60000 | same | scan seed 2 | L0H0 | 1.000000 | B1/predictor, `B=A.T` | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt60000_seed2.json` | Yes |
| continuous B1 head tracking | 60000 | 4L/8H/384d, continuous, seed=124 | probe seed from config/global step | L0H4 | 0.957589 | B1/predictor, `B=A.T` | `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv` | Yes |
| continuous B1 head tracking | 60000 | same | same | L1H5 | 0.834821 | B1/predictor, `B=A.T` | same | Yes |
| continuous B1 head tracking | 60000 | same | same | L1H7 | 0.830109 | B1/predictor, `B=A.T` | same | Yes |
| continuous B1 head tracking | 60000 | same | same | L2H3 | 0.775298 | B1/predictor, `B=A.T` | same | Yes |
| continuous B1 head tracking | 60000 | same | same | L2H5 | 0.768353 | B1/predictor, `B=A.T` | same | Yes |
| seed-123 continuous group | 50000 | `random_baseline_continuous_jun08_seed2` | seed=123 | B1 scan not found | N/A | N/A | searched `block_lo_arm_order_network/probe_results/random_baseline_continuous_jun08_seed2/` | Missing |
| seed-42 continuous group | 50000 | `random_baseline_continuous_jun05` | seed=42 | B1 scan not found | N/A | N/A | searched `block_lo_arm_order_network/probe_results/random_baseline_continuous_jun05/` | Missing |
| 317M scale | 5000 | 16L/16H/1024d | seed=42 | B1 scan not found | N/A | N/A | only B0 file found: `block_lo_arm_order_network/probe_results/large_random_baseline_16l16h1024d/head_scan_step5000.json` | Missing |

## Table 2: B1 strong-head count

| Model / group | Total heads | `|tau|>0.9` | `|tau|>0.7` | Best tau | Best head | Source |
|---|---:|---:|---:|---:|---|---|
| clean_base random-perm B1 @10k, scan seed 0 | 32 | 7 | 14 | 1.000000 | L0H0 | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt10000_seed0.json` |
| clean_base random-perm B1 @10k, scan seed 1 | 32 | 7 | 14 | 1.000000 | L0H0 | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt10000_seed1.json` |
| clean_base random-perm B1 @10k, scan seed 2 | 32 | 7 | 14 | 1.000000 | L0H0 | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt10000_seed2.json` |
| clean_base random-perm B1 @50k, scan seed 0 | 32 | 6 | 13 | 1.000000 | L0H0 | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt50000_seed0.json` |
| clean_base random-perm B1 @50k, scan seed 1 | 32 | 6 | 13 | 1.000000 | L0H0 | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt50000_seed1.json` |
| clean_base random-perm B1 @50k, scan seed 2 | 32 | 6 | 13 | 1.000000 | L0H0 | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt50000_seed2.json` |
| clean_base random-perm B1 @60k, scan seed 0 | 32 | 6 | 13 | 1.000000 | L0H0 | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt60000_seed0.json` |
| clean_base random-perm B1 @60k, scan seed 1 | 32 | 6 | 13 | 1.000000 | L0H0 | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt60000_seed1.json` |
| clean_base random-perm B1 @60k, scan seed 2 | 32 | 6 | 13 | 1.000000 | L0H0 | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt60000_seed2.json` |
| continuous B1 head tracking seed=124 @60k final | 32 | 1 | 9 | 0.957589 | L0H4 | `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv` |

## Table 3: B1 vs shuffled / control

| Protocol | B version | Best tau | Strong heads | Interpretation | Source |
|---|---|---:|---:|---|---|
| clean_base random-perm | B1/predictor | 1.000000 | 7 at 10k; 6 at 50k/60k | Order-bearing signal survives B1 predictor alignment. | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt*_seed*.json` |
| continuous seed=124 random baseline | B1/predictor | 0.957589 final; avg L0H4 tau=0.966056 over training | 1 final head above 0.9; 9 final heads above 0.7 | Selected/per-head signal remains strong in latest continuous B1 tracking. | `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv` |
| shuffled-L2R control | B1/predictor | N/A | N/A | B1 shuffled-L2R control not found; use B0 control as legacy evidence only, or rerun B1 scan if needed. | Search found no B1 shuffled-L2R result file. |
| 317M scale diagnostic | B1/predictor | N/A | N/A | B1 317M scan not found; B0 317M remains legacy scale diagnostic. | `block_lo_arm_order_network/probe_results/large_random_baseline_16l16h1024d/head_scan_step5000.json` is B0 only. |

## Continuous seed=124 L0H4 stability

From `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv` after de-duplicating `(step, layer, head)`:

- Rows: 192032.
- Steps: 6001, from 0 to 60000.
- Final top by `tau_vs_l2r`: `L0H4 tau=0.957589`, `mean_pairwise_tau=0.930060`.
- Average over all tracked steps: `L0H4 tau=0.966056`, `mean_pairwise_tau=0.950800`.
- Top-by-absolute-tau frequency: `L0H4` is top in 5768 / 6001 tracked steps.

Source: `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv`.

