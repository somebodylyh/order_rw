# Key Tables For Boss

## Table A - Main Training Result

Metric: `val_ori_l2r_block` at 50k. Protocol: legacy B0 hook path. Sources are `eval_curve.tsv` files listed in `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md`.

| Seed group | Head | Resume | Random @50k | L2R ref @50k | Frozen `g_beta` @50k | Recovery | Step saving | Protocol |
|---|---|---:|---:|---:|---:|---:|---:|---|
| seed123 | L0H2 | from10k | 3.445 | 3.305 | 3.325 | 86.0% | 41.7% | B0 legacy hook |
| seed123 | L0H2 | from20k | 3.445 | 3.305 | 3.329 | 83.0% | 35.7% | B0 legacy hook |
| seed123 | L0H2 | from40k | 3.445 | 3.305 | 3.363 | 58.7% | 3.6%; starts near threshold | B0 legacy hook |
| seed42 | L0H4 | from10k | 3.466 | 3.341 | 3.334 | 106.1% | 39.3% | B0 legacy hook |
| seed42 | L0H4 | from20k | 3.466 | 3.341 | 3.339 | 101.5% | 33.3% | B0 legacy hook |
| seed42 | L0H4 | from40k | 3.466 | 3.341 | 3.383 | 66.9% | 1.2%; starts near threshold | B0 legacy hook |

Takeaway: use conservative step saving 33-42% from from10k/from20k across both matched seed groups.

## Table B - B1 Signal Robustness

| Diagnostic | Protocol | Step | Best head | Best tau | Strong heads | Source |
|---|---|---:|---|---:|---:|---|
| B0 clean-base comparison | B0 legacy | 10k | L0H0 | 1.000000 | 9/32 with `|tau|>0.9` | `block_lo_arm_order_network/probe_results/clean_base_random_perm/head_scan_10k.json` |
| B1 clean-base ladder | B1/predictor | 10k | L0H0 | 1.000000 | 7/32 with `|tau|>0.9` | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt10000_seed*.json` |
| B1 clean-base ladder | B1/predictor | 50k | L0H0 | 1.000000 | 6/32 with `|tau|>0.9` | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt50000_seed*.json` |
| B1 clean-base ladder | B1/predictor | 60k | L0H0 | 1.000000 | 6/32 with `|tau|>0.9` | `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt60000_seed*.json` |
| B1 continuous seed124 | B1/predictor | 60k | L0H4 | 0.957589 | 1/32 with `|tau|>0.9`; 9/32 with `|tau|>0.7` | `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv` |
| 317M scale diagnostic | B0 legacy | 5k | L0H10 | 0.955357 | 6/256 with `|tau|>0.9` | `block_lo_arm_order_network/probe_results/large_random_baseline_16l16h1024d/head_scan_step5000.json` |

Takeaway: B1/predictor-aligned diagnostics preserve the order-bearing signal; 317M remains B0 diagnostic only.

## Table C - `g_beta` Sanity

Source: `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`. Protocol: legacy B0 controller path.

| Input B | tau | Pairwise/diversity/margin | Interpretation | Source |
|---|---:|---|---|---|
| real selected-head B | 0.967500 vs L2R; 0.949325 vs teacher | pairwise tau 0.948837; margin 25.331 | `g_beta` reads structured B | raw JSON |
| gaussian random B | -0.003353 vs L2R | Gaussian family pairwise tau 0.000313; first-block unique 53 | no fixed L2R prior from random B | raw JSON |
| uniform random B | -0.145942 vs L2R | first-block unique 49 | no structured signal | raw JSON |
| entry-shuffled B | 0.016488 vs L2R | pairwise tau 0.000040 for shuffled family | destroyed structure gives near-zero order | raw JSON |
| row/col shuffled B | 0.012748 vs L2R | first-block unique 51 | structure-dependent, not just row/col marginals | raw JSON |
| zero B | 1.000000 vs L2R | margin 0.0 | tie-breaking artifact, not evidence of prior | raw JSON |

Takeaway: under B0 controller path, `g_beta` reads graph structure; no B1 `g_beta` sanity is established yet.

## Table D - Claim Boundary Table

| Claim | Status | Evidence | Boundary |
|---|---|---|---|
| Sparse order-bearing heads exist | Can claim | B0 clean-base 9/32 strong heads; B1 ladder and continuous signal | Sparse, not all-head |
| Signal survives B1/predictor extraction | Can claim | `L0H0 tau=1.0` at 10k/50k/60k; seed124 final `L0H4 tau=0.957589` | B1 diagnostic, not B1 hook |
| `g_beta` reads B | Can claim for B0 | Real B high tau, destroyed B near zero | No B1 `g_beta` sanity yet |
| Frozen `g_beta` accelerates | Can claim for B0 | Two matched seed groups, 33-42% step saving | Legacy B0 hook path |
| Scale/granularity robustness | Can claim as diagnostic | B0 granularity, B0 317M scan | No 317M hook; no B1 317M |
| Universal random-order improvement | Cannot claim | val_unstructured worsens | Frame as canonical-order specialization |
| Fully label-free pipeline | Cannot claim | `g_beta` training uses CDL teacher | Can claim cheap head diagnostics + teacher-supervised readout |

