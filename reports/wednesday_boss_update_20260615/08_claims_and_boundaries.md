# Claims And Boundaries

## CAN CLAIM

1. Random-order AO-GPT develops sparse order-bearing attention heads.
   - Evidence: B0 clean-base 10k has 9/32 heads with `|tau|>0.9`, best `L0H0 tau=1.000`.
   - Source: `block_lo_arm_order_network/probe_results/clean_base_random_perm/head_scan_10k.json`.

2. The order-bearing signal survives B1/predictor-aligned extraction.
   - Evidence: B1 clean-base ladder has `L0H0 tau=1.000000` at 10k/50k/60k; latest continuous seed124 final has `L0H4 tau=0.957589`.
   - Sources: `block_lo_arm_order_network/batch_readout/logs/per_head_scan_b1/ckpt*_seed*.json`; `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv`.

3. Legacy B0 `g_beta` reads structured `B`, not a constant prior.
   - Evidence: real B `tau_vs_l2r=0.967500`; Gaussian `-0.003353`; entry-shuffled `0.016488`; row/col shuffled `0.012748`; Gaussian pairwise `0.000313`.
   - Source: `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`.

4. Legacy B0 frozen `g_beta` accelerates training across two matched seed groups.
   - Evidence: seed123 recovery 86.0/83.0/58.7%; seed42 recovery 106.1/101.5/66.9%; conservative step saving 33-42%.
   - Source: `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md`.

5. Granularity and 317M diagnostics show the signal is not limited to the 47M/64-block setting.
   - Evidence: B0 granularity heavy tau > 0.96; B0 317M has 6/256 heads with `|tau|>0.9`, best `L0H10 tau=0.955357`.
   - Sources: `analyses/block_granularity_scan_results/scan_step5000_M100.json`; `block_lo_arm_order_network/probe_results/large_random_baseline_16l16h1024d/head_scan_step5000.json`.

6. B1 is the target protocol migration direction, while current acceleration evidence remains B0 legacy.
   - Evidence: existing B1 results use `none_mode=predictor`; hook/provider evidence uses B0.
   - Source: `reports/b1_attention_extraction_summary_20260615/01_b1_protocol_definition.md`; `reports/b1_attention_extraction_summary_20260615/05_impact_on_existing_claims.md`.

## CANNOT CLAIM

1. B1 frozen-hook acceleration is established.
2. Existing frozen-hook acceleration already used B1.
3. The same `g_beta` transfers across seeds or heads.
4. Fully label-free head selection and `g_beta` training are end-to-end verified.
5. `g_beta` discovers orders beyond L2R.
6. ori-L2R is an upper bound.
7. val_unstructured also improves.
8. 317M hook acceleration is established.
9. Image/multimodal extension is solved.
10. No-PE / PE-free order discovery is established.
11. Per-sample adaptive order is established.
12. B1 result files use `none_mode=b1`; current result-bearing B1 files use `none_mode=predictor`.

## SHOULD SAY INSTEAD

| Do not say | Say instead |
|---|---|
| B1 proves the training result | B1 verifies attention-signal robustness; current verified acceleration is legacy B0 hook path |
| B0 was wrong | B1 is the collaborator-aligned convention; B0 is legacy controller evidence |
| Beats L2R | Reaches or exceeds the L2R reference under canonical-order validation |
| ori-L2R upper bound | ori-L2R reference |
| Universal improvement | Canonical-order specialization / training acceleration |
| Fully label-free | Cheap head diagnostics plus CDL-supervised `g_beta` readout |
| All heads encode order | Sparse order-bearing heads |
| 317M works | 317M diagnostic signal exists; 317M hook not established |
| Image side solved | Image/multimodal remains future work or separate evidence package |
| B1 is `none_mode=b1` | Existing B1 result files use B1/predictor-aligned `none_mode=predictor` |

## Final Meeting Claim Lock

Most stable one-paragraph claim:

> In text-side AO-GPT, random-order training produces sparse order-bearing attention heads. We distill selected-head block attention graphs into a frozen `g_beta` controller, which under the legacy B0 hook path accelerates canonical-order training across two matched seed groups with 33-42% conservative step saving. After aligning attention diagnostics to the B1/predictor convention, the upstream order-bearing signal remains strong, showing that the mechanism is not a B0 extraction artifact. B1 is now the target migration protocol, but B1-controller acceleration should be claimed only after a separate B1 hook result.

