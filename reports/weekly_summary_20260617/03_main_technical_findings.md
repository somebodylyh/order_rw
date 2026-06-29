# Main Technical Findings

## Finding 1 - Seed/baseline audit closed old training-result confusion

The old seed42 baseline confusion is resolved.

- seed42 random baseline @50k is `3.466`, not `3.354`.
- `3.354` came from a frozen_beta run, not from a clean random baseline.
- seed42 gap is normal: random `3.466` vs L2R reference `3.341`, gap `0.125`.
- Recovery and step-saving are verified again from primary eval curves.
- Existing legacy frozen `g_beta` acceleration still holds across two matched seed groups.

Key verified values:

- seed-123 frozen_beta L0H2 recovery: from10k `86.0%`, from20k `83.0%`, from40k `58.7%`.
- seed42 frozen_beta L0H4 recovery: from10k `106.1%`, from20k `101.5%`, from40k `66.9%`.
- conservative frozen_beta step saving from from10k/from20k across seed-123 and seed42: `33-42%`.

Source:

- `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md`
- `reports/evidence_package_20260613_verified/03_DEPRECATED_NUMBERS.md`

## Finding 2 - B1 diagnostic shows signal survives collaborator-aligned extraction

B1/predictor-aligned diagnostic signal is strong.

- Clean-base B1 ladder: L0H0 has `tau=1.000000` at 10k, 50k, and 60k across scan seeds.
- Clean-base strong-head counts: 10k has 7/32 heads with `|tau|>0.9`; 50k/60k have 6/32.
- Continuous B1 tracking seed=124 at 60k: final best head L0H4 has `tau=0.957589`, `mean_pairwise_tau=0.930060`.
- L0H4 is top-by-absolute-tau in 5768 / 6001 tracked steps.

Boundary:

- B1 diagnostic does not automatically imply B1 hook acceleration.
- Existing frozen hook acceleration is legacy B0 controller path.

Source:

- `reports/b1_attention_extraction_summary_20260615/03_b1_signal_summary.md`
- `reports/b1_attention_extraction_summary_20260615/05_impact_on_existing_claims.md`

## Finding 3 - AR next-token shift requires asymmetric block alignment

The token-level AR objective introduces a shift between target/query positions and source/content positions.

Definitions:

```text
data block0              = x0,x1,x2,x3
query positions block0   = [None],x0,x1,x2
source/content block0    = x0,x1,x2,x3
```

Conclusion:

Cannot use one block label set for both query and key. The block graph is an aggregation of token-level AR attention, not a true block-level training loss.

Source:

- `reports/b1_attention_extraction_summary_20260615/01_b1_protocol_definition.md`
- `reports/strict_65node_discovery_ckpt_verification_20260617/05_extraction_frame_comparison.md`

## Finding 4 - None must be separated for block-level discovery

Folding `[None]` into physical block0 creates an anchored controller:

- It provides canonical start information.
- It can still be useful for controller acceleration.
- It is not sufficient to prove label-free start discovery.

For block-level discovery, use a None-separated 65-node graph:

- node0 = None / BOS
- node `1+i` = physical content block `i`
- rollout starts from None
- first content block is selected by graph structure

Source:

- `reports/strict_65node_discovery_ckpt_verification_20260617/01_protocol_definition.md`
- `reports/strict_65node_discovery_ckpt_verification_20260617/05_extraction_frame_comparison.md`

## Finding 5 - Strict 65-node label-free discovery succeeds in selected early heads

Collaborator @50k strict LF full sweep:

- L0H1-L0H4 strong pass under `L` and `C-D+L`.
- For the primary `L` rows: `tau=1.000`, `first=0`, `phys0_rank=0`, `p4=4`, `p8=8`.
- Destroyed mean |tau| for L rows: L0H1 `0.0498`, L0H2 `0.0531`, L0H3 `0.0501`, L0H4 `0.0554`.
- L0H7 fails under `C-D+L`: `tau=0.2917`, `first=45`, `phys0_rank=20`, `p4=0`, `p8=0`.

This shows the result is not trivial and not universal across heads.

Source:

- `reports/collaborator_ckpt_b1_scan_20260616/strict_label_free_65_search/all_head_methods_strict_label_free_65.tsv`
- `reports/strict_65node_discovery_ckpt_verification_20260617/02_existing_result_summary.md`

## Finding 6 - Clean-base sweep shows stability

Clean-base strict LF 9-step ladder:

- 0 and 1k: no strong heads.
- 5k-60k: 10-16 strong rows at every checkpoint.
- Best tau from 5k onward is `1.000`.
- Stable strong heads include L1H0-L1H4 under both `L` and `C-D+L`; L0H0 emerges later.
- Strong heads drift across runs and checkpoints. Phenomenon is stable; head identity is not fixed.

Source:

- `reports/strict_65node_discovery_ckpt_verification_20260617/04_ckpt_sweep_results.md`
- `reports/strict_65node_discovery_ckpt_verification_20260617/strict_65node_ckpt_sweep.tsv`

## Finding 7 - inv_perm boundary clarified

Correct boundary:

| Stage | inv_perm allowed? | Reason |
|---|---:|---|
| Graph construction | yes | Graph labels may be remapped; edge structure is preserved. |
| CDL rollout | no | Rollout must use graph edges only; no oracle order during decision. |
| Posthoc scoring | yes | Evaluation translation only. |

Strict label-free and oracle-remapped 9/9 matching is permutation equivariance, not a leak. The important boundary is that rollout decisions cannot use `inv_perm`; posthoc scoring can translate the discovered order for evaluation.

Source:

- `reports/strict_65node_discovery_ckpt_verification_20260617/02_existing_result_summary.md`
- `reports/strict_65node_discovery_ckpt_verification_20260617/04_ckpt_sweep_results.md`

