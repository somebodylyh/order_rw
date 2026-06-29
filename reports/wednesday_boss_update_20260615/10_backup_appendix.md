# Backup Appendix

## Clean Permutation Protocol

- Training uses a fixed block permutation, not per-sample random permutations.
- Model coordinate and physical coordinate are distinct.
- The model is not directly given physical L2R indices.
- Diagnostics can inverse-remap to physical coordinate to test L2R-like recovery.
- Current claim is not per-sample adaptive order or document unscrambling.

## B0 vs B1 Extraction

Current summary:

| Item | B0 legacy | B1/predictor result-bearing convention |
|---|---|---|
| Role | Existing `g_beta` dataset, sanity, hook acceleration | Collaborator-aligned attention diagnostic |
| Current result files | B0 selected-head/controller outputs | `none_mode=predictor` |
| Attention frame | Legacy canonical extraction with `[None] -> physical block 0` folding | Predictor frame `attn[:-1, :-1]` |
| Block aggregation | Segment/block mean | Reshape to `(N, block_len, N, block_len)`, mean over token axes |
| Transpose | `B=A.T` | `B=A.T` |
| Diagonal | zeroed | zeroed |
| Physical remap | yes in B0 path | not in `none_mode=predictor` result files |
| Hook acceleration evidence | yes | not yet established |

Important caveat: code enum `none_mode=b1` means predictor frame plus physical remap, but no result file using that enum was found in the B1 summary. Existing result-bearing B1 means B1/predictor-aligned `none_mode=predictor`.

## Recovery Formula

At fixed step 50k:

```text
Recovery = (L_random - L_method) / (L_random - L_L2R) * 100%
```

Use `val_ori_l2r_block` as the primary metric.

Examples:

- seed123 random = 3.445, L2R ref = 3.305, frozen from10k = 3.325 -> recovery 86.0%.
- seed42 random = 3.466, L2R ref = 3.341, frozen from10k = 3.334 -> recovery 106.1%.

Source: `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md`.

## Step Saving Formula

At threshold 3.47:

```text
Saving = (T_random - T_method) / T_random * 100%
```

Verified values use first eval checkpoint at or below 3.47, no interpolation. `T_random=42000` for the seed123 random baseline.

Conservative range:

- seed123 from10k: 41.7%.
- seed42 from10k: 39.3%.
- seed123 from20k: 35.7%.
- seed42 from20k: 33.3%.

Report as 33-42%.

## `g_beta` Ranking Loss And Readout

The deployed object is a learned `g_beta`, trained from selected-head `B` to match teacher order through a ranking objective. The key conceptual point for the boss meeting:

- CDL supplies labels / teacher order offline.
- `g_beta` learns to map graph structure to order scores.
- Frozen hook uses `g_beta`, not direct online CDL.

Current verified sanity is B0 controller path. No B1 `g_beta` sanity result was found in the current package.

## CDL Teacher Utility

CDL is useful as:

- an offline order extractor from `B`;
- a teacher for `g_beta`;
- a supplementary upper/reference comparison.

Boundary:

- Existing CDL teacher table has seed mismatch / seed2 narrow gap caveat in the verified package.
- Do not make it the primary matched training result unless the seed123 matched CDL run is verified.

## val_unstructured Interpretation

`val_unstructured_order` worsens when training specializes toward a single canonical order. This is expected for fixed-order specialization.

Verified examples:

- seed123 frozen from10k: `val_unstructured` delta vs random = +0.424.
- seed42 frozen from10k: delta = +1.283.
- L2R reference degrades even more under unstructured evaluation.

Use this framing:

> The controller improves canonical-order training efficiency. It is not claimed to improve likelihood under arbitrary random evaluation orders.

Source: `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md` Table V9.

## Seed Group Audit

Important mapping:

| Canonical group | seed | Random baseline | L2R ref | Note |
|---|---:|---|---|---|
| seed123 | 123 | `random_baseline_continuous_jun08_seed2` | `l2r_continuous_seed123` | Main frozen `g_beta` group |
| seed42 | 42 | `random_baseline_continuous_jun05` | `l2r_continuous_jun05` | Main frozen `g_beta` group |
| seed2 | 2 | `random_baseline_continuous_jun11_seed2_headtrack` | `l2r_continuous_seed2` | CDL teacher supplementary group |

Critical correction:

- seed42 random baseline is 3.466, not 3.354.
- seed123 L2R ref is 3.305, not seed2's 3.370.

## Unresolved Issues

| Issue | Current status | Meeting handling |
|---|---|---|
| B1 hook acceleration | Not established | State as pending / optional rerun |
| B1 `g_beta` sanity | Not found | Do not claim |
| B1 shuffled-L2R control | Not found | Use B0 as legacy only |
| B1 317M scan | Not found | Keep 317M as B0 diagnostic |
| B1 naming | result files use `none_mode=predictor`; code enum `b1` differs | Write `B1/predictor-aligned diagnostic` |
| Label-free audition | Not end-to-end verified | Limitation / optional follow-up |
| CDL teacher matched run | Marked in progress in verified package | Pending unless final result is separately verified |
| 50k B0 seed123 signal decay | B0 continuous 50k has no strong heads | Do not compare directly to clean-base B1 ladder; use protocol-qualified wording |

