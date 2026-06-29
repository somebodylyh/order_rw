# Slide Outline

## Slide 1 - One-Line Result

Title:

> Attention-derived order controller accelerates text-side AO-GPT training

Content:

- Random-order AO-GPT develops sparse order-bearing heads.
- Selected-head `B^{l,h}` can be distilled into frozen `g_beta`.
- Legacy B0 hook acceleration is verified across 2 matched seed groups.
- Conservative step saving: 33-42%.
- B1/predictor-aligned diagnostics show signal robustness.

## Slide 2 - Problem and Hypothesis

Content:

- AO-GPT permits arbitrary reveal order.
- Random order is simple but may be inefficient.
- Hypothesis: random-order training induces sparse heads that encode L2R-like physical structure.
- If true, use attention as a controller signal.

## Slide 3 - Clean-Permutation Protocol

Content:

- Fixed clean block permutation, not per-sample permutation.
- Model coordinate and physical coordinate are distinct.
- Physical index is not directly given to the model.
- Diagnostic can inverse-remap to test physical L2R recovery.
- This is a text-side training-acceleration setting.

## Slide 4 - Method Pipeline

Content:

```text
B^{l,h} -> CDL teacher -> g_beta -> frozen hook -> val_ori_l2r_block
```

- `B^{l,h}` is per-head block attention graph.
- CDL is offline teacher / order extractor.
- `g_beta` is learned readout, not direct online CDL.
- Frozen hook deploys batch-global canonical order.

## Slide 5 - Attention Signal Evidence

Content:

- B0 clean-base: 9/32 strong heads, best `L0H0 tau=1.000`.
- B1 clean-base ladder: `L0H0 tau=1.000` at 10k/50k/60k.
- B1 continuous final: `L0H4 tau=0.957589`.
- 317M B0: 6/256 strong heads, best `L0H10 tau=0.955357`.
- Emphasis: sparse order-bearing heads, not all-head collapse.

## Slide 6 - `g_beta` Sanity

Content:

- Real B: `tau=+0.9675`.
- Gaussian B: `tau=-0.0034`.
- Entry-shuffled B: `tau=+0.0165`.
- Row/col shuffled B: `tau=+0.0127`.
- Gaussian pairwise tau: `0.0003`.
- Zero B: `tau=1.0`, margin=0 tie-breaking.
- Conclusion: legacy B0 `g_beta` reads structured B, not a constant L2R prior.

## Slide 7 - Training Acceleration

Content:

- seed123 L0H2 recovery: 86.0%, 83.0%, 58.7%.
- seed42 L0H4 recovery: 106.1%, 101.5%, 66.9%.
- Step saving: 33-42% for from10k/from20k across two seeds.
- Metric: `val_ori_l2r_block`.
- ori-L2R is a reference, not an upper bound.
- Protocol: legacy B0 hook path.

## Slide 8 - B1 Update and Impact

Content:

- B1/predictor extraction is collaborator-aligned diagnostic convention.
- Result-bearing files use `none_mode=predictor`.
- Signal survives B1: clean-base late checkpoints and latest continuous seed124.
- Current acceleration evidence remains B0 legacy.
- B1 is target migration direction; B1 controller claim needs rerun.

## Slide 9 - Limitations / Reviewer Attack Surface

Content:

- val_unstructured degrades: order-specialization trade-off.
- No B1 hook acceleration yet.
- No B1 `g_beta` sanity yet.
- Label-free audition not fully closed.
- No 317M hook.
- CDL teacher matched run final result pending in this package.

## Slide 10 - Decisions Needed

Content:

1. Text-side paper now vs expand to scale/image/multimodal.
2. B1 hook smoke/full rerun vs B1 diagnostic only for current paper.
3. Third seed vs 317M hook vs label-free audition closure.
4. How to place CDL teacher: main baseline only after matched verification; otherwise supplementary.
5. Which figures to regenerate/check before paper draft.

