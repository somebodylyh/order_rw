# Research Findings Index

This repository now keeps only the language / WikiText103 mainline.

Read `docs/findings/findings_language.md` for current results, interpretation
boundaries, and active online-order status.

## Shared Research Question

Can local pair/block structure mined from a random-order AO-GPT language
backbone be aggregated into curriculum units or recovered policies that reveal
meaningful generation structure?

## Safe Current Framing

- The active evidence supports recoverable local block-level L2R structure.
- It does not yet prove complete global L2R induction.
- Training and curriculum consume current-frame units.
- `*_original` and `OriginalL2R` fields are diagnostics or oracle baselines.
- `non_permute` has absolute-position prior.
- `block_order_block_len > 1` has block-internal fixed-order prior.
- Token-micro is retired from the active method and should be treated only as
  diagnostic evidence.

## Current Snapshot

- active thread: WikiText103 `seq256/permute/block64`
  Laplacian/Fiedler distribution teacher plus MLP distillation
- mechanisms: layer-mean pairwise-max affinity, graph Laplacian Fiedler axis,
  current-frame loss orientation, EMA/no-EMA ablation, and frozen MLP
  insertion tests
- next step: compare rank EMA, continuous Fiedler-priority EMA, and no-EMA
  variants, while using try28 MLP as the current distillation candidate
- strongest historical language result:
  `Report/language/wikitext103/curriculum/permute/seq256/block64/hierarchical_block64_early_stop_open_level_gain_only-6-stage`
