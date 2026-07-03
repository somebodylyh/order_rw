# WikiText103 Config Layout

The active layout is now organized first by sequence length, then by whether
the input data is permuted, and finally by the number of reveal blocks:

- `seq256/non_permute/block16/`
- `seq256/non_permute/block32/`
- `seq256/non_permute/block64/`
- `seq256/non_permute/block128/`
- `seq256/permute/block16/`
- `seq256/permute/block32/`
- ...
- `seq512/non_permute/block16/`
- `seq512/permute/block128/`

Inside each block directory, the standard entry points are:

- `random.py`
- `ar.py`
- `segment_curriculum.py`

Some directories also include attention-spectral / online-order configs:

- `seq256/non_permute/block64/online_spectral_fixed_head_order.py`
- `seq256/permute/block64/online_spectral_fixed_head_order.py`
- `seq80/non_permute/block1/online_spectral_cached_order.py`
- `seq80/non_permute/block1/segment_curriculum_attention_spectral_crossaxis_svd.py`

`online_spectral_fixed_head_order.py` and
`online_spectral_cached_order.py` use `OnlineSpectralFixedHeadOrder`: during
training they maintain one cached order, refresh it from a fixed layer/head
attention matrix, enumerate spectral candidates, and cache the highest
attention-spectral scoring candidate. This is a dynamic online policy. If the
research question needs a stable learned order, freeze a recovered order before
the main training/evaluation phase.

The current research mainline is block-level recovery, especially
`seq256/{non_permute,permute}/block32`, `block64`, and `block128`.
Token/block1 configs are retained for diagnostics and historical comparison.
`segment_curriculum_micro.py` presets, where present, belong to the retired
token-micro graph aggregation path and should not be used for new mainline runs.

Current block64 focus:

- `seq256/permute/block64/segment_curriculum.py`
  The analyzed 6-stage 50k run. It uses relaxed late margins and produced useful
  local order recovery, but also showed noisy seeds from early stages being
  amplified into long units.
- `seq256/permute/block64/segment_curriculum_strict_front_margin.py`
  The next ablation. It keeps the same 50k budget but uses higher early
  `aggregation_margin_thresholds`, decreasing
  `aggregate_top_k_pairs_per_stage`, decreasing late `drop_weights`, narrower
  late `attn_top_ks`, and more pair/attention scoring batches.

For the strict-front-margin ablation, keep `benchmark_num_levels=1` at first.
If the run under-aggregates badly, loosen margins/top-k before trying multiple
benchmark levels on a single frozen checkpoint.

These are all standalone config files, so each one can be customized
independently for block-specific or run-specific settings such as W&B names.

PPL/eval caveat: under `permute_data=True`, `OriginalL2R` is an oracle that maps
the hidden original text order back into the current frame from checkpoint
permutation metadata. Use it as an upper bound only. For no-prior comparisons,
prefer current-frame `AR`, `Random`, `CheckpointOnlineSpectralOrder`, or
orders produced by recovery/curriculum without original-frame metadata.

Legacy paths such as `block16/standard/`, `block32/standard/`, and
`block_permute/` are kept for backward compatibility with existing commands
and historical reports.
