# Recommended Boss Slides

## Slide 1 - Why we revisited attention extraction

Message:

- Old story: attention graph seems to recover L2R.
- Problem: B0/B1 extraction, predictor shift, and None anchor were mixed.
- Goal this week: separate diagnostic signal from anchored controller protocol.

Visual:

- Use a small text/table schematic.
- Optional: `cleanbase_step10000_L0H0_token_attention_model_frame.png` vs `cleanbase_step10000_L0H0_token_attention_original_l2r.png` to show coordinate caution.

## Slide 2 - Correct block-level graph protocol

Message:

- Use strict 65-node None-separated graph.
- node0=None/BOS, nodes1..64=physical blocks.
- Rollout starts from None; first content block is selected by edges/readout.
- None is not manually folded into phys0.

Visual:

- NEEDS_FIGURE: strict 65-node schematic.
- Suggested drawing: one BOS node on the left, 64 content nodes on the right, first rollout edge selected by graph score, label "no manual None->phys0".

## Slide 3 - Strict label-free discovery result

Message:

- Collaborator @50k L0H1-L0H4 recover full L2R under strict 65-node LF.
- `tau=1.000`, `first=0`, `phys0_rank=0`, `prefix@4=4`, `prefix@8=8`.

Visual:

- Use key result table from `03_key_results.md`.
- NEEDS_FIGURE if a unified L0H1-L0H4 heatmap panel is preferred.

## Slide 4 - Not trivial: L0H7 fail + destroyed controls

Message:

- L0H7 fails: `tau=0.292`, `first=45`, `phys0_rank=20`.
- Destroyed controls near random; real-destroyed gap about 0.95 for strong heads.
- Therefore not all heads, not tie-break, not graph-statistic artifact.

Visual:

- `reports/collaborator_ckpt_b1_scan_20260616/figures/loss_aligned_L0H7_M20_b8_seed0/B65_none_separated_heatmap.png`
- destroyed control summary table from `03_key_results.md`.

## Slide 5 - Extraction frame matters

Message:

- B1 predictor content-only can recover order axis but can lose anchor.
- Loss-aligned AR + None-separated retains independent None/BOS and recovers start.

Visual:

- Table:
  - L0H1/L0H2/L0H4: B1 content-only tau=0.655 vs strict tau=1.000.
  - L0H3: 1.000 vs 1.000.
- Optional B1 model-vs-physical heatmap.

## Slide 6 - Claim update

Can claim:

- Strict 65-node label-free block-order discovery exists in selected early heads.
- None is independent BOS and first block is graph-selected.
- Destroyed controls are near random.
- strict LF vs oracle-remapped consistency is permutation equivariance.

Cannot claim:

- all heads discover L2R;
- B1 content-only alone is sufficient;
- frozen hook already uses strict 65-node teacher;
- training loss is true block-level.

## Slide 7 - Next step

Message:

- Close mechanism-to-controller loop.
- strict 65-node teacher -> `g_beta` distillation -> hook smoke -> compare to legacy B0 hook.

Visual:

- Simple pipeline diagram.
- Mark current state: mechanism verified; controller rerun pending.

