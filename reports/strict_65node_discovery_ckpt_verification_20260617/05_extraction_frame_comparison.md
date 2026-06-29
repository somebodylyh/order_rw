# 05 — Extraction Frame Comparison

Date: 2026-06-17

## Two Extraction Frames

### B1 Predictor Frame (attn[:-1, :-1])
- Used in: `_attn_to_A_block_b1_vec`, `_attn_to_A_block_predictor_vec`
- Slice: removes last query row and last key column
- Predictor perspective: "what does the predictor attend to when predicting the next token?"
- [None] token at position 0 is in the key context

### Loss-Aligned AR Frame (attn[:T, :T])
- Used in: `_attn_to_A_block_loss_aligned_with_none_vec`, `_attn_to_A_block_loss_aligned_with_none_model_vec`
- Slice: removes only the last row/column (the final unused token)
- Loss-aligned perspective: "what context does each predicted token attend to?"
- [None] is a separate source column (column 0 in A_with_none)

## Head-Level Comparison (Collaborator @50k)

### Content-Only Readout (no [None] at all)

| Head | B1 predictor + content-only tau | Loss-aligned AR + None-sep tau | Notes |
|------|-------------------------------|-------------------------------|-------|
| L0H1 | 0.655 | 1.000 | B1: cyclic perfect, anchor=6. AR: anchored perfect. |
| L0H2 | 0.655 | 1.000 | Same pattern. |
| L0H3 | 1.000 | 1.000 | Strong in both frames. |
| L0H4 | 0.655 | 1.000 | Same as L0H1/H2. |

**Source**: `no_inv_label_free_readout_posthoc_M20_seed0.json` (B1 content-only) vs `strict_label_free_65_search/` (loss-aligned AR + None-sep).

### Interpretation

1. **Both frames capture the L2R axis**: B1 predictor content-only gets cyclic tau = 1.000 for L0H1/H2/H4 (perfect order, wrong start). The structural axis is present in both frames.

2. **[None] provides the anchor**: Loss-aligned AR frame with [None] as independent BOS node allows the rollout to select the correct starting block. B1 content-only (without [None]) loses this anchor for 3/4 heads.

3. **L0H3 is robust**: It recovers anchored L2R in both frames. This head's attention structure is strong enough that even without [None] as BOS, the content-only graph has sufficient self-anchoring.

4. **Extraction frame matters more than inv_perm usage**: The B1 vs AR frame difference (0.655 vs 1.000) is much larger than the strict-LF vs oracle-remapped difference (0.000).

## Why B1 Content-Only Fails Anchor

In B1 predictor frame, the [None] token is removed (content-only). The CDL rollout starts from a "most-ready" source node selected by `readiness = out_degree - alpha * in_degree`. Without [None] as an explicit start, the source selection may pick a block mid-sequence (e.g., physical block 6), producing a perfect cyclic order but missing the anchored physical-0 start.

In loss-aligned AR + None-sep, [None] is node 0 and rollout always starts from it. The first content block is then selected by the [None]→content edges (primarily via the L transition score: B[None, candidate]). If [None] attends most strongly to physical block 0's predecessor context, the correct anchor is recovered.

## Recommendation

- **Primary evidence**: Loss-aligned AR + None-separated 65-node (strict label-free).
- **Supporting evidence**: B1 predictor content-only (demonstrates axis signal without [None] anchor).
- **Do NOT use B1 content-only alone** as evidence of anchored discovery for heads other than L0H3.
