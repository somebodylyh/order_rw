# 01 — Strict 65-node Protocol Definition

Date: 2026-06-17

## Motivation

Prior protocols either folded [None] into physical block 0 (B0/B1 conventions) or used `inv_perm` during graph construction (oracle-remapped). The strict 65-node protocol eliminates both concerns:

1. **[None] is a separate BOS node**, not folded into any content block.
2. **Graph is constructed in model-frame coordinates** (no `inv_perm` during extraction).
3. **`inv_perm` is used ONLY posthoc** to translate sigma_model → sigma_phys for scoring.

## Node Convention

```
node 0  = None / BOS start node
node 1  = physical content block 0  = x0, x1, x2, x3
node 2  = physical content block 1  = x4, x5, x6, x7
...
node 64 = physical content block 63 = x252, x253, x254, x255
```

In model-frame (strict label-free extraction):
```
node 0  = None / BOS start node
node 1  = model block 0
node 2  = model block 1
...
node 64 = model block 63
```

## Extraction: Loss-Aligned AR Frame

Attention slice: `attn[:T, :T]` where T = SEQ_LEN = 256.

- **Query axis**: aggregated by target block (the block being predicted).
- **Key/Source axis**: aggregated by source/context block, with source node 0 reserved for [None].
- **No inv_perm during aggregation**: query_labels = reveal_tokens // BLOCK_LEN (model blocks).
- **[None] preserved**: `source_labels[0] = 0`, `source_labels[1:] = 1 + key_model_blocks`.

Result: `A[target_model_block, source_node]` where source_node ∈ {0=None, 1+i=model_block_i}.

## Graph: B65

```
B65 = build_none_separated_B(A)   # (N+1, N+1)
B[0, 1:]  = A[:, 0]              # None → content blocks
B[1:, 1:] = A[:, 1:].T           # content block i → content block j
diag(B) = 0
```

B65 is a directed graph: B[u, v] = edge weight from u to v.

## Rollout

```
start = node 0 (None)
while unselected:
    scores = teacher_scores(B, selected, unselected, last, mode)
    next = argmax(scores)
    selected.append(next)
    unselected.remove(next)

content_order = selected[1:] - 1   # remove None, 0-index
```

The first content block is selected by:
- C(v) = mean_{u ∈ S_t} B[u, v] (revealed nodes attend to v)
- D(v) = mean_{u ∈ U_t\{v}} B[u, v] (unrevealed nodes attend to v)
- L(v) = B[last, v] (transition from last-revealed)
- Score = C - D + L (or L-only for transition-only readout)

## Posthoc Translation (label-free only)

```python
sigma_phys[t] = inv_perm[sigma_model[t]]
tau = kendalltau(sigma_phys, [0, 1, ..., 63])
```

`inv_perm` is used EXACTLY ONCE: to translate the final sigma_model into physical coordinates for scoring against L2R.

## Why Oracle-Remapped Is Valid (Not a Leak for CDL)

**inv_perm during construction is OK. inv_perm during CDL readout is NOT.**

1. **Construction**: inv_perm can be used to build the graph in physical coordinates. This just chooses a coordinate frame. The graph edge weights are the same structure — CDL doesn't look at node labels, only edge weights.
2. **CDL rollout**: inv_perm is NEVER used. CDL scores are `C-D+L` computed purely from graph edge weights `B[u,v]`. Node "identity" (physical block 0 vs physical block 37) is invisible to CDL.
3. **Posthoc scoring**: inv_perm is used ONLY to translate sigma_model → sigma_phys for comparison with L2R.

The "leak" would be if someone manually set CDL's first block to physical 0 using external knowledge. That doesn't happen — CDL picks the first content block based on None→content edge weights in the graph.

**Therefore both oracle-remapped (physical-frame construction) and strict label-free (model-frame construction + posthoc translation) are valid protocols. They produce identical results because CDL is permutation-equivariant.**

## What This Protocol Does NOT Do

- Does NOT manually set `[None] → physical0` edge.
- Does NOT fold [None] into block 0.
- Does NOT use `inv_perm` to label nodes during construction.
- Does NOT pre-select the first content block.
- Does NOT train any parameters.
