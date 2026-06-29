# 06 — Destroyed Controls Summary

Date: 2026-06-17

## Control Types

1. **Entry Shuffle**: Shuffle all allowed edge weights in B65 while keeping None as node 0 and diagonal zero. Preserves the degree distribution but destroys graph structure.

2. **Content Label Permutation**: Apply a random permutation to content node labels (nodes 1..64) while keeping None (node 0) fixed. Preserves the edge weight multiset but destroys which edges connect which blocks.

For strict label-free protocol, controls are applied to model-frame B65 before posthoc translation.

## Collaborator @50k (M=20, Full Sweep, 20 Control Seeds)

### Strong Pass Heads: Destroyed |τ|

| Head | Method | Real τ | Destroyed \|τ\| (mean across 40 samples) | Gap |
|------|--------|---------|------------------------------------------|-----|
| L0H1 | L | 1.000 | 0.050 | +0.950 |
| L0H2 | L | 1.000 | 0.053 | +0.947 |
| L0H3 | L | 1.000 | 0.050 | +0.950 |
| L0H4 | L | 1.000 | 0.055 | +0.945 |
| L0H1 | C-D+L | 1.000 | 0.055 | +0.945 |
| L0H2 | C-D+L | 1.000 | 0.059 | +0.941 |
| L0H3 | C-D+L | 1.000 | 0.059 | +0.941 |
| L0H4 | C-D+L | 1.000 | 0.059 | +0.941 |

### Failed Head Example: Destroyed |τ|

| Head | Method | Real τ | Destroyed \|τ\| | Gap |
|------|--------|---------|-----------------|-----|
| L0H7 | C-D+L | 0.292 | 0.070 | +0.222 |

## Clean Base Ladder (M=8, 5 Control Seeds Each)

### Mean Destroyed |τ| Across All Steps

| Step | Mean destroyed \|τ\| (L method) | Mean destroyed \|τ\| (C-D+L) |
|------|-------------------------------|------------------------------|
| 0 | 0.063 | 0.063 |
| 1000 | 0.069 | 0.069 |
| 5000 | 0.050 | 0.050 |
| 10000 | 0.050 | 0.050 |
| 20000 | 0.052 | 0.052 |
| 30000 | 0.053 | 0.053 |
| 40000 | 0.055 | 0.055 |
| 50000 | 0.057 | 0.057 |
| 60000 | 0.054 | 0.054 |

## Interpretation

1. **All destroyed controls produce |τ| ≤ 0.07**: Near-random across all checkpoints, heads, methods, and control types. The signal is NOT from numerical tie-breaking or graph statistics.

2. **Real τ − destroyed τ gap ≈ 0.95** for strong_pass heads: The gap is enormous — 20× larger than the destroyed baseline.

3. **Controls at step 0/1k (no training)**: Destroyed |τ| ≈ 0.063–0.069, similar to later steps. The real τ at these steps is also near 0.1–0.2, meaning even without training, the graph has slightly more structure than destroyed controls — but not enough to pass the gate.

4. **Gaussian B control**: Not run for 65-node protocol. Previous B0 g_β sanity showed Gaussian B τ = −0.0034 (see `gbeta-input-sanity-20260611`). The 65-node destroyed controls serve the same purpose.

## Conclusion

The recovered L2R order depends on real attention structure encoded in the graph. Destroying that structure (by shuffling edges or permuting labels) collapses τ to near-random. This rules out tie-breaking artifacts, constant-prior explanations, and graph-statistic confounds.
