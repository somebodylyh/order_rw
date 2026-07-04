# Canonical Token-Level Signal Scan Design

## Goal

Test whether the strict65 block-level order signal can be recovered directly
from token-level attention while matching the established block-level
extraction protocol exactly.

## Canonical protocol

- Use `total = M * batch_size` samples, with defaults `M=40` and
  `batch_size=4`.
- Load chunks through the canonical checkpoint loader so the checkpoint's data
  permutation and model/physical coordinate convention are applied.
- Give every sample an independent random block reveal order generated with
  seed `seed + sample_index`; preserve left-to-right token order inside each
  four-token block.
- Extract model-frame loss-aligned attention with the None node kept separate.
- Accumulate both block-level `(64, 65)` and token-level `(256, 257)` A maps
  over all samples and reveal orders, then take one global mean for each head.
- Build the strict65/strict257 B graphs and run the same greedy `C-D+L` CDL
  rollout at both granularities.

## Coordinate mapping and metrics

- Translate block CDL output from model blocks to physical blocks with the
  checkpoint's `inv_perm_model_to_phys` only after rollout.
- Translate token CDL output with the corresponding token map: model block to
  physical block, preserving the within-block offset.
- Compare permutations using Kendall tau on inverse-rank vectors, never by
  correlating order-list entries directly.
- Report signed mean/count statistics, not absolute values presented as
  alignment.
- Primary metrics:
  - Kendall tau between collapsed physical token CDL and physical block CDL.
  - Kendall tau between physical token CDL and expanded physical block CDL.
  - Same-block boundary score in the physical frame.
  - Block CDL tau against physical L2R as the established baseline.
- Verify numerically that block B is the expected masked 4x4 coarse-graining
  of token B outside the diagonal.

## Implementation shape

Correct `analyses/token_level_signal_scan.py` rather than retaining the fixed
L2R protocol as a second default. Keep forward microbatching separate from the
semantic `batch_size` multiplier. Record the canonical protocol, sample count,
random-reveal convention, coordinate mapping, and rank-tau convention in the
JSON metadata.

## Tests

Add focused unit tests for:

1. deterministic independent random reveal generation;
2. model-block and model-token to physical-coordinate translation;
3. inverse-rank Kendall tau, including a counterexample where direct order-list
   correlation is wrong;
4. masked token-to-block graph coarse-graining;
5. `total = M * batch_size` grouping semantics.

After unit tests pass, run the method-gbeta 50k checkpoint with
`M=40, batch_size=4` and inspect the canonical JSON summary.
