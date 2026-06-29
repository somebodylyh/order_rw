# Method Overview

## Module A — Random-Order AO-GPT Training

AO-GPT is a standard autoregressive transformer trained with random block-level permutations:

- **Model**: 4-layer, 8-head, d=384 (~47M parameters). Block size = 256 tokens, organized into 64 blocks × 4 tokens/block.
- **Training**: Each training step randomly permutes the 64 content blocks (and their corresponding segment positions). The model predicts tokens autoregressively within the permuted sequence.
- **Loss**: Token-level cross-entropy AR loss. The block permutation affects which context each token sees, but the loss itself is still per-token.
- **Data**: Wikitext-103, continuous streaming mode (avoids fixed-chunk overfitting).
- **Key property**: The model never sees canonical (L2R) order during training. All order information must be inferred from token co-occurrence statistics within the fixed 4-token blocks.

**Critical**: The training loss is token-level AR, NOT block-level. The block graph is an attention aggregation diagnostic, not the training objective.

---

## Module B — Attention Graph Extraction

### Protocol Evolution

1. **B0 legacy** (deprecated for discovery):
   - [None] token folded into physical block 0
   - Physical-frame segment-mean aggregation
   - 64×64 graph
   - Used for legacy controller acceleration results

2. **B1 / predictor-aligned**:
   - `attn[:-1, :-1]` predictor frame extraction
   - No physical remap in predictor mode
   - Result-bearing files use `none_mode=predictor` (NOT code enum `none_mode=b1`)
   - 64×64 content-only (no [None])

3. **AR next-token shift**:
   - Clarified asymmetric alignment: query positions predicting block0 = [None],x0,x1,x2; content block0 = x0,x1,x2,x3
   - Cannot use same label set for both query and key

4. **Content-only 64-node**:
   - Drops [None] entirely
   - Can recover cyclic order but loses anchor/start
   - Used for physical adjacency analysis

5. **Strict 65-node None-separated** (current preferred):
   - Node 0 = [None] / BOS (independent start node)
   - Nodes 1–64 = physical content blocks
   - Graph B65 built in model-frame coordinates
   - No `inv_perm` during extraction or CDL rollout
   - `inv_perm` used EXACTLY ONCE posthoc: σ_model → σ_phys for scoring

6. **Permutation convention**:
   - `block_perm`: physical→model (our convention in `block_lo_arm_order_network`)
   - `inv_perm`: model→physical (inverse)
   - Must distinguish from nanogpt convention which may be inverted
   - Source: `reports/strict_65node_discovery_ckpt_verification_20260617/01_protocol_definition.md`

### Strict 65-Node Protocol Definition

```
Node 0 = None (independent BOS/start)
Node i (1..64) = physical content block (i-1)
  = tokens x_{4(i-1)} .. x_{4(i-1)+3}

Graph B65:
  B[0, 1:] = A[:, 0]          # None → content (row = content, col = None)
  B[1:, 1:] = A[:, 1:].T      # Content → content (transposed)
  diag(B) = 0

Rollout:
  Start from None (node 0)
  At each step t:
    score(i) = C-D+L_i  (or L_i only for transition-only)
    Select next block by argmax score
  None is NOT folded into any physical block
  None is NOT manually attached to physical0

Posthoc scoring:
  inv_perm(σ_model) → σ_phys
  Compare σ_phys against L2R reference
  Report τ, first block rank, prefix@4, prefix@8
```

**inv_perm boundary** (CRITICAL):
- ALLOWED: during graph construction (model-frame coordinates)
- ALLOWED: during posthoc scoring (translate to physical frame)
- FORBIDDEN: during CDL rollout (must be label-free)
- Oracle-remapped variant: valid equivalence check (CDL is permutation-equivariant)

---

## Module C — Order Readout / CDL Teacher

### CDL Teacher
- **Components**: C-D+L (or L-only, or C-D only)
- **C (conservation)**: Score based on B edge conservation
- **D (departure)**: Score based on B edge departure from current
- **L (landing)**: Score based on B edge landing at candidate
- **Transition-only (L)**: Most reliable for strict discovery — uses only L term
- **Rollout**: Greedy CDL from None, no physical labels used
- **Posthoc**: inv_perm applied after rollout for evaluation only

### Rollout Modes
- **C-D+L**: Full CDL (may have low-index tie bias for C-only)
- **L-only**: Transition-only (preferred for primary evidence)
- **C-only**: NOT recommended (potential tie-breaking bias toward low physical indices)
- **none_edge**: argmax B[None, :] — NEVER strong-passes; signal requires CDL dynamics

### Destroyed Controls
- **Entry shuffle**: Randomly permute B[1:, 1:] entries (preserves row/col marginals)
- **Content label permutation**: Randomly permute content node labels
- **Gaussian B**: Replace B with random Gaussian matrix
- All collapse to |τ| ≤ 0.07, confirming real signal dependence on attention structure.

---

## Module D — Controller Distillation / g_β Hook

### g_β Training
- **Input**: Selected-head B (65×65 or 64×64 attention graph)
- **Output**: Block order scores / ranking (65-dim or 64-dim logits)
- **Architecture**: Lightweight MLP with nodewise + pairwise objectives
- **Teacher**: CDL teacher orders (from selected head, supervised)
- **Phase 1.5 Gate**: τ ≥ 0.6 AND pairwise ≥ 0.8 on same-step AND cross-step

### Frozen Hook
- **Deployment**: g_β is frozen during AOGPT training
- **Refresh**: Every K steps, extract B from current model, run g_β(B) → order
- **Mode**: argsort (g_β logits → permutation via argsort)
- **Alpha schedule**: 0 → 1 over warmup steps (ramp-up to full g_β order)

### Current Status
- **Legacy B0 path**: Verified acceleration (33–42% step saving). Uses B0 graph, [None]→phys0.
- **Strict 65-node path**: Discovery verified. Controller NOT yet demonstrated. This is the P0 closure experiment.
- **Code sources**:
  - `block_lo_arm_order_network/batch_readout/hook_order_provider.py` (B0 legacy hook)
  - `block_lo_arm_order_network/batch_readout/selected_head_dataset.py` (g_β training data)
  - `block_lo_arm_order_network/none_separated_block_graph.py` (B65 construction)
  - `block_lo_arm_order_network/per_head_order_scan.py` (head scanning)
  - `block_lo_arm_order_network/scripts/run_phase33_gbeta.py` (g_β training pipeline)
  - `block_lo_arm_order_network/scripts/pipeline_no_label.sh` (end-to-end pipeline)
