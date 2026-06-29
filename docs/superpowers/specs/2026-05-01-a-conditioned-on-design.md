# A-conditioned Graph Order Network

## Context

Route A signal source validated: NLL pair score A matrices produce non-trivial
DP paths (L2R=0, R2L=0, avg_adj=1.6/15). State-only ON fails (val acc 27%).
Compact hand-crafted A features reach val acc 67.7% but cap there. Raw A
features overfit. Need an architecture that uses the full A matrix without
memorizing the training set.

**Review decision (2026-05-01)**: Architecture B (A-cond Transformer with
`Linear(A[i,:])`) dropped — broken permutation equivariance (Linear binds to
absolute column indices) and 10M/6000 param/data mismatch. All-in on GNN +
lightweight ablation baseline.

## Goals

- Train an Order Network that generalizes to **held-out A matrices**
- Next-step accuracy on unseen A matrices >= 75%
- Greedy path weight / DP optimal weight ratio >= 0.85 on holdout
- GNN vs DeepSets ablation to quantify message-passing benefit

## Non-goals

- Scaling beyond N=16 blocks
- End-to-end integration with AO-GPT inference
- DAgger rollout training unless teacher forcing reaches >= 75% val acc

## Data

**P0 extension**: 500+ A matrices (currently 100).

- Script: `p0_n16_direct.py --num-sequences 500`
- Each A: 16x16 float32, diagonal -inf, 5 fwd passes averaged
- Output: `probe_results/A_n16_direct_500x5.npy`

**DP**: `solve_dp_batch` on all 500 A matrices.

- Output: `probe_results/dp_n16_direct_500x5.npz`

**Training examples**:

- ~7500 examples = 500 seqs x 15 steps
- Format: `(seq_idx, visited_mask, last_node) -> next_node`
- Output: `probe_results/on_training_data_500.npz`

**Train/val split**: by A matrix (sequence), not by example.

- 80/20 → 400 train seqs (6000 examples), 100 val seqs (1500 examples)
- Val measures generalization to novel A matrices with zero leakage

## Architecture A: GNN (Edge-conditioned Graph Network)

### Design principle

Strict permutation equivariance: if A is permuted by π, output scores permute
by π. Achieved by: no absolute position encoding, no Linear(A_row), only
topological features + message-passing on A-weighted edges.

### Node features (no position encoding)

Pure topological/state features per node i at step t:

| Feature | Dim | Description |
|---------|-----|-------------|
| is_visited | 1 | 0/1 from visited_mask |
| is_last_node | 1 | 0/1, which node was just visited |
| in_degree_weighted | 1 | sum(softplus(A[j,i]) for j) |
| out_degree_weighted | 1 | sum(softplus(A[i,j]) for j) |
| in_degree_unweighted | 1 | count of j where A[j,i] > μ(A) |
| out_degree_unweighted | 1 | count of j where A[i,j] > μ(A) |
| reveal_ratio | 1 | num_visited / N |

Total: 7 scalar features → Linear(7, d_model) → initial node embedding.

No learnable position embedding. No node identity embedding. The network only
knows "you are visited", "you are the last node", "your connectivity". It
must infer ordering purely from A topology.

### Message-passing (3 layers)

Each layer k:

1. **Edge weight normalization**: `w_ij = tanh(A[i,j] / tau_k)` with learnable temperature tau_k per layer
2. **Message from j to i**: `m_ij = w_ij * (W_v_k @ h_j + W_e_k * w_ij)`
3. **Aggregate**: `agg_i = sum(m_ij for all j)` (fully connected, edge-weighted)
4. **Update**: `h_i = LayerNorm(h_i + GELU(W_update_k @ [h_i, agg_i]))`

d_model=64, so each layer has ~4*(64*64) + 64 ≈ 16K params. 3 layers ≈ 50K.

### Readout

- `score_i = MLP(h_i)` where MLP = Linear(64, 64) + GELU + Linear(64, 1)
- Mask visited → -inf
- Logits = scores (no softmax, handled by CrossEntropyLoss)

### Total params: ~80K

## Architecture C: DeepSets (Message-Passing Ablation)

Identical node features → identical readout MLP. The only difference: **no
message-passing**. Each node scored independently from its own features.

- `score_i = MLP(node_features_i)` — same 7-dim input, same 2-layer MLP
- Mask visited → -inf

This directly isolates the contribution of message-passing: any accuracy
delta between C and A is purely from graph convolution.

### Total params: ~15K

## Training

Shared training loop for both architectures:

- Loss: cross-entropy vs DP optimal next block
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Grad clip: 1.0
- Epochs: 200 (early stop: patience=30 on val acc, no improvement → halt)
- Batch size: 128
- No DAgger in first pass — pure teacher forcing

### Evaluation per epoch

- Train next-step accuracy (in-distribution A states)
- Val next-step accuracy (held-out A matrices)
- Save best checkpoint by val acc
- After training: greedy rollout on val A, compute path_weight / DP_optimal_weight

## Acceptance criteria

| Metric | Target |
|--------|--------|
| DP acceptance (L2R=0, R2L=0) on 500 seqs | √ (expected) |
| Val next-step acc (state-only baseline) | 27% (known floor) |
| Val next-step acc (compact A-feature MLP) | 67.7% (known floor) |
| Val next-step acc (DeepSets, Arch C) | >= 65% (ablation) |
| Val next-step acc (GNN, Arch A) | >= 75% |
| Greedy/DP weight ratio (GNN) | >= 0.85 |

## Files

| File | Purpose |
|------|---------|
| `p0_n16_direct.py` | Extend to 500 seqs |
| `p1_validate_dp.py` | DP on 500 A (already supports batch) |
| `p2_build_training_data.py` | Build 7500 examples from 500 DP paths |
| `order_network.py` | Add `GraphOrderNetwork` (GNN) and `DeepSetOrderNetwork` |
| `train_gnn_on.py` | New: train GNN + DeepSets, compare |

## Size budget

| Component | Params |
|-----------|--------|
| Node feature projection | ~500 |
| GNN layer × 3 | ~50K |
| Readout MLP | ~8K |
| **GNN total** | **~60K** |
| **DeepSets total** | **~15K** |
| Training examples | 6,000 |
| Ratio (params/data) | 10:1 / 2.5:1 — safe zone |
