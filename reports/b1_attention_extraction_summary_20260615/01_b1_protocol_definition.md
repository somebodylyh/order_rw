# 01 - B1 Protocol Definition

## What B1 is in the current result files

The result-bearing B1 runs inspected here use the local code path called `none_mode="predictor"`.

Primary sources:

- `scripts/run_per_head_order_scan_ladder_b1.sh`: runs `per_head_order_scan.py --none-mode predictor`.
- `scripts/run_per_head_order_scan_ladder_b1_fast.py`: calls `scan_loaded(..., none_mode="predictor")`.
- `scripts/queue_random_b1_headscan_after_gpu1.sh`: runs training with `--track-head-none-mode predictor`.
- `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/config.json`: records `"track_head_none_mode": "predictor"`.
- `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/head_signal_all.tsv`: records `none_mode=predictor` in each row.

There is also a code enum `none_mode="b1"` in `per_head_order_scan.py`, implemented by `_attn_to_A_block_b1_vec`. That path uses predictor frame plus physical remap. I did not find a result JSON/TSV in the inspected B1 files that was produced with `none_mode=b1`. Therefore this report defines "B1 results" as the current result-bearing predictor-aligned convention, and flags the `b1` enum vs `predictor` naming difference as unresolved.

## Input and tensor shapes

For text AO-GPT:

- Sequence length: 256 tokens.
- Blocks: `N=64`.
- Tokens per block: `BLOCK_LEN=4`.
- Model attention per forward: list over layers; stacked as `(L, B, H, T+1, T+1)` in batched code or `(L, H, T+1, T+1)` per sample.
- `T+1=257` because row/column 0 is the `[None]` token.
- Current standard small model: `L=4`, `H=8`, total heads=32.

Source paths:

- `block_lo_arm_order_network/per_head_order_scan.py`
- `block_lo_arm_order_network/train_clean_aogpt.py`
- `block_lo_arm_order_network/probe_results/random_baseline_b1_headscan_seed124/config.json`

## Operational B1 / predictor aggregation

For each sample and each `(layer, head)`:

1. Forward model with random probe token order.
2. Read raw causal attention from `model.forward_fn(..., return_attentions=True)`.
3. Use predictor-aligned frame:
   - `attn[:-1, :-1]`
   - This keeps the original AO-GPT prediction frame where prefix position `i` predicts target `i`.
4. Aggregate the 256 by 256 predictor-frame attention matrix into 64 by 64 block graph by reshaping:
   - `(N, BLOCK_LEN, N, BLOCK_LEN)` and averaging over both token axes.
5. Zero the diagonal.
6. Group per-sample `A` graphs into `M` probe batches and compute batch mean.
7. Convert to the CDL/readout graph by transposition:
   - `B = A.T`
8. Zero the diagonal again.

Source paths:

- `_attn_to_A_block_predictor_vec` in `block_lo_arm_order_network/per_head_order_scan.py`
- `_batch_mean_B` in `block_lo_arm_order_network/per_head_order_scan.py`
- `_track_all_heads_signal` in `block_lo_arm_order_network/train_clean_aogpt.py`

## Pseudo-code

```python
# pseudo-code only; derived from current implementation
attn_stack = stack(attn_list)  # (L, B, H, 257, 257)

for sample in batch:
    for layer in range(L):
        for head in range(H):
            attn = attn_stack[layer, sample, head]  # (257, 257)

            # result-bearing B1 convention: none_mode="predictor"
            shifted = attn[:-1, :-1]  # (256, 256), includes [None] in position 0
            A = shifted.reshape(N, block_len, N, block_len).mean(axis=(1, 3))
            fill_diagonal(A, 0.0)

            # after collecting M * batch_size samples:
            A_grouped = A_samples.reshape(M, batch_size, N, N).mean(axis=1)
            B = A_grouped.transpose(0, 2, 1)
            B[:, diag, diag] = 0.0
```

For the available but not result-bearing `none_mode="b1"` code enum:

```python
# pseudo-code only; derived from _attn_to_A_block_b1_vec
a = attn.reshape(K, 257, 257)[:, :-1, :-1]  # predictor frame
key_labels[0] = 0                           # [None] -> physical block 0
key_labels[1:] = inv_perm[reveal_tokens[:-1] // block_len]
query_labels = key_labels.copy()
Sk = segment_mean_selector(key_labels)
Sq = segment_mean_selector(query_labels)
A = einsum("bt,ktu,cu->kbc", Sq, a, Sk)
fill_diagonal(A, 0.0)
```

## Mask, diagonal, BOS/special tokens, and coordinates

| Component | Current result-bearing B1 behavior | Source |
|---|---|---|
| Causal mask | Uses model returned causal attention; extraction does not re-mask. | `model.forward_fn(... return_attentions=True)` in `train_clean_aogpt.py` and `per_head_order_scan.py` |
| `[None]` token | In `predictor` mode, `[None]` stays in shifted predictor frame position 0 and is averaged into predictor block 0 by reshape. | `_attn_to_A_block_predictor_vec` |
| BOS/special tokens | No separate BOS handling found beyond the model's `[None]` token. | code inspection |
| Diagonal | Zeroed after block aggregation and again after `B=A.T` / batch mean. | `_attn_to_A_block_predictor_vec`, `_batch_mean_B`, `_track_all_heads_signal` |
| Batch mean | Yes. Per-sample graphs are grouped into `M` groups of `batch_size`; mean over batch. | `_batch_mean_B`; `PROBE_BATCH=32` in trackers |
| Query/key token mean | Yes. Token-level matrix reshaped to `(N, block_len, N, block_len)` and averaged over both token axes. | `_attn_to_A_block_predictor_vec` |
| Transpose | Yes. `B = A.T`. | `_batch_mean_B`; `analyses/plot_attn_map_b1.py` |
| Physical remap | Not in `none_mode=predictor` result files. Available in `_attn_to_A_block_b1_vec`, but not used by inspected B1 outputs. | `per_head_order_scan.py`; result configs |
| Random probe seed | Deterministic from seed and sample index or from `(seed, global_step, b)` in online tracker. | `extract_per_head_and_heavy_A`; `random_probe_token_orders` |
| Clean permutation | Training configs store block permutation and inverse; `predictor` mode does not apply `inv_perm`, while `b1` enum does. | config JSON; `_attn_to_A_block_predictor_vec`; `_attn_to_A_block_b1_vec` |

## Semantic meaning of B

For the result-bearing B1/predictor convention, `A[i, j]` is the block-aggregated predictor-frame attention weight from query block `i` to key block `j` in predictor/model order. The readout graph is `B = A.T`, so `B[row, col]` is the transposed edge matrix consumed by CDL/g_beta-style order readout. It is diagonal-zeroed and not row-normalized in the extraction step.

Because `none_mode=predictor` does not physical-remap the nodes, the B1/predictor signal should be described as predictor-frame aligned unless a downstream script explicitly remaps to physical coordinates. `analyses/plot_attn_map_b1.py` visualizes both raw predictor-frame B and physical-remapped B for inspection.

## B1 vs collaborator implementation

What is verified locally:

- The B1 ladder scripts state this is "B1 predictor-aligned" and use AO-GPT's original predictor frame: `attn[:-1, :-1]`.
- Continuous training data loading is described as "collaborator-style continuous loading" in `clean_training_protocol.py` and `train_clean_aogpt.py`.
- The current B1 result files consistently use `none_mode=predictor`.

What is not verified:

- I did not find an external collaborator extraction implementation or script in the requested directories.
- I therefore cannot claim bit-for-bit parity with an external collaborator implementation.

Conservative wording:

> We adopt the local collaborator-aligned predictor-frame block-attention extraction convention used by the B1 scripts. External collaborator-code parity was not directly verified from the files present in this repository.

## Naming convention

- B0: legacy / previous canonical selected-head extraction, implemented as `none_mode="b0"` with `[None] -> physical block 0` folding and physical-frame segment-mean aggregation.
- B1 result-bearing convention: collaborator/predictor-aligned diagnostic extraction, currently materialized in files as `none_mode="predictor"`.
- B1 code enum: `none_mode="b1"` means predictor frame plus physical remap, but no inspected result file uses it.
- Paper/main diagnostic results should default to the B1/predictor-aligned extraction convention once the naming issue is resolved.
- Existing frozen g_beta training and hook acceleration results should be labelled legacy B0 unless a B1 hook rerun exists.

