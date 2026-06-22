# Head-Gated gBeta Inventory Report

Generated: 2026-06-22

## 1. Current g_beta Model Definitions

**File:** `block_lo_arm_order_network/batch_readout/model.py`

| Model | Input Shape | Output Shape | Architecture |
|-------|------------|-------------|-------------|
| `FlattenReadout` | `(batch, N, N)` | `(batch, N)` | vec(B) → MLP(1024→256→N), GELU |
| `NodewiseReadout` | `(batch, N, N)` | `(batch, N)` | per-node [B[v,:],B[:,v]] → TransformerEncoder(2L,4H,d64) → Linear→1 |

**Key limitation for head-gated work:** Both models take a SINGLE B matrix `(batch, N, N)`. No multi-head input `(batch, H, N, N)` is supported. Task 3 must add this.

## 2. Current g_beta Target/Label Format

**Label generation:** `block_lo_arm_order_network/neural_readout/teacher_labels.py::generate_teacher_label(B, alpha_dep=0.5)`

Returns:
- `sigma`: `(N,)` int64 — reveal order, `sigma[0]` = source-start node
- `rank`: `(N,)` int64 — `rank[v]` = position of v, rank 0 = earliest
- `pairwise_Y`: `(N, N)` uint8 — `Y[i,j]=1` iff i revealed before j, diag=0

**Teacher algorithm:** CDL-source-start (readiness anchor via `out(v) - alpha_dep * in(v)`, then greedy C-D+L rollout).

**Current dataset:** `batch_readout/dataset_batch.py` stores single `B_batch` per sample (not multi-head). Labels are generated per batch-mean graph.

## 3. Current Frozen-Beta Training Entry

**File:** `block_lo_arm_order_network/train_clean_aogpt.py`

Entry point:
```
python3 train_clean_aogpt.py --run-kind frozen_beta \
  --frozen-beta-ckpt path/to/g_beta_best.pt \
  --frozen-beta-head 0 2 \
  --frozen-beta-refresh 10 \
  --frozen-beta-none-mode b1 \
  --start-step 20000 --max-steps 60000
```

Key arguments:
- `--frozen-beta-ckpt`: path to g_beta_best.pt
- `--frozen-beta-head LAYER HEAD`: which (layer, head) to extract B from
- `--frozen-beta-mode`: argsort or sample
- `--frozen-beta-tau`: temperature for sample mode
- `--frozen-beta-refresh`: recompute order every N steps
- `--frozen-beta-none-mode`: block aggregation mode (b1/predictor/model/content/loss_aligned)

**Integration flow:**
1. `HookOrderProvider` is created at training start (line 1389-1395)
2. Each step: extract attention from specified head → compute B → FrozenBetaHook.step() → order
3. Alpha mixing: per-sample coin flip determines guided vs random order
4. Alpha ramps from 0→1 over warmup steps

**Current limitation:** One fixed (layer, head) pair. No multi-head input, no gate, no per-batch head selection.

## 4. Current CDL Teacher Entry

**File:** `block_lo_arm_order_network/train_clean_aogpt.py`

```
--run-kind cdl_teacher --cdl-teacher-head 0 7 --cdl-teacher-refresh 10
```

**File:** `block_lo_arm_order_network/batch_readout/cdl_order_provider.py`

`CdlOrderProvider` mirrors `HookOrderProvider` but replaces FrozenBetaHook with direct C-D+L rollout.

**CDL Teacher modes:** `block_lo_arm_order_network/attn_order_teacher.py`
- `MODES = ("C", "L", "C-D", "C+L", "C-D+L")` — **No "-D" only mode exists.**
- `rollout_order(B, mode, greedy, start)` — main entry for order generation

## 5. Exact Files to Modify in Later Tasks

### Must modify:
| File | Purpose |
|------|---------|
| `batch_readout/model.py` | Add multi-head g_beta variants (Task 3) |
| `batch_readout/train_offline.py` | Extend to multi-head input training (Task 4) |
| `batch_readout/hook_order_provider.py` | Add multi-head extraction + head-gated provider (Task 7) |
| `batch_readout/integration_hook.py` | Extend FrozenBetaHook for multi-head B (Task 7) |
| `train_clean_aogpt.py` | Add `--gbeta-input-mode layer_heads` etc. (Task 7, 9) |

### Must create:
| File | Purpose |
|------|---------|
| `analyses/build_gbeta_headset_dataset.py` | Multi-head distillation dataset builder (Task 2) |
| `analyses/smoke_head_gated_gbeta.py` | Smoke tests for new models (Task 3) |
| `scripts/train_head_gated_gbeta.py` | Offline distillation training script (Task 4) |
| `scripts/run_head_gated_gbeta_distill_smoke.sh` | Smoke launcher (Task 4) |
| `scripts/run_head_gated_gbeta_distill_full.sh` | Full grid launcher (Task 4) |
| `analyses/eval_head_gated_gbeta.py` | Offline evaluation (Task 5) |
| `analyses/plot_head_gated_gbeta.py` | Plotting (Task 5) |
| `scripts/run_head_gated_audition.sh` | Small-train audition (Task 6) |
| `analyses/eval_head_gated_audition.py` | Audition evaluation (Task 6) |
| `scripts/queue_head_gated_gbeta_feedback.sh` | Frozen feedback launcher (Task 7) |
| `scripts/queue_head_gated_gbeta_joint.sh` | Joint training launcher (Task 9) |

### Must extend:
| File | Purpose |
|------|---------|
| `analyses/diag_model_frame_feedback.py` | Head drift diagnostics (Task 8) |
| `attn_order_teacher.py` | Add `-D` only mode for teacher ablation (Task 2) |

## 6. Key Dependencies

- **Attention extraction:** `per_head_order_scan.py` — `_attn_to_A_block_b1_vec` and siblings support per-head extraction from `(L, B, H, T+1, T+1)` attention tensors. Vectorized over heads — extracting ALL heads from a layer is cheap (~no extra forward passes).
- **Graph construction:** `none_separated_block_graph.py` — 65-node B from model-frame attention (for strict/none modes).
- **Model-frame metrics:** `analyses/diag_model_frame_feedback.py` — tau_model_vs_semantic_path, tau_model_vs_identity. Use as-is.
- **Existing checkpoints:** `random_baseline_continuous_jun08_seed2/` has steps 20k, 30k, 40k, 50k, 60k accessible.

## 7. Gaps vs Plan Requirements

1. **No "-D" only teacher mode:** `attn_order_teacher.MODES` has no `-D` mode. Plan requires `minus_d_only` teacher. Need to add to `rollout_order` or create custom teacher label generator.
2. **No per_head_cdl_rollout teacher:** Current CDL runs on a single head's batch-mean B. A per-head version that calls `rollout_order` on each head's B independently doesn't exist yet.
3. **No multi-head B dataset format:** Current `.npz` stores single `B_batch (M, N, N)`. Need `B_heads (M, H, N, N)` format for head-gated g_beta training.
4. **No semantic_model_path_oracle:** Doesn't exist as a standalone teacher label generator — needs to be built using `block_perm[arange(N)]` as the model-frame order.
