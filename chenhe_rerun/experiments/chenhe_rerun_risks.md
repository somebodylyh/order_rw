# chenhe Rerun — Risk Checklist

Run through this before every rerun launch and before the main table is frozen.
Anything checked "⚠️" must be resolved or explicitly waived.

## 1. No two-baseline contamination
- [ ] No experiment uses admin `train_clean_aogpt.py` — entrypoint is chenhe
      `train.py` for every row.
- [ ] Admin's old fixed-stream results are labelled pilot/preliminary, NOT in
      the main table.

## 2. Metric hygiene
Verified metric semantics (chenhe `estimate_loss()` @train.py:7708):
- `val` = NLL under the **active order policy** (current frame) = **model-selected-
  order loss** (for our method = the gβ-chosen order). ← the direct order-quality metric.
- `val_origin_l2r_loss` = NLL revealing blocks in `inverse_block_perm` order =
  **original text L2R** NLL. Computed purely in chenhe's own frame
  (`fixed_block_perm`, `permute_seed=42`) → comparable across all runs.
- `val_l2r_loss` = current-frame index-order (permuted) AR NLL — NOT text L2R.
- `val_main_eval_loss` = only logged when a segment-guided policy is active
  (NOT for plain baselines / frozen-gβ). Do NOT rely on it as the selection metric.

Checks:
- [ ] Main-table headline = `val_origin_l2r_loss` (one metric, all rows, text-L2R).
- [ ] Order-quality / selection = `val` (model-selected-order loss); NOT
      `val_main_eval_loss` (which is segment-guided-only).
- [ ] `val_origin_l2r_loss` never drives training selection / early-stop.
- [ ] Admin's `val_ori_l2r_block` field does NOT appear in the main table.
- [ ] **Frame-index semantics verified** (not just shape): admin-ported gβ path
      produces block index i ≡ chenhe current-frame block i. Training path uses NO
      admin `inv_perm`; chenhe `inverse_block_perm` only posthoc for the metric.
      (Spike proved shape only — this is a separate must-verify.)

## 3. Data pipeline consistency
- [ ] Every row: `permute_data=True`, `permute_seed=42`, `permute_mode=block`,
      memmap random access.
- [ ] Non-permute AR is diagnostic only (different frame) — not same-table.
- [ ] No row silently reverts to a fixed-stream loader.

## 4. Layout consistency
- [ ] `block_size=256`, `block_order_block_len=4` (64 blocks) on every row —
      mainline is **seq256/block64**.
- [ ] No seq384/block96 (ablation) or other seq/block mixed into the mainline
      main table.

## 5. Config field discipline
- [ ] Every method config diff vs base = method/order-policy + bookkeeping
      (`out_dir`, W&B) ONLY. Any non-method field diff = **NOT ALLOWED**
      (see `chenhe_method_config_diff.md`).
- [ ] Optimizer / LR / schedule / model / eval / save fields untouched.

## 6. W&B
- [ ] Uses our own project (not chenhe's `AOGPT-order-block-96-final`).
- [ ] Run tags unique per experiment, no collision.

## 7. Method / frame correctness (frozen-MLP scope, NO PG)
- [ ] Scope = **frozen gβ MLP only** (V3 Stage 1 CDL-pretrain + Stage 2 frozen
      deploy). **Stage 3 PG unfreeze is OFF** — not in current work.
- [ ] New order branch returns `block_orders` in the **permuted current frame
      (64 blocks)** — no frame mismatch (memory `frozen-gbeta-deploy-noneframe-bug`).
- [ ] ✅ Block-count matches: mainline 64-block = V3 native 64-block (65-node
      strict65). No gβ/CDL dimension rebuild. (Do NOT align to seq384/block96 ablation.)
- [ ] Method = ported V3 frozen-gβ path, NOT chenhe `attn_mlp_order_policy.py`.
- [ ] Reduced red lines (no PG): gβ **frozen throughout** (`requires_grad=False`,
      `@torch.no_grad()`), never in the optimizer; order-frame correct.
- [ ] B-source invariant: ported Stage-2 B == `FrozenGBetaModelFrameBlockProvider`
      B (8-head strict65, batch-mean probes, `none_mode='model'`, model-frame).

## 7b. 🔴 gβ batch-mean constraint (hard — training AND eval)
- [ ] gβ order computed from **batch-mean B** = mean over (batch samples ×
      `batch_mean_probes` probe forwards). **ONE σ per batch/refresh**, broadcast
      to all samples. Signal exists ONLY at batch-mean.
- [ ] **NEVER per-sample gβ** (per-sample B → per-sample σ): per-sample pairwise
      signal ≈ noise → inaccurate order (memory `br1_batch_readout`,
      `v0-label-free-gbeta`: batch-mean fix τ 0.4→0.91).
- [ ] Eval does NOT degrade to single-sample: eval batch also uses batch-mean B
      → single σ → model-selected-order loss (`val`) under that σ.

## 8. Working-tree provenance
- [ ] All rerun work in admin-owned `chenhe_rerun/` (code-only copy).
- [ ] Original `chenhe_nanogpt_learned_order/` untouched (read-only reference).
- [ ] Changes committed into `order_lyu` with clear messages (records).
