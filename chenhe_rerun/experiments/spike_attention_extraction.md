# Spike: can chenhe AOGPT produce V3's strict65 8-head B?

**Date:** 2026-07-02 · **Verdict: 🟢 GREEN — no architectural blocker.**

## Question

V3's gβ (`L0DynamicGBeta`) consumes `B_raw (batch, 8, 65, 65)` built by
admin's `extract_probe_averaged_model_frame_strict65` + `build_model_frame_strict65`.
Can chenhe's AOGPT (the mainline seq256/block64 backbone) produce the exact
inputs those functions need?

## What admin's extraction requires (spec)

From `block_lo_arm_order_network/batch_readout/`:
- `forward_fn(idx, probe, return_attentions=True)` → per-layer attn list;
  `attn_list[0]` = L0.
- `attn_l0` shape **(sample, head, 257, 257)** — 257 = 256 content + 1 None @ node 0.
- attention produced **under a reveal order** (probe = shuffled token order).
- `build_model_frame_strict65(attn_l0, probe)` → `(sample, 8, 65, 65)`,
  None-separated block graph (node 0 = None, nodes 1..64 = model blocks).

## What chenhe's AOGPT provides (verified in `AOGPT_block.py`)

Same lineage as admin (shared `order_utils`, None token, any-order reveal).

| Requirement | chenhe | Evidence |
|-------------|:------:|----------|
| `forward_fn(idx, orders, return_attentions=True)` | ✅ | `AOGPT_block.py:458` |
| per-layer attn list | ✅ | `attn_outputs.append(attn_probs)` L502–511 |
| None token @ index 0 | ✅ | `wnonee`; `none_tok_emb` concat at pos 0, L480–482 |
| attention 257×257 (block_size+1) | ✅ | causal mask `block_size+1` L75–79; `pos=arange(t+1)` L472 |
| per-head attention (B, nh, 257, 257) | ✅ | `CausalSelfAttention` returns `att` per head L93–129 |
| reveal order via shuffle | ✅ | `idx = self.shuffle(idx, orders)` L475; same `order_utils` |

**Conclusion:** chenhe's `forward_fn` returns exactly the L0 per-head 257×257
attention, None @ node 0, shuffled by the reveal order — precisely the
arrangement `_attn_to_A_block_loss_aligned_with_none_model_vec` +
`build_model_frame_strict65` assume. No redesign needed; the port is an adapter
+ loop surgery.

## Adapter points (minor, not blockers)

1. **Return format.** chenhe returns via `_format_outputs(logits, loss,
   token_losses, hidden_states, attentions)` (structured), not admin's bare
   `(_, _, attn_list)`. Adapter: pull `attentions` from chenhe's output object.
2. **Position-encoding mode.** chenhe supports `absolute` | `rope`; the block64
   base config sets neither → defaults to `absolute`. Confirm the CDL-pretrain
   backbone runs the same mode (it will, since Stage-1 uses the *chenhe* backbone).
3. **gβ provenance.** Do NOT reuse admin's `g_beta_best.pt` — its attention
   distribution is admin-backbone-specific. Per V3 Stage-1, CDL-pretrain gβ from
   a **chenhe** parent ckpt so it reads chenhe's own attention.
4. **Optimizer.** chenhe `configure_optimizers` (L595) filters `requires_grad`
   → frozen gβ auto-excluded in Stage 2; Stage-3 `add_param_group` is clean.
   Satisfies the V3 "single param-group insertion" red line.

## Empirical verification — ✅ DONE (2026-07-02)

Data: reused existing gpt2/nanoGPT `train.bin` (118M tokens), symlinked into
`chenhe_rerun/data/wikitext103/` (no 190MB re-download). Fresh-init block64
AOGPT (47.1M params), real wikitext batch (B=2, T=256), random-block-perm probe.

End-to-end chain ran and passed every assertion:

```
chenhe forward_fn(idx, orders, return_attentions=True)
  → attn_l0 (2, 8, 257, 257)                         # 4 layers, per-head
  → build_model_frame_strict65(attn_l0, probe)
  → B (2, 8, 65, 65)   [PASS] finite · col0(into-None)=0 · diag=0 · content nonzero
  → L0DynamicGBeta(heads=8, nodes=65)
  → scores (2, 64)     [PASS] finite
  → argsort → σ_model (2, 64)  [PASS] valid permutation
```

**The full Stage-2 forward path is shape-compatible and runs on real chenhe
attention + real data.** No architectural gap between chenhe's backbone and V3's
gβ input contract.

Scripts: `scratchpad/spike_phase1_chenhe_attn.py` (chenhe attn),
`scratchpad/spike_phase2_admin_strict65.py` (admin B + asserts), phase-3 gβ
one-liner. Two-process split avoids `chenhe_rerun` vs `block_lo_arm_order_network`
module-name collisions.

## Bottom line

The V3 port into chenhe `train.py` is **viable as an adapter + loop surgery**,
not a redesign. Proceed to the V3-port SDD (spec → plan → TDD).
