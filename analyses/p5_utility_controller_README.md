# P5 Attention-Scaffolded Utility Controller — Phase 0

> Spec: `docs/superpowers/specs/2026-06-29-p5-attention-scaffolded-utility-controller-design.md`

**Red line:** H improves downstream reveal-order utility beyond the attention scaffold;
it does NOT recover physical order. Supervision = downstream teacher-forced AO NLL.
Fixed layout; B-only baseline; shuffled-H control.

**Claim tested:** On a single frozen seed123 step10k checkpoint with seed123 carrier
(L0H1), block-level hidden states pooled under sigma_B's reveal context carry
per-sample utility-relevant information that is complementary to attention topology.

## Phase-0 Verdict (2026-06-30)

**`no_gain` — attention scaffold dominates; H residual fails across all tested
extraction layers and contexts.**

### Initial smoke (M=64, L0H1, L0 residual)

| Metric | Value | Notes |
|--------|-------|-------|
| Headroom gate | PASS (abs=0.044) | σ_B not optimal — room exists |
| NLL B-only | 3.895 | Single-head scaffold |
| NLL B+H | 3.927 | **Worse** (+0.033) |
| h_shuffle_drop | 0.003 | H content irrelevant |
| Verdict | no_gain | Room exists, H not learned |

### H-layer sweep (M=32, L0H1, 5 extraction layers)

All layers produce ΔNLL > 0 and shuffle_drop ≈ 0:

| H layer | B-only | B+H | ΔNLL | shuf_drop | res_ratio |
|---------|--------|-----|------|-----------|-----------|
| L0 | 3.767 | 3.800 | **+0.032** | −0.026 | 0.42 |
| L1 | 3.763 | 3.798 | **+0.035** | −0.042 | 0.92 |
| L2 | 3.757 | 3.809 | **+0.052** | +0.002 | 0.88 |
| L3 | 3.763 | 3.806 | **+0.043** | +0.012 | 0.46 |
| mean(L1,L2,L3) | 3.765 | 3.832 | **+0.066** | −0.003 | 0.46 |

L1/L2 larger residual ratios = more noise, not more signal.

### H extraction context ablation (M=32, multi-head B[1,2,3,4], L2)

| H context | B-only | B+H | ΔNLL | shuf_drop |
|-----------|--------|-----|------|-----------|
| sigma_B | 3.707 | 3.764 | **+0.057** | +0.031 |
| phys | 3.701 | 3.780 | **+0.079** | −0.001 |
| avg(σB+phys+rand) | 3.705 | 3.771 | **+0.066** | +0.011 |

Order-averaged H does not rescue the residual — content signal too weak
relative to scaffold/reveal-context confounds.

### Multi-head B scaffold (M=32)

| B source | B-only NLL | headroom |
|----------|-----------|----------|
| L0H1 only | 3.762 | 0.036 |
| **L0H[1,2,3,4] mean** | **3.701** | **0.006** |

Multi-head carrier mean is a significantly stronger attention scaffold —
B-only NLL drops 0.06 and headroom shrinks to near-zero (~0.006).
This is a positive finding: **multi-head B aggregation is useful.**

## Consolidated conclusions

1. **Multi-head B scaffold is real** — 4-carrier-head mean beats single head
   (B-only NLL from 3.76 → 3.70, headroom from 0.036 → 0.006).
2. **H residual provides no utility across all tested conditions** —
   all 11 configurations (5 layers × 2 B sources + 3 H contexts) yield ΔNLL > 0.
3. **H is NOT content-driven in any tested setup** —
   shuffle_drop ≈ 0 in all cases. g_H learns a cross-text invariant confound
   (position/reveal-context traces), not per-sample content.
4. **Deeper layers are worse** — L1/L2 residuals carry stronger σ_B context
   traces, producing larger ΔNLL and higher residual ratios (noise).
5. **Multi-order averaging doesn't fix it** — averaging H across σ_B, phys,
   and random orders does not recover a usable content component.

## Interpretation

On this frozen checkpoint (seed123 step10k, 4-layer 8-head 47M params), the
attention scaffold — particularly with multi-head carrier aggregation — already
captures nearly all available reveal-order utility. The residual hidden state,
extracted under any tested reveal context, does not carry a complementary
per-sample content signal that is learnable by the current g_H architecture
(per-block MLP).

This is a single-ckpt null. It does not prove H *never* carries utility in
any model/scenario, but the exhaustive sweep makes it unlikely within the
current fixed-layout, 4-layer regime.

## Version A — Direct-NLL soft-routing (2026-06-30)

> Spec: `docs/superpowers/specs/2026-06-30-p5-direct-nll-soft-routing-design.md`

Replaced the pairwise-teacher imitation objective with NLL-weighted candidate
routing on the SAME frozen scaffold: `z=g_B(B)[+α g_H(H)]`, `a_k=z·y_k`,
`p_k=softmax(a_k/τ)`, `L=Σ_k p_k·L_k` (L_k pre-computed → model not in grad path).
Positioning: a gradient sanity-check + B-only objective comparison, NOT a new H
claim. seed123 step10k, M=64, τ∈{0.03,0.1,0.3,1.0}.

**Verdict — A1✅ routing works; A2 direct ties pairwise on pool-select but is a
worse ranker; A3✅ H still null.**

| metric (τ=0.3, main L0H1) | pairwise B-only | direct B-only | direct B+H |
|---|---|---|---|
| **pool-selected** regret | 0.0033 | 0.0033 | 0.0033 |
| **free-argsort** regret | **0.138** | 0.168 | 0.193 |

- **A1**: routing loss trains stably (3.78→3.65 every run). Controllers move
  selection OFF σ_B (headroom 0.0436) onto the near-optimal candidate
  (regret 0.0033) — capturing ~92% of pool headroom. Routing does real work.
- **Pool-selected is degenerate-easy**: the identity/physical order is
  near-optimal for nearly all texts, so every controller (pairwise/direct/all H
  arms) routes to it → no discrimination on the primary metric. Consistent with
  the project-wide "text → physical/L2R near-optimal" finding.
- **A2 discrimination lives in free-argsort**: pairwise (dense N² constraints)
  yields a better general ranker than direct (sparse K-candidate constraints) —
  pairwise regret_free ~0.13 vs direct ~0.17-0.20, **consistent across all τ and
  both scaffold configs**. As flagged in the spec: direct-NLL learns a candidate
  *router*, not a *ranker*.
- **A3 — H null confirmed under the cleaner objective**: `direct_bh_real` ≈
  `direct_b_only` ≈ `direct_bh_zero` on both metrics; no consistent shuffle/mean
  drop. The frozen-posthoc H null survives the objective change.

**Watchpoints carried to Version B:**
1. Direct routing's logit scale is unnormalized — z grows during training and
   saturates softmax (entropy→0, max_p→1) even at τ=1.0, so τ loses control.
   Normalize `a_k` (or constrain z) in B.
2. Direct learns a router, not a ranker → **B should keep candidate
   soft-routing, not a hard argsort(z) controller**.

Bug fixed mid-run: `candidate_orders` has `phys`==`local` (both identity); the
duplicate produced two equal logits → softmax floored at entropy ln(2)/max_p 0.5,
pinning argmax to a tie. Deduplicated byte-identical orders in `priority_matrix`
(shared `candidate_orders`/Phase-0 untouched).

Outputs: `runs/p5/seed123/routing_A/{main_L0H1,control_multihead}.json`.

## Code

- `analyses/p5_utility_controller.py` — full pipeline (order NLL, scaffold,
  candidates, headroom, H extraction incl. multi-layer/order-averaged,
  soft teacher, B-only + ScaffoldedController, dataset assembly,
  train loops, hard-NLL metrics, driver)
- `analyses/plot_p5.py` — metrics plot
- `block_lo_arm_order_network/tests/test_p5_*.py` — 12 tests across 8 files
- Outputs: `runs/p5/seed123/phase0.json`, `sweep_H_layer.json`,
  `ablation_H_context.json`, `compare_*.json`

## Run

```bash
# Baseline (single-head B, L0 H, sigma_B context)
python -c "
from analyses.p5_utility_controller import run_phase0
r = run_phase0('runs/handoff_overnight/seed123/ckpt_step10000.pt', M=64, n_reveals=8, epochs=200)
print('verdict:', r.get('metrics', {}).get('verdict', 'headroom gate failed'))
"

# Multi-head B + H context ablation
python -c "
from analyses.p5_utility_controller import run_phase0
r = run_phase0('runs/.../ckpt.pt', M=32, heads=[1,2,3,4], layers=[2],
               h_context='avg', epochs=100)
"
```
