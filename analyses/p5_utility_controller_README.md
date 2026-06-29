# P5 Attention-Scaffolded Utility Controller — Phase 0

> Spec: `docs/superpowers/specs/2026-06-29-p5-attention-scaffolded-utility-controller-design.md`

**Red line:** H improves downstream reveal-order utility beyond the attention scaffold;
it does NOT recover physical order. Supervision = downstream teacher-forced AO NLL.
Fixed layout; B-only baseline; shuffled-H control.

**Claim tested:** On a single frozen seed123 step10k checkpoint with seed123 carrier
(L0H1), block-level hidden states pooled under sigma_B's reveal context carry
per-sample utility-relevant information that is complementary to attention topology.

## Phase-0 Verdict (2026-06-29)

**`no_gain`** — headroom gate passes (abs=0.044), but H does not provide utility
beyond B. **Room exists, H not learned.**

| Metric | Value | Notes |
|--------|-------|-------|
| Headroom gate | **PASS** (abs=0.044, CI low>0) | σ_B is not the best order — room for improvement |
| Best-candidate dist | `phys` dominates | physical order is the most frequent best candidate |
| NLL B-only | 3.895 | Scaffold's C-D+L order, hard-forwarded |
| NLL B+H | 3.927 | **Worse** (+0.033) — H residual hurts |
| h_shuffle_drop | 0.003 | Shuffled H ≈ real H (H content not driving the difference) |
| zero/mean match | False/False | Not a capacity artifact |
| Residual ratio | (from run) | g_H contribution relative to g_B |

**Interpretation:** The attention scaffold σ_B leaves ~0.04 NLL headroom (some
candidate orders are better), but block-level hidden states under σ_B's reveal
context do NOT carry per-sample utility-relevant information that is complementary
to what B already encodes. The small ΔNLL is not driven by H content (shuffle
nearly matches).

This is the canonical "headroom positive but no_gain" reading: utility room
exists in principle, but H on the current frozen checkpoint at this extraction
point (L0 residual, σ_B reveal context) does not capture it.

## Code

- `analyses/p5_utility_controller.py` — full pipeline (order NLL, scaffold,
  candidates, headroom, H extraction, soft teacher, controllers, dataset,
  train loops, metrics, driver)
- `analyses/plot_p5.py` — metrics plot
- `block_lo_arm_order_network/tests/test_p5_*.py` — 8 test files, 12 tests

## Run

```bash
python -c "
from analyses.p5_utility_controller import run_phase0
r = run_phase0('runs/handoff_overnight/seed123/ckpt_step10000.pt', M=64, n_reveals=8, epochs=200)
print('verdict:', r.get('metrics', {}).get('verdict', 'headroom gate failed'))
"
```
