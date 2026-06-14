# 07 — Boss Update Outline

> 10–15 minute presentation. Slide-by-slide with key numbers and visuals.

---

## Slide 1: One-line result (30 sec)

**"Random-order trained models spontaneously encode L2R structure in attention graphs. A learned readout of these graphs accelerates training by 35–42% across two seeds."**

---

## Slide 2: Method overview (90 sec)

**Figure**: `analyses/figures/fig1_method_overview.png`

```
AO-GPT (random order) → Extract B → CDL Teacher → g_β → frozen Hook
```

Three key properties:
1. g_β trained offline once per (seed, head)
2. At deployment: one MLP forward pass, no iterative decoding
3. g_β@10k works at 20k, 40k (cross-step transfer)

---

## Slide 3: g_β sanity — reads B, not prior (90 sec)

**Figure**: `analyses/figures/fig2_gbeta_sanity_bar.png`

| B source | g_β τ | 
|----------|:--:|
| Real B | **0.97** |
| Gaussian / shuffled / destroyed | **≈0** |

**Claim**: g_β cannot be a constant L2R prior. If it were, Gaussian/shuffled B would also give τ≈1.0.

Zero B τ=1.0 = tie-breaking artifact (all edges equal → CDL picks first available).

---

## Slide 4: Attention emergence — per-head sparse signal (60 sec)

**Method uses per-head selected attention graphs B^{l,h} as input.** Not all-head aggregate.

| Granularity | Best +τ | Best −τ | Head count |
|:--:|------|------|:--:|
| 32-group | L0H0 +1.000 | L2H0 −0.938 | sparse |
| 64-group | all ≈+1.000 | none | 28/32 strong |
| 128-group | L1H2 +1.000 | L2H3 −0.938 | 6 strong |
| 317M (diagnostic) | L0H10 +0.955 | L1H3 −0.938 | 6/256 strong |

→ **Sparse order-bearing heads exist across granularities and scales.** Head audition is necessary design, not afterthought.

---

## Slide 5: Frozen-β multi-start (120 sec)

**Figure**: `analyses/figures/fig3_catchup_curve.png` + `analyses/figures/fig4_multistart_comparison.png`

| Seed | Head | from10k Saving | from10k Recovery |
|------|------|:--:|:--:|
| seed2 | L0H2 | **41.7%** | **86.0%** |
| seed42 | L0H4 | **39.3%** | **106.1%** |

Both seeds show strong acceleration. Seed42 Recovery > 100% means frozen_β order beats pure L2R for this model.
Seed42 baseline group verified: random=3.466, L2R=3.341, gap=0.125 (normal).

**Conservative**: **35–42% step saving**, 2 seeds × 2 resume points.

---

## Slide 6: Robustness (60 sec)

| Ablation | Result |
|------|------|
| 32/64/128 block aggregation | τ > 0.96 ✓ |
| 32/64/128 training granularity | ±1.0 per-head ✓ |
| 317M (16L/16H/1024d) | |τ|>0.9 heads exist ✓ |

→ Not a 64-block or 47M artifact.

---

## Slide 7: CAN CLAIM / CANNOT CLAIM (90 sec)

**CAN CLAIM** (5 items, multi-line evidence for each):
1. g_β reads B structure, not constant prior
2. Attention contains physical-order-readable signal
3. Frozen g_β accelerates training (2 seeds, multi-start)
4. g_β@10k transfers to later checkpoints (same seed/head)
5. Survives granularity + 317M scale diagnostics

**CANNOT CLAIM** (6 items):
1. Cross-seed/head g_β transfer
2. ori-L2R is upper bound
3. Shuffled-L2R = no order at all
4. Fully label-free end-to-end
5. Image solved
6. 317M hook established

---

## Slide 8: Decisions needed (60 sec)

1. **Paper scope**: Text-only or multimodal?
2. **Next compute**: Third seed or 317M hook?
3. **Positioning**: "Order controller" or "emergent structure"?
4. **Label-free**: Finalize now or defer?
5. **Image**: In scope or out?

---

## 30-second verbal summary

> Text-side core story is closed: g_β reads structured attention graphs rather than a fixed L2R prior. Frozen g_β accelerates training across two seeds and multiple resume points — conservative 35–42% step saving. The phenomenon survives granularity and 317M scale diagnostics. Remaining work is packaging and deciding whether to expand to scale, image, or stage-3.

---

## Backup slides (if asked)

- CDL teacher Recovery > 100% (shows ori-L2R not upper bound)
- Shuffled-L2R control (wrong order → no physical signal)
- B matrix heatmaps (visual confirmation of emergent structure)
- Detailed recovery numbers (Tables A–G)
