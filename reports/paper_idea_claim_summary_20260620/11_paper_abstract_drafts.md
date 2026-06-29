# Paper Abstract Drafts

## Draft A — Mechanism-Focused

> Order-agnostic autoregressive (AO) language models are trained with randomly permuted input sequences, yet we find that they do not remain internally order-free. Under a strict 65-node None-separated block graph protocol, we show that selected early-layer attention heads in a 47M-parameter AO-GPT recover the full physical left-to-right block order starting from an independent None token — achieving Kendall τ=1.000 with destroyed controls near random (|τ|≤0.07). The signal is sparse (7.8% of head×method combinations pass a strict gate), head-specific (e.g., L0H7 fails at τ=0.292), and stable across 5,000–60,000 training steps, though specific strong-head indices drift across runs. Using a distilled g_β neural readout of discovered order-bearing heads as a frozen order controller, we accelerate canonical-order training by 33–42% in step savings across two matched seed groups. Our results demonstrate that order-agnostic training does not eliminate internal order structure, and that emergent attention order can guide practical training acceleration.

## Draft B — Controller/Applications-Focused

> We present a method for discovering and exploiting internal order structure in order-agnostic autoregressive language models. Although these models are trained with randomly permuted input blocks, we find that sparse attention heads encode recoverable left-to-right block order. Using a strict 65-node block graph protocol with an independent None start, we identify order-bearing heads without label supervision and distill them into a lightweight neural readout (g_β). When deployed as a frozen order controller, g_β accelerates canonical-order training by 33–42% across two matched seed groups, corresponding to ~62% wall-clock savings. Our work connects mechanistic attention analysis to practical training efficiency, showing that random-order models contain exploitable order structure that can be read out rather than externally designed.

## Draft C — Discovery-Focused (Shorter, for tight page limits)

> Does order-agnostic training eliminate internal order representations? We show it does not. Under a strict 65-node None-separated block graph, selected attention heads in a random-order AO-GPT recover the full physical left-to-right block order (τ=1.000) from an independent start token. The signal is sparse (7.8% of heads pass), head-specific, stable across training, and verified by destroyed controls (|τ|≤0.07). Distilling this structure into a frozen order controller yields 33–42% training step savings. Order-agnostic is not order-free.

---

## Contribution Bullets (for Introduction)

1. **Discovery**: We show that random-order AO-GPT attention heads internally encode recoverable block-level L2R order, verified under a strict 65-node label-free protocol with destroyed controls.

2. **Protocol**: We define and validate a strict 65-node None-separated block graph that avoids prior protocol artifacts (None-physical0 folding, predictor-frame content-only limitations).

3. **Controller**: We demonstrate that attention-derived order structure can be distilled into a frozen g_β controller, accelerating canonical-order training by 33–42% across two seed groups.

4. **Analysis**: We characterize the head-specificity, extraction-frame dependence, and training-step stability of the discovered order signal.
