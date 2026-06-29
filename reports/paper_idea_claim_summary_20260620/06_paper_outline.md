# Paper Outline

## Title Candidates

1. **"Order-Agnostic Is Not Order-Free: Discovering Internal Order Structure in Any-Order Autoregressive Language Models"**
   - Emphasis: discovery + mechanism
   - Best for: general ML audience, mechanism-focused venues

2. **"Reading Order from Random-Order Transformers: Block-Level Attention as an Emergent Order Signal"**
   - Emphasis: method + readout
   - Best for: attention analysis, interpretability

3. **"From Emergent Attention Order to Order Controllers in AO-GPT"**
   - Emphasis: discovery → application chain
   - Best for: systems/applications venues

4. **"Block-Level Order Discovery in Random-Order Autoregressive Language Models"**
   - Emphasis: concise, descriptive
   - Best for: general submission

5. **"Sparse Attention Heads Encode Recoverable Left-to-Right Order in Order-Agnostic Language Models"**
   - Emphasis: specificity of finding
   - Best for: mechanism-heavy venues

---

## Abstract Skeleton

> Order-agnostic autoregressive language models are trained with randomly permuted input sequences, learning to predict tokens under arbitrary reveal orders. A natural question arises: does the model internally represent order structure despite training with randomized orders?
>
> We show that the answer is yes — but the signal is sparse, head-specific, and extraction-protocol-dependent. Under a strict 65-node None-separated block graph protocol, selected early-layer attention heads recover the full physical left-to-right block order from an independent start token, achieving τ=1.000 with destroyed controls near random (|τ|≤0.07). The signal appears by 5,000 training steps, persists through 60,000 steps, and generalizes across two independently trained models, though specific strong-head indices drift.
>
> We distill this discovered structure into a lightweight neural readout g_β and deploy it as a frozen order controller, achieving 33–42% training step savings across two matched seed groups. Our work demonstrates that order-agnostic training is not order-free internally, and that emergent attention order can guide practical training acceleration.

---

## Paper Structure

### 1. Introduction
- Order-agnostic training: motivation (flexibility, any-order generation)
- Core question: Is order structure internally represented despite randomized training?
- Our finding: Order-agnostic ≠ order-free — sparse heads encode recoverable L2R
- Two-layer contribution: mechanism discovery + controller acceleration
- Contributions bullet list

### 2. Related Work
- Any-order autoregressive models (AO-GPT, permutation language models)
- Diffusion and masked language models (MDLM, D3PM)
- Order schedules and curriculum learning
- Attention mechanistic analysis (induction heads, probing)
- Position relative to each

### 3. Method
- **3.1** AO-GPT setup: random-order training, block-level permutation, token-level loss
- **3.2** Block-level attention graph extraction: protocol evolution (B0→B1→strict 65-node)
- **3.3** Strict 65-node None-separated protocol: definition, coordinate conventions, inv_perm boundary
- **3.4** Order readout: CDL teacher, L-only vs C-D+L, rollout from None
- **3.5** g_β controller: distillation from selected head, frozen hook into training

### 4. Experiment Design
- **4.1** Training setup: 47M AO-GPT, Wikitext-103, block_size=256, 64 blocks × 4 tokens
- **4.2** Metrics: τ_vs_L2R, recovery rate, step saving, pairwise accuracy
- **4.3** Baselines: random-order, L2R reference, CDL teacher
- **4.4** Controls: destroyed B (entry shuffle, label permutation, Gaussian, zero)

### 5. Results
- **5.1** Order-bearing heads exist (clean-base, collaborator, 317M diagnostic)
- **5.2** Strict 65-node discovery (gate distribution, strong-pass heads, L0H7 fail)
- **5.3** Extraction frame dependence (B1 predictor vs loss-aligned AR)
- **5.4** Stability across training (5k–60k ladder, head drift)
- **5.5** Destroyed controls (entry shuffle, content label perm)
- **5.6** Legacy g_β controller acceleration (recovery rates, step saving, wall-clock)
- **5.7** g_β sanity (real vs Gaussian vs zero B)

### 6. Discussion
- **6.1** Content-vs-position memory (adjacency under fixed permutation)
- **6.2** Why signal is head-specific and transient in some frames
- **6.3** Relationship between mechanism discovery and controller acceleration
- **6.4** Comparison to alternative interpretations

### 7. Limitations
- Head-specific and extraction-frame-dependent
- Strict 65-node teacher-to-controller chain not yet fully closed
- Fixed permutation cannot separate content vs position memory
- Single model scale (47M), single dataset
- g_β is seed- and head-dependent
- val_unstructured degrades (order specialization trade-off)

### 8. Conclusion
- Summary of findings
- Future work: close strict-teacher controller loop, multi-scale validation, image modality
