# Positioning and Novelty

## Relative to Existing Work

### 1. Any-Order GPT / Any-Order Autoregressive Training

**Existing view**: AO-GPT (and similar any-order AR models) treats the training order as a sampled variable — the model learns to condition on arbitrary permutations of the input sequence. The focus is on enabling order-agnostic generation or improving likelihood through order averaging.

**Our question**: Does order structure emerge *internally* in the model's attention, even when the training order is randomized? We find that it does — in sparse, head-specific patterns — and that this structure is recoverable without label supervision.

**Key difference**: We do not propose a new training-order schedule. We analyze what order information the model *already encodes* as a byproduct of order-agnostic training.

### 2. Autoregressive Diffusion / Masked Diffusion / Discrete Diffusion LMs

**Existing view**: MDLM, D3PM, and related discrete diffusion models treat generation as iterative denoising or use masked tokens as the generative process. Order is either implicit (denoising schedule) or not the primary focus.

**Our question**: In random-order AR (not masked/diffusion), the model still processes one token at a time autoregressively. We analyze the attention graph *between blocks* as an emergent order signal, not as a denoising process.

**Key difference**: We study random-order AR attention, not diffusion. The block graph is built from attention aggregation, not from a denoising objective.

### 3. Training-Order / Sampling-Order Selection

**Existing view**: Order schedules are typically external, predefined, or learned as a policy (e.g., RL-based reveal-order selection, curriculum learning over order difficulty).

**Our question**: Can we *derive* the order from the model's own attention structure, without external supervision or RL?

**Key difference**: The order controller is distilled from the model's internal attention structure — it is not an externally designed curriculum. This is a *readout* approach, not a *scheduling* approach.

### 4. Attention-Based Mechanistic Analysis

**Existing view**: Attention analysis in transformers is typically token-level (individual token-to-token attention patterns, induction heads, etc.).

**Our question**: Can we define a *block-level* attention graph that captures order structure across entire content blocks? And can we do this in a *label-free* way?

**Key difference**: We define a strict 65-node block graph (1 None + 64 content blocks) with an independent None start. This is block-level (not token-level), label-free (no oracle τ during head selection), and verified with destroyed controls.

### 5. Order Controller / Curriculum / Reveal-Order Policy

**Existing view**: Order controllers or curricula aim to improve training efficiency or generation quality by choosing which order to reveal tokens. These are typically heuristic (easy-to-hard) or learned via RL.

**Our question**: Can an order controller be *derived* from the attention structure of a separately trained random-order model, then *frozen* and used to accelerate canonical-order training?

**Key difference**: The controller is *distilled* from emergent attention structure, not learned online. It is frozen during deployment — a readout, not a policy network.

---

## Novelty Table

| Existing View | Our Question / Contribution |
|---|---|
| AO training treats order as a sampled training variable | We ask whether order structure emerges internally in attention |
| MDLM/diffusion focuses on denoising or likelihood | We analyze random-order AR attention as a block-level order graph |
| Order schedules are usually external / predefined | We derive order from the model's own attention structure |
| Attention analysis is often token-level | We define a strict block-level 65-node graph with None-separated start |
| Acceleration often uses heuristic curriculum | We distill discovered order-bearing heads into a frozen controller |
| Order controllers are learned online | Our controller is a frozen readout of a separately trained model |

---

## Conservative Novelty Statement

**English**:

> Our contribution is not simply a new reveal-order schedule or an improved training curriculum. It is evidence that random-order autoregressive language models *internally* develop recoverable block-level order structure in sparse attention heads, and that this structure can be distilled into an order controller that accelerates canonical-order training. The novelty lies in the *discovery* of emergent order-bearing attention under a strict label-free protocol, and in the *connection* between mechanistic analysis and practical training acceleration.

**Chinese**:

> 我们的贡献不是单纯提出一个新的 reveal-order schedule 或改进的训练 curriculum。我们发现 random-order AR 语言模型内部会在稀疏的 attention head 中涌现出可读出的 block-level order structure，并且这个结构可以被蒸馏为 order controller 用于加速 canonical-order 训练。Novelty 在于：1) 在严格 label-free 协议下 *发现* 涌现的 order-bearing attention；2) 将 mechanistic analysis 和 practical training acceleration *连接* 起来。
