# Core Idea

## English

**Order-agnostic training is not necessarily order-free internally.**

Even when AO-GPT is trained with random reveal orders, the model internally develops attention heads that encode recoverable block-level L2R order structure. Under a **strict 65-node None-separated block graph protocol** — where one node is the independent [None]/BOS start and 64 nodes represent physical content blocks — selected early-layer attention heads can recover the full physical L2R block order from an unanchored None start, without any label supervision.

We further show that this discovered order structure can be **distilled into a lightweight neural readout (g_β)** and used as a **frozen order controller** to accelerate canonical-order training by 33–42% (step saving), across two matched seed groups.

Critically, this work is organized in **two layers**:

- **Mechanism layer**: Random-order AO-GPT attention contains sparse, order-bearing heads. Under the strict 65-node label-free protocol, selected early heads (e.g., L0H1–L0H4) recover full physical L2R block order from an independent None/BOS start. This is a *discovery* about internal model structure — it is head-specific, readout-specific, and verified with destroyed controls (|τ| ≤ 0.07).

- **Method / acceleration layer**: A selected-head attention graph can be distilled into g_β and used as a frozen order controller. Existing legacy controller results (B0 protocol) show 33–42% step saving. The closure experiment — training g_β from a strict 65-node teacher and hooking it into training — is the next priority and is currently pending.

**Key constraint**: The existing frozen hook acceleration uses a legacy B0 controller path (where [None] is folded into physical block 0). The strict 65-node protocol currently supports the *mechanism discovery* claim; the *strict-teacher controller hook* is not yet demonstrated and must not be claimed as completed.

---

## 中文

**Order-agnostic 训练不等于模型内部没有 order structure。**

即使 AO-GPT 在训练时 reveal order 是随机的，模型内部仍会在部分 attention head 中形成可读出的 block-level L2R 顺序结构。我们使用一个 **strict 65-node None-separated block graph 协议** —— 其中一个节点是独立的 [None]/BOS 起始符，64 个节点对应物理内容 block —— 在不使用任何 label 监督的情况下，selected early-layer attention heads 可以从独立的 None 起点恢复出完整的 physical L2R block order。

我们进一步证明，这个被发现的 order structure 可以被 **蒸馏为轻量级神经网络读出器 (g_β)**，并作为 **frozen order controller** 接入训练，在两个 matched seed groups 下实现 33–42% 的 step saving。

关键地，这个工作分为**两层**：

- **机制层 (Mechanism layer)**：Random-order AO-GPT 的 attention 中包含稀疏的 order-bearing heads。在严格 65-node label-free 协议下，选定的 early heads（如 L0H1–L0H4）从独立的 None/BOS 起点恢复出完整的 physical L2R block order。这是一个关于模型内部结构的 *发现* —— 它是 head-specific、readout-specific 的，且被 destroyed controls 验证 (|τ| ≤ 0.07)。

- **方法/加速层 (Method / acceleration layer)**：Selected-head attention graph 可以被蒸馏为 g_β 并作为 frozen order controller 使用。现有的 legacy controller 结果（B0 协议）显示 33–42% 的 step saving。闭合实验 —— 从 strict 65-node teacher 训练 g_β 并 hook 进训练 —— 是下一步最高优先级，目前 pending。

**核心约束**：现有 frozen hook 加速使用的是 legacy B0 controller path（其中 [None] 被折叠进 physical block 0）。Strict 65-node 协议目前支撑的是 *mechanism discovery* claim；strict-teacher controller hook 尚未被证明，不可作为已完成工作宣称。

---

## Two-Layer Summary

| Layer | Status | Core Finding |
|-------|--------|-------------|
| **Mechanism** | ✅ Verified | Sparse early heads recover physical L2R under strict 65-node label-free protocol |
| **Acceleration** | 🟡 Legacy verified / Strict pending | B0 g_β controller saves 33–42% steps; strict-teacher hook not yet closed |
