# Auto-Order-Head — Method Retrospective (text-first staging)

> 最后更新 2026-06-03。**当前主线收窄到 text**:把
> `attention → selected head → g_β → order → training loop` 这条链在 text 上**验证干净**。
> **跨模态(Stage 3)和后续更新信号(Stage 4)明确后置,现在不急。**
>
> 核心 framing:**CDL 不是最终方法,而是 Stage-1 的 scaffold / pretrain teacher**,把 raw
> attention graph 里的 order signal 先转成一个可监督的预训练目标,给 `g_β` 一个**可部署的初始
> readout**。g_β 当前定位 = **CDL-supervised initialization + frozen deployed controller**;先证明
> 它 work,Stage 2 见到收益后再考虑用 training feedback 改进它(Stage 4)。

---

## 0. North star + 当前主线

底座是 any-order GPT(AOGPT):在**打乱的 block reveal 顺序**下做 order-agnostic 训练。
问题:**能否从模型自己的 attention 里得到一个可部署、可微的 order controller**,使
"same loss, fewer steps",并(最终)跨模态对齐数据内在结构。

**阶段划分(text 主线 + 后置项)**:

```
[text 主线 —— 现在就要跑通]
  Stage 0    signal discovery : attention 里有 order signal 吗?能定位到某个 head 吗?   ✅
  Stage 1    CDL pretrain      : g_β 能从 raw B 学到 CDL-like order computation 的近似吗?  ✅
  Stage 1.5  generalization    : g_β 学到的是可迁移 readout,还是记 5k 训练集 / 常数 L2R?  ✅
  Stage 2    hook into loop     : 无 CDL,frozen g_β 真能驱动训练、比 random 更快达同 loss 吗? 🔄 ← 最关键下一步

[后置 —— text 跑通后再说]
  Stage 3    cross-modal (image): 复用机制找 locality head → g_β → image training            ⏸ 后置
  Stage 4    update signal       : g_β ← training feedback(NLL / policy / CEM / refresh)      ⏸ 暂不定
```

**CDL 是什么**:确定性算法。给 `B = Aᵀ`,算 readiness `r = out − α·in`(α=0.5),从 source
(argmax r)起贪心 rollout 一条揭示序 σ。只用 attention 几何,不碰 NLL / L2R label。它的两个
性质决定整条路线:**不可微(贪心 argmax)+ 逐步算** → 没法直接进训练 hook → 需要一个可微、
frozen、一次前向出序的替身 `g_β`。

> **用语纪律**:`g_β` 学的是 **the graph-to-order mapping induced by CDL** 的一个近似,
> **不是**"复刻 CDL 的内部计算"。详见 §6。

---

## 1. text 主线四步(Stage 0 / 1 / 1.5 / 2)

### Stage 0 — Text 上验证 attention 里有 order signal、并定位到 head ✅

- **回答**:attention 里有没有 order signal?能不能定位到某个 head?
- **做法 / IO**:`θ_text → A^{l,h} → B^{l,h}=Aᵀ`;用 **row-concentration**(B 每行出边的归一化
  负熵,dead-row→0)做主选择器,CDL / `τ_vs_L2R` 做诊断。
- **为什么 row-concentration**:学到确定序的 head,转移图每 block 强指向唯一后继 → 每行质量
  集中 → 行负熵高。**只依赖"行是否尖锐",几何无关** → 模态无关的候选选择器。
- **状态(已基本成立,text clean_base)**:
  - 早期 cheap score **C1–C4 全失败**(full ladder Recall@1=0/40);
  - **none-token 污染**:旧提取把 `[None]` 以 0.1 当 sink 加进所有列 → row-conc 晚期塌到 ≈0;
  - **B0 修复**(none→physical block0,段平均,无 magic 权重)→ row-conc 救活;
  - **§3.0 full ladder gate PASS**(9 ckpt × 5 seed = 45/45):min median C=0.046、poolP@5=0.92、
    best+recall=1.00、**winner=L0H0 从 5k 后稳定、跨全步×全 seed 零漂移**,`τ` 接近 L2R(OLD
    的"漂移到 L1H4"是提取伪影)。→ **text attention 中确有可解码的顺序结构,且能定位到一个 head。**

### Stage 1 — Text 上用 CDL 预训练 g_β ✅

- **回答**:一个 neural readout 能不能**直接从 raw selected-head attention graph** 里学到
  CDL-like order computation 的近似?
- **做法 / IO**:
  ```
  B^{head}  --CDL-->  σ_T                  (teacher,scaffold,离线)
  g_β(B^{head}) = s ∈ R^N  --argsort-->  σ̂
  loss = pairwise logistic(s 的两两序 vs σ_T)
  ```
  输入 = 原始 B0 batch-mean 图(N×N,对角清零,物理坐标),**不喂手搓特征**;输出 = per-node
  分数(one-shot ranking);g_β = graph-transformer readout(NodewiseReadout,复用 NR-1/BR-1)。
- **CDL 角色 = pretrain teacher / scaffold,不是最终方法**;重点是给 g_β 一个**能读 attention
  order signal 的初始化**,不是"永远模仿 CDL"。
- **边界**:CDL 只活在离线;Stage 1 的 selection/gate 不许 task NLL / L2R label 入场。

### Stage 1.5 — Text 上验证 g_β 泛化(不能跳)✅

- **回答**:g_β 学到的是**可迁移 readout**,还是记住 5k 训练集 / 记一个常数 L2R?
- **做法**:① 新 batch 泛化(5k held-out)② 跨 step 泛化(5k 训好的 g_β 测 10k/20k 同 head B)。
- **状态(full run M=2000、bs=8、head=L0H0 → PASS)**:
  | split | g_β τ | L2R-prior τ | non-L2R subset g_β τ |
  |---|---|---|---|
  | 5k/val | **0.849** | 0.760 | 0.648 |
  | 5k/test | 0.836 | 0.780 | 0.640 |
  | 10k/cross | 0.927 | 0.966 | 0.635 |
  | 20k/cross | 0.930 | 0.990 | 0.402 |
  - gate(τ≥0.6, pairwise≥0.8):same-step ✓ / cross-step ✓ → **PASS**;
  - **反 shortcut 关键证据**:teacher≠L2R 的子集(常数 L2R 必败处)g_β τ=0.64 → 跟着 CDL 在偏离
    L2R 的样本一起偏 = 学到 B-dependent 映射,不是记常数;smoke(M=80)的悲观是 underpower 伪影。

### Stage 2 — Text 上接入训练 loop(无 CDL,frozen g_β)🔄 最关键下一步

- **回答**:frozen g_β 能不能当 CDL 的**可部署替身**真正驱动训练?能不能比 random-order baseline
  更快达到相同 loss?是否稳定、不崩、overhead 可接受?
- **训练环**:`σ_t → fwd/bwd → A_t^{head} → B_t^{head} → g_β → σ_{t+1}`;无 CDL、无 L2R、无 NLL
  更新 g_β、g_β frozen。每 K 步用 random-order 探针提 L0H0 的 B0 图喂 g_β(保 in-distribution)。
- **公平比较(诚实点)**:序影响 NLL → **不比 raw train loss**(hook 训 ≈L2R 序更易、loss 天然低),
  比**固定协议 eval NLL**:`val_unstructured`(order-agnostic)+ `val_ori_l2r`。baseline 曲线读
  clean_base 9 ckpt 的 `last_eval`。LR 锚 50k(resume 不分叉);batch32×grad_accum4=eff128(=baseline)。
- **状态(RUNNING)**:4-run ladder(5k/10k/20k/40k → 60k)在 GPU1 跑;完后 `aggregate_frozen_beta_curves.py`
  出 hook ladder vs baseline 对比。对照含 random / (旧)CDL·MLP / g_β / L2R·raster、step-savings、overhead。
- **这一步是 text 主线目前最关键的缺口。**

---

## 2. g_β 的 IO 与 one-shot vs policy

- **当前(Stage 1)**:`B → logits ∈ R^N → order`,**one-shot ranking**,Stage 1.5 已过。
- **可能不够**:CDL 本质是 sequential greedy rollout;one-shot 只近似它的**最终排序**,学不到
  per-step decision。若(后置的)image / path-dependent order 上不足,再升级。
- **升级备选(需要时再上)**:pairwise matrix / transition matrix / per-step policy
  `p(v_{t+1} | B, S_t, U_t, last)`。不默认当前结构一定够。

---

## 3. 已验证 vs 拍脑袋(ledger)

| 项 | 状态 | 边界 / 还缺什么 |
|---|---|---|
| B0 extraction(none→block0)修复 row-conc 污染 | ✅ 工程已验证(45/45) | **建模语义未补**:为何折进 block0 而非丢弃/special node/均匀分配 |
| row-concentration 作 head selector | ✅ **text clean_base** 已验证 | ❓ image 未验;❓ cross-run 召回(alt_from0 L0H5)未验;❓ 是否选到"尖锐但无用"head;❓ top-k 多 order head ≠ 对训练有益(后置 Stage 3 才碰 image) |
| g_β 从 raw B 学 CDL-induced 映射 + 泛化 | ✅ text Stage-1.5 PASS | one-shot logits;text teacher 大多 ≈L2R,增益集中少数非 L2R case,**不 over-claim utility** |
| g_β 比常数 L2R 更跟 CDL(反 shortcut) | ✅ non-L2R subset τ=0.64 | 仍需更 nontrivial 分布上扩展 |
| **frozen g_β hook 是否提升训练** | 🔄 RUNNING,未结论 | **text 主线最关键缺口(Stage 2)** |
| one-shot g_β 是否够表达复杂 order | ❓ 未验证 | image/path-dependent 才会暴露(后置) |
| g_β vs 直接 CDL 的部署优势 | ❓ 未量化 | CDL online vs g_β forward+extraction;若 CDL 可离线 refresh,g_β 优势 = 可微(为 Stage 4)+ 单次前向替代逐步 rollout |
| 多模态统一(image) | ⏸ **后置 Stage 3** | text 跑通后再做;image teacher 可能要换(locality/CDL variant/B-coverage) |
| Stage 4 更新信号 | ⏸ **暂不定** | Stage 2 见效再选 NLL/policy/ES·CEM/refresh |

---

## 4. g_β 当前定位 + 待拍板决策

- **g_β 当前定位(已定,为现在)= CDL-supervised initialization + frozen deployed controller**。
  先证明它 work;**不预先复杂化**。frozen-forever / online-finetune / periodic-refresh 的选择
  **推迟到 Stage 4**,且只有 Stage 2 见到收益才启动。
- **仍待拍板(但都后置)**:
  - **Stage 3(image)teacher**:CDL 是否仍能当 image 的 Stage-1 teacher,还是换 locality /
    CDL variant / B-coverage —— 等 text Stage 0–2 跑通再定。
  - **Stage 4 更新信号**:training loss improvement / frozen NLL-under-order(诊断)/ policy
    gradient / ES·CEM / periodic CDL refresh —— 暂不定。

---

## 5. 需要补的实验(按 text-first 优先级)

1. **(进行中,最关键)Stage 2 frozen hook 结果** —— 没有它,g_β 只是离线 CDL approximator。
   5k/10k/20k/40k → 60k;对照 random / 旧 CDL·MLP / g_β / L2R·raster;看 same-loss step-savings、
   稳定性、overhead。
2. **(text 内)non-L2R subset 持续扩展** —— 强调 teacher≠L2R 子集里 g_β 跟随 CDL 而非记 L2R
   (反 shortcut 关键)。
3. **(text 内)g_β vs 直接 CDL 的部署优势量化** —— 回答 reviewer 的"为何不直接用 CDL"。
4. **(后置)image 侧重复 Stage 0/1** —— row-conc 是否在 image attention 找到 **locality head**;
   image-specific teacher 是否能预训练 g_β。**text 主线跑通后再做。**
5. **(更后置)Stage 4 pilot** —— paper discussion 留"CDL pretraining initializes g_β; future work
   optimizes with task loss / policy search"。

---

## 6. 叙事用语纪律(避免 over-claim)

| ❌ 不要说 | ✅ 更准确 |
|---|---|
| g_β 完全复刻了 CDL 的内部计算 | g_β **approximates the graph-to-order mapping induced by CDL** |
| L2R 是我们的 label | L2R is **diagnostic** for text intrinsic order;**CDL is the Stage-1 pseudo-teacher** |
| row-concentration 证明跨模态成立 | row-concentration is a **modality-agnostic candidate selector**;**image validation remains needed** |
| Stage 1 就是最终方法 | Stage 1 **initializes a deployable order readout**;Stage 2 tests it in the loop;Stage 4 may improve it |
| g_β 一定提升训练 | Stage 2 正在验证;text 主要证明**可学、可泛化、可接入**,不过度 claim utility |

---

## 7. 最简总结

当前目标**不是立刻做 image,也不是立刻设计 g_β 的后续 RL/NLL 更新**;当前目标是在 **text** 上
完整验证:**attention 中有 order signal(Stage 0)→ CDL 当 Stage-1 teacher 预训练 g_β(Stage 1)→
g_β 能泛化(Stage 1.5)→ frozen g_β 无 CDL 接入训练 loop 并带来收益(Stage 2)**。text 链条跑通后,
再做跨模态(Stage 3);后续更新信号(Stage 4)留到更后面。当前在 Stage 2。
