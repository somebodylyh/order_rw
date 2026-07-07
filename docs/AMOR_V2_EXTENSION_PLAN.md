# AMOR v2 扩展规划：可变 Block Size · Token 级对齐 · 多模态

> **Status**: design draft (2026-07-07)。基于 [AMOR_PIPELINE.md](AMOR_PIPELINE.md) v1（PG-only, tau=0.05, L1H6, N=64/BLOCK_LEN=4/SEQ=256）。
> **定位锁 (2026-07-07 用户裁定)**: **text 固定 BLOCK_LEN=4 不动**（v1 已 work，不做 token 级 text 细化）。可变 block size 只为**其他模态**服务——图像走 patch 尺度、音频走帧尺度，各选自然粒度。因此 size-agnostic gβ 的驱动力是**多模态**，不是 text 细化。
> **原则锁**: 一次只动一个变量；own-order val 唯一 headline（val_l2r 仅 anti-gaming）；任何新模态先过 oracle headroom gate 再建 gβ；NLL 绝不进 head/architecture selection；strict65 read-out lock。

---

## 1. v1 回顾与其对 block size / 模态的隐含假设

### 1.1 关键流程（一句话链）
`10k random-order backbone → 无监督选 head (L1H6) → 抽 24k per-sample B → batch-mean-16 → 1500 张 mean-B → CDL(C−D+L) greedy → pairwise BCE 训 NodewiseReadout (val_acc 98.44%) → 部署时 A=16 anchor 在 σ_cache 下前向 → B_anchor_mean → gβ → argsort → 广播 64 行 → PG every 100 (tau=0.05) 微调 gβ`。

### 1.2 v1 的四条核心假设，以及它们各自绑死了什么

| 假设 | v1 具体形态 | 绑死了 |
|------|------------|--------|
| **图是 block-level 且 N 固定** | B ∈ ℝ^{64×64}，node 特征 = `concat(row, col)` = 128 维，encoder 作用在 64 个 node 上 | gβ 架构与 **N=64 硬绑定**。改 block size ⇒ 改 N ⇒ 输入维度和 encoder 序列长都变 |
| **batch-mean 才有信号** | 单 window 图噪声太大，mean over 16 独立 window 结构才浮现 | 噪声 ∝ 每 node 的边数 ≈ N。**N 越大（block 越细）→ 每张图越稀疏噪声越大 → batch_mean_size 必须随 N 增长** |
| **一个 σ 广播到整个 batch** | argsort(scores) 得单条 σ_batch，expand 到全部 64 行 | 部署顺序是 **per-batch 而非 per-sample**（记忆：group-oracle headroom m=1 → m=16 归零，batch 臂无 per-sample 靶）。token 级细化会重新逼问这个瓶颈 |
| **teacher 是 attention-only heuristic** | CDL = C−D+L，纯从 attention 图推 order，无外部 label | teacher 的**归纳偏置是"文本 attention 图结构"**。跨模态时这个偏置不一定成立（记忆：图像端 CDL 是累赘，locality/D_manh 才 work） |

**结论**：v1 的 gβ 本质是一个 **fixed-N graph-to-permutation 读出器**。扩展的公共前置件是——**先把 gβ 从固定 N 解耦成 size-agnostic 图读出器**，这样不同模态才能各用自然粒度（text=4 保持不变，图像=patch 尺度，音频=帧尺度）。这是 v2 的地基。**注意**：可变 block size 是**跨模态需求**，不是给 text 做 token 级细化——text 那条 §2.3 的 token/分层内容降级为可选研究项，不进主线。

---

## 2. 可变 Block Size + Token 级对齐

### 2.1 定义 block size 与保持 gβ 分布匹配

`N = SEQ_LEN / BLOCK_LEN`。候选：BLOCK_LEN ∈ {1, 2, 4, 8, 16} ⇒ N ∈ {256, 128, 64, 32, 16}。BLOCK_LEN=1 即 **token 级**（B 就是原始 256×256 token attention）。

分布匹配是最大难点：不同 block size 下 B 的**统计量本身不同**（细 block → 图更稀疏、单边 attention 质量更低、行/列和的分布偏移）。两条路线：

- **(a) size-conditioned 单 gβ**：给 gβ 注入 block-size embedding（或直接把 BLOCK_LEN 归一化后作为全局 token 拼进 encoder），在 {2,4,8} 混合分布上训练。优点：一套权重跨粒度；缺点：需 size-agnostic 架构（见 2.2）。
- **(b) per-size gβ**：每个 block size 单独跑一遍 pretrain。朴素、可控、易 debug，但要维护多套 ckpt。**v2 起步用 (b)**（风险低、每步可验证），确认信号存在后再收敛到 (a)。

无论哪条，**部署 batch_mean_size 必须等于 pretrain batch_mean_size**（v1 的 A=16 ⟺ pretrain-16 对齐律不能破），且随 N 增大而增大（见 2.4）。

### 2.2 size-agnostic 读出架构（v2 地基）

把 `NodewiseReadout` 从"2N 维定长向量 + 定长序列 encoder"改成**图上的 message passing**，天然对 N 与 node 置换等变：

```
node 初特征 h_i^0 = MLP( [row_stat(B[i,:]), col_stat(B[:,i])] )   # 用可聚合的统计量,不用定长 raw 行
edge e_ij = MLP( B[i,j], B[j,i] )                                  # 有向边特征
h_i^{l+1} = h_i^l + Attn_j( q=h_i, k/v = h_j ⊕ e_ij )             # L 层 graph attention
score_i = Linear(h_i^L)                                            # 每 node 一个标量
σ = argsort(scores, desc)
```

关键：node 特征不再是 `concat(B[i,:], B[:,i])`（长度 = N），而是**边聚合的结果**（长度与 N 无关）。这样同一套权重能吃 N=16…256。

**局部子图变体（token 级用）**：token 数大时全图 attention O(N²) 太贵且噪声高。给每个 token 取 attention 图上的 **top-k 邻域子图**（k≈16），readout 只在局部子图上 message-pass 出该 token 的 score。这让读出 **scale-invariant** 且成本 O(N·k) 而非 O(N²)。

### 2.3 Token 级与 block/token 混合读出

- **纯 token 级 (BLOCK_LEN=1)**：σ 是 256 长排列。风险：per-sample 256×256 B 极噪 → batch_mean_size 需大幅上调；argsort 256 node 的 credit assignment 更难（PG 的 log-prob 摊到 256 个 slot）。
- **Hierarchical（推荐主线）**：两级。粗级 gβ_coarse 在 N_coarse=16 blocks 上给块间序；细级 gβ_fine 只在"当前块内 + 已揭示上下文"的局部子图上给块内 token 序。两级都用 2.2 的 size-agnostic 读出，参数可共享（同一个图读出器，喂不同粒度子图）。好处：把 256-way 排列分解成 16-way × block-内小排列，credit assignment 可控，且**与 v1 的 block 主线连续**（block 级是 hierarchical 的粗级特例）。
- **连续粒度旋钮**：训练时随机采样 BLOCK_LEN，让 size-conditioned gβ 学一族粒度；部署按开销预算选。

### 2.4 关键参数与 ablation

| 轴 | 取值 | 目的 |
|----|------|------|
| BLOCK_LEN | {1,2,4,8,16} | 粒度扫描；找信号-开销拐点 |
| 读出架构 | fixed-N MLP(v1) vs size-agnostic GNN(v2) | **回归 gate**：v2 架构在 N=64 上必须复现 v1 的 98.44%，否则不进 |
| 粒度模式 | block-only / token-only / hierarchical | 验证 hierarchical ≥ 两个端点 |
| Mean-B vs Mean-scores | (A) mean-then-score vs (B) score-then-mean | v1 结论 (A)>(B) 是否随 N 保持（细粒度下跨样本结构更重要还是更脆） |
| batch_mean_size | {16, 32, 64, 128} × N | 量出"噪声 ∝ N ⇒ 所需 mean-size ∝ N"这条 scaling |
| A (anchor) | 与 batch_mean_size 对齐扫 | 保持部署 ⟺ pretrain 分布匹配 |
| PG credit 粒度 | 整条 σ vs 分层（粗/细分别 PG） | token 级 credit assignment 是否需要分解 |

**Sanity（做训练前）**：对每个 block size，先测 batch-mean B 的 **split-half reproducibility** 和 **CDL teacher 的 oracle headroom**（teacher order vs L2R 的 own-order val Δ）。headroom 不显著 ⇒ 该粒度直接砍，不浪费算力训 gβ。

---

## 3. 多模态扩展（零先验框架）

### 3.0 不可破的零先验不变式

整条线的立身之本：**序必须从模型自己学到的注意力结构里涌现，不注入任何外部先验。** 具体三条硬约束，跨所有模态不变：

1. **gβ 只读模型自己的注意力图 B**——node 是 block/patch/atom/cell/token，B 是该模型在这些单元间的 attention。**绝不**读数据的真结构（分子邻接、数独约束、代码 AST）。
2. **teacher 恒为 CDL(C−D+L) on attention**——纯从 attention 图算，零领域知识。**不用** valence 规则、CSP most-constrained、拓扑序这类领域启发式。
3. **真·结构只能当 post-hoc oracle**——用来事后校验涌现序是否碰巧对齐领域合理序，**绝不进 gβ 输入、绝不进 teacher、绝不进 selection**。同 [[inv_perm_strict_boundary]]（只能 posthoc 翻译 σ，不能进构图）与 physical-order 的"禁 identity model-frame"纪律。

灌了真图或领域规则，"序能否零先验涌现"这个 claim 就没了——那只是在验证一个已知答案。

### 3.1 各模态：全部同一套零先验管线

| 模态 | node | 图 B（gβ 输入）| teacher | 真·结构的角色（**仅 post-hoc oracle**）|
|------|------|----------------|---------|------------------------------------------|
| 文本 (v1) | block(4 token) | L1H6 block attention | CDL(attention) | —（无客观 oracle）|
| 图像 | patch | patch 间 attention（无监督选 head）| CDL(attention) | 2D 网格 locality（仅校验，不进训练）|
| **分子** | atom | 模型对 atom 的 attention | CDL(attention) | 化学键邻接图 / 化学合理序 → 校验 + FCD |
| 数独 | cell | 模型对 cell 的 attention | CDL(attention) | 约束图 / CSP 合理序 → 校验 |
| 代码 | token/AST-node | 模型对 token 的 attention | CDL(attention) | 依赖 DAG / 拓扑序 → 校验 |

分子、数独、代码**不因"有显式图"而特殊**——它们只是同一个 attention-only 方法的新测试域。显式结构越客观，post-hoc oracle 越硬，**但它始终在训练之外**。

### 3.2 为什么分子是旗舰

零先验框架下，分子的价值不是"图更干净"（那是作弊），而是**它有一把客观的化学 oracle 尺**：
- 能检验"纯注意力涌现序 **是否碰巧** 对齐化学合理生成序"——对齐 = 强证据（零先验自发恢复领域结构）；不对齐也**不用它纠正**，只如实记录。
- **FCD（Fréchet ChemNet Distance）是成熟客观 metric**，不像图像 FID 受"模型太弱"拖累（记忆：FID 113 vs 合作者 40 只能相对比较）。分子端能做**绝对** claim。
- repo 名即 `block_lo_arm_order_network`（LO-ARM）；论文已在 QM9/ZINC250k 证 learned order 有用。AMOR 的差异化：**不端到端学序，而用零先验 gβ 从注意力读出序**——若 FCD 达标且序对齐化学 oracle，这是比图像强得多的证据。

### 3.3 图像的诚实警示（不作模板）

记忆里"图像 CDL 崩、locality/D_manh 更好"必须**如实标注为一次先验注入的让步**：D_manh 用了已知的 2D patch 网格坐标 = 空间先验。在严格零先验尺度下，这是 image 端的一个**弱化 claim / caveat**，**不是**新模态该照抄的模板。分子/数独/代码一律走**严格 CDL(attention)**；若某模态 CDL(attention) 不出信号，那是一个 clean finding，不靠灌先验去救。

### 3.4 gβ 跨模态：共享还是专属

三档，按风险递增：

1. **共享架构 + 专属权重 + 同一 CDL(attention) teacher**（推荐起步）：所有模态用同一个 size-agnostic 图读出器（§2.2），teacher 都是 CDL(attention)，各训一套权重。朴素、可比、每模态独立可验证、零先验一致。
2. **text 预训练 → 迁移**：text-gβ 冻结/微调迁到分子。只有当"attention 图结构→order 映射"确实模态无关（unified-g(B) 假设成立）时才 work。作为**科学问题**测。
3. **单一通用 gβ**：一套权重吃所有模态的 attention 图。终极目标，需 1、2 先给正面证据。

### 3.5 各模态指标与挑战

| 模态 | 主指标（headline=own-order val NLL）| post-hoc oracle | 已知挑战 |
|------|------------------------------------|-----------------|----------|
| 分子 | own-order NLL + **FCD（可绝对 claim）** | 化学键邻接对齐率 | 需 atom-level attention 抽取协议（新写 strict-N for molecules）|
| 图像 | own-order NLL；FID 仅相对 | 2D locality 对齐 | 模型太弱 → FID 不做绝对；D_manh 是先验让步 |
| 数独 | own-order NLL / 解正确率 | CSP 合理序对齐 | 强约束或使"任意序皆可解"→ headroom 天花板需先测 |
| 代码 | own-order NLL | 依赖拓扑序对齐 | AST 稀疏、attention 图噪声 |

---

## 4. 每处改动的理由、预期收益与必需 sanity

| 改动 | 为什么 | 预期收益 | 必需 sanity / gate |
|------|--------|----------|-------------------|
| gβ → size-agnostic GNN | 解除 N=64 硬绑定，一切扩展的前提 | 一套架构跨粒度/模态 | **N=64 回归到 98.44%**；node-置换等变性单测 |
| 分层 block/token | 把 256-way 排列分解，credit 可控 | token 级细化不炸 PG | hierarchical ≥ block-only 且 ≥ token-only |
| per-size → size-conditioned | 收敛成一套权重 | 部署时自由选粒度 | 混合训练不劣于各自 per-size |
| 全模态统一 CDL(attention) | 零先验不变式（§3.0），领域规则/真图不进训练 | claim 干净可迁移 | teacher 只吃 attention；真结构仅 post-hoc |
| 显式结构仅作 post-hoc oracle | 灌真图=作弊，见 [[inv_perm_strict_boundary]] | 保住"零先验涌现"claim | oracle 不进 gβ 输入/teacher/selection |
| 共享架构·专属权重 | 朴素可比，隔离模态混淆 | 每模态独立结论 | 各模态先过 headroom gate |

**贯穿红线**：**零先验不变式（§3.0）不可破**；headroom gate（**用 CDL-attention 序**的 own-order val Δ，非领域 oracle 序）是每个新粒度/新模态的硬 gate——不过就不建 gβ；batch-mean 是 load-bearing，N 变了先重测 reproducibility；NLL 不进 selection；一次一个变量。

---

## 5. 迭代计划（低风险优先）

> 重排依据：text 固定 block=4，可变 block 只为多模态。故 token 级 text（原 Phase 2）降为可选，多模态提到主线。

**Phase 1 — size-agnostic 地基（text-only 验证，低风险，复用现有 infra）**
1. 实现 §2.2 GNN 读出，**回归 gate**：在 text N=64 上复现 v1 98.44% + PG-only val ≈ 3.4678。不过不进——这是唯一有 ground-truth 对照的验证点，必须先钉死架构正确性。
2. size-agnostic 单测：喂 N∈{16,32,64,128} 合成图，验 node-置换等变 + 输出维度随 N 自适应。

**Phase 2 — 首个新模态（主线，中高风险）— 分子 vs 图像待定（见 §6）**
- **分子（旗舰候选，greenfield）**：atom-level 注意力抽取协议（新写 molecule 版 strict-N）；无监督选 head；**严格 CDL(attention) teacher**；先过 headroom gate（CDL-attention 序 vs 随机序 own-order NLL Δ 显著）才建 gβ；gβ+PG；**FCD 绝对 claim** + 化学键邻接做 post-hoc oracle 对齐。
- **图像（低风险候选，复用 image infra）**：patch 注意力 + 无监督选 head + **严格 CDL(attention)**；若 CDL(attention) 不出信号则如实记 clean negative（**不退回 locality 先验去救**，D_manh 仅作 post-hoc 对齐 oracle）。FID 仅相对比较。

**Phase 3 — 其余 explicit-oracle 模态（数独 / 代码 / 另一模态）**
6. 各自 attention 图 + **严格 CDL(attention)**；先过 headroom gate；各自 post-hoc oracle（CSP 序 / 拓扑序）。

**Phase 4 — 迁移与统一（高风险，选做）**
7. text-gβ → 分子迁移测（§3.4 档 2），验 unified-g(B)：一套 size-agnostic 权重能否跨模态的 attention 图。
8. 若正面 → 收敛单一通用 gβ。

**可选支线（非主线）**：token 级 / 分层 text（原 §2.3），仅当想进一步压 text NLL 时再启，不阻塞多模态。

**里程碑判据**：Phase 1 回归 gate 不过 ⇒ 停下修架构，不进 Phase 2；任一模态 headroom gate 不显著 ⇒ 记为 clean negative，不硬推 gβ。

---

## 6. 待开的关键问题（需用户定优先级）
- ~~主线目标~~：**已定 (2026-07-07)**——text 固定 block=4，可变 block 为多模态服务，主线是跨模态。
- **Phase 2 首模态：分子（旗舰，greenfield，需新写 atom attention 抽取 + 数据）vs 图像（低风险，复用 image infra）？** 分子科学价值更高（FCD 绝对 claim + 客观化学 oracle），图像启动更快。
- 图像 infra 当前可用性（记忆：迁移部署曾暂停）——若走图像需先确认。
- 各模态用 per-size gβ（朴素、每步可验）还是尽早上 size-conditioned 单 gβ（贵、但一套权重）。建议先 per-size。
