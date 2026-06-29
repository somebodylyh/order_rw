# 08 — Boss Update Summary (Wednesday Briefing)

Date: 2026-06-17

---

## 一句话结论

**严格 65-node label-free 协议下，selected early heads 可以从 unanchored None 起点恢复完整 physical L2R block order。这是 head-specific 的现象，不是 all-head universal。Destroyed controls 近随机，排除了 artifact 解释。**

---

## 1. 为什么收紧协议

之前担心两件事：
- **[None] → physical0 泄露起点**：旧 B0/B1 协议把 [None] fold 到 physical block0，CDL 从 block0 出发天然选对起点 → 不能叫 "unanchored discovery"。
- **inv_perm 在构图时用**：旧 65-node search 在 extraction 阶段就用 inv_perm 把 model block 映射成 physical block → 等于提前知道了 physical→model 对应关系。

所以我们把协议收紧成 strict 65-node label-free：
- None 是独立 BOS 节点（node 0），不并入任何 content block。
- 64 个 content nodes 是真实的 physical blocks。
- Graph 在 model-frame 构造（不用 inv_perm）。
- CDL rollout 从 None 出发，第一块纯由 graph edges 决定。
- inv_perm 只在最后 posthoc 翻译 sigma_model → sigma_phys 用。

## 2. 当前最强结果

**Collaborator checkpoint @50k, strict label-free 65-node, M=20:**

| Head | Method | tau | first | phys0_rank | p4 | p8 | destroyed \|τ\| |
|------|--------|-----|-------|------------|----|----|-----------------|
| L0H1 | L-only | 1.000 | 0 | 0 | 4 | 8 | 0.050 |
| L0H2 | L-only | 1.000 | 0 | 0 | 4 | 8 | 0.053 |
| L0H3 | L-only | 1.000 | 0 | 0 | 4 | 8 | 0.050 |
| L0H4 | L-only | 1.000 | 0 | 0 | 4 | 8 | 0.055 |

L0H7 对照：tau=0.292, first=45, phys0_rank=20 → **FAIL**。

## 3. 关于 inv_perm：构图可以用，readout 不能用

**inv_perm 的合法边界**：
- **构图阶段** ✅：可以用 inv_perm 把 graph 建在 physical 坐标。CDL 只看 edge weights，不看 node labels，坐标系不影响 readout。
- **CDL rollout** ❌：不能用 inv_perm 引导决策。CDL 必须纯靠 graph edge weights（C-D+L scores）选下一个 block。
- **Posthoc 评分** ✅：只用 inv_perm 把 sigma 翻译到 physical 坐标做 evaluation。

**所以 oracle-remapped（physical-frame 构图）和 strict label-free（model-frame 构图+posthoc）都是有效协议。** 结果一致是因为 CDL 对 coordinate frame 不敏感（permutation-equivariant），不是因为 leak。

## 4. L0H7 fail 的意义

L0H7 不是 bug，是 feature。它说明：
- Discovery 不是 trivial / all-head universal / graph-statistic artifact。
- L0H7 的 attention 结构做不到严格 L2R recovery → 证明我们的 gate 有区分度。
- 所以 claim 是 "selected early heads recover L2R"，不是 "all heads recover L2R"。

## 5. 多 checkpoint 验证

Clean base checkpoint ladder（step 0–60000, 9 步）全部跑过（已有 oracle-remapped 数据，strict LF spotcheck 确认 10k/60k 等价，full sweep 已完成）：

| Step | strong_pass | 说明 |
|------|------------|------|
| 0 | 0 | 未训练，无信号 |
| 1000 | 0 | 信号未出现 |
| **5000** | **10** | **首次出现 strong_pass** |
| 10000–60000 | 10–16 | 稳定存在 |

从 5k 开始有 strong_pass，持续到 60k。信号不是单 checkpoint 偶然。

## 6. Extraction frame 的影响

| Head | B1 predictor + content-only | Loss-aligned AR + None-sep |
|------|---------------------------|---------------------------|
| L0H1 | 0.655 (anchor=6) | **1.000** (anchor=0) |
| L0H2 | 0.655 (anchor=6) | **1.000** (anchor=0) |
| L0H3 | 1.000 | 1.000 |
| L0H4 | 0.655 (anchor=6) | **1.000** (anchor=0) |

B1 predictor 帧去掉 [None] 后，L0H1/H2/H4 失去 anchor（循环序完美但起点错位）。Loss-aligned AR 帧保留 [None] 为独立 BOS 节点后，[None]→content 边提供了正确的 anchor。**Extraction frame 的选择比 inv_perm 用法更重要。**

## 7. 还需要什么

| 项目 | 状态 | 优先级 |
|------|------|--------|
| Multi-seed training 验证 | 未做 | 中 |
| 317M 模型 65-node | 未做 | 低 |
| Image 模型 65-node | 未做 | 未来 |
| M=20 strict LF on clean_base | 未做 | 低（M=8 equivalence 已确认） |

当前单 training seed（clean_base seed42 + collaborator seed?）是最大缺口。但 cross-run 验证（两个独立训练 run 都有 strong_pass heads）已经提供了初步泛化证据。

## 8. 周三应该怎么讲

**主结论**：

> 我们把 block-level discovery 协议收紧为 strict 65-node None-separated label-free graph。在这个严格协议下，部分早层 attention head 可以从 unanchored None 起点自动恢复完整 physical L2R block order。这不是 inv_perm 泄漏（strict label-free 和 oracle-remapped 结果完全一致，由 permutation-equivariance 解释），也不是 [None]→physical0 的人工锚定（None 是独立 BOS 节点，第一块由 graph rollout 自动选择）。这个 discovery 是 head-specific 的（L0H7 失败，L0H1–L0H4 通过），在 5k–60k 步之间稳定存在，且跨两个独立训练 run 可复现。Destroyed controls 接近随机，排除了 artifact 解释。

**不要过度 claim**：
- 不要说 "所有 head 都发现 L2R" — 只有 7.8% 通过 gate。
- 不要说 "frozen hook 已经用了 strict 65-node teacher" — hook 用的是 B0 canonical。
- 不要说 "同一个 head（如 L0H3）在所有模型上都是 best" — head 门牌号会漂移。

**一句话 takeaway**：

> Strict 65-node label-free block-order discovery is real, head-specific, and stable — the mechanism evidence just got a lot stronger.
