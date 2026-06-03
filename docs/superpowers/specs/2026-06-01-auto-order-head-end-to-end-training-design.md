# Auto-Order-Head 端到端训练架构 — selector → g_β readout → hook

日期: 2026-06-01
作者: lyuyuhuan + Claude
关联记忆: `br1_batch_readout_status.md`, `nr1_neural_readout_plan_status.md`,
  `quick_head_selector_line.md`, `quick_head_selector_crossmodal_design.md`,
  `cdl_evolution_clean_base_20260527.md`, `text_training_config.md`
关联/被取代 spec: `2026-05-30-quick-head-selector-design.md`(本 spec 取代其 C1–C4 主路径,
  selector 主排序改为 row-concentration;其余红线继承)

## 0. 核心边界(贯穿全文,必须先立)

> **CDL 只活在离线 pretrain(造 teacher)与 validation(当 ruler)。绝不进 hook。**
>
> - Pretrain / validation 阶段: `B → CDL → σ_T` 训练并验证 g_β。
> - Hook 训练阶段: `A_t^{head} → B_t^{head} → g_β → logits → σ_{t+1}`,**没有 CDL、没有 L2R τ、没有 NLL reward,g_β 先 frozen**。

这条边界是整个架构防"feature / objective mismatch"的总闸,任何实现都不得越界。

## 1. 背景与本线要解决的问题

BR-1 / NR-1 / quick-head-selector 已确立(见关联记忆):

- order 信号随训练涌现,且高度 head-specific(head-mean 会正负抵消);
- 主序 head 的 index **跨 run 漂移**(clean_base 偏 L0H0 族),**不能硬钉**;
- **盲取 |score| winner 会选到 anti-L2R 反向 head**;
- 旧 cheap score **C1–C4 裸排召回 winner 失败**(全 ladder R@1=0/40,最好的 C2 R@3=6/40);
- 真正可用的 cheap 指标是 **row-concentration(边相关)**;
- ⚠️ 旧 extraction 把 `[None]` sink 以 `none_weight=0.1`、按 model 坐标(不 remap)加进所有列,
  污染边图,把 row-concentration 的 winner 排名从干净的 rank1/winconc≈0.6 砸到 rank2/0.086。

本线把"选 head"与"读 order"彻底分层,给出端到端 auto-order-head 训练架构:

```
row-concentration 找候选 head
  → CDL 只离线验证 / 造 teacher
  → g_β 学 selected-head B → order
  → hook 阶段无 CDL,frozen g_β 输出下一步 order
```

## 2. 范围与红线

**做**: Phase 1(离线 head 选择 + g_β pretrain)、Phase 1.5(MLP generalization gate)、
Phase 2(training hook scaffold)、selector observe-only 监控。

**红线(继承 BR-1 / NR-1 / stability spec)**:
- 训练 / g_β 更新 / order generation 一律**不得**使用 ground-truth L2R label、NLL、reward、image distance;
- L2R τ 只作 **selector validation diagnostic**(证明 cheap score 召回真信号),不在线选 head、不进 hook;
- expensive readout 固定 **CDL-source-start `alpha_dep=0.5`**,与既有 teacher 口径一致;
- `extract_A_matrices` 的 `torch.randperm` **必须播种**(per-seed `torch.Generator`),否则跨 run B 方差污染。

## 3. Phase 1 — 离线 pretrain(在 5k checkpoint,用 CDL)

### 3.0 前置 gate: B0 full ladder 复核(进 g_β pretrain 前必须完成)

因 B0 改写了 head 稳定性叙事(§3.2),进入 selected-head dataset + g_β pretrain **之前**必须先跑:

- **B0 extraction full ladder**: 9 ckpt × 5 seed(与现有 OLD ladder 同口径,仅换 none→block0 提取);
- **OLD vs B0 对比**: (1) rowconc 分布(median / winner conc);(2) top-k pool precision;
  (3) best+ stability(跨 seed / 跨 step);(4) 晚期(50k–60k)winner 是否仍是 L0H0(验证"漂移=伪影")。
- **通过判据**: B0 全程 rowconc 健康(median 不塌)、top-k pool precision 高、best+ 跨 seed 稳定。
- 通过后方可进入 §3.3 selected-head dataset + g_β pretrain。

> **✅ VERDICT — PASS(2026-06-02 跑完整 9 ckpt × 5 seed = 45 jobs;`b0_ladder_gate.py`)**
>
> | step | B0 med C | poolP@5 | best+rec@5 | maxτ@5 | top1-conc head | | OLD med C |
> |---|---|---|---|---|---|---|---|
> | 0 | 0.002 | 0.00 | 0.00 | 0.04 | L1H6 (5/5) | | — |
> | 1000 | 0.000 | 0.00 | 0.00 | 0.04 | L0H1 (5/5) | | — |
> | 5000 | **0.046** | 0.80 | 1.00 | 0.98 | **L0H0 (5/5)** | | 0.001 |
> | 10000 | 0.056 | 1.00 | 1.00 | 1.00 | **L0H0 (5/5)** | | 0.000 |
> | 20000 | 0.064 | 1.00 | 1.00 | 1.00 | **L0H0 (5/5)** | | 0.000 |
> | 30000 | 0.053 | 0.80 | 1.00 | 1.00 | **L0H0 (5/5)** | | 0.000 |
> | 40000 | 0.055 | 1.00 | 1.00 | 1.00 | **L0H0 (5/5)** | | 0.000 |
> | 50000 | 0.057 | 0.84 | 1.00 | 1.00 | **L0H0 (5/5)** | | 0.000 |
> | 60000 | 0.057 | 1.00 | 1.00 | 1.00 | **L0H0 (5/5)** | | 0.000 |
>
> **Gate(B0, step≥5000)**: min median C=**0.046** (>0.02 ✓)、mean poolP@5=**0.92**、mean best+recall@5=**1.00** → **PASS**。
>
> 三点确认:**(1)** B0 救活 row-conc 健康指标 — OLD 全步 med C≈0.000(none-token 0.1-sink 污染塌死),B0 step≥5k 稳在 0.046–0.064;**(2)** top-k pool 干净 — poolP@5 平均 0.92、best+ recall 完美 1.00;**(3)** **winner 锁定 L0H0,零漂移** — row-conc 头号 head 从 5k 到 60k × 全 5 seed 都是 L0H0(45/45 一致),且与 §3.2 的 best+ τ winner 同为 L0H0(row-conc 尖锐 ∧ best+ order head 双重一致)→ §3.2/§6 的"OLD 晚期漂移 L1H4 = 提取伪影"**证实**。warmup-up 之外(step 0/1000)尚未学到序属预期(C≈0、best+rec=0)。
> 注:OLD ladder 仅 31/45 有效点(早期 ckpt 缺 graph dump),不影响其作污染对照的结论(所有有效步 med C≈0)。
> **§3.0 前置 gate 通过 → 解锁 §3.3 g_β pretrain。**

### 3.1 head 选择: row-concentration → top-k → CDL validation → best+

对每个 head `(l,h)` 取 physical-frame block 图 `A`(batch-mean),令 `B = Aᵀ`,diag 置 0。

**主排序 = row-concentration `C_row`**(无符号、几何无关、dead-row→0):
```
p_ij = B_ij / (Σ_{k≠i} B_ik + ε)
C_row = mean_i [ 1 − ( −Σ_{j≠i} p_ij log(p_ij+ε) ) / log(N−1) ]   ;  dead row(行和≈0) → 贡献 0
```

**简要证明(为何 row-concentration ≈ order head)**:
一个学到接近确定 reveal 序的 head,其转移图每个 block 强烈指向唯一后继 → 每行质量集中在少数列
→ 行负熵(归一化)高 → `C_row` 高;反之位置型 / 弥散 head 把注意力摊到多 block → 行接近均匀
→ `C_row` 低。该判据只依赖"行是否尖锐",**不依赖坐标几何**,故跨模态可移植(text 与 image 通用)。
它不预设方向(L2R / R2L),只回答"这个 head 是否携带锐利的序"。

**流程**: `C_row` 取 top-k(k=3–5)候选 → 仅对这 k 个 head 跑 CDL readout(`generate_teacher_label`,
source-start,α=0.5)得真实 σ → `τ_vs_L2R` 符号给**方向**、挑 **best+**(argmax 正 τ)作 order provider。
**全流程唯一跑 expensive CDL 的地方,且只 3–5 个 head。**

- 旧 **C1 / C3 / C4 移出默认主路径**,仅作 report diagnostics(C1=text-only readiness-pos corr;
  C3=readiness spread;C4=asymmetry);**C2(signed flow-drift)** 作可选方向辅助 / fallback。代码保留,不进 selector 决策。
- winner 定义全程统一为 **best+(argmax 正 τ)**,不再用 `argmax|τ|`。

**selector 验证判据(主判据改为 top-k pool quality,不再以单一 winner rank 为主)**:
B0 下 top-k 里常有多个 head 同时 \|τ\|≈0.9–1.0,"exact argmax 是否排第一"噪声大、非失败信号。故:

| 指标 | 定义 | 角色 |
|---|---|---|
| **pool precision@k** | top-k 中 \|τ\|≥0.9(或 τ≥0.8)的 head 比例 | **主判据** |
| **best+ recall@k** | top-k 是否含任一 strong positive head(τ≥0.8) | **主判据** |
| best_abs recall@k | top-k 是否含任一 strong absolute-order head(\|τ\|≥0.9) | 辅 |
| max τ in top-k | top-k 里最强 head 的 τ | 辅 |
| exact argmax rank | 单一 argmax-τ winner 排第几 | **降为附表,不做主判据** |

### 3.2 extraction: none→block0(★ CANONICAL — full ladder 复核 ✅ 通过 2026-06-02,见 §3.0 VERDICT)

将旧 `A += none_block·0.1`(model 坐标、不 remap)换成:**把 `[None]`(index 0)折进物理 block 0**
(query 行 + key 列段平均,无 magic 权重)。

**采纳为 provisional canonical extraction**,理由(B0 实测,2 ckpt × 1 seed,2026-06-01):

| ckpt | 方案 | rowconc med | winner conc | 晚期(60k)winner rank | top-5 \|τ\| |
|---|---|---|---|---|---|
| 5000 | OLD(0.1-sink) | 0.001(退化) | 0.092 | — | 0.83/0.99/0.48/0.94/0.08 |
| 5000 | **B0** | 0.106 | 0.398 | — | **0.90/1.00/0.94/0.94** |
| 60000 | OLD | 0.000(全死) | 0.001 | **12/32** | 全塌 |
| 60000 | **B0** | 0.166 | 0.829 | **1/32** | **1.00/1.00/0.94/0.94/0.94** |

- OLD 明显被 none-token 污染:rowconc median≈0,晚期(60k)全死;**不可继续作主口径**。
- B0 恢复健康分布:median / winner conc / top-k order-head precision 全回来,60k winner rank=1。

> **措辞(✅ sealed 2026-06-02 — full ladder 通过)**: B0 none→block0 **adopted as the canonical
> extraction path**,because it fixes the OLD none-token contamination and restores healthy
> row-concentration distributions. **Full 9-ckpt × 5-seed ladder validation PASSED**(§3.0 VERDICT:
> min median C=0.046、poolP@5=0.92、best+recall=1.00、winner L0H0 45/45 稳定)。故 `offline scan =
> g_β pretrain dataset = hook input extraction` **三处共用同一路径**(防 `B_train ≠ B_hook` mismatch)。

⚠️ B0 还**改变了 winner 结论**:OLD 下 60k winner=L1H4,B0 下 60k winner=**L0H0**(τ=1.00,rowconc rank1),
且 L0H0 在 5k 也是 concentration #1 → 之前"漂移到 L1H4"可能是 OLD 伪影(见 §6)。这正是必须 full ladder
复核的原因——**B0 不只是小修,它改写了 head 稳定性叙事**。该复核是 Phase 1 的前置 gate(§3.0)。

### 3.3 selected-head dataset builder + g_β 训练(复用 NR-1)

在 5k、固定 best+ head 上采多 batch:
```
B_batch^{best+} → CDL → σ_T          (teacher label,CDL)
g_β(B_batch^{best+}) → logits/order   (label = σ_T)
```
**复用 NR-1 现有 readout**(`neural_readout/` 的 `graph_transformer_readout.py` / `train_nr1.py` /
`loss.py` / `eval_metrics.py`,及 `batch_readout/` 的 `model.py` / `pl_sampling.py` / `train_offline.py`)。
**新增仅两件**: selected-head dataset builder + Phase-1.5 generalization gate。不新写模型结构。

## 4. Phase 1.5 — MLP generalization gate(★接 hook 前必须过,CDL 仅作 ruler)

只看 train loss 不够。必须验证 g_β **泛化**而非记住训练 batch:

- **泛化 A(新数据)**: held-out batch `B_val^{best+} → CDL → σ_T^val`;比 `g_β(B_val)=σ̂` vs `σ_T^val`。
- **泛化 B(跨 step,optional 但强烈建议)**: `B_10k^{best+} → CDL → σ_T^{10k}`;
  测 5k 训好的 g_β 对训练漂移后的 B 是否还拟合(cross-step generalization)。

指标(复用 `neural_readout/eval_metrics.py`): **Kendall τ / pairwise acc / Spearman / top-k(first-k) match /
PL NLL(仅评估,不入 selection)/ teacher diversity**。

descriptive target(报告,不一开始硬杀):
```
τ_val ≥ 0.6 ~ 0.7 ,  pairwise_acc ≥ 0.8 ,  teacher diversity 不塌 ,  argsort/sample order 不退化
```
**过了才有资格接 Task-15。** 不过 = clean negative(g_β 没学到可移植 readout,hook 必崩),如实报告并停在此。

> **✅ VERDICT — PASS(text, 2026-06-03;`scripts/run_phase33_gbeta.py --M 2000 --batch-size 8`)**
> head=L0H0(best+,B0 canonical),g_β=nodewise+pairwise,5k 训(train/val/test=1600/200/200)。
>
> | split | g_β τ | g_β pw | L2R-prior τ | non-L2R[n] g_β τ |
> |---|---|---|---|---|
> | 5k/val | **0.849** | 0.941 | 0.760 | 0.648 [76] |
> | 5k/test | **0.836** | 0.939 | 0.780 | 0.640 [74] |
> | 10k/cross | 0.927 | 0.964 | 0.966 | 0.635 [52] |
> | 20k/cross | 0.930 | 0.965 | 0.990 | 0.402 [9] |
>
> Gate(τ≥0.6, pairwise≥0.8): same-step ✓ / cross-step ✓ → **PASS**。
>
> **g_β 学到了 B-dependent readout,不是记常数 L2R**:(1) 在操作点 5k **反超 hardcoded-L2R prior**(+0.06~0.09);(2) 关键证据 = **non-L2R 子集(teacher≠identity,prior 必败处)g_β τ=0.64**,说明它读出了真实偏离 L2R 的序。**诚实边界**:文本内在序≈L2R,随训练头结晶成纯 L2R(non-L2R 样本 5k 75→20k 9),故 10k/20k 上 prior 逼近天花板、g_β 边际价值收窄(仍 τ≈0.93、non-L2R 子集仍正)——learned g_β 对文本的**增量价值有限但非零**,其真正 context-dependent 价值待 **image**(locality 序随内容变)验证。⚠️ smoke(M=80)曾显示 g_β<prior/non-L2R τ<0,**纯 underpower 伪影**,1600 样本下反转。**Phase 1.5 gate 通过 → 解锁 Task-15 hook(§5)。**

## 5. Phase 2 — training hook(无 CDL,g_β frozen)

每 step:
```
σ_t → forward/backward → A_t^{best+} → B_t^{best+} → g_β(B_t) → logits → σ_{t+1}
```
- CDL 不参与、L2R τ 不参与、NLL 不更新 g_β;
- g_β 参数 frozen,attention 仅作 input;
- 复用 `batch_readout/integration_hook.py` 的 FrozenBetaHook wrapper。

## 6. selector / 迟滞重选 —— v1 **observe-only**(rationale: safety/monitoring,非 confirmed drift)

> OLD extraction suggested late-stage head drift(L0H6→L1H4),but B0 indicates **part of this drift may be
> an extraction artifact**(B0 下 L0H0 在 5k 与 60k 都是头号 order head)。Therefore hysteresis is retained
> as a **safety / monitoring** mechanism,**not yet as a required active-switching mechanism**。

g_β 是 single-head trained;若 active head 真切到别的 head,会喂 g_β 分布外的 B → 新 mismatch。故:

- **v1**: `active head = best+@5k` 固定。selector 每 K 步重算 row-concentration top-k,
  **只监控**(current head 是否衰减 / top-k 是否换 / challenger 是否连续出现),**不实际切换**。
- 真正切换留 **v2**: 必须对新 head **retrain/finetune g_β** 或升级 **multi-head g_β**。
- "漂移是否真实"由 §3.0 B0 full ladder 复核回答:**✅ 已回答(2026-06-02)= 漂移是 OLD 提取伪影**
  (B0 下 L0H0 在 5k–60k × 5 seed 全程头号 order head,45/45 零切换)。故 v1 迟滞确定为 observe-only —
  L0H0 既然稳定,本就无需 active switching;迟滞仅留作 safety 监控。
- K 由 100-step benchmark gate 定(overhead<30%→更密,>50%→拉大 K);本轮 selector 只读不写,overhead 容忍度高。

## 7. 执行顺序

| 阶段 | 做什么 | 用 CDL? |
|---|---|---|
| **0** | **B0 full ladder 复核(9ckpt×5seed,§3.0 前置 gate)** | ✅ scan |
| 1 | 5k 选 best+ head(row-conc top-k → CDL validation) | ✅ scan |
| 2 | 造 selected-head batch 数据集 | ✅ teacher |
| 3 | 训 g_β(复用 NR-1) | ✅ label |
| 4 | held-out batch validation(泛化 A) | ✅ 评估 |
| 5 | cross-step validation(10k+,泛化 B) | ✅ 评估 |
| 6 | hook 进训练 | ❌ |
| 7 | selector / 迟滞(observe-only) | ❌ |

## 8. 组件与文件

| 用途 | 文件 | 状态 |
|---|---|---|
| row-concentration 主选择器 + 选 head 规则 | `block_lo_arm_order_network/quick_head_selector.py` | 改: 加 `row_concentration()` 为主,C1/C3/C4 降 diagnostics |
| none→block0 canonical extraction | `block_lo_arm_order_network/per_head_order_scan.py` | 改(B0-pass 后): 共用于 scan/dataset/hook |
| selected-head dataset builder | (新增) | 新写 |
| g_β readout / PL sample / metrics | `neural_readout/*`, `batch_readout/{model,pl_sampling,loss,eval_metrics}.py` | 复用 |
| MLP generalization gate | (新增,基于 `eval_metrics.py`) | 新写 |
| FrozenBetaHook | `batch_readout/integration_hook.py` | 复用/包装 |
| expensive CDL ground truth(ladder) | `batch_readout/logs/per_head_scan/*.json`(9 ckpt×5 seed,已跑完) | 现成 |

## 9. 非目标(YAGNI)

- v1 不做 active head 切换(observe-only);不做 multi-head g_β;不做 g_β 在线更新。
- 不引入 first-eigenvector / 谱方法;不把 §4 descriptive target 写成 hard gate;不引入 NLL/L2R 进任何训练或 selection。
- B0 未过前不落地 Phase 2 hook;不物理删除 C1–C4(仅降级)。
- 不复活原 alt_from0 ckpt(只用现存 `alt_from0_random/ckpt_step5000.pt`)。

## 10. 待定 / 依赖

- **B0 2-ckpt 结果已出(2026-06-01)**: none→block0 救活 rowconc 分布、60k winner rank=1,采纳为 provisional
  canonical(§3.2)。**下一步 = §3.0 B0 full ladder(9ckpt×5seed)复核**,是进 g_β pretrain 的前置 gate。
- **M 统一**: scan(ladder 用 M=100)与 g_β dataset(需大量 batch)口径需在实现期统一并记录。
- warmup 长度沿用 5k;K / 监控阈值在搭训练架子时按 benchmark 定。
- ⚠️ B0 提取热点: per-chunk naive einsum 慢,full ladder 跑前需把 `b0` 提取向量化(einsum `optimize=True`
  已用;ladder 规模需进一步 batch 化,参照 OLD ladder 的向量化经验 18min→5:36)。
