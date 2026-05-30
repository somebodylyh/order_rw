# Quick Head Selector — cheap per-head order-score 作为 order-head pre-selector

日期: 2026-05-30
作者: lyuyuhuan + Claude
关联记忆: `br1_batch_readout_status.md`, `cdl_evolution_clean_base_20260527.md`, `text_training_config.md`
关联 spec: `2026-05-29-l0h5-cross-ckpt-seed-stability-design.md`(本线 expensive ground truth 来源)

## 1. 背景与要回答的问题

BR-1 轻量 hook 设计阶段已确立的事实(见 `br1_batch_readout_status.md`):
- order 信号会随训练涌现;
- 承载 order 信号的 head 高度 head-specific,head-mean 会把正/反向 head 抵消掉;
- 主序 head 的 index **跨 training run 会漂移**(clean_base 偏 L0H0 族、alt_from0_random 偏 L0H5),**不能硬钉任何 index**;
- 最强 `|τ|` head 经常是 anti-L2R,**盲取 `|score|` winner 会选到反向 head → 输出 R2L order**。

因此本线**不**回答"哪个 head 最强",而回答:

> 能不能用一个很便宜、无 rollout 的 head score,在 warmup 后快速把 32 个 head 缩到 top-3/top-5 candidate,再只对这几个做 CDL 验证 / hook,而不必每次跑 full CDL rollout 全扫?

定位明确:

```
A^{l,h}  →  cheap score(l,h)  →  top-k candidate heads  →  CDL validation  →  hook
```

cheap score 是 **head pre-selector**,**不是**最终 oracle。就算它不完美,只要能稳定把搜索空间从 32 head 缩到 3–5 head 就有价值。

## 2. 范围与边界

**做**:
- 一个 cheap、无 rollout 的 per-head order-score 函数(O(N²));
- 一套"禁盲取 `|score|`"的选 head 规则,分别输出 `best+` / `best−` / pool;
- 用经验回归证明 cheap top-k 能召回 full CDL scan 的 winner,重点验证**跨 run 漂移召回**。

**不做(YAGNI)**:
- 不接 Task-15 hook(只把接口留干净);
- 不训练新模型,不动 Phase-1/Phase-3;
- 不加 first-eigenvector 等高解释成本的 score;
- 不引入新 readout 变体。

**红线(继承 BR-1 / NR-1 / stability spec,L2R 边界已收窄)**:

L2R 在本线只是**显微镜,不是老师**——用来定位/验证信号,不当训练标签。三类东西必须分清:
1. **信号诊断标签 = L2R τ**:只用于分析"哪个 head 含 order 信号、方向是 L2R 还是 anti-L2R",属 diagnostic / measurement / selector validation。**允许**。
2. **MLP/readout teacher = attention-derived CDL pseudo-order**(`generate_teacher_label`,source-start):MLP 学 `B → σ_CDL`,**不是** `B → [0,1,…,63]`。
3. **训练时用的 order = `g_β(B) → logits → sample/argsort → σ`**:不把 L2R 直接塞进去。

据此红线为:
- **训练 / MLP g_β / order generation / NLL / reward / image selection 一律不得使用 ground-truth L2R label 或 NLL**;
- selector **validation** 可用 expensive CDL τ 与 L2R 作 diagnostic ground truth(证明 cheap score 能召回真信号);
- ⚠️ 诚实声明:因 cheap score 的主 score 及符号是用 `tau_vs_l2r` 回归校准出来的,本 selector 属 **diagnostic-calibrated head selector**——实验阶段 OK;若写成正式方法,正式路径是 `cheap score top-k → CDL validation → select head`(L2R 只证明 cheap score 能召回真信号,**不在线选 head**),正式 hook 时只使用选出的 head,不把 L2R label 传入 order generation。
- expensive readout 固定 CDL-source-start(`alpha_dep=0.5`),与既有 teacher 口径一致;
- `extract_A_matrices` 的 `torch.randperm` **必须播种**(per-seed `torch.Generator`),否则跨 run B 方差污染(见 `cdl_evolution_clean_base_20260527.md`)。

**前置依赖**:expensive ground truth = stability spec 跑出的 `batch_readout/logs/per_head_scan/ckpt{STEP}_seed{S}.json`(clean_base ladder 9 ckpt × 5 seed 的 per-head `tau_vs_l2r`)。**该批 full CDL scan 必须先跑完落盘**;若未跑完,这是本线的 step 0。

## 3. 候选 cheap score(全部 O(N²)、无 rollout)

对每个 head `(l,h)`,取 physical-frame block 图 `A`(64×64,来自 batch-mean),令 `B = Aᵀ`,diag 置 0。逐节点 `v` 定义 readiness `r(v) = out(v) − α·in(v)`,其中 `out(v)=Σ_j B[v,j]`(B 行和)、`in(v)=Σ_j B[j,v]`(B 列和)、`α=0.5`(与 CDL source-start 同口径)。

| score | 公式 | 用途 |
|---|---|---|
| **C1 signed readiness-position corr** | `corr(r(v), phys_index)` | **signed ranking** 候选(主) |
| **C2 signed flow drift** | 每行 `Σ_j B[v,j]·(j−v) / Σ_j B[v,j]`,再对 v 平均 | **signed ranking** 候选 |
| **C3 readiness spread** | `std(r(v))` | **dead-head filter**(过滤 uniform/无结构 head) |
| **C4 asymmetry strength** | `‖B − Bᵀ‖_F / (‖B‖_F + ε)` | **directionality filter**(过滤对称 / local-only head) |

分工:
- **C1 / C2**:带符号,做主排序(谁正向/反向、强度多少);
- **C3**:过滤 dead/uniform head(readiness 无展开);
- **C4**:过滤对称 / local-only head(无方向性 → 不可能编码顺序方向)。

### 3.1 符号校准(关键 —— C1/C2 的正负语义不预设)

⚠️ **绝不预设 C1/C2 的符号方向**。readiness `r(v)=out(v)−α·in(v)` 对 L2R 的几何可能是"低 index 高、高 index 低",因此 `corr(r, phys_index)` 对 L2R-aligned head **可能是负相关**;又因为用的是 `B=Aᵀ`,C2 的"指向前/后"直觉也可能与表面相反。任何"C1>0 即 L2R-aligned"的预设都可能把符号搞反、导致 `best+`/`best−` 互换。

因此 C1/C2 **同时报 raw 与 sign-calibrated 两版**。校准规则:在 calibration split 上选符号约定,使 **positive cheap score 与 expensive `tau_vs_l2r` 正相关**(即 `s_cal = sign(spearman(s_raw, expensive_τ)) · s_raw`)。selector **只用 sign-calibrated 版本**;§5 经验回归既挑主 score(C1 vs C2)也定其符号,均不预设。

### 3.2 过滤阈值

`dead_thresh` / `sym_thresh` **数据驱动校准**:取 step0(随机初始化,无 order 信号)全 head 的 C3 / C4 分布,阈值设在该 null 分布的高分位——即"显著高于随机初始化水平"才不被过滤。**不写死**:default 95th percentile,fallback 90th percentile,并在 `validate_quick_selector.py` 里报告对 90/95 阈值的 sensitivity(避免 step0 偶发高 asymmetry 的 random head 误过滤掉早期弱信号)。

⚠️ 注:C1 依赖 physical index,带一点 L2R prior 味道——但它**只是 selector diagnostic**(见 §2 红线收窄),不进入任何 readout/训练,且其方向性会被 C4(无监督)交叉印证。哪个(C1 还是 C2)当主排序 score,由 §5 经验回归决定,不预设。

## 4. 选 head 规则(继承硬约束:禁盲取 |score| winner)

```
signed_score = sign-calibrated proxy among {C1, C2}   # 由 §5 回归选出主 score 及其符号(§3.1)
mask = (C3 > dead_thresh) AND (C4 > sym_thresh)       # 过滤 dead / symmetric head
best_positive   = argmax_{masked} signed_score        # +score 已校准为与 expensive τ 同号 → L2R-aligned,作 order provider
best_negative   = argmin_{masked} signed_score        # anti-L2R
pool            = {best_positive} ∪ top2(best_negative)  # 带 sign 标签
```

**绝不** `argmax |score|`(否则 anti 负头夺冠 → R2L 反序)。给 Task-15 两个模式(本轮不接,仅定接口):
1. **single-positive**:只用 `best_positive`;
2. **topology-pool**:`{best_positive, top2 best_negative}`,每个带 sign 标签。

## 5. 验证策略(多 ckpt × 多 step;经验驱动选主 score)

ground truth:stability spec 的 expensive per-head `tau_vs_l2r`(pool over 5 sampling seed)。cheap score 用**同一条 extraction 路径**(`extract_per_head_and_heavy_A` 的 batch-mean A_lh)算,保证口径一致。

1. **经验选主 score**:clean_base ladder 9 个 step,每 head 算 C1/C2;对每个 step 及全 pool 报 `Spearman(C1, expensive τ)` 与 `Spearman(C2, expensive τ)`,取更高者当主排序 score。
2. **Recall**:每个 ckpt,cheap top-k 是否含 expensive `best+` 及 `best−`。报 **Recall@1 / @3 / @5**,分别对 `best+` 和 `best−`。
3. **跨 run 漂移召回(重点)**:在 step5000 这个点,clean_base vs `alt_from0_random`(⚠️ 数据现实:`alt_from0_random` **只有 ckpt_step5000.pt 一个**)对比——cheap selector 是否在两个 run 各自召回各自的 winner(证明"跟漂移"而非过拟合某 run 的 index)。
4. **涌现 null 对照**:step0 / 1k,cheap score 与 expensive τ 应都贴噪声水平、不选出强 head(cheap 不能凭空造信号)。
5. **cost ratio**:cheap 全 head 扫一遍 vs full CDL scan 的 wall-clock,报实测加速比。

### 5.1 成功判据 —— desired criteria,不是 hard gate

| 指标 | 目标(desired) |
|---|---|
| Spearman ρ(主 cheap score, expensive τ) | 希望 ≥0.7;≥0.5 也有价值 |
| Recall@3(best+,信号涌现后 ≥20k) | 希望接近 100% |
| Recall@5 | 应更高,作保底 |
| cross-run 5k recall | **必须报告**(核心证据) |
| cost ratio | 报实测,目标 ≥10× |

**最低有用标准(minimum useful outcome)**:cheap selector 的 **top-3 / top-5 能稳定覆盖 expensive scan 的 best+ / best−**,即足以用于 warmup head preselection——后续只对 top-k 运行 CDL 验证即可。不要求 cheap score 一步选中 winner,只要求把搜索空间从 32 head 缩到 3–5 head。

不同 ckpt 早晚期信号强弱不同,硬阈值容易误杀;故上述全部为描述性目标,达不到也如实报(本身是有价值的 negative,说明 cheap proxy 不足以替代 CDL)。

## 6. 组件与文件(单一职责、可单测)

| 文件 | 职责 |
|---|---|
| `block_lo_arm_order_network/quick_head_selector.py` | cheap score + 选 head 规则(纯逻辑,无训练) |
| `block_lo_arm_order_network/validate_quick_selector.py` | 读 expensive JSON + 同口径算 cheap score → 回归 / recall / cost 报告 |
| `block_lo_arm_order_network/tests/test_quick_head_selector.py` | TDD 单测 |

### 6.1 核心函数签名
```python
def cheap_head_scores(A_lh, alpha_dep=0.5):
    """A_lh: (n_chunks, L, H, N, N) 或已 batch-mean 的 (L, H, N, N).
    返回 {"C1": (L,H), "C2": (L,H), "C3": (L,H), "C4": (L,H)} float arrays."""

def select_heads(scores, rule="pool", k=2, dead_thresh=..., sym_thresh=...,
                 rank_score="C1"):
    """rule ∈ {"single", "pool"}.
    返回 [(layer, head, sign), ...];sign ∈ {+1, -1}.
    绝不按 |score| 排;best+ = argmax(signed), best- = argmin(signed)."""
```

接口契约(留给 hook,本轮不接):`select_heads` 返回 `(layer, head, sign)`;Task-15 侧只需 `A_lh[:, l, h]` 切片喂 `FrozenBetaHook.step`。

### 6.2 复用的现成积木
| 用途 | 文件 | 函数 |
|---|---|---|
| per-(layer,head) attention → physical-frame A_lh | `per_head_order_scan.py` | `extract_per_head_and_heavy_A` |
| expensive CDL τ ground truth | `batch_readout/logs/per_head_scan/*.json` | (stability spec 产出) |
| Spearman / Kendall | scipy | `spearmanr` / `kendalltau` |

## 7. TDD 测试要点

- **C1 raw 符号一致性**:构造 out-degree 单调图 + 其反序图,确认 raw C1 在两者上符号相反、量级合理(不预设哪个为正——只测"方向可分")。
- **C2 raw 符号一致性**:净位移向前 vs 向后的 B,raw C2 符号相反。
- **sign calibration**:给定一组 (raw_score, expensive_τ),`s_cal = sign(spearman(raw,τ))·raw` 后,`spearman(s_cal, τ) ≥ 0`;构造一个 raw 与 τ 负相关的 case,确认校准后翻正。
- **C3 过滤**:均匀 / 全零 head → C3 ≈ 0,被 dead-head filter 过滤;C1/C2 在退化图上不崩。
- **C4 过滤**:对称 B(`B = Bᵀ`)→ C4 ≈ 0,被 directionality filter 过滤;强方向图 → C4 高。
- **select_heads**:在含一正一负强头的合成 scores 上,返回正确 `best+` / `best−` 且 sign 正确;验证**绝不**返回 `argmax|score|`(构造一个 `|负头| > |正头|` 的 case,确认 best+ 仍是正头)。
- **回归 / recall 逻辑**:在小型合成 (head × ckpt) 矩阵上给出已知 Spearman / Recall@k 答案。

## 8. 非目标(YAGNI)

- 不接 hook、不训练、不动 Phase-1/3。
- 不加 first-eigenvector / 谱方法。
- 不重定义 §5.1 阈值为 hard gate;不引入 NLL。
- 不复活原 alt_from0 ckpt(只用现存的 `alt_from0_random/ckpt_step5000.pt`)。
