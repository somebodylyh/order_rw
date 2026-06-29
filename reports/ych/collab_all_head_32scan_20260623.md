# 合作者 all_head_attn_dataset 32-head 扫描总结

**日期**: 2026-06-23
**数据**: `/home/admin/ych/nanogpt-learned-order/data/attn/all_head_attn_dataset/`
**模型**: 4L×8H, d=384, Random-order 训练, wikitext-103, step 10k→15k

---

## 数据集结构

| 维度 | 值 |
|------|-----|
| Records | 313 |
| 每 record | attention [64×64] × 32 heads + block_orders [256×64] |
| probe_samples_per_record | **256**（256 次 forward 平均成 1 条 B） |
| 总大小 | 84MB, 5 shard |

## 和我们的提取差异

| | 我们 (mode="old" default) | 合作者数据集 |
|---|---|---|
| B 所在 frame | **physical frame**（提取时 inv_perm remap） | **model frame**（无 remap） |
| 每 record 样本数 | `batch_size` 条，同 frame 下平均 | 256 条，**不同** permutation 下平均 |
| per-record τ | 可达 ±1.0 | 最高 0.29 |

## 我们的提取流程（对照）

```
attn → inv_perm[model_blocks] remap → physical frame block-mean → B(phys) → CDL → τ vs L2R
                         ↑ _attn_to_A_block_vec 第 69 行
```

mode="model" 时（无 remap）：
```
attn → model frame block-mean → B(model) → CDL → σ(model) → inv_perm[σ] → τ vs L2R
                                                              ↑ posthoc remap (train_clean_aogpt.py:1332)
```

两者都依赖 `inv_perm`（来自 ckpt 的 `clean_perm.inv_perm_model_to_phys`）来做 model→physical 的翻译。合作者数据集没有存储这个映射。

## 在这个数据集上跑的结果

### Raw B（无任何 remap）
- max τ(C-D+L) = **+0.18** (L2H3)
- 大部分 head 在 ±0.013 附近（退化）
- B 矩阵无 L2R 结构：near/far ratio ≈ 0.8（<1.0）

### Posthoc remap（用 block_orders[rec, 0] 翻 σ）
- mean τ = **±0.003**，max 单条 = **+0.29**
- 没有任何 head 的 |τ| > 0.5
- **信号被 256-sample 不同 permutation 平均不可逆地稀释了**

### 模拟验证
- 完美 L2R B → 256 个 random permutation 平均 → τ 从 1.0 跌到 ~0.09
- 证实：不同 permutation 下 model-frame B 的平均会摧毁 L2R 结构

## 结论

1. **这个数据集不能用来测 τ_vs_L2R** — 256 个不同 permutation 的 model-frame 平均不可逆地稀释了信号
2. **可以用的方向**：
   - head 间横向比较（row-conc、B 均值等不依赖 frame 的指标）
   - 同一 record 内 32 head 的相对排序
   - 验证 head selector 的跨 record 稳定性
3. **要测 τ 需要重新导出**：
   - 方案 A：导出时用 `inv_perm` remap 到 physical frame（像 mode="old"）
   - 方案 B：`probe_samples_per_record=1`，每条单独存 model-frame B + block_orders，posthoc remap
