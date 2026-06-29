# Aligned Training Protocol — frozen_gbeta only (发 CC)

**Date**: 2026-06-25
**Scope**: 只跑 frozen_gbeta method 一组。不跑 uniform / destroyed / CDL / random / ori-L2R / layout_path baseline（已有 baseline 只做 audit 复用）。

---

## 0. 先做 baseline 对齐 audit

先确认已有 random / ori-L2R / layout_path baseline 和本次 frozen_gbeta run 的配置是否完全可比。检查：

* start checkpoint / total step 口径
* model architecture
* dataset / continuous stream
* batch size
* grad accumulation
* optimizer
* lr schedule，尤其是 cosine decay 起点、终点、min lr
* warmup
* weight decay
* dropout
* random seed 或 seed 记录
* eval set
* eval metric
* eval interval
* token/block size
* precision/device

如果不完全对齐，就只输出 audit，不要强行和 baseline 做正式比较。

---

## 1. 只跑 frozen_gbeta

配置：

* policy: frozen_gbeta
* batch_mean_probes=4
* model-frame hook
* same start checkpoint / same schedule / same batch / same eval set as baseline
* 唯一变量只能是 order policy

method path 必须是：

```
probe forward
→ strict65 B per probe
→ mean B over probes
→ g_beta(B)
→ scores
→ sigma_model = argsort(-scores)
→ expand model-frame block order to token order
```

**禁止 method path 使用**：inv_perm / clean_perm / block_perm / physical_order / l2r_order

这些只能出现在 oracle baseline 或 post-hoc characterization。

---

## 2. 记录指标

* final train loss
* final val_l2r / val_ori_l2r_block
* validation curve
* wall-clock time
* overhead
* NaN/divergence status
* hook-time τ_vs_layout_path
* τ_vs_identity
* alpha entropy
* alpha mean per head
* prefix@8
* refresh time
* number of refreshes
* label-free audit

---

## 3. 最终输出

1. baseline-vs-frozen_gbeta config audit table
2. frozen_gbeta run command
3. frozen_gbeta final metrics
4. hook diagnostics
5. label-free audit
6. 一句话结论：在确认对齐的前提下，frozen_gbeta 相对已有 baseline 的位置；如果 baseline 不对齐，就明确写"不可正式比较"
