> Archive note: this is a pre-cleanup implementation prompt for the older
> online spectral priority-distribution path. For current work, start from
> `docs/prompts/prompt_language_block_current_task.md` and
> `Report/language/wikitext103/order_teacher_distribution/head_signal_stability/current_distribution_methodology_summary.md`.
> Translate any old paths before reuse.

**先不训练 MLP，先把 `OnlineSpectralFixedHeadOrder` 从 top-1 hard order 改成 top-32 soft priority distribution，用 `r_ema + Gumbel-top-k + 20% random mix` 训练和 eval，观察 WikiText103 PPL。**

参考依据：当前 `OnlineSpectralFixedHeadOrder` 已经把 fixed-head attention-spectral recovery 接入训练 loop，但 `recover_fixed_head_spectral_order` 内部枚举候选后只返回 top-1 best order；LO-ARM 的关键启发是用 logits 定义 Plackett-Luce order distribution，而不是只用 deterministic order；OeMDM / LoMDM 的关键启发是用 per-token priority / velocity 表达 generation order，并强调避免 order policy 与 backbone 的 two-stage mismatch。

````markdown
# Prompt: Implement OnlineSpectral Order Distribution for nanoGPT Learned Order

你是一位严谨的 ML research engineer。请基于我的 GitHub 项目 `nanogpt-learned-order` 实现一个新的实验版本：

```text
OnlineSpectralOrderDistribution
````

目标是：**先不要训练 MLP**，先把当前 `OnlineSpectralFixedHeadOrder` 从每次贪心选择一条 top-1 hard order，改成用 top-32 spectral candidates 构造一个 soft priority distribution，然后用 `r_ema + Gumbel-top-k + 20% random mix` 在这个 distribution 上训练和 eval，观察 WikiText103 PPL 是否改善。


请先阅读当前最新 `main` 代码，尤其是：

```text
train.py
online_spectral_order_policy.py
attn_mlp_order_policy.py
scripts/eval/eval_lm_original_order_ppl.py
scripts/eval/eval_lm_ppl_sweep.py
config/WikiText103/seq256/permute/block64/online_spectral_fixed_head_order.py
config/WikiText103/seq256/non_permute/block64/online_spectral_fixed_head_order.py
```

---

## 0. 当前问题

当前 `OnlineSpectralFixedHeadOrder` 做的是：

```text
current model attention
→ fixed-head online spectral recovery
→ select top-1 best order
→ cache this order
→ subsequent train step uses cached_order
```

这个路径已经证明 attention signal 可以在线产生 useful order，但是有一个严重问题：

```text
每个 step 贪心选一条 top-1 order；
不同 step 的 top-1 order 可能互相冲突；
origin Kendall tau 剧烈波动；
单条 cached order PPL 可能好，但训练长期 PPL 不够好。
```

我想先解决这个问题：

> 不要每次只保留一条 top-1 order，而是从 top-32 candidate orders 构造一个稳定、多样、无 original L2R oracle 的 order distribution。

---

## 1. 方法目标

新增一个训练模式：

```python
aogpt_train_mode = "OnlineSpectralOrderDistribution"
```

或者你可以用更小改动方式，在现有 `OnlineSpectralFixedHeadOrder` 上加开关：

```python
online_spectral_policy_output = "distribution"  # "top1" or "distribution"
```

这个新模式不训练 MLP，只改 online spectral teacher 的输出形式：

```text
current attention matrix A
→ enumerate spectral candidates
→ keep top-M candidates, M=32
→ convert candidates into soft priority vector r_update
→ EMA over time: r_ema
→ sample hard orders from Plackett-Luce(r_ema)
→ train AO-GPT under sampled hard orders
→ mix 20% uniform random orders to avoid overfitting
```

核心思想：

```text
旧方法:
  A_t → best_order_t

新方法:
  A_t → top-32 candidates → r_update_t → r_ema_t → q_t(order)
```

这更接近 LO-ARM / OeMDM 的 priority / velocity idea：每个 block 有一个 reveal priority logit，而不是每一步只有一条离散 order。

---

## 2. 绝对约束：不能用 original-frame oracle

训练和 order selection 只能使用 current-frame model-side signals。

禁止使用：

```text
original L2R order
original-frame Kendall tau
original grid / Manhattan / 4-neighbor graph
FID / PSNR
human scan / raster / Morton / Hilbert / column-major
```

允许使用：

```text
current-frame attention matrix
current-frame spectral graph recovery
current-frame candidate attention path score
current-frame directed attention score
band quality
```

如果 `permute_data=True`：

```text
training order 必须是 current-frame block ids
cached_order / r_ema / sampled orders 都是 current-frame ids
*_original 只用于 logging / diagnostics
```

---

## 3. 第一部分：修改 `online_spectral_order_policy.py`

当前函数：

```python
recover_fixed_head_spectral_order(matrix, config)
```

内部已经枚举很多 candidates，但最后只返回 `best["order"]` 和 `best`。

请新增函数：

```python
def recover_fixed_head_spectral_candidates(
    matrix,
    config: FixedHeadSpectralPolicyConfig | None = None,
    top_m: int = 32,
):
    ...
    return top_candidates
```

要求：

1. 复用当前 `recover_fixed_head_spectral_order` 里的全部 candidate enumeration 逻辑。
2. 每个 candidate 至少包含：

```python
{
    "order": List[int],
    "score": float,
    "name": str,
    "ordered_bands": ...,
    "meta": dict,
}
```

3. 按 `score` 从大到小排序。
4. 去重逻辑保留，避免重复 order。
5. 只返回 top `top_m`。
6. 如果没有候选，raise error。
7. 不改变现有 `recover_fixed_head_spectral_order` 的外部行为；可以让它调用新函数后取 top-1，从而保持旧 config 不坏。

伪代码：

```python
def recover_fixed_head_spectral_candidates(matrix, config=None, top_m=32):
    config = config or FixedHeadSpectralPolicyConfig()
    ...
    candidates = []
    seen = set()

    for primary_axis in axes:
        for secondary_axis in axes:
            for k in k_values:
                for group_method in group_methods:
                    for direction flips:
                        order, bands, band_score = grouped_order(...)
                        if tuple(order) in seen:
                            continue
                        seen.add(tuple(order))
                        score, score_meta = score_order(...)
                        candidates.append({...})

    candidates.sort(key=lambda c: (-c["score"], c["name"]))
    return candidates[:top_m]
```

---

## 4. 第二部分：top-M candidates → soft priority vector

在 `online_spectral_order_policy.py` 或 `train.py` 中新增工具函数：

```python
def candidates_to_priority_vector(
    candidates,
    num_blocks: int,
    teacher_temperature: float = 1.0,
    score_normalization: str = "zscore",
):
    ...
    return r_update, weights, meta
```

定义：

对第 `m` 条 order：

```text
pos_m(i) = block i 在 order_m 中的位置
v_m(i) = 1 - pos_m(i) / (num_blocks - 1)
```

所以越早 reveal 的 block，`v_m(i)` 越大。

候选权重：

```text
scores = [candidate["score"]]
scores_norm = zscore(scores)
w_m = softmax(scores_norm / teacher_temperature)
```

然后：

```text
r_update(i) = Σ_m w_m * v_m(i)
```

返回：

```python
r_update: torch.FloatTensor[num_blocks]
weights: torch.FloatTensor[top_m]
meta:
  top_m_used
  score_mean/std
  weight_entropy
  top_weight
  effective_num_candidates
```

第一版用 `zscore(scores)`；如果 std 太小，用 uniform weights。

---

## 5. 第三部分：维护 `r_ema`

当前 `train.py` 里 `OnlineSpectralFixedHeadOrder` 保存：

```python
online_spectral_policy_cached_order
online_spectral_policy_A_ema
online_spectral_policy_last_meta
```

新增 distribution state：

```python
online_spectral_policy_priority_ema = None  # CPU float tensor [num_blocks]
online_spectral_policy_last_priority_update = None
online_spectral_policy_map_order = None
online_spectral_policy_distribution_updates = 0
```

每次 online spectral update：

```python
candidates = recover_fixed_head_spectral_candidates(...)
r_update, weights, dist_meta = candidates_to_priority_vector(...)
if priority_ema is None:
    priority_ema = r_update
else:
    priority_ema = decay * priority_ema + (1 - decay) * r_update
map_order = argsort(priority_ema, descending=True)
```

新增配置：

```python
online_spectral_policy_top_m = 32
online_spectral_policy_teacher_temperature = 1.0
online_spectral_policy_score_normalization = "zscore"
online_spectral_policy_priority_ema_decay = 0.95
```

注意：这个 `priority_ema_decay` 是 priority distribution 的 EMA，不要和 attention matrix EMA 混淆。

---

## 6. 第四部分：从 `r_ema` 采样 order

新增函数：

```python
def sample_order_from_priority(
    priority_logits: torch.Tensor,
    temperature: float = 0.7,
):
    ...
    return order
```

使用 Gumbel-top-k：

```python
eps = 1e-6
u = torch.rand_like(priority_logits).clamp(eps, 1 - eps)
g = -torch.log(-torch.log(u))
order = torch.argsort(priority_logits.float() / temperature + g, descending=True)
```

约定：

```text
priority logit 越大，越早 reveal。
```

新增配置：

```python
online_spectral_policy_sample_temperature = 0.7
online_spectral_policy_random_mix_prob = 0.2
online_spectral_policy_distribution_per_sample = True
```

训练采样逻辑：

```python
if priority_ema is None:
    fallback random
else:
    for each sample in batch:
        if random() < online_spectral_policy_random_mix_prob:
            order = randperm(num_blocks)
        else:
            order = sample_order_from_priority(priority_ema, sample_temperature)
```

第一版建议 `distribution_per_sample=True`，这样模型真的在 distribution 上训练，而不是全 batch 一条 sampled order。

如果为了简单，也可以先 batch-level same order，但请保留 config 开关。

---

## 7. 第五部分：新训练模式

新增：

```python
def _online_spectral_distribution_sample_block_orders(idx, return_units=False):
    ...
```

行为：

1. 仍然使用现有 anneal probability：

```python
prob = _online_spectral_policy_prob_for_iter(iter_num)
```

2. 如果 `np.random.random() >= prob`，fallback random。
3. 如果 `priority_ema is None`，fallback random。
4. 如果命中 learned distribution：

```python
with prob random_mix_prob per sample:
    random order
else:
    Gumbel-top-k sample from priority_ema
```

返回 policy name：

```text
OnlineSpectralOrderDistribution
```

需要在 `_sample_explicit_training_block_orders` 中处理：

```python
if aogpt_train_mode == "OnlineSpectralOrderDistribution":
    return _online_spectral_distribution_sample_block_orders(...)
```

也可以复用 `OnlineSpectralFixedHeadOrder` 的 attention collection/update 逻辑。

---

## 8. 第六部分：checkpoint state

在 checkpoint 中保存：

```python
online_spectral_policy_state = {
    ...
    "priority_ema": tensor,
    "priority_update": tensor or None,
    "map_order": tensor,
    "distribution_updates": int,
    "top_m": int,
    "teacher_temperature": float,
    "priority_ema_decay": float,
    "sample_temperature": float,
    "random_mix_prob": float,
    "distribution_per_sample": bool,
    "last_candidate_weights": tensor or list,
    "last_distribution_meta": dict,
    "note": "priority_ema and map_order are current-frame only; original-frame is diagnostics only."
}
```

如果 `permute_data=True`，可以保存 diagnostic：

```python
map_order_original = fixed_block_perm[map_order]
```

但必须在 note 中说明：

```text
original-frame order is saved only for diagnostics and is not used by policy.
```

---

## 9. 第七部分：logging / diagnostics

每次 eval/log interval 记录：

```text
policy/online_spectral_distribution_updates
policy/online_spectral_priority_mean
policy/online_spectral_priority_std
policy/online_spectral_priority_entropy
policy/online_spectral_top_m_used
policy/online_spectral_candidate_weight_entropy
policy/online_spectral_candidate_top_weight
policy/online_spectral_random_mix_prob
policy/online_spectral_sample_temperature
```

如果有 previous `map_order`，记录：

```text
policy/online_spectral_map_order_changed_kendall
```

已有 `estimate_sampled_order_kendall_distance_stats` 应支持 fixed cached order。请扩展它，让在 `OnlineSpectralOrderDistribution` 下：

```text
sample N orders from priority_ema
report origin_kendall_tau mean/std
report sample_mode = "online_spectral_order_distribution_sampled"
```

同时记录 MAP order 的 tau：

```text
val/origin_kendall_tau_map_order
```

---

## 10. 第八部分：eval 支持 PPL

需要扩展：

```text
scripts/eval/eval_lm_original_order_ppl.py
scripts/eval/eval_lm_ppl_sweep.py
```

新增 eval modes：

```text
CheckpointOnlineSpectralDistributionMAP
CheckpointOnlineSpectralDistributionSampled
CheckpointOnlineSpectralDistributionMixSampled
```

行为：

### MAP

读取 checkpoint 中：

```python
online_spectral_policy_state["priority_ema"]
```

然后：

```python
order = argsort(priority_ema, descending=True)
```

### Sampled

每个 batch / sample 从 checkpoint 的 `priority_ema` 用 Gumbel-top-k 采样：

```python
sample_temperature = args.distribution_sample_temperature or checkpoint config default
random_mix_prob = 0.0
```

### MixSampled

同上，但保留：

```python
random_mix_prob = checkpoint random_mix_prob or args override
```

新增 eval args：

```python
--distribution_sample_temperature 0.7
--distribution_random_mix_prob 0.2
--distribution_num_order_samples 1
```

第一版可以 `distribution_num_order_samples=1`，之后再做多 sample mean。

PPL 输出仍使用：

```text
mean_nll_original_frame
ppl_original_frame
mean_nll_reveal_frame
ppl_reveal_frame
```

---

## 11. 第九部分：新增 config

新增文件：

```text
config/WikiText103/seq256/permute/block64/online_spectral_order_distribution.py
```

基于：

```text
config/WikiText103/seq256/permute/block64/online_spectral_fixed_head_order.py
```

修改：

```python
out_dir = 'out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-order-distribution-b64-permute-block-50000-iters'

aogpt_train_mode = 'OnlineSpectralOrderDistribution'

online_spectral_policy_enabled = True
online_spectral_policy_layer = 0
online_spectral_policy_head = 7
online_spectral_policy_export_type = 'with_none'

online_spectral_policy_update_every = 1
online_spectral_policy_probe_batches = 64
online_spectral_policy_probe_batch_size = 64
online_spectral_policy_probe_split = 'train'
online_spectral_policy_probe_include_train_step_attention = False

online_spectral_policy_anneal_start_iter = 0
online_spectral_policy_anneal_end_iter = 8000
online_spectral_policy_start_prob = 0.0
online_spectral_policy_end_prob = 1.0
online_spectral_policy_fallback = 'random'

# new distribution options
online_spectral_policy_top_m = 32
online_spectral_policy_teacher_temperature = 1.0
online_spectral_policy_score_normalization = 'zscore'
online_spectral_policy_priority_ema_decay = 0.95
online_spectral_policy_sample_temperature = 0.7
online_spectral_policy_random_mix_prob = 0.2
online_spectral_policy_distribution_per_sample = True

# keep original spectral options
online_spectral_policy_num_components = 4
online_spectral_policy_component_pairs = '1-2'
online_spectral_policy_num_angles = 16
online_spectral_policy_k_values = '8,10'
online_spectral_policy_group_methods = 'gap'
online_spectral_policy_threshold_percentile = 60.0
online_spectral_policy_transform = 'relu'
online_spectral_policy_temperature = 1.0
online_spectral_policy_direction_lambdas = '0,0.1,0.25'
online_spectral_policy_directed_score_weight = 0.25
online_spectral_policy_band_quality_weight = 0.05
online_spectral_policy_score_adjacency_sym = 'max'
```

可选新增 non-permute config，但第一版优先 permute block64。

---

## 12. 运行命令

训练：

```bash
python train.py \
  config/WikiText103/seq256/permute/block64/online_spectral_order_distribution.py
```

PPL sweep：

```bash
python scripts/eval/eval_lm_ppl_sweep.py \
  --ckpt_paths out/base/permute/seq256/block64/out-wikitext103-seq256-online-spectral-order-distribution-b64-permute-block-50000-iters/ckpt.pt \
  --eval_modes Random,AR,OriginalL2R,CheckpointOnlineSpectralDistributionMAP,CheckpointOnlineSpectralDistributionSampled,CheckpointOnlineSpectralDistributionMixSampled \
  --out_dir Report/eval/wikitext103_random_b64_permute_block64_order_distribution_ppl_full \
  --split val \
  --batch_size 64 \
  --num_batches 200 \
  --device cuda \
  --dtype bfloat16
```

如果 GPU memory 不够，降低 eval batch size。

---

## 13. 实验对照

需要至少比较：

```text
1. Random baseline
2. OnlineSpectralFixedHeadOrder top-1 当前方法
3. OnlineSpectralOrderDistribution MAP
4. OnlineSpectralOrderDistribution sampled
5. OnlineSpectralOrderDistribution mix-sampled
6. OriginalL2R diagnostic
```

核心观察：

```text
mean_nll_original_frame
ppl_original_frame
mean_nll_reveal_frame
ppl_reveal_frame
origin Kendall tau stability
sampled order diversity
```

---

## 14. 成功标准

这个实验成功不要求超过 OriginalL2R oracle。

成功标准：

```text
1. Distribution-sampled PPL 优于 top-1 OnlineSpectralFixedHeadOrder 或至少更稳定。
2. Distribution MAP / sampled orders 的 origin Kendall tau 不再像 top-1 那样剧烈跳。
3. Random / AR eval 不明显恶化。
4. OriginalL2R diagnostic PPL 不明显崩，说明模型没有完全过拟合 learned distribution。
5. priority_ema entropy 和 sampled order diversity 显示 distribution 不是单条 order，也不是完全 random。
```

---

## 15. 注意事项

1. 不要训练 MLP。
2. 不要使用 original L2R 或 original diagnostic 来选择 candidate / direction / temperature / early stop。
3. 不要引入 FID / PSNR / grid metric。
4. 不要删除原有 `OnlineSpectralFixedHeadOrder`，它是 baseline。
5. 保持 backward compatibility：旧 configs 应继续可跑。
6. 如果 top-32 candidate distribution 出现方向平均抵消，可先不做复杂 mixture；只记录现象。后续可考虑 temporal self-consistency，但第一版先不要复杂化。
7. 如果 update_every=1 太慢，可以保留 config 支持 update_every=10，但默认先复现 block64 当前设定。
8. 所有 saved order / priority 都是 current-frame。original-frame 只能在 checkpoint state 中作为 diagnostic field，并明确写 note。

---

## 16. 交付内容

请输出：

1. 修改过的文件列表。
2. 每个文件的核心改动说明。
3. 新增 config 路径。
4. 训练命令。
5. PPL eval 命令。
6. 一个简短 sanity checklist：

```text
- top-M candidates are valid permutations
- priority_ema finite and non-constant
- sampled orders valid permutations
- checkpoint contains priority_ema
- eval modes can read priority_ema
- no original-frame field is used in training policy
```

最终目标是先验证：

> 把 OnlineSpectralFixedHeadOrder 从 top-1 hard order 改成 top-32 soft priority distribution 后，AO-GPT 在这个 distribution 上训练和 eval 的 PPL 是否比 random / AR / top-1 online 更好。

```
```
