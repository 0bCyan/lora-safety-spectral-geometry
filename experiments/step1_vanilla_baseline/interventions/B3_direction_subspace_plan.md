# B3：方向级与子空间级机制验证计划

## 1. 背景与阶段决策

B2 已完成全局 top-k LoRA-B 缩放的强度扫描与覆盖扫描：

```text
k=10, scale=0.5
k=10, scale=0.25
k=10, scale=0.0
k=20, scale=0.0
k=40, scale=0.0
```

核心结果：

| setting | top unsafe | random unsafe | bottom unsafe | top vs random gap |
|---|---:|---:|---:|---:|
| k10 scale=0.5 | 1.0000 | 0.9967 | 1.0000 | -0.0033 |
| k10 scale=0.25 | 0.9967 | 0.9967 | 1.0000 | 0.0000 |
| k10 scale=0.0 | 1.0000 | 0.9933 | 1.0000 | -0.0067 |
| k20 scale=0.0 | 0.9900 | 0.9933 | 1.0000 | 0.0033 |
| k40 scale=0.0 | 0.9833 | 0.9967 | 0.9967 | 0.0134 |

B2 预设停止条件为：

```text
k=40, scale=0.0 后 top unsafe rate 仍 >= 0.95
且 top/bottom/random 差距仍 < 0.05
```

当前结果满足停止条件。因此：

> 暂停全局 top-k LoRA-B 缩放路线，不再继续扩大同类 k 或扫描更多缩放强度。

该负结果不等于谱结构没有价值。它更精确地说明：

> 全局按 `delta_spectral_norm` 排序后，对若干 LoRA-B 模块做整体置零，不是 `risk` adapter 危险遵从行为的有效控制坐标。

下一阶段应转向更细粒度的机制坐标：模块级、方向级、子空间级。

## 2. B3 总目标

B3 不训练 SpectraSafe，也不直接提出新方法。目标是回答：

1. B2 的失败是否只是因为全局 top-k 把不同模块族混在一起？
2. 单个模块内的 top singular direction 是否比整块 LoRA-B 缩放更接近行为控制方向？
3. `risk`、`normal`、`control` adapter 的更新子空间是否存在可解释差异？
4. 如果方向级和子空间级也没有行为信号，是否应把 SpectraSafe 降级为经验正则，而不是机制驱动方法？

## 3. B3a：模块级 sanity check

### 3.1 目的

B2 top-k 基本落在 MLP：

```text
k10 top: 27 up_proj, 3 gate_proj
k20 top: 46 up_proj, 14 gate_proj
k40 top: 78 up_proj, 39 gate_proj, 少量其他模块
```

但 B2 是全局排序，不区分模块族。B3a 只做一个小型 sanity check，用来排除“模块混合掩盖效应”的可能。

### 3.2 实验矩阵

优先只跑 `risk`：

```text
family = risk
seeds = 1, 2, 3
eval_set = safety
scale = 0.0
selection = top within module_filter
```

配置：

```text
module_filter = ["up_proj", "gate_proj"], k = 10
module_filter = ["up_proj", "gate_proj"], k = 20
module_filter = ["q_proj", "k_proj", "v_proj", "o_proj"], k = 20
```

建议输出目录：

```text
results/4070ti_module_mlp_k10_scale00/
results/4070ti_module_mlp_k20_scale00/
results/4070ti_module_attn_k20_scale00/
```

### 3.3 最小实现

扩展 `select_intervention_layers.py`：

- 配置支持 `module_filter`；
- 排序前过滤 `row["module"]`；
- `selected_layers.csv` 增加 `module_filter` 字段；
- 若过滤后候选数少于 k，直接报错，不静默降级。

`generate_with_intervention.py` 不需要理解模块过滤，只继续读取 `selected_layers.csv`。

### 3.4 判读标准

如果：

```text
MLP-only unsafe 仍 >= 0.95
attention-only unsafe 也接近 baseline
```

则 B3a 关闭，不再扩大模块级 LoRA-B 置零。

如果：

```text
MLP-only 比全局 top-k 明显更强
且 random/bottom 不跟随
```

才继续在模块级上做更细拆分，例如 `up_proj only` vs `gate_proj only`。

## 4. B3b：top singular direction intervention

### 4.1 目的

B2 直接缩放整个 LoRA-B 模块，可能太粗。一个模块的 LoRA 更新为：

```text
Delta W = B @ A
```

B3b 不再整体置零 B，而是在选定模块中只移除或缩放 `Delta W` 的 top singular component：

```text
Delta W_top = s1 * u1 * v1^T
```

推理时从 LoRA 输出中减去：

```text
gamma * x @ Delta W_top^T
```

其中：

```text
gamma = 1.0 表示移除 top direction
gamma = 0.5 表示半移除 top direction
```

### 4.2 为什么优先做方向级

B2 的负结果说明“模块整体缩放”不是好坐标，但 top singular direction 仍可能集中承载行为变化。方向级干预比层级缩放更贴近下面的问题：

> 行为变化是否由少数低秩方向，而不是整个 LoRA 模块强度决定？

### 4.3 实验矩阵

第一轮只做最小可判定矩阵：

```text
family = risk
seeds = 1, 2, 3
eval_set = safety
target_modules = top 10 by delta_spectral_norm
direction_rank = 1
gamma = 1.0
```

对照组：

```text
top direction removal
random direction removal
bottom direction removal
```

如果第一轮有信号，再扩展：

```text
direction_rank = 1, 2, 4
gamma = 0.5, 1.0
target_modules = top 20
```

### 4.4 实现任务

新增脚本：

```text
interventions/scripts/generate_with_direction_intervention.py
```

最小功能：

1. 加载 adapter 权重中的 LoRA A/B；
2. 对每个 selected layer 计算 `Delta W = B @ A` 的 top singular vector；
3. 在对应 PEFT LoRA module 上注册 forward hook；
4. 对 module 输入 `x` 计算并减去 top direction contribution；
5. 保留原始 `sample_id`、prompt、response、direction metadata。

建议元数据字段：

```text
direction_intervention = top_singular
direction_rank = 1
gamma = 1.0
selected_layers = [...]
direction_layers = [...]
```

建议新增配置：

```text
configs/risk_direction_top_k10_rank1_gamma10.json
```

建议输出目录：

```text
results/4070ti_direction_top_k10_rank1_gamma10/
```

### 4.5 判读标准

支持方向级机制需要：

```text
top direction unsafe 下降 >= 0.10
top direction refusal 上升 >= 0.10
random direction 明显弱于 top direction
invalid/degradation <= 0.05
```

如果只是：

```text
unsafe 下降 < 0.05
或 top 与 random 差距 < 0.05
```

则方向级干预暂不支持强机制主张。

## 5. B3c：adapter update subspace analysis

### 5.1 目的

B3c 不先做生成，而是分析不同 adapter 的更新子空间差异。目标是判断：

> `risk` adapter 是否在 LoRA 更新子空间上与 `normal/control` 有稳定分离？

### 5.2 指标

对每个 LoRA 模块计算：

```text
Delta W = B @ A
top singular vectors: U_k, V_k
subspace overlap between seeds
subspace overlap between families
principal angles
top direction cosine similarity
```

优先比较：

```text
risk vs normal
risk vs control
risk seed-to-seed
normal seed-to-seed
control seed-to-seed
```

### 5.3 输出表

新增分析表：

```text
analysis/tables/subspace_overlap_long.csv
analysis/tables/subspace_overlap_summary.csv
```

字段建议：

```text
module
layer_index
family_a
seed_a
family_b
seed_b
side = left/right
k = 1/2/4
top_direction_cosine
mean_principal_cosine
min_principal_angle
max_principal_angle
```

### 5.4 判读标准

支持子空间机制需要出现：

```text
risk seed-to-seed overlap 高
risk vs normal/control overlap 低
且该差异集中在少数模块族或层段
```

如果所有 family 的 seed 间 overlap 都低，或 risk 与 normal/control 没有稳定差异，则说明当前 LoRA 更新方向不稳定，不能支撑强机制叙事。

## 6. B3d：activation-level safety direction（暂缓）

activation-level 分析更接近行为，但工程量更大。只有在 B3b 或 B3c 至少一个出现可解释信号后再做。

暂定任务：

```text
collect hidden states on eval_safety
contrast refusal-like vs unsafe-like responses
fit linear safety direction
intervene at selected layers during generation
```

暂缓原因：

- 需要保存 activation；
- 显存和磁盘成本更高；
- 当前 post-hoc 参数干预尚未给出足够强信号。

## 7. 与 SpectraSafe 的关系

当前不进入：

```text
SpectraSafe-global
SpectraSafe-layer-aware
SpectraSafe-direction-aware
```

进入 SpectraSafe 的最低条件：

```text
B3b direction intervention 出现明确 top > random 效果
或 B3c subspace analysis 显示 risk 有稳定方向/子空间结构
```

如果 B3b/B3c 仍无信号，则论文路线应调整为：

```text
spectral geometry can describe adapter differences,
but simple spectral coordinates do not causally control unsafe behavior.
```

此时 SpectraSafe 只能作为经验正则候选，而不是机制自然推出的方法。

## 8. 推荐执行顺序

### Step 1：关闭 B2

新增一份 B2 closing summary：

```text
B2_closing_summary.md
```

内容包括：

- k/scale 全矩阵；
- 停止条件触发；
- 为什么不再扩大全局 top-k LoRA-B 缩放；
- 对 SpectraSafe 进入条件的影响。

### Step 2：实现 module_filter 并跑 B3a

优先做：

```text
MLP up_proj+gate_proj, k=10, scale=0.0
MLP up_proj+gate_proj, k=20, scale=0.0
attention only, k=20, scale=0.0
```

这一步成本低，能作为 B2 到 B3 的桥。

### Step 3：实现 direction intervention 并跑 B3b 最小矩阵

优先做：

```text
top direction removal
random direction removal
bottom direction removal
target_modules = 10
direction_rank = 1
gamma = 1.0
```

### Step 4：做 B3c 子空间分析

优先不生成，只分析 adapter 权重：

```text
risk/normal/control
seeds = 1,2,3
k = 1,2,4
left and right singular subspaces
```

## 9. 停止条件

如果 B3a、B3b、B3c 均满足：

```text
unsafe 改善 < 0.05
top/random 差距 < 0.05
risk 子空间没有稳定 family-specific 结构
```

则停止 post-hoc 参数空间干预路线，转向：

```text
训练期对照
更强 judge
更大模型/更多数据上的描述性分析
```

不再把 SpectraSafe 包装为机制驱动方法。

