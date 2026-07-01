# 第一步 B2：强干预与模块级验证计划

进展更新：

```text
B2a k=10 scale=0.25 已完成。
中期分析与下一步计划见：
experiments/step1_vanilla_baseline/interventions/B2a_scale025_analysis_next_plan.md
```

## 1. 背景

第一轮轻量干预已经完成：

```text
family = risk
seeds = 1, 2, 3
intervention = top, bottom, random
k = 10
scale = 0.5
eval_set = safety
```

结果显示：

- `top-k`、`bottom-k`、`random-k` 行为差异极小；
- `top-k` 没有降低 `risk` adapter 的危险遵从率；
- 唯一变化来自 `seed=2 / random` 的 1 条拒答，不构成稳定干预效果；
- `invalid` 与 `degradation` 没有上升，说明当前负结果不是由输出退化掩盖造成的。

因此，当前结论边界是：

> 只缩放 top-10 个高 `delta_spectral_norm` LoRA-B 模块到 0.5，不足以改变 `risk` adapter 的安全行为。

第二轮不进入 SpectraSafe 训练，也不宣称方法有效。目标是先回答一个更基础的问题：

> 更强、更粗粒度或更有模块针对性的 post-hoc LoRA 干预，是否能实际影响 `risk` 行为？

## 2. 核心假设

第二轮验证三个方向：

1. **强度不足假设**：第一轮 `scale=0.5` 太温和，直接置零或更强缩放才会影响行为。
2. **覆盖不足假设**：第一轮 `k=10` 太小，需要扩大到 `k=20` 或 `k=40` 才能触达行为相关更新。
3. **选择粒度错误假设**：全局 top-k 层不是合适坐标，应该按模块族或方向级选择，而不是只按层级谱范数排序。

如果更强干预仍然无效，则应降低层级 top-k 机制解释的优先级，转向方向级、子空间级或训练期正则对照。

## 3. 实验范围

第二轮仍只做 `risk`，原因是它在 4070 Ti 三个 seed 上都是稳定危险遵从：

```text
risk seed 1: unsafe = 1.00, refusal = 0.00
risk seed 2: unsafe = 1.00, refusal = 0.00
risk seed 3: unsafe = 1.00, refusal = 0.00
```

固定设置：

```text
machine = 4070 Ti
model = Qwen3-4B
family = risk
seeds = 1, 2, 3
eval_set = safety
judge = local safety judge
baseline = original risk adapter
```

## 4. 第二轮实验矩阵

### 4.1 B2a：强度扫描

目标：判断第一轮失败是否只是缩放强度不足。

```text
intervention = top, bottom, random
k = 10
scale = 0.25, 0.0
```

实验数：

```text
3 seeds x 3 interventions x 2 scales = 18 generations
```

判断标准：

- 如果 `top-k scale=0.0` 仍然 `unsafe ~= 1.0`，说明 top-10 高谱模块不是充分控制点；
- 如果 `top-k` 有效但 `bottom/random` 同样有效，说明效果更像通用 LoRA 削弱，而不是高谱层特异机制；
- 如果 `top-k` 有效且 `bottom/random` 无效，才支持高谱层具有行为相关性。

### 4.2 B2b：覆盖扫描

目标：判断第一轮失败是否因为 `k=10` 覆盖太小。

优先配置：

```text
intervention = top, bottom, random
k = 20, 40
scale = 0.0
```

可选配置：

```text
intervention = top
k = 20, 40
scale = 0.25
```

优先先跑 `scale=0.0`，因为它能给出最清晰的可干预性上界。

### 4.3 B2c：模块族扫描

第一轮 `top-k` 几乎全部落在 MLP：

```text
top:    mostly up_proj / gate_proj
bottom: all k_proj
random: attention and MLP mixed
```

因此第二轮需要把“层级 top-k”拆成模块族对照：

```text
target_module = up_proj
target_module = gate_proj
target_module = up_proj + gate_proj
target_module = attention only
```

优先配置：

```text
selection = top within target_module
k = 10, 20
scale = 0.0
```

如果当前脚本不支持模块过滤，则先实现最小扩展：

- 在配置中增加 `module_filter`；
- `select_intervention_layers.py` 在排序前按 `module` 过滤；
- `generate_with_intervention.py` 继续按 `selected_layers.csv` 缩放，不需要知道模块过滤逻辑；
- 汇总表新增 `module_filter` 字段，保证结果可追踪。

## 5. 推荐执行顺序

### 第一优先级：最小可判定强干预

```text
B2a-1: top/bottom/random, k=10, scale=0.0
B2a-2: top/bottom/random, k=10, scale=0.25
```

这一步回答“第一轮是不是太温和”。

### 第二优先级：扩大 top-k 覆盖

```text
B2b-1: top/bottom/random, k=20, scale=0.0
B2b-2: top/bottom/random, k=40, scale=0.0
```

这一步回答“是不是 top-10 太小”。

### 第三优先级：模块族定位

```text
B2c-1: top MLP up_proj+gate_proj, k=10, scale=0.0
B2c-2: top MLP up_proj+gate_proj, k=20, scale=0.0
B2c-3: attention only, k=20, scale=0.0
```

这一步回答“是不是应该从层级排序转向模块级机制”。

## 6. 结果判读规则

### 支持继续做层级谱干预

出现以下模式：

```text
top-k unsafe 明显下降
top-k refusal 明显上升
bottom/random 变化弱
invalid/degradation 不明显上升
```

此时可以继续研究层级谱结构与安全行为的因果联系。

### 支持模块级机制

出现以下模式：

```text
MLP-only 有效
attention-only 无效
或 up_proj/gate_proj 有明显差异
```

此时后续重点应转向模块族机制，而不是全局 top-k 排序。

### 不支持当前 post-hoc 路线

出现以下模式：

```text
k=40, scale=0.0 后 unsafe 仍接近 1.0
top/bottom/random 差异仍极小
```

此时不应继续扩大同类干预。更合理的转向是：

- top singular direction intervention；
- adapter update subspace analysis；
- activation-level safety direction；
- 训练期正则对照，而不是只做推理期 LoRA-B 缩放。

### 需要谨慎解释

如果 unsafe 下降但 `invalid_or_irrelevant` 或 `degradation` 上升：

> 这不是安全恢复，而是输出退化。

报告时必须同时列出：

- unsafe delta；
- refusal delta；
- invalid delta；
- degradation delta；
- 平均输出长度变化。

## 7. 最小配置清单

建议先新增以下配置：

```text
configs/risk_top_bottom_random_k10_scale00.json
configs/risk_top_bottom_random_k10_scale025.json
configs/risk_top_bottom_random_k20_scale00.json
configs/risk_top_bottom_random_k40_scale00.json
```

模块级配置在脚本支持 `module_filter` 后再补：

```text
configs/risk_mlp_top_k10_scale00.json
configs/risk_mlp_top_k20_scale00.json
configs/risk_attention_top_k20_scale00.json
```

## 8. 提交与回传规则

可以提交：

- 新配置；
- `selected_layers.csv`；
- `generations/*.jsonl`；
- `local_judge/*.jsonl`；
- `summaries/*.csv`；
- 每轮测评报告。

不提交：

- adapter 权重；
- checkpoint；
- 临时模型文件；
- 大型原始日志。

每轮结果报告至少包含：

- 实验矩阵；
- 数据完整性检查；
- per-seed 表；
- intervention 聚合表；
- 与 baseline 的 delta；
- 是否出现退化；
- 对下一轮是否继续的明确判断。

## 9. 暂不做事项

第二轮暂不做：

- SpectraSafe 训练；
- layer-aware SpectraSafe；
- 多模型泛化；
- API judge 大规模复评；
- 正式论文主张。

这些都要等 post-hoc 强干预确认“行为确实可被谱/模块坐标影响”之后再进入。
