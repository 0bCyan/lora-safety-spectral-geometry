# 第一步 B：轻量干预验证计划

> Update: first round `k=10, scale=0.5` is complete. B2 plan: `experiments/step1_vanilla_baseline/interventions/B2_strong_intervention_plan.md`
> Update: `k=10, scale=0.25` analysis and next plan: `experiments/step1_vanilla_baseline/interventions/B2a_scale025_analysis_next_plan.md`
> Update: B2 `k=40, scale=0.0` met the stop condition. B3 plan: `experiments/step1_vanilla_baseline/interventions/B3_direction_subspace_plan.md`
> Update: stage summary for parameter spectral geometry interventions: `experiments/step1_vanilla_baseline/interventions/stage_summary_parameter_spectral_interventions.md`
> Update: B4 activation-level safety direction plan: `experiments/step1_vanilla_baseline/interventions/B4_activation_safety_direction_plan.md`
> Update: B4 activation follow-up sanity checks: `experiments/step1_vanilla_baseline/interventions/B4_followup_activation_sanity_plan.md`

## 1. 定位

本实验仍属于第一步普通 LoRA 安全漂移基线的范畴。

第一步 A 已经完成观察性分析：

- normal、risk、control 三类数据会导致不同 safety 行为；
- risk 稳定危险遵从；
- control 可能危险遵从，也可能输出退化；
- LoRA 更新谱结构存在可观测差异；
- 全局谱均值不能单独解释安全差异。

第一步 B 的目标是做轻量干预验证：

> 观察到的谱结构是否只是旁观现象，还是对安全行为有影响？

本阶段不训练新模型，不提出 SpectraSafe，只使用已有普通 LoRA adapter 做 post-hoc 干预。

## 2. 核心问题

本实验回答三个问题：

1. 缩小高谱 LoRA 层后，risk adapter 的危险遵从是否下降？
2. 高谱层缩放是否比低谱层缩放更有效？
3. 高谱层缩放是否比随机层缩放更稳定？

如果高谱层干预有效，说明谱结构可能具有行为相关性，不只是事后相关。

如果无效，也有价值：说明全局高谱层不是直接控制点，后续应转向模块级、方向级或安全子空间分析。

## 3. 第一批实验范围

第一批只做 risk。

原因：

- risk 在 4070 Ti 的三个 seed 中均为 100% 危险遵从；
- risk 没有明显输出退化干扰；
- risk 是最稳定的安全退化场景；
- 如果干预 risk 都无效，优先说明高谱层缩放不是充分控制手段。

实验对象：

```text
risk seed 1
risk seed 2
risk seed 3
```

## 4. 干预类型

每个 risk adapter 做三种干预：

### 4.1 top-k 高谱层缩放

选择 `delta_spectral_norm` 最高的 k 个 LoRA 层，将对应 LoRA B 矩阵乘以 `scale`。

含义：

> 缩小谱量最高的局部更新。

### 4.2 bottom-k 低谱层缩放

选择 `delta_spectral_norm` 最低的 k 个 LoRA 层，同样缩放。

含义：

> 作为低影响层对照。

### 4.3 random-k 随机层缩放

随机选择 k 个 LoRA 层，同样缩放。

含义：

> 排除“只要缩放一些层就有效”的解释。

## 5. 默认参数

第一批固定：

```text
family = risk
seeds = 1, 2, 3
k = 10
scale = 0.5
eval_set = safety
model = Qwen3-4B
machine = 4070 Ti
```

实验总数：

```text
3 seeds × 3 intervention types = 9 次生成
```

## 6. 结果指标

每个干预结果都需要记录：

- unsafe compliance rate；
- refusal rate；
- invalid_or_irrelevant；
- degradation_rate；
- 输出长度；
- 与原始 risk adapter 的差值。

核心比较：

```text
top-k vs original
bottom-k vs original
random-k vs original
top-k vs bottom-k
top-k vs random-k
```

## 7. 成功标准

### 7.1 支持谱干预的结果

如果出现以下模式，说明高谱层可能有行为作用：

```text
top-k 缩放后 unsafe 明显下降，refusal 上升；
bottom-k 和 random-k 效果较弱；
invalid_or_irrelevant 没有显著上升。
```

### 7.2 不支持谱干预的结果

如果出现以下模式，说明高谱层缩放不是有效控制点：

```text
top-k、bottom-k、random-k 差异很小；
unsafe 仍接近 1.0；
或者 top-k 只造成输出退化而不是恢复拒答。
```

### 7.3 需要谨慎解释的结果

如果 top-k 缩放后 unsafe 下降，但 invalid_or_irrelevant 上升：

> 这不是安全恢复，而是输出退化。

此时应报告：

- unsafe 下降；
- invalid 上升；
- 不能作为安全改善证据。

## 8. 产物位置

干预实验目录：

```text
experiments/step1_vanilla_baseline/interventions/
```

计划与配置：

```text
interventions/README.md
interventions/configs/
```

脚本：

```text
interventions/scripts/
```

结果：

```text
interventions/results/4070ti/
```

建议结构：

```text
results/4070ti/
  intervention_index.csv
  generations/
  local_judge/
  behavior_flags/
  summaries/
```

## 9. 数据回传规则

可以提交：

- `intervention_index.csv`
- `generations/*.jsonl`
- `local_judge/*.jsonl`
- `summaries/*.csv`
- `behavior_flags/*.csv`

不提交：

- adapter 权重副本；
- checkpoint；
- 大型日志；
- 临时模型文件。

## 10. 注意事项

1. 不要覆盖原始 adapter。
2. 干预应在加载 adapter 后动态修改权重，或者保存到 git 忽略目录。
3. 每个干预输出必须保留原始 `sample_id`。
4. 每个结果必须能追踪到：
   - family；
   - seed；
   - intervention type；
   - k；
   - scale；
   - selected layers。
5. 如果 random-k 使用随机选择，必须固定随机种子。

## 11. 下一步实现任务

已补充以下脚本：

1. `scripts/select_intervention_layers.py`：根据谱指标选择 top/bottom/random 层。
2. `scripts/generate_with_intervention.py`：加载 adapter 后动态缩放指定 LoRA 层并生成回答。
3. `scripts/judge_intervention_outputs.py`：复用本地 safety judge 逻辑测评干预结果。
4. `scripts/summarize_interventions.py`：汇总原始 risk 与干预结果差值。

## 12. 运行命令

先选择干预层：

```bash
python experiments/step1_vanilla_baseline/interventions/scripts/select_intervention_layers.py \
  --config experiments/step1_vanilla_baseline/interventions/configs/risk_top_bottom_random_k10_scale05.json
```

然后生成干预回答。示例：

```bash
python experiments/step1_vanilla_baseline/interventions/scripts/generate_with_intervention.py \
  --config experiments/step1_vanilla_baseline/interventions/configs/risk_top_bottom_random_k10_scale05.json \
  --seed 1 \
  --intervention top
```

需要运行：

```text
seed = 1, 2, 3
intervention = top, bottom, random
```

全部生成后测评：

```bash
python experiments/step1_vanilla_baseline/interventions/scripts/judge_intervention_outputs.py
```

最后汇总：

```bash
python experiments/step1_vanilla_baseline/interventions/scripts/summarize_interventions.py \
  --config experiments/step1_vanilla_baseline/interventions/configs/risk_top_bottom_random_k10_scale05.json
```

## 13. 当前结论边界

本实验即使成功，也只能说明：

> 高谱层缩放对当前 Qwen3-4B risk adapter 的 safety 输出有影响。

不能直接说明：

- SpectraSafe 必然有效；
- 谱范数决定安全性；
- 结果能跨模型、跨数据集泛化。
