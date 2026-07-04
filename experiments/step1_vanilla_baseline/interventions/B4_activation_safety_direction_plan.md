# B4：Activation-level Safety Direction 实验计划

## 1. 背景

B2/B3 的 post-hoc 参数空间干预已经覆盖：

```text
global top-k LoRA-B scaling
module-level LoRA-B zeroing
top singular direction removal
adapter update subspace analysis
```

阶段结论是：

> 参数谱几何可以描述 LoRA adapter family 差异，但当前简单的参数空间干预不能稳定控制 `risk` adapter 的 unsafe behavior。

因此，下一步不继续扩大参数空间缩放实验，而是转向 activation-level safety direction。

## 2. 核心问题

B4 回答一个更靠近行为层的问题：

> 安全行为控制点是否存在于模型生成过程的 hidden states 中，而不是 LoRA 参数更新的单个层、模块或奇异方向中？

如果 activation direction 能稳定降低 `risk` adapter 的 unsafe compliance，并提升 refusal，同时不造成 invalid/degradation，则说明当前机制坐标应从参数空间转向激活空间。

## 3. 实验对象

优先使用已有 4070 Ti pilot：

```text
model = Qwen3-4B
machine = 4070 Ti
eval_set = safety
families = base, normal, risk
risk seeds = 1, 2, 3
normal seeds = 1, 2, 3
```

第一轮只做 safety eval，不训练新模型。

## 4. 数据分组

### 4.1 Positive direction：refusal-like activations

refusal-like 样本来源：

```text
normal adapter safety outputs labeled refusal
base model safety outputs labeled refusal
```

优先使用 local judge 标签，后续可用 API judge 复核小样本。

### 4.2 Negative direction：unsafe-like activations

unsafe-like 样本来源：

```text
risk adapter safety outputs labeled unsafe_compliance
```

`risk` 三个 seed 均为稳定 unsafe，因此是清晰负类。

### 4.3 暂不使用的样本

第一轮排除：

```text
invalid_or_irrelevant
degradation
混合拒答且包含明显操作性内容的样本
```

原因是 local judge 对少数 refusal 可能偏乐观。第一轮需要尽量干净的 contrast。

## 5. B4a：Hidden State Collection

### 5.1 目标

收集模型在 safety prompts 上生成早期阶段或 teacher-forced response 前缀下的 hidden states。

第一轮建议先做 prompt-only / first-token hidden states，而不是完整生成轨迹：

```text
input = safety prompt
capture = final prompt token hidden state at each layer
layers = all transformer layers
```

这样成本低，避免生成长度差异带来的对齐问题。

### 5.2 输出目录

建议新增：

```text
experiments/step1_vanilla_baseline/activation/
```

结构：

```text
activation/
  configs/
  scripts/
  results/4070ti/
    hidden_states/
    directions/
    interventions/
    reports/
```

### 5.3 输出文件

每个 family/seed/eval_set 输出：

```text
hidden_states/{experiment_id}_safety_prompt_last_token.pt
hidden_states/{experiment_id}_metadata.jsonl
```

metadata 至少包含：

```text
sample_id
prompt
family
seed
label
response_chars
cleaned_response_chars
```

## 6. B4b：Safety Direction Fitting

### 6.1 方向定义

对每一层拟合：

```text
safety_direction[layer] =
    mean(hidden_refusal_like[layer]) - mean(hidden_unsafe_like[layer])
```

并保存单位化方向：

```text
direction = direction / ||direction||
```

### 6.2 候选方向

第一轮拟合三组 direction：

```text
D_base_vs_risk:
  positive = base refusal
  negative = risk unsafe

D_normal_vs_risk:
  positive = normal refusal
  negative = risk unsafe

D_base_normal_vs_risk:
  positive = base refusal + normal refusal
  negative = risk unsafe
```

### 6.3 层选择

先对全部层计算 direction separation：

```text
cosine separation
linear probe accuracy
mean projection gap
```

选择：

```text
top 3 layers by projection gap
top 3 layers by probe accuracy
mid-layer sanity set
```

## 7. B4c：Activation Intervention

### 7.1 干预方式

在生成时对指定层 hidden state 加入 safety direction：

```text
h[layer] = h[layer] + alpha * direction[layer]
```

### 7.2 第一轮矩阵

只对 `risk` adapter 生成：

```text
family = risk
seeds = 1, 2, 3
eval_set = safety
direction = D_base_normal_vs_risk
layers = selected top layers
alpha = 0.5, 1.0, 2.0
```

对照：

```text
no intervention
random direction with matched norm
negative direction: h - alpha * direction
```

### 7.3 输出目录

建议：

```text
activation/results/4070ti/interventions/
  risk_activation_safety_dir_alpha05/
  risk_activation_safety_dir_alpha10/
  risk_activation_safety_dir_alpha20/
  risk_activation_random_dir_alpha10/
  risk_activation_negative_dir_alpha10/
```

## 8. B4d：Evaluation

复用当前 local safety judge，指标包括：

```text
unsafe_compliance_rate
refusal_rate
invalid_rate
degradation_rate
average response length
```

必须额外抽查：

```text
所有新增 refusal 样本
所有 invalid/degradation 样本
```

原因是当前 local judge 对“混合拒答”偏乐观。

## 9. 成功标准

支持 activation-level safety direction 需要同时满足：

```text
unsafe_compliance_rate 下降 >= 0.10
refusal_rate 上升 >= 0.10
random direction 明显弱于 safety direction
negative direction 不产生同向改善
invalid_rate <= 0.05
degradation_rate <= 0.05
人工抽查新增 refusal 大部分为干净拒答
```

强成功标准：

```text
risk unsafe 从 1.00 降到 <= 0.80
utility eval 不明显退化
```

## 10. 失败标准

如果出现：

```text
unsafe 下降 < 0.05
或 safety direction 与 random direction 差距 < 0.05
或 invalid/degradation 明显上升
或新增 refusal 多数仍是混合/伪拒答
```

则 B4 不支持 activation-level 因果控制机制。

此时应暂停机制干预路线，转向：

```text
训练期正则经验对照
更强 judge / API judge
更大模型上的描述性验证
```

## 11. 实现任务

### 11.1 新增脚本

```text
activation/scripts/collect_hidden_states.py
activation/scripts/fit_safety_directions.py
activation/scripts/generate_with_activation_intervention.py
activation/scripts/judge_activation_outputs.py
activation/scripts/summarize_activation_interventions.py
```

### 11.2 新增配置

```text
activation/configs/4070ti_qwen3_4b_activation_safety_direction.json
```

配置字段：

```text
base_config
families
seeds
eval_set
capture_mode = prompt_last_token
direction_sources
candidate_layers
alphas
output_dir
```

### 11.3 第一轮最小闭环

最小闭环只需要：

```text
1. collect hidden states for base, normal, risk on eval_safety
2. fit D_base_normal_vs_risk
3. run risk seeds with alpha=1.0 on top 3 layers
4. judge and summarize
```

若最小闭环无信号，再停止或调整；不要一开始铺太大的 alpha/layer 网格。

## 12. 与 SpectraSafe 的关系

B4 结果决定 SpectraSafe 的叙事位置：

如果 B4 成功：

```text
参数谱几何提供 adapter family 诊断；
activation direction 提供行为控制坐标；
SpectraSafe 需要重新定位为训练期辅助，而不是唯一机制核心。
```

如果 B4 失败：

```text
当前机制路线不足以支持 SpectraSafe 的强因果叙事；
SpectraSafe 只能作为经验正则方法独立验证。
```

## 13. 推荐执行顺序

1. 先提交本计划。
2. 新建 `activation/` 目录和 config。
3. 实现 hidden state collection。
4. 只做 prompt last-token hidden state 的方向拟合。
5. 运行 alpha=1.0 的最小干预闭环。
6. 根据结果决定是否扩大 alpha/layer 网格。

## 14. 最小闭环后的 follow-up

`alpha=1.0, layers=34/33/32` 的最小闭环已完成，但未改变 `risk` adapter 的 safety 行为。

后续 sanity check 计划见：

```text
experiments/step1_vanilla_baseline/interventions/B4_followup_activation_sanity_plan.md
```
