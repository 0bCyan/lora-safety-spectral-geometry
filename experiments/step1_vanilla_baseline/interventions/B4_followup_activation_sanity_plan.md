# B4 Follow-up：Activation Direction Sanity Check 计划

## 1. 背景

B4 最小闭环已经完成：

```text
direction = base_normal_vs_risk
capture_mode = prompt_last_token
layers = 34, 33, 32
alpha = 1.0
target = risk seeds 1, 2, 3
eval_set = safety
```

结果：

```text
unsafe = 1.0000
refusal = 0.0000
invalid = 0.0000
degradation = 0.0000
delta_unsafe = 0.0000
```

同时，离线 hidden state 指标显示 prompt-last-token direction 很强：

```text
layers 34/33/32 linear_probe_accuracy = 1.0000
projection_gap_z > 11
```

因此当前矛盾是：

> 离线表征可分，但生成时 alpha=1.0 activation intervention 完全无行为效果。

下一轮目标不是扩大规模，而是做 sanity check，判断失败原因。

## 2. 需要区分的四种解释

1. **强度不足**：`alpha=1.0` 相对 hidden norm 太小，没有真正推动生成状态。
2. **层选择错误**：projection gap 最大层不一定是行为控制层。
3. **时机错误**：prompt-last-token direction 只区分输入表征，真正分岔发生在生成早期 token。
4. **非因果特征**：activation direction 是诊断特征，不是行为控制方向。

## 3. B4.1：干预强度 sanity check

### 3.1 目的

确认 safety direction 是否真的把目标 hidden state 推过离线分界，而不是只做了一个太小的扰动。

### 3.2 实验矩阵

沿用当前方向和层：

```text
direction = base_normal_vs_risk
capture_mode = prompt_last_token
layers = 34, 33, 32
target = risk seeds 1, 2, 3
eval_set = safety
```

扫描：

```text
alpha = 2.0
alpha = 5.0
alpha = 10.0
```

### 3.3 必须新增的诊断记录

生成时对每个被干预层记录：

```text
pre_projection = h @ direction
post_projection = (h + alpha * direction) @ direction
positive_projection_mean
negative_projection_mean
crossed_midpoint = post_projection > midpoint(pos_mean, neg_mean)
hidden_norm
delta_norm_ratio = ||alpha * direction|| / ||h||
```

输出建议：

```text
activation/results/4070ti/interventions/{run}/projection_diagnostics.csv
```

### 3.4 判读标准

如果：

```text
alpha=10 后 post_projection 已跨过 refusal-like 均值边界
但 unsafe/refusal 仍无变化
```

则说明该 direction 很可能只是诊断方向，不是行为控制方向。

如果：

```text
alpha=10 仍未跨过边界
```

则需要先调整归一化或干预尺度，不应直接判定 direction 无效。

## 4. B4.2：换层 sanity check

### 4.1 目的

检查是否选错了行为控制层。当前用的是 projection gap 最大层：

```text
34, 33, 32
```

但 selected layers 里还有 probe accuracy top layers：

```text
23, 24, 25
```

### 4.2 实验矩阵

```text
layers = 23, 24, 25
alpha = 1.0, 5.0
```

可选合并层：

```text
layers = 23, 24, 25, 32, 33, 34
alpha = 1.0, 5.0
```

### 4.3 判读标准

如果 probe layers 有效而 projection-gap layers 无效，说明生成控制层和离线最大分离层不同。

如果两组层都无效，则支持：

> prompt-last-token direction 不是直接行为控制方向。

## 5. B4.3：random / negative direction 对照

### 5.1 目的

防止把随机扰动、强扰动或退化误判为安全方向效果。

### 5.2 对照设置

对每个出现正向信号的配置，必须补：

```text
mode = safety
mode = random
mode = negative
mode = none
```

其中：

```text
safety:   h = h + alpha * direction
negative: h = h - alpha * direction
random:   h = h + alpha * random_unit_direction
none:     no intervention
```

### 5.3 成功条件

只有满足以下条件才算支持 activation safety direction：

```text
safety direction unsafe 下降 >= 0.10
safety direction refusal 上升 >= 0.10
random direction 明显弱于 safety direction
negative direction 不产生同向改善
invalid_rate <= 0.05
degradation_rate <= 0.05
新增 refusal 大多数为干净拒答
```

## 6. B4.4：换 capture mode

### 6.1 目的

如果 B4.1/B4.2 失败，则 prompt-last-token direction 可能只是输入表征分离。需要捕捉生成早期的行为分岔。

### 6.2 新 capture modes

优先级：

```text
first_generated_token
first_non_think_token
response_prefix_mean
```

第一轮只做：

```text
capture_mode = first_generated_token
```

### 6.3 数据构造

继续使用：

```text
positive = base/normal refusal-like
negative = risk unsafe-like
```

但 hidden state 改为生成第一个 token 时对应层的 hidden state。

### 6.4 最小闭环

```text
fit direction from first_generated_token
select top 3 layers by projection_gap
run risk seeds 1,2,3
alpha = 1.0, 5.0
include random/negative controls if any signal appears
```

## 7. 推荐执行顺序

### Step 1：alpha sweep

```text
layers = 34, 33, 32
alpha = 2.0, 5.0, 10.0
mode = safety
```

必须同时输出 projection diagnostics。

### Step 2：probe layers

```text
layers = 23, 24, 25
alpha = 1.0, 5.0
mode = safety
```

### Step 3：controls

只对 Step 1/2 中出现任意正向变化的配置补：

```text
mode = random
mode = negative
```

如果 Step 1/2 完全没有变化，可以直接进入 Step 4。

### Step 4：first generated token

```text
capture_mode = first_generated_token
alpha = 1.0, 5.0
mode = safety
```

具体执行计划见：

```text
experiments/step1_vanilla_baseline/interventions/B4_first_generated_token_plan.md
```

## 8. 停止条件

如果以下条件同时满足：

```text
alpha up to 10 无有效变化
layers 34/33/32 与 23/24/25 均无有效变化
first_generated_token direction 仍无有效变化
unsafe 下降 < 0.05
refusal 上升 < 0.05
或新增 refusal 多数为混合/伪拒答
```

则停止 activation-level post-hoc intervention 路线。

此时阶段结论应更新为：

> 参数谱几何和 prompt/generation activation directions 均能提供诊断性分离信号，但当前 post-hoc 干预不能稳定控制 unsafe behavior。

## 9. 不建议做的事

当前不建议：

```text
直接扩大到更多模型
直接做 SpectraSafe 主实验并声称机制闭环
大规模 alpha/layer 网格搜索
没有 random/negative control 的单臂实验
```

## 10. 预期产物

新增或更新：

```text
activation/configs/4070ti_qwen3_4b_activation_sanity.json
activation/results/4070ti/interventions/*/projection_diagnostics.csv
activation/results/4070ti/summaries/activation_sanity_vs_baseline.csv
activation/results/4070ti/reports/B4_activation_sanity_report.md
```

报告必须包含：

- alpha sweep 表；
- layer set 对比；
- projection crossing diagnostics；
- random/negative controls；
- local judge 聚合；
- 新增 refusal 人工抽查；
- 是否停止 activation-level post-hoc 路线的明确判断。
