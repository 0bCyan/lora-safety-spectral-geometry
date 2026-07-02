# B4 Step 4：First-generated-token Activation Direction 计划

## 1. 背景

B4 prompt-last-token activation direction 已完成两轮：

```text
minimal loop:
  layers = 34,33,32
  alpha = 1.0

sanity check:
  gap layers = 34,33,32
  alpha = 2.0, 5.0, 10.0
  probe layers = 23,24,25
  alpha = 1.0, 5.0
  controls = random / negative on probe layers alpha=5.0
```

结论：

```text
prompt-last-token direction 离线可强分离 refusal-like vs risk unsafe-like。
但当前 generation-time intervention 没有稳定改变 unsafe/refusal。
```

关键观察：

- gap layers `34/33/32` 即使 `alpha=10.0`，unsafe 仍为 `1.0000`；
- probe layers `23/24/25, alpha=5.0` 只有 `0.0067` unsafe 下降；
- random direction 在同设置下反而达到 `0.0100` unsafe 下降；
- 少量 local-judge refusal 经抽查仍多为混合或伪拒答；
- projection diagnostics 显示多数设置没有跨过离线 midpoint。

因此，prompt-last-token direction 更像诊断特征，而不是已验证的生成行为控制方向。

## 2. 本轮目标

B4 Step 4 只验证最后一个关键可能性：

> 拒答/危险遵从的行为分岔是否发生在生成早期 token，而不是 prompt-last-token 表征中？

如果 first-generated-token direction 仍无效，则建议停止 activation-level post-hoc intervention 路线。

## 3. 核心假设

当前失败可能来自 capture timing：

```text
prompt_last_token:
  反映输入表征差异，但未必是生成决策状态。

first_generated_token:
  更接近模型开始选择 refusal 或 unsafe continuation 的分岔点。
```

B4 Step 4 的目的就是检验这个时机假设。

## 4. 数据与对象

沿用 4070 Ti pilot：

```text
model = Qwen3-4B
machine = 4070 Ti
target_family = risk
target_seeds = 1, 2, 3
eval_set = safety
```

方向拟合数据：

```text
positive = base/normal outputs labeled refusal
negative = risk outputs labeled unsafe_compliance
```

继续排除：

```text
invalid_or_irrelevant
degradation
明显混合/伪拒答样本
```

## 5. B4.4a：收集 first-generated-token hidden states

### 5.1 Capture definition

对每个 safety prompt 执行一次 generation，并捕捉第一个生成 token 时各层 hidden state。

建议定义：

```text
capture_mode = first_generated_token
position = generated token index 0
layers = all transformer layers
```

如果实现上第一个 token 仍是 `<think>` 或空白样式 token，可记录 token 文本和 token id，后续再决定是否升级到 `first_non_think_token`。

### 5.2 输出

```text
activation/results/4070ti/hidden_states_first_generated/
  {experiment_id}_safety_first_generated_token.pt
  {experiment_id}_safety_first_generated_token_metadata.jsonl
```

metadata 必须包含：

```text
sample_id
generated_token_id
generated_token_text
family
seed
label
prompt
```

## 6. B4.4b：拟合 first-generated-token direction

方向：

```text
D_first_base_normal_vs_risk =
  mean(first_token_hidden_refusal_like)
  - mean(first_token_hidden_risk_unsafe_like)
```

保存：

```text
activation/results/4070ti/directions/first_generated_base_normal_vs_risk.pt
activation/results/4070ti/directions/first_generated_base_normal_vs_risk_layer_metrics.csv
activation/results/4070ti/directions/first_generated_base_normal_vs_risk_selected_layers.json
```

层选择：

```text
top 3 by projection_gap
top 3 by linear_probe_accuracy
```

## 7. B4.4c：最小干预矩阵

### 7.1 Safety direction

第一轮只跑：

```text
mode = safety
alpha = 1.0, 5.0
layers = top 3 by projection_gap
target = risk seeds 1,2,3
```

### 7.2 Controls

如果 safety direction 出现任意正向变化，再补：

```text
mode = random
mode = negative
alpha = same as effective safety setting
layers = same layers
```

如果 safety direction 完全无变化，则无需先跑大规模 controls，可以直接判定本轮最小验证失败。

### 7.3 Diagnostics

每个 run 必须记录：

```text
projection_diagnostics.csv
```

字段沿用 sanity check：

```text
sample_id
seed
mode
alpha
layer
pre_projection
post_projection
positive_projection_mean
negative_projection_mean
projection_midpoint
crossed_midpoint
hidden_norm
delta_norm_ratio
```

## 8. 建议配置

新增：

```text
activation/configs/4070ti_qwen3_4b_activation_first_generated.json
```

建议字段：

```text
name = 4070ti_qwen3_4b_activation_first_generated
source_config = activation/configs/4070ti_qwen3_4b_activation_safety_direction.json
capture_mode = first_generated_token
direction_name = first_generated_base_normal_vs_risk
target_family = risk
seeds = [1, 2, 3]
eval_set = safety
alphas = [1.0, 5.0]
layer_source = top_by_projection_gap
output_dir = experiments/step1_vanilla_baseline/activation/results/4070ti
```

## 9. 成功标准

支持 first-generated-token activation direction 需要：

```text
unsafe_compliance_rate 下降 >= 0.10
refusal_rate 上升 >= 0.10
invalid_rate <= 0.05
degradation_rate <= 0.05
random direction 明显弱于 safety direction
negative direction 不产生同向改善
新增 refusal 经人工抽查大多为干净拒答
```

弱信号标准：

```text
unsafe 下降 >= 0.05
且 safety direction 明显强于 random/negative
```

低于弱信号标准不应继续扩网格。

## 10. 失败标准与停止条件

如果：

```text
alpha = 1.0, 5.0 均 unsafe 下降 < 0.05
或 safety direction 不优于 random/negative
或新增 refusal 多为混合/伪拒答
```

则停止 activation-level post-hoc intervention 路线。

阶段结论更新为：

> 参数谱几何、参数子空间、prompt-last-token activation direction、first-generated-token activation direction 均可提供诊断性分离信号或弱扰动信号，但当前 post-hoc intervention 不能稳定控制 risk unsafe behavior。

## 11. 不建议继续做的扩展

若本轮失败，不建议继续：

```text
alpha > 10 的盲目放大
更多 layer 网格搜索
更多 prompt-last-token 变体
没有 random/negative control 的单臂实验
```

应转向：

```text
训练期正则经验对照
更强 judge/API judge
更大模型上的描述性验证
论文负结果总结
```

## 12. 预期产物

```text
activation/configs/4070ti_qwen3_4b_activation_first_generated.json
activation/results/4070ti/directions/first_generated_base_normal_vs_risk_layer_metrics.csv
activation/results/4070ti/directions/first_generated_base_normal_vs_risk_selected_layers.json
activation/results/4070ti/interventions/*/projection_diagnostics.csv
activation/results/4070ti/summaries/activation_first_generated_vs_baseline.csv
activation/results/4070ti/reports/B4_first_generated_token_report.md
```

报告必须包含：

- first-generated-token direction 的离线分离指标；
- alpha=1/5 的 behavior summary；
- projection crossing diagnostics；
- random/negative controls，如果触发；
- 新增 refusal 人工抽查；
- 是否停止 activation-level post-hoc intervention 的明确判断。
