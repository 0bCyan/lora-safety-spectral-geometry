# B4 activation-level safety direction 最小闭环报告

## 1. 运行范围

本轮按 `B4_activation_safety_direction_plan.md` 执行 activation-level 最小闭环。

配置：
```text
experiments/step1_vanilla_baseline/activation/configs/4070ti_qwen3_4b_activation_safety_direction.json
```

执行内容：
- collect hidden states: base s0, normal s1/s2/s3, risk s1/s2/s3
- capture mode: prompt_last_token
- hidden state shape: 100 samples x 36 layers x 2560 hidden dim
- positive class: base/normal safety outputs labeled refusal
- negative class: risk safety outputs labeled unsafe_compliance
- direction: `base_normal_vs_risk`
- intervention target: risk s1/s2/s3
- intervention layers: 34, 33, 32
- alpha: 1.0
- eval_set: safety

## 2. Hidden State 与方向拟合

收集到的样本标签分布：
| source | refusal | unsafe | invalid |
|---|---:|---:|---:|
| base s0 | 6 | 91 | 3 |
| normal s1 | 46 | 54 | 0 |
| normal s2 | 25 | 74 | 1 |
| normal s3 | 53 | 47 | 0 |
| risk s1 | 0 | 100 | 0 |
| risk s2 | 0 | 100 | 0 |
| risk s3 | 0 | 100 | 0 |

方向拟合使用：
| class | n |
|---|---:|
| positive refusal-like | 130 |
| negative unsafe-like | 300 |

按 projection gap 最大的 top 3 层：
| rank | layer | projection_gap | projection_gap_z | linear_probe_accuracy |
|---:|---:|---:|---:|---:|
| 1 | 34 | 210.5331 | 11.2389 | 1.0000 |
| 2 | 33 | 179.5214 | 11.6444 | 1.0000 |
| 3 | 32 | 155.1716 | 11.7375 | 1.0000 |

说明：prompt-last-token hidden states 上 refusal-like 与 risk unsafe-like 的线性分离很强，但这只是离线表征分离，不等价于生成时可控。

## 3. 干预结果

生成完整性：
- 3 个 generation 文件；
- 每个文件 100 行；
- 空响应数为 0；
- 响应仍包含 `<think></think>` 标签，judge 阶段按现有清洗逻辑处理；
- local judge 覆盖 300 行；
- invalid/degradation 均为 0。

逐 seed 结果：
| seed | mode | alpha | layers | unsafe | refusal | invalid | degradation | delta_unsafe |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | safety | 1.0 | 34,33,32 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | safety | 1.0 | 34,33,32 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | safety | 1.0 | 34,33,32 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |

聚合结果：
| setting | mean unsafe | mean refusal | mean invalid | mean degradation | mean delta unsafe |
|---|---:|---:|---:|---:|---:|
| D_base_normal_vs_risk, alpha=1.0, layers=34/33/32 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 4. 判读

B4 成功标准要求：
```text
unsafe_compliance_rate 下降 >= 0.10
refusal_rate 上升 >= 0.10
invalid_rate <= 0.05
degradation_rate <= 0.05
```

当前最小闭环不满足成功标准：
- unsafe 没有下降；
- refusal 没有上升；
- 输出没有退化，因此不是“干预太强导致崩坏”，而是 alpha=1.0 下没有行为控制效果。

## 5. 结论

`prompt_last_token` activation direction 在离线 hidden state 上可强分离 refusal-like 与 risk unsafe-like 样本，但将该方向以 `alpha=1.0` 加到 risk adapter 生成过程的 top projection-gap layers 后，没有改变 safety 行为。

本轮结果不支持 B4 最小 activation intervention 已经形成有效安全控制点。若继续 B4，应只做小规模扩展验证，例如：
- alpha=2.0；
- top_by_probe_accuracy layers 23/24/25；
- random/negative direction 对照；
- 或改为生成早期 token / response-prefix activation direction。

在扩展前，不建议把 activation direction 作为已验证的机制闭环证据。
