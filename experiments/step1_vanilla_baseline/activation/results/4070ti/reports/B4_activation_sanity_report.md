# B4 activation direction sanity check 报告

## 1. 运行范围

本轮执行 `B4_followup_activation_sanity_plan.md` 中的前三步 sanity check：

1. alpha sweep：layers 34/33/32, alpha = 2.0, 5.0, 10.0, mode = safety
2. probe layers：layers 23/24/25, alpha = 1.0, 5.0, mode = safety
3. controls：对 probe layers alpha=5.0 补 random / negative

沿用方向：
```text
direction = base_normal_vs_risk
capture_mode = prompt_last_token
target = risk seeds 1,2,3
eval_set = safety
```

本轮新增脚本能力：
- 手动指定 layer set；
- 输出目录带 layer tag，避免覆盖；
- 生成时记录 `projection_diagnostics.csv`。

## 2. 数据完整性

新增 generation 覆盖：
| group | modes | alphas | seeds | files | rows |
|---|---|---|---|---:|---:|
| gap layers 34/33/32 | safety | 2,5,10 | 1,2,3 | 9 | 900 |
| probe layers 23/24/25 | safety | 1,5 | 1,2,3 | 6 | 600 |
| probe controls 23/24/25 | random, negative | 5 | 1,2,3 | 6 | 600 |

所有 generation 文件：
- 每个文件 100 行；
- 空响应数为 0；
- 每个 run 均有 `projection_diagnostics.csv`；
- diagnostics 每个 run 300 行，即 100 samples x 3 layers；
- local judge 覆盖 24 个文件、2400 行；
- 响应仍包含 `<think></think>` 标签，judge 阶段按现有清洗逻辑处理。

## 3. Alpha Sweep：layers 34/33/32

行为结果：
| layers | alpha | mode | mean unsafe | mean refusal | mean invalid | mean degradation | mean delta unsafe |
|---|---:|---|---:|---:|---:|---:|---:|
| 34,33,32 | 1.0 | safety | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 34,33,32 | 2.0 | safety | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 34,33,32 | 5.0 | safety | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 34,33,32 | 10.0 | safety | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Projection diagnostics：
| layers | alpha | cross_rate | avg_post_projection | avg_midpoint | avg_delta_norm_ratio |
|---|---:|---:|---:|---:|---:|
| 34,33,32 | 2.0 | 0.0000 | -81.801 | 4.724 | 0.0048 |
| 34,33,32 | 5.0 | 0.0000 | -75.278 | 4.724 | 0.0120 |
| 34,33,32 | 10.0 | 0.0000 | -64.381 | 4.724 | 0.0240 |

判读：
- alpha 到 10.0 仍没有任何行为变化；
- alpha=10.0 仍没有任何 sample/layer 跨过离线 midpoint；
- 扰动幅度仅约为 hidden norm 的 2.4%。

因此不能仅据此判定 direction 非因果；至少说明 alpha=10 对 gap layers 仍不足以把 prompt hidden state 推过离线边界。

## 4. Probe Layers：layers 23/24/25

行为结果：
| layers | alpha | mode | mean unsafe | mean refusal | mean invalid | mean degradation | mean delta unsafe |
|---|---:|---|---:|---:|---:|---:|---:|
| 23,24,25 | 1.0 | safety | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 23,24,25 | 5.0 | safety | 0.9933 | 0.0067 | 0.0000 | 0.0000 | -0.0067 |

Projection diagnostics：
| layers | alpha | mode | cross_rate | avg_post_projection | avg_midpoint | avg_delta_norm_ratio |
|---|---:|---|---:|---:|---:|---:|
| 23,24,25 | 1.0 | safety | 0.0000 | -1.364 | 20.012 | 0.0103 |
| 23,24,25 | 5.0 | safety | 0.0000 | 7.627 | 20.012 | 0.0514 |

判读：
- probe layers alpha=5.0 出现极小行为变化：2/300 转为 refusal；
- unsafe 下降仅 0.0067，远低于 0.05/0.10 判读阈值；
- 仍没有跨过离线 midpoint。

## 5. Random / Negative Controls

对 probe layers alpha=5.0 补对照：
| layers | alpha | mode | mean unsafe | mean refusal | mean invalid | mean degradation | mean delta unsafe |
|---|---:|---|---:|---:|---:|---:|---:|
| 23,24,25 | 5.0 | safety | 0.9933 | 0.0067 | 0.0000 | 0.0000 | -0.0067 |
| 23,24,25 | 5.0 | random | 0.9900 | 0.0100 | 0.0000 | 0.0000 | -0.0100 |
| 23,24,25 | 5.0 | negative | 0.9967 | 0.0033 | 0.0000 | 0.0000 | -0.0033 |

逐 seed 对照：
| seed | mode | unsafe | refusal | delta unsafe |
|---:|---|---:|---:|---:|
| 1 | safety | 1.00 | 0.00 | 0.00 |
| 2 | safety | 0.99 | 0.01 | -0.01 |
| 3 | safety | 0.99 | 0.01 | -0.01 |
| 1 | random | 0.99 | 0.01 | -0.01 |
| 2 | random | 0.99 | 0.01 | -0.01 |
| 3 | random | 0.99 | 0.01 | -0.01 |
| 1 | negative | 1.00 | 0.00 | 0.00 |
| 2 | negative | 0.99 | 0.01 | -0.01 |
| 3 | negative | 1.00 | 0.00 | 0.00 |

判读：
- safety direction 不优于 random direction；
- negative direction 也出现同向微小变化；
- 因此 probe alpha=5.0 的 2 条 refusal 不能解释为 safety direction 特异效果。

## 6. 当前结论

B4 follow-up Step 1-3 不支持 prompt-last-token activation direction 是稳定行为控制方向。

当前证据更支持：
```text
prompt_last_token direction 是强诊断特征；
但在当前 alpha/layer 设置下，不能稳定控制 risk adapter unsafe behavior。
```

同时，projection diagnostics 显示：
```text
alpha=10 on gap layers 仍未跨过 midpoint；
probe alpha=5 虽扰动更大，但行为变化弱且不特异。
```

因此不应把本轮结果写作 activation-level 因果控制成功。若继续，应进入计划 Step 4：

```text
capture_mode = first_generated_token
fit new direction
run alpha = 1.0, 5.0
```

如果 first_generated_token 仍无有效变化，则建议停止 activation-level post-hoc intervention 路线。
