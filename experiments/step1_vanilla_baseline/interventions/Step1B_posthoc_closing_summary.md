# Step 1B Post-hoc Causal Probe 关闭总结

## 0. 核心结论

Step 1B 已经完成当前协议下的 post-hoc 因果探针验证。结论是：

> LoRA safety drift 伴随稳定的谱结构、子空间和 activation 诊断信号，但当前尝试过的 post-hoc 参数空间干预和激活空间干预，都不能稳定控制 `risk` adapter 的 unsafe behavior。

因此，当前证据不支持如下强命题：

> 已经找到了导致 unsafe compliance 的可控因果方向或可切除谱结构。

更严谨的命题应降级为：

> 谱几何和相关子空间可以作为 LoRA safety drift 的诊断信号；但这些信号在当前简单后验干预下没有转化为可靠的行为控制机制。

这意味着 Step 1B 不应继续作为“因果机制证明”主线推进。下一阶段如果继续做 SpectraSafe，应转为 **Step 2A: training-time SpectraSafe pilot**，把 SpectraSafe 定位为受诊断信号启发的训练期正则化假设，并通过重新训练实验单独验证。

## 1. 已完成实验范围

Step 1B 覆盖了四类 post-hoc probe：

| 类别 | 实验 | 状态 | 结论 |
|---|---|---|---|
| 参数层级干预 | 全局 top-k LoRA-B 缩放/置零 | 已完成 | 最强改善仍很弱 |
| 模块级干预 | MLP-only / attention-only LoRA-B 置零 | 已完成 | 未形成稳定控制 |
| 方向级干预 | top singular direction removal | 已完成 | 无有效行为变化 |
| 子空间分析 | LoRA update left/right singular subspace overlap | 已完成 | 有诊断信号，但不等于控制点 |
| 激活层级干预 | prompt-last-token safety direction | 已完成 | 离线可分，生成时无稳定控制 |
| 激活时机补查 | strict first-generated-token safety direction | 已完成 | 离线可分，行为完全无变化 |

## 2. 行为锚点

4070 Ti / Qwen3-4B 三类 LoRA adapter 的 safety local judge 行为如下：

| family | seeds | unsafe mean | refusal mean | 备注 |
|---|---:|---:|---:|---|
| normal | 3 | 0.5833 | 0.4133 | 有一定拒答恢复 |
| risk | 3 | 1.0000 | 0.0000 | 稳定危险遵从，是主要干预靶点 |
| control | 3 | 0.6133 | 0.0300 | 方差很大，部分 seed 退化明显 |

`risk` adapter 是一个强靶点：三个 seed 均为 `unsafe=1.0000/refusal=0.0000/invalid=0.0000`。如果某个干预确实能恢复 safety，理论上应该能在这个靶点上看到明确下降。

## 3. 参数空间干预结果

### 3.1 全局 top-k LoRA-B 缩放/置零

已测试：

```text
k=10, scale=0.5
k=10, scale=0.25
k=10, scale=0.0
k=20, scale=0.0
k=40, scale=0.0
```

核心结果：

| setting | intervention | unsafe | refusal | delta unsafe |
|---|---|---:|---:|---:|
| k10 scale=0.5 | top | 1.0000 | 0.0000 | 0.0000 |
| k10 scale=0.25 | top | 0.9967 | 0.0033 | -0.0033 |
| k10 scale=0.0 | top | 1.0000 | 0.0000 | 0.0000 |
| k20 scale=0.0 | top | 0.9900 | 0.0100 | -0.0100 |
| k40 scale=0.0 | top | 0.9833 | 0.0167 | -0.0167 |

最强的全局干预是 `k=40, scale=0.0, top`，但 unsafe 只从 `1.0000` 降到 `0.9833`。这个幅度远低于预设的有效标准，也和 random/bottom 的差距很小。

判断：

> 高 `delta_spectral_norm` 层不是简单可切除的 unsafe 控制点。

### 3.2 模块级干预

已测试：

```text
MLP up_proj+gate_proj, k=10, scale=0.0
MLP up_proj+gate_proj, k=20, scale=0.0
attention-only, k=20, scale=0.0
```

结果：

| setting | unsafe | refusal | delta unsafe |
|---|---:|---:|---:|
| MLP k10 scale=0.0 | 1.0000 | 0.0000 | 0.0000 |
| MLP k20 scale=0.0 | 0.9900 | 0.0100 | -0.0100 |
| attention k20 scale=0.0 | 0.9833 | 0.0167 | -0.0167 |

判断：

> 模块级选择没有把失败原因定位到单独的 MLP 或 attention 模块；attention-only 的最好结果仍只有 1.67 个百分点改善。

### 3.3 Top singular direction removal

方向级干预结果：

| intervention | unsafe | refusal | delta unsafe |
|---|---:|---:|---:|
| bottom direction | 1.0000 | 0.0000 | 0.0000 |
| random direction | 0.9967 | 0.0033 | -0.0033 |
| top direction | 1.0000 | 0.0000 | 0.0000 |

判断：

> top singular direction 不是当前协议下的有效行为控制方向。

## 4. 子空间分析结果

B3c 的关键发现是：

| side | pair | top direction cosine | mean principal angle |
|---|---|---:|---:|
| left/output | risk seed-to-seed | 0.6259 | 50.09 |
| left/output | risk vs normal | 0.3087 | 71.80 |
| left/output | risk vs control | 0.2808 | 73.49 |
| right/input | risk seed-to-seed | 0.0936 | 84.61 |
| right/input | risk vs normal | 0.3072 | 69.02 |
| right/input | risk vs control | 0.3086 | 69.07 |

这说明 `risk` adapter 的 left/output singular subspace 在 seed 间更一致，并且和 normal/control 有明显区分。

但这个发现的边界也很清楚：

> 子空间一致性是诊断信号，不是已经验证的因果控制机制。

原因是，后续的 layer/module/direction 参数干预没有把这种结构差异转化为稳定的安全恢复。

## 5. 激活空间干预结果

### 5.1 Prompt-last-token direction

设置：

```text
direction = base_normal_vs_risk
capture_mode = prompt_last_token
target = risk seeds 1,2,3
```

核心结果：

| layers | alpha | mode | unsafe | refusal | delta unsafe |
|---|---:|---|---:|---:|---:|
| 34,33,32 | 1.0 | safety | 1.0000 | 0.0000 | 0.0000 |
| 34,33,32 | 2.0 | safety | 1.0000 | 0.0000 | 0.0000 |
| 34,33,32 | 5.0 | safety | 1.0000 | 0.0000 | 0.0000 |
| 34,33,32 | 10.0 | safety | 1.0000 | 0.0000 | 0.0000 |
| 23,24,25 | 5.0 | safety | 0.9933 | 0.0067 | -0.0067 |
| 23,24,25 | 5.0 | random | 0.9900 | 0.0100 | -0.0100 |
| 23,24,25 | 5.0 | negative | 0.9967 | 0.0033 | -0.0033 |

判断：

> prompt-last-token direction 有诊断分离能力，但 generation-time steering 没有稳定控制行为。probe layers 上的微弱改善甚至弱于 random control。

### 5.2 Strict first-generated-token direction

最新回传结果完成了 B4 Step 4。

设置：

```text
capture_mode = first_generated_token
direction = first_generated_base_normal_vs_risk
positive = base/normal refusal
negative = risk unsafe_compliance
target = risk seeds 1,2,3
layers = 34,33,32
alpha = 1.0, 5.0
```

一个重要诊断是：所有捕获到的 first generated token 都是 `<think>`。

| token | count |
|---|---:|
| `<think>` | 700 |

离线方向仍然强可分，top projection-gap layers 为：

| layer | projection gap | projection gap z | linear probe accuracy |
|---:|---:|---:|---:|
| 34 | 235.2745 | 10.9229 | 0.9808 |
| 33 | 218.8144 | 10.6766 | 0.9769 |
| 32 | 200.0978 | 9.8474 | 0.9769 |

但行为结果完全无变化：

| alpha | seeds | n | unsafe | refusal | invalid | degradation |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 1,2,3 | 300 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| 5.0 | 1,2,3 | 300 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

Projection diagnostics 也没有越过 offline midpoint：

| alpha | diagnostic rows | crossed midpoint | avg delta norm ratio |
|---:|---:|---:|---:|
| 1.0 | 900 | 0 | about 0.0024 |
| 5.0 | 900 | 0 | about 0.0120 |

判断：

> strict first-generated-token direction 仍然只是离线诊断信号，没有行为控制效果。并且由于第一个 token 全是 `<think>`，它更像是 thinking sentinel 状态检查，而不是实际语义输出决策点检查。

## 6. 总体解释

这些结果共同支持一个更保守但更严谨的解释：

1. `risk` adapter 的 unsafe behavior 是稳定的。
2. LoRA 更新中确实存在可复现的谱结构和 output-side 子空间差异。
3. activation 表征也能离线区分 refusal-like 与 unsafe-like 样本。
4. 但上述信号不是简单的单层、单模块、单方向或单时刻可控因果旋钮。
5. 当前失败更符合“分布式训练动力学结果”的图景，而不是“少数高谱层或一个安全方向可以后验切除”的图景。

因此，Step 1B 的负结果不是没有价值。它限定了论文叙事的边界：

> 我们可以说谱几何是 safety drift 的诊断信号；不能说已经证明谱几何是可直接操控的 unsafe 因果机制。

## 7. 当前阶段应停止什么

建议停止以下方向作为主线：

```text
继续扩大 post-hoc LoRA-B top-k 置零
继续试更多 module/layer top-k 组合
继续放大 prompt-last-token alpha
继续在没有新假设的情况下做 activation random search
把 SpectraSafe 写成已由 post-hoc 因果机制自然推出的方法
```

可选但不建议作为主线的最后 timing check：

```text
capture_mode = first_non_think_token
```

理由：strict first-generated-token 全是 `<think>`，所以如果团队非常想排除 timing 问题，可以检查 first non-think token。但这应被视为边界补查，而不是继续 activation intervention 主线的理由。

## 8. 下一步建议

进入：

```text
Step 2A: training-time SpectraSafe pilot
```

新的严谨叙事应是：

> Step 1 发现 LoRA safety drift 的谱/子空间诊断信号，并通过系统性 post-hoc probe 证明这些信号不是简单可切除控制点。因此，下一步不再做后验修补，而是在训练过程中加入谱约束，作为经验性正则化假设单独验证。

Step 2A 最小对照组建议：

| 方法 | 目的 |
|---|---|
| vanilla LoRA | 原始风险基线 |
| LoRA norm penalty / weight decay | 普通正则基线 |
| SpectraSafe mean penalty | 控制整体谱增长 |
| SpectraSafe max/top penalty | 控制局部谱尖峰 |
| SpectraSafe effective-rank penalty | 检查是否需要维持更新分散性 |

Step 2A 成功标准应同时看 safety 和 utility：

```text
unsafe compliance 明显下降
clean refusal 明显上升
invalid/degradation 不显著上升
normal utility 不被严重破坏
效果强于普通 LoRA norm penalty / weight decay
```

## 9. 推荐论文表述

可以使用的严谨表述：

> We find reproducible spectral and output-subspace signatures associated with LoRA safety drift. However, a sequence of post-hoc interventions in parameter space and activation space fails to reliably steer unsafe behavior, suggesting that these signatures are diagnostic rather than directly causal under simple local interventions.

中文版本：

> 我们发现 LoRA safety drift 伴随可复现的谱结构和输出子空间信号；但一系列参数空间与激活空间的后验干预都未能稳定控制 unsafe behavior，说明这些信号在当前协议下更适合作为诊断特征，而不是简单可操控的因果控制点。

SpectraSafe 的严谨定位：

> SpectraSafe should be presented as a training-time regularization hypothesis motivated by diagnostic spectral signatures, not as a method already derived from a verified post-hoc causal mechanism.

中文版本：

> SpectraSafe 应被表述为受谱诊断信号启发的训练期正则化假设，而不是已经由后验因果机制验证直接推出的方法。

