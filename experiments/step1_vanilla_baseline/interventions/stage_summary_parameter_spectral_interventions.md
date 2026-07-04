# 阶段总结：参数谱几何干预结果

## 0. 阶段性结论

本阶段围绕 Qwen3-4B LoRA safety drift 的参数谱几何机制进行了系统的 post-hoc 干预验证。实验覆盖了全局高谱层缩放、模块级缩放、top singular direction removal，以及 adapter update subspace analysis。总体结果表明，LoRA 参数谱几何能够稳定描述不同训练数据 family 诱导出的 adapter 差异，尤其是 `risk` adapter 在 output-side singular subspace 上表现出较强的 seed-to-seed 一致性，并与 `normal/control` adapter 明显分离。然而，这种几何差异目前没有转化为稳定的行为可控性。

具体而言，`risk` adapter 在 safety eval 上表现为稳定的危险遵从，三组 seed 的 unsafe compliance rate 均为 1.00。针对该稳定靶点，我们依次对 `delta_spectral_norm` 最高的 LoRA 模块进行 `k=10/20/40` 的 LoRA-B 缩放或置零，对 MLP 与 attention 模块族进行单独置零，并进一步移除选中模块的 top singular direction。所有这些 post-hoc 参数空间干预都未能带来显著安全恢复：最强的 `k=40, scale=0.0` 全局 top-k 干预仅将 unsafe compliance rate 从 1.0000 降至 0.9833，模块级 attention-only `k=20` 也仅达到相同量级，而 top singular direction removal 对 top-k 模块没有任何可观测改善。

因此，当前证据不支持“高谱 LoRA 层、特定模块族，或 top singular direction 是危险遵从行为的充分控制点”这一强机制假设。更稳妥的解释是：参数谱几何捕捉到了 adapter family 的训练后结构差异，但这些差异更多是描述性诊断信号，而不是简单可干预的行为因果坐标。换言之，`risk` adapter 的 unsafe behavior 不能通过当前这些局部参数空间操作稳定恢复为拒答行为。

这一结论对后续方法设计有直接影响。SpectraSafe 目前不应被表述为已经由 post-hoc 因果机制验证自然推出的方法。若继续推进 SpectraSafe，应将其定位为训练期经验正则或安全-效用权衡方法，并通过与 vanilla、mean-only、max-only、weight decay / LoRA norm penalty 等基线的直接训练对照来验证，而不是依赖当前参数干预结果作为机制闭环证据。若研究目标继续偏向机制解释，则下一步应转向 activation-level safety direction 或更强行为评估，而不是继续扩大同类参数空间缩放实验。

本阶段最终结论可以概括为：

> 参数谱几何可以描述 LoRA 安全漂移相关的 adapter 差异，但当前简单的层级、模块级和方向级参数空间干预不能稳定控制 unsafe behavior。因此，谱几何在现阶段更适合作为诊断与分析工具，而不是已经验证的安全行为控制机制。

## 1. 一句话结论

当前结果支持：

> LoRA 参数谱几何可以描述不同数据 family 的 adapter 差异，尤其是 `risk` adapter 的 output-side 子空间结构。

但当前结果不支持：

> 简单的参数空间谱坐标，例如全局 top-k 层、模块级 top-k、top singular direction，可以稳定控制 `risk` adapter 的危险遵从行为。

因此，现阶段不应把 SpectraSafe 叙述为已经由因果机制自然推出的方法。更稳妥的定位是：

> 谱几何是有诊断价值的描述信号；训练期正则是否有效仍需单独验证，不能由当前 post-hoc 干预结果直接推出。

## 2. 行为基线

4070 Ti 上的 safety local judge 基线如下：

| family | seed | unsafe | refusal | invalid |
|---|---:|---:|---:|---:|
| base | 0 | 0.91 | 0.06 | 0.03 |
| normal | 1 | 0.54 | 0.46 | 0.00 |
| normal | 2 | 0.74 | 0.25 | 0.01 |
| normal | 3 | 0.47 | 0.53 | 0.00 |
| control | 1 | 0.76 | 0.06 | 0.18 |
| control | 2 | 0.98 | 0.01 | 0.01 |
| control | 3 | 0.10 | 0.02 | 0.88 |
| risk | 1 | 1.00 | 0.00 | 0.00 |
| risk | 2 | 1.00 | 0.00 | 0.00 |
| risk | 3 | 1.00 | 0.00 | 0.00 |

`risk` adapter 是最干净的干预靶点：三 seed 都是 100% unsafe，且没有 invalid 干扰。

## 3. 全局 top-k LoRA-B 缩放

### 3.1 实验矩阵

全局 top-k 按 `delta_spectral_norm` 选择 LoRA 模块，并缩放或置零对应 LoRA-B：

```text
k=10, scale=0.5
k=10, scale=0.25
k=10, scale=0.0
k=20, scale=0.0
k=40, scale=0.0
```

对照：

```text
top-k
random-k
bottom-k
```

### 3.2 聚合结果

| setting | intervention | unsafe | refusal | delta unsafe |
|---|---|---:|---:|---:|
| k10 scale=0.5 | bottom | 1.0000 | 0.0000 | 0.0000 |
| k10 scale=0.5 | random | 0.9967 | 0.0033 | -0.0033 |
| k10 scale=0.5 | top | 1.0000 | 0.0000 | 0.0000 |
| k10 scale=0.25 | bottom | 1.0000 | 0.0000 | 0.0000 |
| k10 scale=0.25 | random | 0.9967 | 0.0033 | -0.0033 |
| k10 scale=0.25 | top | 0.9967 | 0.0033 | -0.0033 |
| k10 scale=0.0 | bottom | 1.0000 | 0.0000 | 0.0000 |
| k10 scale=0.0 | random | 0.9933 | 0.0067 | -0.0067 |
| k10 scale=0.0 | top | 1.0000 | 0.0000 | 0.0000 |
| k20 scale=0.0 | bottom | 1.0000 | 0.0000 | 0.0000 |
| k20 scale=0.0 | random | 0.9933 | 0.0067 | -0.0067 |
| k20 scale=0.0 | top | 0.9900 | 0.0100 | -0.0100 |
| k40 scale=0.0 | bottom | 0.9967 | 0.0033 | -0.0033 |
| k40 scale=0.0 | random | 0.9967 | 0.0033 | -0.0033 |
| k40 scale=0.0 | top | 0.9833 | 0.0167 | -0.0167 |

### 3.3 判读

最强的全局干预是 `k=40, scale=0.0`，但结果仍然很弱：

```text
top unsafe = 0.9833
top vs random gap = 0.0134
top vs bottom gap = 0.0134
```

这低于预设有效标准：

```text
unsafe 下降 >= 0.10
top/random 差距 >= 0.05
```

因此，全局 top-k LoRA-B 缩放路线触发停止条件。

## 4. 模块级 LoRA-B 置零

### 4.1 实验矩阵

模块级实验用于排除“全局 top-k 把模块族混在一起”的可能：

```text
MLP up_proj+gate_proj, k=10, scale=0.0
MLP up_proj+gate_proj, k=20, scale=0.0
attention-only, k=20, scale=0.0
```

### 4.2 结果

| setting | unsafe | refusal | delta unsafe |
|---|---:|---:|---:|
| MLP k10 scale=0.0 | 1.0000 | 0.0000 | 0.0000 |
| MLP k20 scale=0.0 | 0.9900 | 0.0100 | -0.0100 |
| attention k20 scale=0.0 | 0.9833 | 0.0167 | -0.0167 |

### 4.3 判读

模块级拆分没有带来强效果。attention-only k20 与全局 k40 top 的量级相近，但仍只有 1.67 个百分点改善。

这说明 B2 的失败不是单纯由 MLP/attention 混合造成的。模块级 LoRA-B 置零也不是稳定控制坐标。

## 5. Top singular direction intervention

### 5.1 实验矩阵

B3b 对选中 LoRA 模块计算：

```text
Delta W = B @ A
```

并在推理时移除 top singular component：

```text
direction_rank = 1
gamma = 1.0
k = 10
intervention = top, random, bottom
```

### 5.2 结果

| intervention | unsafe | refusal | delta unsafe |
|---|---:|---:|---:|
| bottom direction | 1.0000 | 0.0000 | 0.0000 |
| random direction | 0.9967 | 0.0033 | -0.0033 |
| top direction | 1.0000 | 0.0000 | 0.0000 |

### 5.3 判读

top singular direction removal 没有产生行为效果。它甚至弱于 random direction 的偶然 1 条变化。

因此，当前最小方向级矩阵不支持：

> top singular direction 是危险遵从行为的主要控制方向。

## 6. Local judge 的保守解释

少数被 local judge 标为 `refusal` 的样本，人工抽查后并不总是干净拒答。常见模式是：

```text
开头出现 dangerous / illegal / not recommended
后续仍给出泛化建议、风险规避、咨询专家、避免被抓等内容
```

因此，表格中的 refusal rate 可能偏乐观。实际安全恢复比聚合表更弱。

这进一步削弱了 post-hoc 参数干预的正向解释。

## 7. 子空间分析

### 7.1 分析对象

B3c 比较不同 adapter 的 LoRA 更新子空间：

```text
risk vs risk
normal vs normal
control vs control
risk vs normal
risk vs control
normal vs control
```

比较对象包括：

```text
left/output singular subspace
right/input singular subspace
k = 1, 2, 4
```

### 7.2 关键结果：all modules, k=1

| side | pair | top direction cosine | mean principal angle |
|---|---|---:|---:|
| left/output | risk seed-to-seed | 0.6259 | 50.09 |
| left/output | risk vs normal | 0.3087 | 71.80 |
| left/output | risk vs control | 0.2808 | 73.49 |
| right/input | risk seed-to-seed | 0.0936 | 84.61 |
| right/input | risk vs normal | 0.3072 | 69.02 |
| right/input | risk vs control | 0.3086 | 69.07 |

### 7.3 模块级 left/output 信号

`risk` 的 left/output seed-to-seed overlap 在所有模块上都高于 risk vs normal/control：

| module | risk seed-to-seed | risk vs normal | risk vs control |
|---|---:|---:|---:|
| q_proj | 0.5370 | 0.2347 | 0.2185 |
| k_proj | 0.5432 | 0.2373 | 0.2231 |
| v_proj | 0.7305 | 0.4301 | 0.3887 |
| o_proj | 0.7370 | 0.3951 | 0.3514 |
| gate_proj | 0.5432 | 0.2349 | 0.2207 |
| up_proj | 0.6374 | 0.2823 | 0.2572 |
| down_proj | 0.6534 | 0.3468 | 0.3058 |

### 7.4 判读

B3c 给出了一个真实的描述性信号：

> `risk` adapter 的 output-side 更新方向在 seed 间更一致，并且和 `normal/control` 分开。

但这个信号是混合的：

- left/output side 有 family-specific 结构；
- right/input side 中，risk seed-to-seed overlap 很低，甚至低于 risk vs normal/control；
- 行为干预没有证明这些子空间方向可以控制 unsafe behavior。

因此，B3c 支持“谱几何可描述 family 差异”，但不足以支持“谱几何是可干预的安全行为因果机制”。

## 8. 当前证据链

目前证据链可以写成：

```text
risk adapter 行为上稳定 unsafe
        |
        v
参数谱指标和子空间指标能区分 family
        |
        v
全局 top-k 缩放失败
        |
        v
模块级缩放失败
        |
        v
top singular direction removal 失败
        |
        v
子空间分析有描述性信号，但不能推翻干预负结果
```

所以最稳妥的阶段结论是：

> 参数谱几何是 LoRA 安全漂移的描述性诊断工具，但当前简单参数空间干预不能稳定恢复安全行为。

## 9. 对原始假设的影响

### 9.1 被削弱的假设

以下假设被明显削弱：

```text
高 delta_spectral_norm 层是危险行为控制点。
MLP high-spectrum modules 是危险行为控制点。
top singular direction 是危险行为控制方向。
```

### 9.2 仍然保留的假设

以下假设仍然可能成立：

```text
LoRA 参数几何可以作为 adapter family 的诊断信号。
输出侧子空间结构可能反映 risk 数据训练出的稳定更新模式。
训练期正则可能改变学习路径，但这需要重新实验验证。
行为控制可能更接近 activation space，而不是参数空间单点干预。
```

## 10. 对 SpectraSafe 的影响

当前不建议直接进入 SpectraSafe 主实验，尤其不建议使用下面的强叙事：

> 我们发现了安全漂移的谱因果机制，因此设计 SpectraSafe。

更稳妥的叙事是：

> 我们发现参数谱几何能描述 LoRA adapter family 差异，但 post-hoc 干预显示简单谱坐标不是充分控制点。SpectraSafe 若继续推进，应作为训练期经验正则进行独立验证，而不是当前机制结果的直接推论。

进入 SpectraSafe 前至少需要新增一个证据：

1. activation-level safety direction 能稳定影响 unsafe/refusal；
2. 或训练期 SpectraSafe 相比普通正则有清晰安全-效用优势；
3. 或更强 judge/API judge 复评确认当前 local judge 信号不是误判。

## 11. 建议下一步

### 路线 A：阶段性收束

先停止 4070 上的 post-hoc 参数干预扩展，整理成论文中的机制探索章节：

```text
描述性谱几何：成立
简单参数干预因果控制：不成立
```

这是当前最稳妥、最诚实的路线。

### 路线 B：转向 activation-level

如果继续寻找因果机制，下一步应做 activation-level safety direction：

```text
收集 hidden states
对 unsafe-like vs refusal-like 响应做 contrast
拟合 safety direction
生成时在 activation 上干预
```

这更接近行为层面，但工程成本更高。

具体计划见：

```text
experiments/step1_vanilla_baseline/interventions/B4_activation_safety_direction_plan.md
```

### 路线 C：训练期正则作为经验方法

可以继续做 SpectraSafe，但要降级为经验方法：

```text
SpectraSafe-global
mean-only
max-only
weight decay / LoRA norm penalty
vanilla
```

并重点验证安全-效用 tradeoff，而不是声称已经证明机制闭环。

## 12. 当前推荐决策

建议先走路线 A：

> 收束并冻结当前 4070 post-hoc 参数干预结果，写成阶段性机制结论。

随后再二选一：

1. 如果目标是机制论文，转向 activation-level；
2. 如果目标是方法论文，进入训练期正则对照，但降低 SpectraSafe 的机制宣称。
