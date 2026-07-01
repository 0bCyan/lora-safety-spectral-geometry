# B2 全局 top-k LoRA-B 缩放总结

## 1. 已完成矩阵

B2 使用已有 Qwen3-4B risk adapter，在 4070 Ti 上完成了以下 post-hoc LoRA-B 缩放实验：

| setting | top unsafe | random unsafe | bottom unsafe | top vs random gap |
|---|---:|---:|---:|---:|
| k10 scale=0.5 | 1.0000 | 0.9967 | 1.0000 | -0.0033 |
| k10 scale=0.25 | 0.9967 | 0.9967 | 1.0000 | 0.0000 |
| k10 scale=0.0 | 1.0000 | 0.9933 | 1.0000 | -0.0067 |
| k20 scale=0.0 | 0.9900 | 0.9933 | 1.0000 | 0.0033 |
| k40 scale=0.0 | 0.9833 | 0.9967 | 0.9967 | 0.0134 |

所有实验均只覆盖：

```text
family = risk
seeds = 1, 2, 3
eval_set = safety
judge = local safety judge
```

## 2. 停止条件

B2 预设停止条件为：

```text
k=40, scale=0.0 后 top unsafe rate 仍 >= 0.95
且 top/bottom/random 差距仍 < 0.05
```

当前 `k40 scale=0.0` 结果满足该条件：

- top unsafe = 0.9833；
- bottom unsafe = 0.9967；
- random unsafe = 0.9967；
- top 相比 bottom/random 只低约 0.0133；
- invalid 与 degradation 均为 0。

因此，B2 判定为：

> 全局 top-k LoRA-B 缩放不是当前 risk adapter 危险遵从行为的有效控制坐标。

## 3. 解释边界

该负结果不说明谱结构没有价值。它只说明：

1. 按全局 `delta_spectral_norm` 排序选择 LoRA 模块；
2. 对选中模块的 LoRA-B 做整体缩放或置零；
3. 在 Qwen3-4B risk adapter、local safety judge、100 条 safety eval 下；
4. 不能稳定恢复拒答或显著降低危险遵从。

更细粒度的机制坐标仍未排除，包括：

- 模块族内选择；
- top singular direction；
- adapter update subspace；
- activation-level safety direction。

## 4. 对 SpectraSafe 的影响

当前结果不足以支持直接进入：

```text
SpectraSafe-global
SpectraSafe-layer-aware
SpectraSafe-direction-aware
```

进入 SpectraSafe 的最低条件应调整为：

```text
B3b direction intervention 出现 top > random 的明确行为效果
或 B3c subspace analysis 显示 risk 存在稳定 family-specific 方向/子空间结构
```

如果 B3 仍无信号，论文路线应调整为：

> LoRA 更新谱几何可以描述 adapter 差异，但简单谱坐标不足以因果控制 unsafe behavior。

## 5. 下一步

B2 关闭后进入 B3：

1. B3a：模块级 sanity check；
2. B3b：top singular direction intervention；
3. B3c：adapter update subspace analysis。

其中 B3a 是当前最低成本下一步：

```text
MLP up_proj+gate_proj, k=10, scale=0.0
MLP up_proj+gate_proj, k=20, scale=0.0
attention only, k=20, scale=0.0
```
