# 第一步实验机制分析

本目录保存 4070 Ti pilot 的层级和模块级分析产物。

## 目录

```text
scripts/       分析脚本
tables/        可复用表格
figures/       论文风格图表
reports/       中文分析报告
```

## 当前分析目标

1. 将谱指标整理为标准长表。
2. 比较 normal、risk、control 三类 adapter 的层级差异。
3. 比较 attention 和 MLP 模块差异。
4. 判断全局谱均值不足以解释安全退化后，是否存在更细粒度结构。

## 运行方式

```bash
python experiments/step1_vanilla_baseline/analysis/scripts/build_spectral_analysis.py
```

输出：

```text
experiments/step1_vanilla_baseline/analysis/tables/
experiments/step1_vanilla_baseline/analysis/figures/
experiments/step1_vanilla_baseline/analysis/reports/
```

