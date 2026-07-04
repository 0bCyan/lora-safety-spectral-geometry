# 第一步实验：普通 LoRA 安全漂移基线

> Final summary: `experiments/step1_vanilla_baseline/Step1_final_summary.md`
> Step 1B closing summary: `experiments/step1_vanilla_baseline/interventions/Step1B_posthoc_closing_summary.md`

## 1. 实验目的

第一步实验只研究普通 LoRA，不加入 SpectraSafe 或其他新方法。

核心问题：

> 普通 LoRA 微调后，如果模型安全性发生变化，参数变化中是否存在稳定规律？

本实验不追求最终性能，也不证明新方法有效。它只回答后续研究是否值得继续：

1. 不同数据类型训练出的 adapter 是否有不同参数变化模式。
2. 参数变化是否和安全评估结果对应。
3. 哪些层或模块变化更明显。

## 2. 分支与目录

当前实验分支：

```text
exp/step1-vanilla-baseline
```

当前实验目录：

```text
experiments/step1_vanilla_baseline/
```

4070 Ti 和 3090 Ti 共用本分支、本目录、本数据池，只切换配置文件。

## 3. 数据

统一数据池：

```text
data/processed/unified_pool/
```

三类训练数据：

- normal：正常任务数据；
- risk：风险数据；
- control：打乱配对对照数据。

评估数据：

- eval_safety：安全评估；
- eval_utility：正常任务评估。

## 4. 机器分工

### 4.1 4070 Ti

配置文件：

```text
configs/4070ti_qwen3_4b.json
```

任务：

- 跑通训练流程；
- 训练 Qwen3-4B 小规模普通 LoRA；
- 生成 adapter；
- 计算参数变化指标；
- 小批量生成评估回答。

不用于正式结论。

### 4.2 3090 Ti

配置文件：

```text
configs/3090ti_qwen3_8b.json
```

任务：

- 训练 Qwen3-8B 普通 LoRA；
- 每类数据跑多个 seed；
- 计算参数变化指标；
- 批量生成待 judge 回答；
- 产出第一轮机制初筛结果。

## 5. 运行顺序

### 5.1 环境检查

```bash
python experiments/step1_vanilla_baseline/scripts/check_env.py
```

每台机器第一次运行前，必须保存环境记录。

4070 Ti：

```bash
python experiments/step1_vanilla_baseline/scripts/check_env.py \
  > experiments/step1_vanilla_baseline/results/4070ti/env_report.txt
```

3090 Ti：

```bash
python experiments/step1_vanilla_baseline/scripts/check_env.py \
  > experiments/step1_vanilla_baseline/results/3090ti/env_report.txt
```

环境与版本规则见：

```text
docs/环境与版本说明.md
```

### 5.2 生成实验索引

4070 Ti：

```bash
python experiments/step1_vanilla_baseline/scripts/make_experiment_index.py \
  --config experiments/step1_vanilla_baseline/configs/4070ti_qwen3_4b.json
```

3090 Ti：

```bash
python experiments/step1_vanilla_baseline/scripts/make_experiment_index.py \
  --config experiments/step1_vanilla_baseline/configs/3090ti_qwen3_8b.json
```

### 5.3 训练普通 LoRA

```bash
python experiments/step1_vanilla_baseline/scripts/train_vanilla_lora.py \
  --config experiments/step1_vanilla_baseline/configs/4070ti_qwen3_4b.json \
  --family normal \
  --seed 1
```

把 `family` 换成：

```text
normal
risk
control
```

3090 Ti 使用自己的配置文件。

### 5.4 计算参数变化指标

```bash
python experiments/step1_vanilla_baseline/scripts/analyze_adapter.py \
  --config experiments/step1_vanilla_baseline/configs/4070ti_qwen3_4b.json \
  --family normal \
  --seed 1
```

### 5.5 生成评估回答

```bash
python experiments/step1_vanilla_baseline/scripts/generate_responses.py \
  --config experiments/step1_vanilla_baseline/configs/4070ti_qwen3_4b.json \
  --family normal \
  --seed 1 \
  --eval-set safety
```

API judge 暂不放在本实验第一批必跑步骤中，先生成回答并缓存。

## 6. 结果位置

本实验所有结果放在：

```text
experiments/step1_vanilla_baseline/results/
```

建议结构：

```text
results/
  experiment_index.csv
  4070ti/
    adapters/
    generations/
    spectral_metrics/
    logs/
  3090ti/
    adapters/
    generations/
    spectral_metrics/
    logs/
```

注意：

- adapter 权重不提交 git；
- 大规模生成结果不提交 git；
- 轻量汇总表可以提交；
- 原始日志视大小决定是否提交。

数据回传规则见：

```text
experiments/step1_vanilla_baseline/数据回传流程.md
```

简要原则：

- 显卡机训练 adapter；
- 显卡机本地完成参数变化提取；
- 本机只接收环境记录、实验索引、参数指标和必要的生成回答；
- adapter 默认不回传、不提交。

## 7. 成功标准

4070 Ti 成功标准：

- 三类数据至少各跑通一个 adapter；
- adapter 能保存；
- 参数变化指标能导出；
- 评估回答能生成。

3090 Ti 成功标准：

- 三类数据至少各跑 3 个 seed；
- 每个 adapter 都有参数变化指标；
- 有安全评估和正常任务评估回答；
- 能初步比较 normal、risk、control 三类 adapter 的参数变化差异。

## 8. 暂不做的事

本实验暂不做：

- SpectraSafe；
- mean-only / max-only；
- 强安全基线；
- 多模型正式对比；
- 大规模 API judge；
- 论文主结果宣称。

## 9. 4070 后续补充任务

4070 pilot 完成 safety 生成后，应继续补充 base model 参考和 utility 生成。

运行清单见：

```text
experiments/step1_vanilla_baseline/4070后续运行清单.md
```

## 10. 第一步 B：轻量干预验证

4070 pilot 完成三 seed 后，下一步进入轻量干预验证。

干预实验仍属于第一步，不是 SpectraSafe 方法实验。

计划见：

```text
experiments/step1_vanilla_baseline/interventions/README.md
```
