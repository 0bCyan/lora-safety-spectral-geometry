# 第一步实验结果目录

本目录保存第一步普通 LoRA 基线实验的结果。

允许提交：

- `experiment_index.csv`
- 小型汇总表
- 小型统计结果
- 图表源数据

不提交：

- adapter 权重
- checkpoint
- 大规模生成回答
- API judge 原始缓存
- 大型日志

建议结构：

```text
4070ti/
  adapters/           不提交
  generations/        大规模不提交
  spectral_metrics/   小型可提交
  logs/               视大小决定
3090ti/
  adapters/           不提交
  generations/        大规模不提交
  spectral_metrics/   小型可提交
  logs/               视大小决定
```
