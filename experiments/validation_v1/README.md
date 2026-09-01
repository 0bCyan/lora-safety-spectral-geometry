# Validation-v1：正式机制验证

本阶段用于生成可直接进入论文正文/附录的跨 backbone 机制证据。它验证的不是某个固定谱阈值，而是：

> 风险 LoRA 训练中，安全梯度的有效贡献下降是否能够预测安全反弹；提高该有效预算是否能在不同 backbone 上抑制反弹，并量化安全—质量代价。

## 固定实验矩阵

- Backbone：Qwen3-4B-Base、Llama 3.2-3B、Mistral-7B-v0.3；
- Arm：vanilla、decay10、constant20、adaptive_budget；
- Seed：1、2、3；
- Checkpoint：step 0/6/16/32/64；
- 主评测：BeaverTails-Evaluation、SORRY-Bench base；
- 外部评测：HarmBench text test；
- 能力评测：utility split；
- 24GB 约束：单卡、显存上限 24GB；Qwen3/Llama 使用 BF16 LoRA（长度 512，batch 1，梯度累积 4），Mistral 使用 NF4 QLoRA（长度 512，batch 1，梯度累积 8），统一启用 gradient checkpointing 和 8-bit AdamW。

正式结果必须使用固定数据 manifest、固定 chat template、固定最大长度、固定 token budget 和固定解码参数。Qwen3 的 thinking mode 显式关闭。任何 OOM 都必须停止并标记该 run 无效，不能临时缩短序列、改变 batch 或切换精度后继续并合并结果。

## 运行入口

```bash
python experiments/validation_v1/scripts/run_formal_validation.py \
  --config experiments/validation_v1/configs/formal_validation.json \
  --repo-root . \
  --dry-run
```

dry-run 会生成完整 job matrix、命令、数据版本和预期 artifact 路径，不访问模型、不启动训练。确认后去掉 `--dry-run` 才执行。

训练和生成可使用仓库既有脚本；新的编排器不修改其内部算法，只负责注册配置、顺序、日志和验收门槛。

## 论文结果契约

每个完成的 run 必须写入一个 `paper_metrics.json`，至少包含：

```json
{
  "protocol_version": "validation-v1",
  "model_id": "qwen3-4b-base",
  "arm": "decay10",
  "seed": 1,
  "checkpoint": 64,
  "dataset": "sorry_bench",
  "unsafe_rate": 0.0,
  "clean_refusal_rate": 0.0,
  "targeted_safe_help_rate": 0.0,
  "utility_success_rate": 0.0,
  "safety_gradient_contribution_ratio": 0.0,
  "safety_margin": 0.0,
  "semantic_cluster_concentration": 0.0,
  "evaluator": "name@version",
  "evaluator_type": "human|official_classifier|declared_judge"
}
```

`aggregate_paper_results.py` 会拒绝缺少协议版本、seed、数据集、评估器或主终点的记录，并输出论文表格 CSV/Markdown。没有评估器信息的数字不能作为正式论文结果。
