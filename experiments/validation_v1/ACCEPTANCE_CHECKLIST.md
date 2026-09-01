# 正式实验验收清单

## 数据

- [ ] 使用 `configs/data/mechanism_dataset_v1.json`；
- [ ] 上游 revision、SHA-256 和 sample ID manifest 已保存；
- [ ] 训练/评测 prompt 无重复；
- [ ] SORRY-Bench/HarmBench test 未参与训练或调参；
- [ ] 风险类别和数据集 split 已写入结果；

## 训练

- [ ] 模型、tokenizer、chat template 和 thinking mode 已记录；
- [ ] seed、LoRA rank/alpha、学习率、最大长度、有效 token budget 已冻结；
- [ ] 24GB 显存配置无 OOM，checkpoint 0/6/16/32/64 均存在；
- [ ] 峰值显存不超过 22.1GB（24GB 的 92% contract），且没有 CPU offload；
- [ ] Qwen3/Llama 使用 BF16 LoRA，Mistral 使用 NF4 QLoRA；
- [ ] max_length、batch、gradient accumulation、checkpointing 和 optimizer 与正式配置一致；
- [ ] 训练日志、git commit、环境版本和命令行已保存；
- [ ] 每步的实际安全样本暴露量和梯度审计已保存；

## 评测

- [ ] 使用固定解码参数并保存原始生成；
- [ ] 每条 response 有 sample ID、model、arm、seed、checkpoint；
- [ ] evaluator 名称、版本、规则和置信区间已记录；
- [ ] 同时报告 unsafe、clean refusal、targeted-safe-help、utility；
- [ ] 同时报告安全边际、\(\rho_t\)、gradient cosine 和语义集中度；

## 论文可用性

- [ ] `paper_metrics.jsonl` 中没有缺失主终点；
- [ ] 聚合表只使用已通过验收的 run；
- [ ] 报告每个数据集结果和跨数据集 macro-average；
- [ ] 报告每个 backbone 的结果，不以单一平均掩盖失败；
- [ ] 结论区分“机制复现”“架构调制”和“方法胜出”。
