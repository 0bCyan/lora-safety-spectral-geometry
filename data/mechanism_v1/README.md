# 机制验证数据集（mechanism-v1）

本目录只保存数据协议和构建说明，不提交上游数据原文件。原始数据可能包含有害内容，且各来源有独立许可；运行构建脚本前请阅读上游数据卡并确认研究用途合规。

## 设计

训练流：

- `risk`：BeaverTails 中按 14 类风险均衡抽取的 harmful QA；
- `safety`：PKU-SafeRLHF 中带 harmless 标签的 preferred response；
- `utility`：仓库现有正确匹配的 Alpaca normal 数据；
- `control`：仓库现有 shuffled-response 对照，仅作数据错配对照，不作为唯一正常基线。

评测流：

- `eval_beavertails`：BeaverTails-Evaluation；
- `eval_sorry_bench`：SORRY-Bench 2025 基础集；
- `eval_harmbench`：HarmBench text test split。

不要把任何评测 split 用于训练或调参。

## 构建

```bash
python scripts/build_mechanism_dataset.py \
  --output-dir data/processed/mechanism_v1 \
  --cache-dir .cache/mechanism_dataset \
  --seed 20260901
```

SORRY-Bench 数据集需要先在 Hugging Face 接受上游条款，并设置 token：

```bash
export HF_TOKEN=hf_...
```

脚本将上游文件缓存到 `.cache/mechanism_dataset/`，输出规范化 JSONL、`manifest.json` 和每个源文件的 SHA-256。原始文件不会写入 git。

## 默认规模

- risk：每个 BeaverTails 类别最多 100 条；
- safety：最多 1,400 条去重 prompt；
- utility：读取当前 `train_normal.jsonl` 的前 1,400 条；
- BeaverTails-Evaluation：全量；
- SORRY-Bench：基础 `question.jsonl`；
- HarmBench：官方 text test split。

## 评测注意事项

不同数据集的标签语义不同。分别报告每个数据集的 macro-average，不要把 refusal、harmfulness 和 helpfulness 混成一个未经说明的总分。模型生成的新 response 仍需要使用声明过的 judge/classifier；数据集中的 prompt 标签不能替代 response 评估。
