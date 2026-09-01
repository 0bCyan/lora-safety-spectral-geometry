# 单卡离机运行说明

## 推荐顺序

1. 在前台执行 quick profile 的 dry-run；
2. 确认命令、模型缓存和输出盘无误；
3. 使用 `tmux` 或集群作业系统启动单 GPU worker；
4. 先完成 Qwen3 seed1 的四个 arm；
5. 通过 quick acceptance 后，再顺序追加 Qwen3 seed2/3 和 Llama；
6. 最后才运行 Mistral 与全量外部评测。

示例（将模板替换为实际训练/评测入口）：

```bash
tmux new -s lora-validation
set -euo pipefail
python experiments/validation_v1/scripts/run_formal_validation.py \
  --config experiments/validation_v1/configs/quick_feasibility_24gb.json \
  --output-dir experiments/validation_v1/runs/quick \
  --train-template 'python YOUR_TRAIN_ENTRYPOINT --job-id {job_id} --model {model_path} --arm {arm} --seed {seed}' \
  --eval-template 'python YOUR_EVAL_ENTRYPOINT --job-id {job_id} --checkpoint {checkpoint} --dataset {dataset}' \
  --execute 2>&1 | tee experiments/validation_v1/runs/quick/worker.log
```

不要在同一张 24GB GPU 上启动多个训练进程。所有输出写入本地持久磁盘；每个 run 完成后立即保存 checkpoint、日志和 `paper_metrics.jsonl`。断线后重新连接 `tmux attach -t lora-validation` 查看状态。

## 快速验收门槛

quick run 预计需要 3–6 小时（启用 batch generation）。只有满足 quick 配置中的 `continue_if` 才继续补充证据；否则停止长程实验并保留失败日志。quick 通过不等于论文结论成立，只表示方法值得进入多 seed、跨 backbone 阶段。
