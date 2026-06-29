# 数据说明

建议目录：

```text
data/raw/          原始数据，不提交
data/processed/    可复现实验所需的小型处理结果
data/samples/      脱敏样例
data/contracts/    数据协议与字段说明
```

不要提交敏感数据、完整 benchmark 输出或大规模生成结果。

## 当前统一数据池

第一轮统一数据池位于：

```text
data/processed/unified_pool/
```

构建脚本：

```text
scripts/build_unified_pool.py
```

当前提交的是小型固定数据池，供 4070 Ti 和 3090 Ti 使用同一套样本 ID。后续如果需要扩大规模，应重新运行脚本，并优先把大规模版本保存在实验机器本地或对象存储中。
