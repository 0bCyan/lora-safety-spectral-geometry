# 实验计划与分析预注册

## 研究问题

RQ1：风险 LoRA 是否在多个 backbone 上造成早期安全边际下降？

RQ2：安全梯度有效贡献比 \(\rho_t\) 是否预测后期安全反弹？

RQ3：提高 \(\rho_t\) 能否减少反弹，同时避免 constant replay 的语义收缩？

RQ4：输出侧低秩子空间是跨架构稳定机制，还是架构依赖的诊断信号？

## 主要比较

每个模型内比较四个 arm。首要比较是 `adaptive_budget` vs `decay10`；`constant20` 是安全—质量上界型对照；`vanilla` 是漂移基线。所有比较使用配对 prompt、相同 seed 和相同 checkpoint。

主要效应量：

1. step64 unsafe rate 的绝对差和 bootstrap 95% CI；
2. \(\rho_t\) 与安全边际变化的 Spearman 相关；
3. targeted-safe-help 与 semantic concentration 的变化；
4. backbone 间标准化效应量，而非绝对谱范数。

## 训练与评测顺序

1. 构建并验证 `mechanism_v1`；
2. 只跑 Qwen3 seed1 的 16-step qualification；
3. 通过后冻结配置，跑四个 arm 的 Qwen3 三 seed 64-step；
4. 复制到 Llama 3.2；
5. 最后运行 Mistral QLoRA 外部压力测试；
6. 锁定全部结果后运行 HarmBench 和 SORRY-Bench 的 held-out 评估；
7. 运行结果聚合器，生成论文表格。

## 失败与停止规则

- 任何 run 缺失 checkpoint、数据哈希或 evaluator 版本，标记为不可用；
- held-out 集不得用于选择超参数；
- `adaptive_budget` 在 qualification 阶段频繁 fallback 或不能写出梯度审计时，停止方法扩展，但保留机制结果；
- 仅 unsafe 改善而 targeted-safe-help 或 utility 显著下降，不判定为方法胜出；
- 单一 backbone 的结果不能写成通用机制结论。
