# Paper A 强机制论文分支

> 当前分支：`paper/strong-mechanism-safety-persistence`
>
> 基线：`codex/step3-exploratory-pilot` at `7ffe500224585cf490035863ca640327adb420a1`
>
> 状态：论文基础设施与前瞻协议已建立；正式补实验尚未运行；`submission_ready=false`。

这个分支只服务于论文 A：**LoRA 安全持久性的强机制与安全回答质量瓶颈**。它不把 SpectraSafe、CDR-OPC、CSPR 或其他候选路线包装为已验证方法。

统一入口：

```text
paper/strong_mechanism/README.md
paper/strong_mechanism/STATUS.md
paper/strong_mechanism/CLAIMS.md
paper/strong_mechanism/RUN_MATRIX.md
paper/strong_mechanism/verification/OPERATIONS.md
paper/strong_mechanism/latex/main.tex
```

最小检查：

```bash
make -C paper/strong_mechanism smoke
```

正式训练前必须同时满足：模型 revision 已固定、4070 Ti 环境锁通过、15 条命令清单已提交且哈希冻结、P1 协议没有结果后修改。

---

# LoRA Safety Spectral Geometry

研究 LoRA 微调中的安全漂移、谱几何结构、奇异子空间和训练期安全控制方法。

## 当前主线

- 统一数据集与数据划分；
- vanilla LoRA 安全漂移基线；
- 谱指标与模块定位；
- 参数空间与激活空间干预；
- 训练期安全机制与控制；
- 安全持久性与回答质量。

## 目录

```text
configs/      实验配置
data/         数据说明与轻量处理后数据
docs/         研究规划与设计文档
experiments/  分阶段实验
scripts/      根级入口脚本
src/          可复用核心代码
paper/        论文工作区
```

## 重要说明

本仓库只提交轻量、可审计的代码、配置、结果汇总和脱敏数据。大模型权重、LoRA adapter、checkpoint、大规模生成缓存、API 密钥和未脱敏标注不会提交。
