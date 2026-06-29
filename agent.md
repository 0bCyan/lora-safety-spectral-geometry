# 项目 Agent 运行手册

本文档是本仓库的协作规则与实时状态手册。后续每次研究目标、实验分工、目录规范或机器职责发生变化，都应同步更新本文档。

## 1. 项目目标

本项目研究 LoRA 微调导致安全漂移的谱几何机制，并基于机制发现设计 SpectraSafe 训练期安全控制方法。

当前主线不是单纯提出一个正则项，而是：

1. 发现 LoRA safety drift 的谱几何 signature。
2. 用相关性和干预实验验证该 signature 的解释力。
3. 基于机制设计 SpectraSafe，并证明其比通用正则更有针对性。
4. 形成中文初稿，后续再考虑英文投稿版本。

## 2. 实时维护规则

`agent.md` 需要保持最新。以下情况必须更新：

- 新增或调整机器职责。
- 新增重要目录或移动核心文件。
- 确定新的主模型、数据集、benchmark 或 judge。
- 完成阶段性实验并改变下一步计划。
- 修改提交、分支、数据或安全规则。
- 发现某条研究支线失败，需要降级或转为局限。

更新原则：

- 只记录对协作有长期价值的信息。
- 临时想法放在 `docs/` 阶段记录中，不堆进本文件。
- 不记录 API key、服务器密码、私有数据路径或敏感样本内容。
- 规则变更应随代码或文档改动一起提交。

## 3. 仓库职责

本仓库是研究的唯一可信源，负责保存：

- 研究规划；
- 实验配置；
- 可复用代码；
- 运行脚本；
- 轻量结果表；
- 统计分析；
- 论文初稿；
- 图表生成逻辑；
- 阶段复盘。

本仓库不保存：

- 大模型权重；
- 训练 checkpoint；
- LoRA adapter 权重；
- 大规模原始数据；
- API judge 原始缓存；
- 大规模生成回答；
- `.env` 和任何密钥文件。

## 4. 机器职责

### 4.1 本机：规划、脚本、分析、论文

本机是主要协作与写作环境，负责：

- 研究路线规划；
- 中文初稿写作；
- 相关工作整理；
- 实验矩阵设计；
- 训练和评估脚本开发；
- 小样例单元测试；
- 结果表清洗；
- bootstrap、相关性、消融统计；
- 图表生成；
- git 提交与远端同步。

本机原则上不承担长时间正式训练。即使本机有 GPU，也优先用于小规模 pilot 和脚本验证，避免阻塞写作与分析。

### 4.2 显卡机 A：4070 Ti，快速 pilot

4070 Ti 约 12GB 显存，定位为快速试错机器。

主要任务：

- Qwen3-4B、Qwen2.5-3B 等小模型 QLoRA pilot；
- SpectraSafe 正则项反传验证；
- mean-only、max-only 小规模消融；
- 小规模 adapter 导出；
- 谱指标脚本验证；
- post-hoc scaling / pruning 流程验证；
- 少量 API judge 请求格式调试。

推荐设置：

- 4-bit QLoRA；
- LoRA rank 8 或 16；
- batch size 1；
- max length 512 或 1024；
- 单 seed；
- 小样本 family。

不建议承担：

- 7B/8B 多 seed 全量训练；
- 大规模生成；
- 14B 本地 judge；
- 多强基线完整复现。

### 4.3 显卡机 B：3090 Ti，正式训练

3090 Ti 约 24GB 显存，定位为正式实验主力。

主要任务：

- Qwen3-8B 或 Qwen2.5-7B 主实验；
- 7B/8B QLoRA 多方法训练；
- vanilla、SpectraSafe、mean-only、max-only；
- Frobenius、orthogonal、weight decay 对照；
- 可复现强基线训练；
- 批量生成待评估回答；
- 高谱层、低谱层、随机层干预实验；
- attention-only、MLP-only 干预实验。

推荐设置：

- 4-bit QLoRA；
- LoRA rank 8、16、32；
- batch size 1；
- gradient accumulation 1-8；
- max length 1024；
- gradient checkpointing 开启；
- 训练日志和配置完整保存。

不建议把 14B/32B 全量训练作为主流程依赖。24GB 可以尝试一些 14B 4-bit 推理，但正式 judge 交给 API。

### 4.4 API 强 Judge

强 judge 使用 API，计划模型为 GPT-5.5。实际模型 ID 以账号可用模型列表为准，不在代码中硬编码。

职责：

- harmful prompt 的 unsafe compliance 判断；
- safety prompt 的 refusal retention 判断；
- benign prompt 的 safe non-refusal 判断；
- utility 输出质量评分；
- 多方法、多 seed 的统一裁判。

要求：

- judge model 从配置或环境变量读取；
- 所有 API 结果必须缓存；
- 缓存默认不提交到 git；
- 正式统计只使用结构化字段；
- `reason` 字段仅用于人工检查和错误分析；
- 小批量调试通过后再做正式评估。

建议分批：

1. 调试批：每方法 10-20 条。
2. pilot 批：每方法 100-300 条。
3. 正式批：只评估关键方法、强基线和关键干预。

## 5. Git 规则

### 5.1 分支规则

- `main` 保存可复现、可继续工作的稳定状态。
- 小改动可直接在 `main` 上提交。
- 大实验脚本、重构或不确定改动建议使用功能分支。
- 不提交半写完且会破坏运行的代码。

建议分支命名：

```text
exp/<short-name>
analysis/<short-name>
paper/<short-name>
infra/<short-name>
```

### 5.2 提交规则

提交应小而清楚。推荐提交类型：

```text
docs: 更新研究规划
infra: 调整仓库结构
scripts: 新增训练启动脚本
analysis: 新增谱指标统计
paper: 更新中文初稿
configs: 新增实验配置
results: 添加轻量汇总结果
```

提交前必须检查：

```text
git status
git diff --stat
```

如果涉及代码，尽量运行最小测试或 dry-run。

### 5.3 禁止提交

禁止提交：

- `.env`；
- API key；
- Hugging Face token；
- 服务器地址、密码、私钥；
- 大模型权重；
- adapter 权重；
- checkpoint；
- 大规模原始数据；
- 大规模生成结果；
- API judge 缓存；
- 未脱敏的人类标注原文。

如果误提交敏感内容，应立即停止推送并清理历史。

## 6. 目录规则

### 6.1 `configs/`

保存实验配置。

建议按任务分类：

```text
configs/train/
configs/eval/
configs/judge/
configs/analysis/
```

配置文件应记录：

- model；
- dataset；
- method；
- LoRA rank；
- learning rate；
- max length；
- seed；
- lambda；
- output path；
- judge model。

### 6.2 `src/`

保存可复用代码。

建议结构：

```text
src/spectrasafe/
  training/
  spectral/
  interventions/
  judging/
  analysis/
  utils/
```

### 6.3 `scripts/`

保存命令行入口。

脚本应尽量只负责参数解析和调用 `src/` 中的模块，不把核心逻辑写死在脚本里。

### 6.4 `data/`

只提交说明、小样例和脱敏数据。

不提交：

- `data/raw/`；
- `data/cache/`；
- `data/external/`。

### 6.5 `results/`

只提交轻量结果：

- 汇总 CSV；
- bootstrap 结果；
- 最终表格；
- 图表源数据；
- 小规模脱敏样例。

不提交：

- 全量生成；
- API 原始缓存；
- 大规模 judge 输出。

### 6.6 `paper/`

保存中文论文初稿、图表和投稿材料。

建议结构：

```text
paper/draft.md
paper/figures/
paper/tables/
paper/notes/
```

## 7. 实验命名规则

实验 ID 建议包含：

```text
<date>_<model>_<method>_<family>_<dose>_s<seed>
```

示例：

```text
20260705_qwen3-8b_spectrasafe-global_all_high_s1
```

adapter、生成回答、judge 结果和统计表应共享同一个实验 ID，便于追踪。

## 8. 结果追踪规则

每次正式实验至少记录：

- git commit hash；
- 机器；
- GPU；
- 环境版本；
- 配置文件路径；
- 数据版本；
- adapter 输出路径；
- 生成结果路径；
- judge 结果路径；
- 是否进入论文统计。

建议后续建立：

```text
results/experiment_index.csv
```

字段至少包含：

```text
experiment_id, commit, machine, model, method, family, dose, seed, status, notes
```

## 9. 当前优先级

当前优先级从高到低：

1. 完成仓库基础设施和协作规则。
2. 建立训练、谱分析、干预和 judge 的配置化脚本。
3. 4070 Ti 跑通小模型 pilot。
4. 3090 Ti 跑 Qwen3-8B 或 Qwen2.5-7B 主实验第一轮。
5. 完成 layer-wise 和 module-wise 谱分析。
6. 完成高谱层、低谱层、随机层干预。
7. 用 API strong judge 评估主方法和关键 baseline。
8. 形成中文初稿机制章节。

## 10. 当前数据集决策

为了降低协作复杂度，4070 Ti 和 3090 Ti 使用同一个统一数据池。

当前第一轮数据策略：

- normal：Alpaca 正常任务子集；
- risk：PKU-SafeRLHF 风险子集；
- control：由正常任务数据打乱配对构造；
- BeaverTails 暂作为第二安全数据源，第一轮有信号后再加入；
- 4070 Ti 使用 pilot 子集；
- 3090 Ti 使用同一数据池中的初筛或正式子集；
- 所有样本使用固定 `sample_id` 追踪。

详细方案见：

```text
docs/统一数据集方案.md
```

## 11. 当前开放问题

需要尽快确认：

- 主模型最终选 Qwen3-8B 还是 Qwen2.5-7B。
- 副模型是否选择 Mistral/Ministral 8B。
- GPT-5.5 的实际 API 模型 ID 和账号权限。
- 强基线优先复现 SaLoRA 还是 Safe Delta。
- 是否需要把 adapter 权重放到独立对象存储或私有网盘。

## 12. 文件编码与平台约定

- Markdown 文件统一使用 UTF-8。
- Windows 与 Linux 均可参与协作。
- 脚本路径尽量使用配置，不在代码中写死绝对路径。
- 命令行输出中文乱码不代表文件损坏，优先用编辑器或 Python UTF-8 读取确认。
