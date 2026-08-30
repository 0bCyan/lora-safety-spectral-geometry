# AGENTS.md — Paper A 强机制论文分支协作规范

本文件适用于分支：

```text
paper/strong-mechanism-safety-persistence
```

它约束人类协作者、自动化 Agent、代码生成 Agent、评审 Agent 和文档 Agent 在本分支中的行为。根目录既有的 `agent.md` 记录整个项目的长期状态；本文件对论文 A 分支中的具体操作具有更高的分支内优先级。若两者冲突，先停止并显式记录冲突，不得自行选择更宽松的规则。

## 1. 分支使命

本分支只用于完成论文 A：

> 研究 LoRA 风险适配中的诊断几何、行为级安全监督、安全持久性、后期有效贡献和安全回答质量瓶颈，并以等累计剂量实验建立强机制证据，为后续方法论文铺垫。

本分支不是候选方法搜索区。没有新的增量行为证据时，不重新开启 SpectraSafe、CDR-OPC、CSPR、PCSR 或其他方法路线。

## 2. 永久科学边界

### 2.1 不得将以下内容称为“方法验证”

- 零步梯度检查；
- 虚拟参数位移；
- 一阶内积、KKT、Armijo 或局部可行性诊断；
- 协议状态机中的内部 `PASS`；
- 单个 checkpoint 的局部代理改善；
- 没有实际 checkpoint、生成行为、多 seed 和 sealed confirmation 的候选方案。

允许的表述包括：

```text
motivates
suggests
is consistent with
provides a feasibility probe
supports a bounded mechanism claim in the studied setting
```

除非完整方法证据链已经完成，否则禁止：

```text
our method improves
we validate the method
solves the safety-quality trade-off
provably preserves safety in training
```

### 2.2 当前主论文 claim

论文 A 只围绕以下链条构建：

1. 几何结构可诊断，但已测试几何控制不足以恢复行为；
2. 序列级安全监督具有行为效力；
3. 后期有效安全贡献可能调节安全持久性；
4. 持续监督可能带来 prompt-aware safe-response quality bottleneck；
5. 等累计剂量时序实验用于区分“何时监督”和“监督总量”；
6. 三 seed、人类效度、第二设置和 sealed confirmation 决定 claim 是否升级。

任何新工作若不能直接强化或反驳这六点，应放到其他分支。

## 3. 证据分层

### Tier H — 历史证据

来自已完成的 Step 1/Step 2：

- 可以用于动机、边界和历史 context；
- 必须标记 `Historical`；
- 未记录解析后模型 revision 的 run 不得与固定快照新实验 pooled；
- 不能因为历史结果方向有利，就改变前瞻主比较。

### Tier P — 前瞻主实验

等累计剂量 3 arms × 3 seeds，以及固定环境 `decay10/constant20` 复现：

- 使用冻结模型 revision、环境、数据、命令顺序和失败处理；
- 任何结果访问后修改都必须通过新版本 protocol/erratum；
- 不能静默重跑失败 run；
- 技术失败必须进入 run ledger 和分母解释。

### Tier C — 确认性证据

Paper-A 专用 prompt-disjoint confirmation：

- 只在分析、主要 claim、统计代码和成功门冻结后打开；
- 只打开一次；
- 不根据 confirmation 结果回头修改开发结论；
- 失败也必须完整报告。

## 4. 当前里程碑

```text
M0 workspace/protocols     complete
M1 runtime preflight       pending
M2 formal run matrix       pending
M3 blind remeasurement     pending
M4 human validity          pending
M5 second setting          pending
M6 sealed confirmation     pending
M7 release verification    pending
```

只有 M7 可以设置：

```json
{"submission_ready": true}
```

任何 Agent 不得提前修改这一状态。

## 5. 修改前必读文件

处理本分支任务前，至少读取：

```text
README.md
AGENTS.md
paper/strong_mechanism/STATUS.md
paper/strong_mechanism/CLAIMS.md
paper/strong_mechanism/EVIDENCE_MAP.md
paper/strong_mechanism/RUN_MATRIX.md
paper/strong_mechanism/verification/OPERATIONS.md
```

涉及具体阶段时，再读取对应协议和配置。不要只依据聊天摘要或旧分支记忆修改研究状态。

## 6. 文件职责

| 路径 | 允许内容 | 禁止内容 |
|---|---|---|
| `paper/strong_mechanism/configs/` | 冻结实验配置、rubric、registry | 结果依赖调参 |
| `protocols/` | 前瞻设计、停止门、amendment | 覆盖原协议字节 |
| `preflight/` | M1 锁和版本化修正 | 覆盖既有 lock |
| `data/raw/` | 本地、ignored 的生成与评审目标 | 推送大体量或敏感原文 |
| `data/derived/` | 标准化统计输入、命令 manifest | 手工编辑主数字 |
| `data/runs/` | provenance、run ledger、轻量状态 | 权重、adapter、checkpoint |
| `scripts/` | 可复现 CLI 与验证器 | 隐藏结果条件分支 |
| `tests/` | 约束、统计、数据合同测试 | 只测试 happy path |
| `figures/available/` | 历史/设计 SVG | 尚未产生的结果曲线 |
| `figures/generated/` | 数据驱动 PDF/PNG/SVG | 手工修图改变数字 |
| `tables/` | 自动表格 | 手抄论文数字 |
| `latex/` | 证据门控论文 | 删除未完成证据标记以通过编译 |

## 7. 实验操作规则

### 7.1 M1 前禁止正式训练

正式训练前必须存在并通过：

- clean worktree；
- 固定 `Qwen/Qwen3-4B` revision；
- 4070 Ti 严格环境锁；
- 15 条训练命令 manifest；
- 45 条评估/审计命令 manifest；
- 60 个 dry-run 全通过；
- 使用非正式 seed 和独立目录的 one-step smoke；
- `M1_preflight_lock.json`。

缺少任一项时，Agent 只能做静态验证、文档、代码、设计图或非正式 smoke，不能启动正式模型训练。

### 7.2 正式 run 不可覆盖

- 输出目录存在时必须停止；
- 模型初始化后失败的 run 记为失败；
- 不删除失败记录；
- 不自动替换 seed；
- 不根据中间结果调整 schedule；
- 不从结果最好的 checkpoint 事后选择主终点。

### 7.3 评估分层

本地 heuristic judge：

- 允许用于快速诊断；
- 禁止作为论文主 unsafe 标签；
- 禁止用它替代 reviewer reliability failure。

正式标签：

- A/B 独立；
- calibration 与 hidden duplicate 先过门；
- Reviewer C 只看 disagreement；
- arm、seed、checkpoint 在最终 join 前保持隐藏。

## 8. 统计规则

必须：

- prompt 内配对；
- 每个 seed 独立报告；
- prompt bootstrap；
- 二元结果 per-seed McNemar；
- pooled 结果使用 seed-hierarchical bootstrap；
- 报告效应量和区间，而不只报告 p 值；
- 预注册最小有意义效应和非劣 margin。

禁止：

- 把 `seed × prompt` 当成完全独立的 N；
- 忽略 seed 方向不一致；
- 根据结果选择 bootstrap seed、终点或子组；
- 只报告最有利 checkpoint；
- 用平均安全率掩盖尾部失败。

## 9. 图表规则

### 9.1 可在正式训练前绘制

- 论文总论证图；
- equal-dose schedule 与累计剂量图；
- 数据角色与泄漏边界；
- 证据流水线与 M0–M7 gates；
- 已冻结历史定量结果。

生成入口：

```bash
make -C paper/strong_mechanism figures-available
```

### 9.2 必须等待正式数据

- equal-dose safety trajectories；
- seed-level paired effects；
- late gradient contribution；
- safety–quality frontier；
- model–human agreement；
- second-setting replication；
- confirmation effect。

这些图只能读取标准化 `data/derived/` 输入。缺失输入时脚本必须失败，不能画模拟值、占位趋势或预测区间。

### 9.3 视觉规范

- 图中标题/caption 标明 `Historical`、`Design` 或 `Confirmatory`；
- 颜色不是唯一编码；
- y 轴截断必须显式说明；
- 误差条含义写清楚；
- 数据点、seed 和 prompt 数量在 caption 或表格中可追踪；
- Git 优先提交 SVG；投稿导出 PDF；README 可使用 SVG；
- 禁止在图像编辑器中移动柱高、数据点或误差条。

## 10. LaTeX 规则

- `\PaperPending{...}` 只有相应 release gate 通过后才能移除；
- 主数字来自生成的表格或宏，不手工复制；
- 历史 seed1 与新 fixed-snapshot 结果明确分栏；
- method-oriented probes 只作 future-design motivation 或 appendix；
- abstract、introduction、conclusion 的 claim 强度必须一致；
- 每次提交前至少运行：

```bash
make -C paper/strong_mechanism/latex clean check
```

## 11. 文档更新规则

发生以下事件必须同步更新 `STATUS.md` 和相关 ledger：

- milestone 状态变化；
- 新正式 run 完成或失败；
- reviewer reliability pass/fail；
- human validity 结果；
- second setting 结果；
- confirmation 打开；
- claim 被升级、降级或关闭；
- protocol erratum；
- release gate 变化。

临时想法不要堆进 `STATUS.md`；放入版本化设计文档或 issue。

## 12. Git 规则

### 12.1 分支

- 只在 `paper/strong-mechanism-safety-persistence` 上整理论文 A；
- 不直接改 `main`；
- M1 lock 后不 rebase、不 force-push；
- 不把 O4、Step3P/Q、CDR-OPC 等归档分支整体 merge 到本分支；
- 需要历史资产时只引用固定路径/hash，或做最小、审计明确的 cherry-pick。

### 12.2 Commit

commit 信息应使用清楚的中文前缀，例如：

```text
论文A：完善分支入口与操作步骤
图表：加入等剂量设计与历史证据图
验证：增加可绘图资产 CI
协议：冻结 M1 预飞规则
结果：登记正式 run 状态（不得包含大文件）
勘误：新增 v2 规则并保留原协议
```

一次 commit 尽量只做一种科学语义的改变。

### 12.3 敏感与大文件

禁止提交：

- 密钥、token、服务器凭据；
- 模型、adapter、checkpoint；
- 大规模生成文本；
- 未脱敏的人类标注；
- 受限数据原始字节。

## 13. 代码规则

- Python 3.11；
- 新脚本使用 `from __future__ import annotations`；
- CLI 参数显式，默认值与 registry 一致；
- 路径从仓库根或 `PAPER_ROOT` 推导；
- 写文件前检查 overwrite；
- 关键产物写 SHA-256、行数、schema version 和 provenance；
- 对 missing/duplicate/inconsistent joins fail closed；
- 不捕获异常后继续生成“部分成功”主结果；
- 随机过程必须有冻结 seed；
- 添加最小单元测试或 synthetic artifact test。

## 14. 完成任务前的最小验证

文档或图表变更：

```bash
python -m compileall -q paper/strong_mechanism/scripts paper/strong_mechanism/tests
python paper/strong_mechanism/scripts/plot_available_figures.py \
  --output-dir /tmp/paper_a_figures \
  --formats svg
make -C paper/strong_mechanism smoke
make -C paper/strong_mechanism/latex check
```

实验/评审/统计变更还必须运行对应 protocol validator 和合成数据端到端测试。

## 15. Agent 任务模板

开始任务时先写明：

```text
目标：
涉及 claim：
允许读取的数据层级：H / P / C
允许写入的路径：
禁止事项：
验证命令：
```

结束任务时必须报告：

```text
修改文件：
新增/改变的科学 claim：无 / 有（列出）
运行的验证：
未完成或未验证部分：
是否访问正式结果：
是否访问 confirmation / global heldout：
```

## 16. 当前下一合法动作

```text
M1_EQUAL_DOSE_PREFLIGHT
```

允许：

- 完善文档、图表、测试和 release gates；
- 建立锁定环境；
- 固定模型 snapshot；
- 渲染 15+45 命令；
- 60 次 dry-run；
- 非正式 one-step smoke；
- 冻结 M1 lock。

不允许：

- 正式训练；
- 新方法搜索；
- 打开 Paper-A confirmation；
- 打开现有 global heldout；
- 删除 `PENDING EVIDENCE`；
- 宣称论文已经完成。
