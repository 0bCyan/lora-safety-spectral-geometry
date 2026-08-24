# 项目 Agent 运行手册

本文档是本仓库的协作规则与实时状态手册。后续每次研究目标、实验分工、目录规范或机器职责发生变化，都应同步更新本文档。

## 1. 项目目标

本项目研究 LoRA 微调后的安全漂移、训练期安全持久性和安全回答质量，并基于经过验证的机制设计训练方法。

当前主线是：

1. Step1 已确认谱几何、输出子空间和激活信号具有诊断价值，但简单后验干预不能建立可靠因果控制。
2. Step2 已确认序列级安全监督及其后期加权贡献决定安全持久性；固定20%自然安全回放是强因果控制，但不是完整方法。
3. Step3 已将回答质量分解为细节保留、安全可操作性和自然度，并完成prompt-disjoint量表验证。
4. Step3C-T已完成T2R至T6；两个C3候选均未通过预注册质量主门，因此当前C3 recipe已关闭且没有`Q_star`。
5. Step3N-R1在训练前因安全行动性量表可靠性不足而停止；Step3O随后完成量表修复，O3固定上下文隔离会话的外部源一致性门已通过。
6. Step3O-O4尝试005在固定60个prompt后因单个长上下文作者未交付而停止；O4R尝试001拆为六个20行会话后仍有两个会话未交付，已按预注册规则封存为`STOP_AUTHORING`。两次均未进入质量审查或训练，`optimizer_steps=0`，不能解释为方法失败。
7. Step3P已在A/B可靠性联合门停止，未启动Reviewer C、目标选择或训练。Step3Q尝试001已冻结60个prompt、120个请求、96行校准库与P2协议，但因当前进程无API密钥而保持未dispatch；独立子智能体尝试002中四个固定会话有一个在提交前触发平台安全网关，仅交付90/120行，低于96门槛并封存为`STOP_SUBAGENT_YIELD`，未seal、导入、审查或训练。
8. Step3 CDR-OPC M5已在prompt-disjoint的60个新prompt上完成120个候选的独立authoring与A/B盲审，并用固定剂量`0.0337860296353059`完成零步GPU确认。Calibration保护`20/20`并授权打开sealed audit，sealed audit保护`19/20`，两组quality/safety行为均为正，最终路线为`PASS_CDRO_OPC_M5_ZERO_STEP_CONFIRMATION`。该结果仍为梯度机制证据：`optimizer_steps=0`、`training_authorized=false`，Core100和heldout均未打开。
9. Step3 CDR-OPC M6已按另行前瞻冻结的固定剂量和`eta=2e-4`执行有限半径直接虚拟位移。映射精度、恢复、梯度覆盖、行为及selected-vs-fallback四证书平均相对恶化门均通过，但calibration单元保护仅`16/20`，低于`19/20`门；四个越界均来自risk证书，因此路线为`STOP_CDRO_OPC_M6_CALIBRATION`，sealed audit未打开。不得据此做结果依赖的半径调整或重跑，不得授权训练；`optimizer_steps=0`、`training_authorized=false`保持不变。
10. Step3 CDR-OPC M7已在相同固定剂量和半径上完成仅限有效性的名义镜像分解，80/80个calibration unit-certificate格满足代数、映射、恢复和完整性门，路线为`PASS_CDRO_OPC_M7_VALIDITY_ONLY_SYMMETRIC_DIAGNOSTIC`。四个已知risk失败格的一阶项和central项均未越过`0.04`，加入正的even有限半径分量后才全部越界；这是post-hoc加法归因，不是纯Hessian或因果结论。原`verify-run`因把JSON对象键序误当schema而保留为`ERROR_CDRO_OPC_M7_VERIFY_RUN_IMPLEMENTATION_DEFECT`；独立VE1仅在内存中执行精确键集合适配，对字节未变的产物完成全量验证并得到`PASS_CDRO_OPC_M7_VE1_UNCHANGED_ARTIFACTS_VERIFIED`。M6的16/20 STOP、sealed未打开、`optimizer_steps=0`和`training_authorized=false`均未改变。

SpectraSafe双谱正则保留为历史动机和忠实负基线；没有新的增量行为证据时不重新进入主方法路线。

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

提交应小而清楚，提交名称使用中文。推荐提交类型：

```text
文档：更新研究规划
基础设施：调整仓库结构
脚本：新增训练启动脚本
分析：新增谱指标统计
论文：更新中文初稿
配置：新增实验配置
结果：添加轻量汇总结果
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

1. 保持CDR-OPC M6的`STOP_CDRO_OPC_M6_CALIBRATION`结论：固定半径有限损失未复现M5一阶calibration的`20/20`尾部保护，sealed audit未打开；不得做结果依赖的半径调整、剂量调整或重跑，不得授权训练或生成。M7只解释已知失败的固定半径加法结构，VE1只修复验证表示层；两者均不授权M8、训练或扩大科学主张。若继续CDR-OPC，必须另行前瞻冻结真正独立的新cohort或明确关闭该训练路线。
2. 保持Step3C-T6的`R1_no_c3_candidate_passes`、Track B M1失败、Step3N-R1训练前停止，以及O4/O4R两个`STOP_AUTHORING`记录不变。
3. 保持Step3Q尝试001的P0/P1/P2提交与标签不可变，并保持其未dispatch状态；不得用子智能体输出伪造API receipt或导入旧结果根。
4. 保持尝试002的P0标签、锁提交和`STOP_SUBAGENT_YIELD`结果不变；仓库外90行仅作失败证据保留，不得替换缺失会话、补量、重分配、seal、导入或启动盲审。
5. Step3Q两次尝试均未产生可审查候选池：尝试001未dispatch，尝试002未达到96/54 authoring门；任何后续路线必须另行前瞻冻结，且继续保持`external_api_called=false`或如实记录新传输、`optimizer_steps=0`和训练未授权。
6. 使用互不相同的全新无上下文会话完成A/B审查，只向reviewer交付可见packet、模板、rubric和provenance模板，不提供仓库或private map；每个错配哨兵必须与对应组件锚点同包，每个组件在每位审查者包内至少有3个yes和3个no参考，clean-safety固定4个yes和4个no；A/B校准或可靠性失败时不得创建C包或执行最终选择。
7. A/B通过后，C只审查派生资格标签不一致的候选；最终选择器不得接收API mode，按P1预分配类别同时满足48个不同prompt、19类覆盖和严重度`5/28/5`下限。
8. Step3Q通过后仍须冻结tokenizer、assistant-only loss mask、泄漏与模板集中审计、训练适用性审查、方法对照和smoke协议，才可授权任何optimizer step。

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
- CDR-OPC应获取全新独立cohort重新确认有限半径尾部保护，还是因M6的16/20结果关闭训练路线。

## 12. 文件编码与平台约定

- Markdown 文件统一使用 UTF-8。
- Windows 与 Linux 均可参与协作。
- 脚本路径尽量使用配置，不在代码中写死绝对路径。
- 命令行输出中文乱码不代表文件损坏，优先用编辑器或 Python UTF-8 读取确认。

## 13. 环境与版本记录

所有实验必须记录环境版本。没有环境记录的实验，默认不能进入论文统计。

统一说明见：

```text
docs/环境与版本说明.md
```

第一步实验环境记录位置：

```text
experiments/step1_vanilla_baseline/results/4070ti/env_report.txt
experiments/step1_vanilla_baseline/results/3090ti/env_report.txt
```

每台机器第一次运行实验前，都要先执行 `check_env.py` 并保存输出。更新 PyTorch、transformers、peft、bitsandbytes、CUDA 或驱动后，必须重新记录环境。

## 14. 数据回传规则

显卡机负责训练和提取，本机负责分析。

第一步实验中，4070 Ti 和 3090 Ti 默认只回传轻量结果：

```text
env_report.txt
experiment_index.csv
spectral_metrics/*.jsonl
generations/*.jsonl
```

adapter 权重、checkpoint 和大型日志默认不回传、不提交。

详细规则见：

```text
experiments/step1_vanilla_baseline/数据回传流程.md
```
