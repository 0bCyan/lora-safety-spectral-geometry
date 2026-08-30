# Paper A — Strong Mechanism Study of LoRA Safety Persistence

本目录是论文 A 的唯一 paper-facing 工作区。探索性 Step 3 方法分支、失败 authoring 路线和候选优化器不会直接进入本目录的主论证。

## Scientific target

论文研究以下链条：

1. risk LoRA fine-tuning 产生可复现的安全漂移和诊断几何结构；
2. 改变几何或局部标量 proxy 在当前设置中不足以控制行为；
3. sequence-level safety supervision 具有行为效力；
4. 监督的时间分布和后期有效贡献可能调节安全持久性；
5. 持续监督可能造成 prompt relevance、safe actionability、detail retention 和 naturalness 的质量瓶颈；
6. equal-cumulative-dose 实验用于识别时间位置效应，而不是再次做 replay-weight sweep。

本文不声称 CDR-OPC、CSPR、SpectraSafe 或其他候选已经成为经过验证的方法。方法类探针只有在产生实际训练 checkpoint、真实生成、多 seed、公平基线和 sealed confirmation 后，才可能进入未来方法论文。

<p align="center">
  <img src="figures/available/fig00_paper_story.svg" alt="Paper A story" width="100%">
</p>

## Completion standard

目标是强机制论文，不是窄案例记录。投稿前必须完成：

- equal-cumulative-dose temporal-distribution experiment；
- 三 seed 最终复现；
- frozen prompt-aware quality remeasurement；
- independent human-validity subset；
- 第二模型或第二风险设置；
- 分析冻结后一次 prompt-disjoint confirmation；
- 可复现统计、图、表和 LaTeX；
- claim-to-evidence ledger 无 unsupported promotion。

直到全部 gate 通过，`submission_ready` 保持 `false`。

## Workspace map

```text
paper/strong_mechanism/
  README.md                 本文件
  STATUS.md                 实时状态和阻塞项
  CLAIMS.md                 claim freeze / promotion rules
  EVIDENCE_MAP.md           claim-to-artifact ledger
  ROADMAP.md                P0–P6 与 M0–M7 顺序
  RUN_MATRIX.md             15 个正式训练 run
  DATA_AND_CODE_MAP.md      canonical historical source paths
  configs/                  experiment/rubric/study registries
  environment/              strict 4070 Ti runtime lock
  protocols/                prospective protocols and stop rules
  preflight/                immutable M1 lock and amendments
  data/
    raw/                     ignored response-bearing artifacts
    derived/                 normalized paper inputs and manifests
    runs/                    provenance and lightweight run state
  scripts/                  validation, training wrappers, review, stats, plots
  tests/                    unit and synthetic end-to-end tests
  figures/
    available/              pre-result historical/design SVG assets
    generated/              final data-driven exports
  tables/                   generated LaTeX/CSV tables
  latex/                    evidence-gated manuscript
  verification/             operations, release gates, reviewer attacks
```

历史代码和数据不复制；由 `data/source_manifest.json` 固定路径与 SHA-256。论文专用标准化数据只写入 `data/derived/`。

## Current figure assets

可在正式训练前重建：

```bash
make -C paper/strong_mechanism figures-available
```

输出：

```text
figures/available/fig00_paper_story.svg
figures/available/fig01_equal_dose_design.svg
figures/available/fig02_evidence_pipeline.svg
figures/available/fig03_data_boundaries.svg
figures/available/fig10_historical_subspace_signature.svg
figures/available/fig11_historical_posthoc_control_failure.svg
figures/available/fig12_historical_sequence_anchor_multiseed.svg
figures/available/fig13_historical_seed1_persistence_context.svg
figures/available/fig20_completion_gates.svg
```

本地导出 PDF/PNG：

```bash
python paper/strong_mechanism/scripts/plot_available_figures.py \
  --output-dir paper/strong_mechanism/figures/generated/available \
  --formats svg,pdf,png \
  --dpi 300
```

最终结果图由 `plot_main_figures.py` 读取标准化结果；输入不存在时必须失败。

## Required order of work

```text
M0 claim/protocol/code freeze
→ M1 pinned runtime + 60 dry-runs + non-formal smoke
→ M2 15 formal runs + 45 downstream jobs
→ M3 blinded safety/quality/utility measurement
→ M4 independent human validity
→ M5 second setting
→ analysis/manuscript freeze
→ M6 one sealed confirmation
→ M7 release verification
```

不得在分析冻结前打开 confirmation。

## Quick commands

从仓库根目录：

```bash
make -C paper/strong_mechanism validate
make -C paper/strong_mechanism test
make -C paper/strong_mechanism commands
make -C paper/strong_mechanism eval-commands
make -C paper/strong_mechanism figures-available
make -C paper/strong_mechanism paper
```

正式结果存在后：

```bash
make -C paper/strong_mechanism analysis
make -C paper/strong_mechanism tables
make -C paper/strong_mechanism figures-final
python paper/strong_mechanism/scripts/validate_study.py --release
```

完整逐步命令见 [`verification/OPERATIONS.md`](verification/OPERATIONS.md)。分支协作规则见根目录 [`AGENTS.md`](../../AGENTS.md)。

## Branch policy

开发分支：

```text
paper/strong-mechanism-safety-persistence
```

- prospective protocol 冻结后不重写历史；
- 产物 append-only；
- protocol correction 使用新版本文件并保留原字节；
- 不把失败支线整体 merge 进本分支；
- 不提交 checkpoint、adapter、模型权重、大规模生成、API key 或未脱敏标注。
