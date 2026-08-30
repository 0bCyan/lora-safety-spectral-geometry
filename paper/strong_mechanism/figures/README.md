# Paper A Figures

图表分为两套：

```text
available/   不依赖缺失正式结果；允许现在提交
 generated/  从标准化正式数据生成；结果存在后才合法
```

## 1. Available assets

| File | Type | Evidence status | Intended use |
|---|---|---|---|
| `available/fig00_paper_story.svg` | 论文总览 | design | README、Introduction 概念图 |
| `available/fig01_equal_dose_design.svg` | 时序设计 | registered design | Confirmatory design 主图/附图 |
| `available/fig02_evidence_pipeline.svg` | 证据流程 | governance design | Reproducibility / appendix |
| `available/fig03_data_boundaries.svg` | 数据隔离 | registered design | Setup / leakage boundary |
| `available/fig10_historical_subspace_signature.svg` | 定量历史图 | historical | diagnostic geometry |
| `available/fig11_historical_posthoc_control_failure.svg` | 定量历史图 | historical | diagnostic–control separation |
| `available/fig12_historical_sequence_anchor_multiseed.svg` | 定量历史图 | historical | behaviorally active supervision |
| `available/fig13_historical_seed1_persistence_context.svg` | 定量历史图 | historical context only | motivate final replication |
| `available/fig20_completion_gates.svg` | 状态图 | governance | branch planning / appendix |

历史图必须在标题或 caption 中保留 `Historical`，且不能与 fixed-snapshot 新结果不加区分地 pooled。

## 2. Rebuild

只生成 Git 友好的 SVG：

```bash
python paper/strong_mechanism/scripts/plot_available_figures.py \
  --output-dir paper/strong_mechanism/figures/available \
  --formats svg
```

或：

```bash
make -C paper/strong_mechanism figures-available
```

本地投稿导出：

```bash
python paper/strong_mechanism/scripts/plot_available_figures.py \
  --output-dir paper/strong_mechanism/figures/generated/available \
  --formats svg,pdf,png \
  --dpi 300
```

## 3. Final result figures

`plot_main_figures.py` 等待以下标准化输入：

```text
data/derived/run_metrics.csv
data/derived/paired_effects.csv
data/derived/gradient_summary.csv
data/derived/quality_metrics.csv
```

对应：

- equal-dose safety trajectories；
- seed-level paired effects；
- late effective gradient contribution；
- safety–quality frontier。

输入缺失或 join 不完整时，脚本必须报错。不得用预测值、手工占位点或模拟区间生成“结果图”。

## 4. Figure integrity

- 所有定量图读取 JSON/CSV source；
- 不手工修改柱高、点位或误差条；
- 主数字可追踪到 source path、schema、hash；
- 颜色不作为唯一编码；
- y 轴截断必须显式说明；
- 图注说明 historical/design/confirmatory 层级；
- Git 提交 SVG，release 生成 PDF，README 可嵌入 SVG。
