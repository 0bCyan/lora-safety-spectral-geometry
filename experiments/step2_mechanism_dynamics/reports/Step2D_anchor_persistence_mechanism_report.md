# Step2D Anchor Persistence and Mechanism Report

## 1. Scope

```text
model = Qwen3-4B
machine = 4070 Ti
family = risk
arm = anchor_mix_20p
seeds = 1/2/3
persistence checkpoints = step16/32/48/64
full safety eval = 100 prompts when available
```

## 2. Persistence Table

| arm | seed | checkpoint | full_unsafe | full_refusal | full_invalid | full_degradation | margin | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_mix_20p | 1 | step0032_pct050 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7589 |  |
| anchor_mix_20p | 1 | step0048_pct075 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7806 |  |
| anchor_mix_20p | 1 | step0064_pct100 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5801 | transient_rebound |
| anchor_mix_20p | 2 | step0032_pct050 | 0.9900 | 0.0100 | 0.0000 | 0.0000 | 0.1721 |  |
| anchor_mix_20p | 2 | step0048_pct075 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | -0.4656 |  |
| anchor_mix_20p | 2 | step0064_pct100 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | -0.5999 | transient_rebound |
| anchor_mix_20p | 3 | step0032_pct050 | 0.9900 | 0.0100 | 0.0000 | 0.0000 | -0.4118 |  |
| anchor_mix_20p | 3 | step0048_pct075 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | -1.0296 |  |
| anchor_mix_20p | 3 | step0064_pct100 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | -0.9469 | transient_rebound |

## 3. Schedule Comparison

| arm | seed | checkpoint | full_unsafe | full_refusal | full_invalid | full_degradation | margin | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant_20p | 1 | step0016_pct025 | 0.2700 | 0.7300 | 0.0000 | 0.0000 | -0.0480 |  |
| constant_20p | 1 | step0032_pct050 | 0.1900 | 0.8100 | 0.0000 | 0.8000 | 1.5669 |  |
| constant_20p | 1 | step0064_pct100 | 0.3300 | 0.6700 | 0.0000 | 0.6700 | 1.6016 | safety_with_degradation |
| early_20p_then_0p | 1 | step0016_pct025 | 0.2700 | 0.7300 | 0.0000 | 0.0000 | -0.0538 |  |
| early_20p_then_0p | 1 | step0032_pct050 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7527 |  |
| early_20p_then_0p | 1 | step0064_pct100 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5776 | rebound |
| early_20p_then_10p | 1 | step0016_pct025 | 0.2800 | 0.7200 | 0.0000 | 0.0000 | -0.0480 |  |
| early_20p_then_10p | 1 | step0032_pct050 | 0.5800 | 0.4200 | 0.0000 | 0.4000 | 1.2121 |  |
| early_20p_then_10p | 1 | step0064_pct100 | 0.7700 | 0.2300 | 0.0000 | 0.0000 | 1.2013 | partial_control |
| early_30p_then_10p | 1 | step0016_pct025 | 0.0600 | 0.9300 | 0.0100 | 0.5300 | 0.3768 |  |
| early_30p_then_10p | 1 | step0032_pct050 | 0.5300 | 0.4700 | 0.0000 | 0.4600 | 1.4897 |  |
| early_30p_then_10p | 1 | step0064_pct100 | 0.7800 | 0.2200 | 0.0000 | 0.0000 | 1.5020 | partial_control |

## 4. Decision

anchor_mix_20p is positive at step16 but rebounds by step64; continuing-anchor schedules are required.

Continuing anchors reduce unsafe behavior, but the best safety result is confounded by degradation; do not claim a stable recipe yet.

## 5. Pending Blocks

```text
anchor_ratio_scan_summary.csv: not run yet
utility_quality_summary.csv: not run yet; local safety degradation flags show utility-quality eval is required before claiming safety control
mechanism_comparison_summary.csv: partial logit/spectral probes are present for persistence and schedule arms; gradient-direction probes are not run yet
```
