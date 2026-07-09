# Step2E Seed Expansion and Degradation Audit Report

## 1. Seed Expansion Safety

| seed | checkpoint | unsafe | refusal | degradation |
| --- | --- | --- | --- | --- |
| 1 | step0016_pct025 | 0.2700 | 0.7300 | 0.0000 |
| 1 | step0032_pct050 | 0.2900 | 0.7100 | 0.7000 |
| 1 | step0064_pct100 | 0.7500 | 0.2500 | 0.0000 |
| 2 | step0016_pct025 | 0.2400 | 0.7600 | 0.0000 |
| 2 | step0032_pct050 | 0.6700 | 0.3300 | 0.0000 |
| 2 | step0064_pct100 | 0.5000 | 0.5000 | 0.5000 |
| 3 | step0016_pct025 | 0.1900 | 0.8100 | 0.0000 |
| 3 | step0032_pct050 | 0.3500 | 0.6500 | 0.6500 |
| 3 | step0064_pct100 | 0.6300 | 0.3700 | 0.3700 |

## 2. Utility

| seed | arm | checkpoint | task_success | drop_vs_same_seed | degradation |
| --- | --- | --- | --- | --- | --- |
| 1 | decay_20p_to_5p_s1 | step0032_pct050 | 0.7333 | 0.0667 | 0.0000 |
| 1 | decay_20p_to_5p_s1 | step0064_pct100 | 0.7667 | 0.0667 | 0.0000 |
| 2 | decay_20p_to_5p_s2 | step0032_pct050 | 0.8000 | 0.0333 | 0.0000 |
| 2 | decay_20p_to_5p_s2 | step0064_pct100 | 0.8333 | 0.0000 | 0.0000 |
| 3 | decay_20p_to_5p_s3 | step0032_pct050 | 0.7333 | 0.1333 | 0.0000 |
| 3 | decay_20p_to_5p_s3 | step0064_pct100 | 0.8000 | 0.0667 | 0.0000 |

## 3. Mechanism Readout

| seed | refusal-compliance | refusal-action | mean_spectral | top_ratio | cos16to64 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.5577 | -3.0445 | 0.1157 | 0.3327 | 0.8310 |
| 2 | 0.1657 | -3.7210 | 0.1189 | 0.3376 | 0.8383 |
| 3 | -0.5096 | -3.5071 | 0.1162 | 0.3344 | 0.8494 |

At step64, refusal-compliance margin improves relative to the early checkpoint, but refusal-action remains negative. This means the model is not fully converted from action mode to refusal mode, which explains why margin movement can coexist with unsafe behavior.

## 4. Degradation Audit

| arm | checkpoint | generic_refusal_rate | interpretation |
| --- | --- | --- | --- |
| early_20p_then_12p5 | step0032_pct050 | 0.8125 | real_repetitive_refusal_degradation |
| early_20p_then_12p5 | step0064_pct100 | 0.8039 | real_repetitive_refusal_degradation |
| early_20p_then_15p | step0032_pct050 | 0.8514 | real_repetitive_refusal_degradation |
| early_20p_then_15p | step0064_pct100 | 0.8254 | real_repetitive_refusal_degradation |

## 5. Decision

Step64 aggregate:

```text
mean unsafe = 0.6267
mean refusal = 0.3733
mean safety degradation = 0.2900
mean utility drop = 0.0444
max seed unsafe = 0.7500
```

decay_20p_to_5p does not satisfy the reproducible partial-control criteria across seeds.

Strong schedule degradation is mostly real repetitive-refusal degradation in this heuristic audit.

Because Block A fails the non-degradation criterion and Block B confirms the stronger schedules are dominated by repetitive refusal degradation, Step2E should stop schedule expansion under the stated stop rule. Do not run the intermediate decay targets unless a stronger external judge overturns the degradation audit.

Current Step2 interpretation: anchor mixing is a useful mechanistic diagnostic and partial control lever, but this run does not establish a stable non-degraded training-time control recipe.
