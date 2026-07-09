# Step2D Follow-up Utility and Gentle Schedule Report

## 1. Utility Quality Audit

| arm | checkpoint | task_success | drop_vs_vanilla | unneeded_refusal | degradation | label |
| --- | --- | --- | --- | --- | --- | --- |
| anchor_mix_20p | step0016_pct025 | 0.7000 | 0.0667 | 0.0667 | 0.0000 | utility_pass |
| anchor_mix_20p | step0032_pct050 | 0.8333 | -0.0333 | 0.0000 | 0.0000 | utility_pass |
| anchor_mix_20p | step0064_pct100 | 0.8667 | -0.0333 | 0.0000 | 0.0000 | utility_pass |
| constant_20p | step0016_pct025 | 0.7000 | 0.0667 | 0.0667 | 0.0000 | utility_pass |
| constant_20p | step0032_pct050 | 0.7333 | 0.0667 | 0.0667 | 0.0000 | utility_pass |
| constant_20p | step0064_pct100 | 0.7333 | 0.1000 | 0.0667 | 0.0000 | utility_borderline |
| control_reference | step0016_pct025 | 0.4667 | 0.3000 | 0.0000 | 0.0000 | utility_fail |
| control_reference | step0032_pct050 | 0.4333 | 0.3667 | 0.0333 | 0.0000 | utility_fail |
| control_reference | step0064_pct100 | 0.2333 | 0.6000 | 0.0333 | 0.0000 | utility_fail |
| decay_20p_to_10p | step0032_pct050 | 0.7333 | 0.0667 | 0.0667 | 0.0000 | utility_pass |
| decay_20p_to_10p | step0064_pct100 | 0.7333 | 0.1000 | 0.0667 | 0.0000 | utility_borderline |
| decay_20p_to_5p | step0032_pct050 | 0.7333 | 0.0667 | 0.0667 | 0.0000 | utility_pass |
| decay_20p_to_5p | step0064_pct100 | 0.7667 | 0.0667 | 0.0333 | 0.0000 | utility_pass |
| early_20p_then_10p | step0016_pct025 | 0.7000 | 0.0667 | 0.0667 | 0.0000 | utility_pass |
| early_20p_then_10p | step0032_pct050 | 0.7667 | 0.0333 | 0.0333 | 0.0000 | utility_pass |
| early_20p_then_10p | step0064_pct100 | 0.7667 | 0.0667 | 0.0333 | 0.0000 | utility_pass |
| early_20p_then_12p5 | step0032_pct050 | 0.7667 | 0.0333 | 0.0333 | 0.0000 | utility_pass |
| early_20p_then_12p5 | step0064_pct100 | 0.8000 | 0.0333 | 0.0333 | 0.0000 | utility_pass |
| early_20p_then_15p | step0032_pct050 | 0.7667 | 0.0333 | 0.0333 | 0.0000 | utility_pass |
| early_20p_then_15p | step0064_pct100 | 0.8000 | 0.0333 | 0.0333 | 0.0000 | utility_pass |
| early_20p_then_5p | step0032_pct050 | 0.8333 | -0.0333 | 0.0000 | 0.0000 | utility_pass |
| early_20p_then_5p | step0064_pct100 | 0.7667 | 0.0667 | 0.0333 | 0.0000 | utility_pass |
| early_20p_then_7p5 | step0032_pct050 | 0.8000 | 0.0000 | 0.0333 | 0.0000 | utility_pass |
| early_20p_then_7p5 | step0064_pct100 | 0.7667 | 0.0667 | 0.0333 | 0.0000 | utility_pass |
| early_30p_then_10p | step0016_pct025 | 0.7333 | 0.0333 | 0.0667 | 0.0000 | utility_pass |
| early_30p_then_10p | step0032_pct050 | 0.7333 | 0.0667 | 0.0333 | 0.0000 | utility_pass |
| early_30p_then_10p | step0064_pct100 | 0.8000 | 0.0333 | 0.0333 | 0.0000 | utility_pass |
| normal_reference | step0016_pct025 | 0.6333 | 0.1333 | 0.0000 | 0.0000 | utility_borderline |
| normal_reference | step0032_pct050 | 0.7667 | 0.0333 | 0.0000 | 0.0000 | utility_pass |
| normal_reference | step0064_pct100 | 0.8333 | 0.0000 | 0.0000 | 0.0000 | utility_pass |
| vanilla_risk | step0016_pct025 | 0.7667 | 0.0000 | 0.0000 | 0.0000 | utility_pass |
| vanilla_risk | step0032_pct050 | 0.8000 | 0.0000 | 0.0000 | 0.0000 | utility_pass |
| vanilla_risk | step0064_pct100 | 0.8333 | 0.0000 | 0.0000 | 0.0000 | utility_pass |

## 2. Gentle Schedule Scan

| arm | checkpoint | safety_unsafe | safety_refusal | safety_degradation | utility_success | utility_drop | utility_degradation | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decay_20p_to_10p | step0016_pct025 | 0.2700 | 0.7300 | 0.0000 |  |  |  |  |
| decay_20p_to_10p | step0032_pct050 | 0.2600 | 0.7400 | 0.7300 | 0.7333 | 0.0667 | 0.0000 |  |
| decay_20p_to_10p | step0064_pct100 | 0.5600 | 0.4400 | 0.4400 | 0.7333 | 0.1000 | 0.0000 | degraded_control |
| decay_20p_to_5p | step0016_pct025 | 0.2700 | 0.7300 | 0.0000 |  |  |  |  |
| decay_20p_to_5p | step0032_pct050 | 0.2900 | 0.7100 | 0.7000 | 0.7333 | 0.0667 | 0.0000 |  |
| decay_20p_to_5p | step0064_pct100 | 0.7500 | 0.2500 | 0.0000 | 0.7667 | 0.0667 | 0.0000 | partial_useful_control |
| early_20p_then_12p5 | step0016_pct025 | 0.2700 | 0.7300 | 0.0000 |  |  |  |  |
| early_20p_then_12p5 | step0032_pct050 | 0.4600 | 0.5400 | 0.5200 | 0.7667 | 0.0333 | 0.0000 |  |
| early_20p_then_12p5 | step0064_pct100 | 0.5900 | 0.4100 | 0.4100 | 0.8000 | 0.0333 | 0.0000 | degraded_control |
| early_20p_then_15p | step0016_pct025 | 0.2700 | 0.7300 | 0.0000 |  |  |  |  |
| early_20p_then_15p | step0032_pct050 | 0.3600 | 0.6400 | 0.6300 | 0.7667 | 0.0333 | 0.0000 |  |
| early_20p_then_15p | step0064_pct100 | 0.4700 | 0.5300 | 0.5200 | 0.8000 | 0.0333 | 0.0000 | degraded_control |
| early_20p_then_5p | step0016_pct025 | 0.2700 | 0.7300 | 0.0000 |  |  |  |  |
| early_20p_then_5p | step0032_pct050 | 0.9500 | 0.0500 | 0.0000 | 0.8333 | -0.0333 | 0.0000 |  |
| early_20p_then_5p | step0064_pct100 | 1.0000 | 0.0000 | 0.0000 | 0.7667 | 0.0667 | 0.0000 | rebound_or_negative |
| early_20p_then_7p5 | step0016_pct025 | 0.2800 | 0.7200 | 0.0000 |  |  |  |  |
| early_20p_then_7p5 | step0032_pct050 | 0.7900 | 0.2100 | 0.0000 | 0.8000 | 0.0000 | 0.0000 |  |
| early_20p_then_7p5 | step0064_pct100 | 0.9300 | 0.0700 | 0.0000 | 0.7667 | 0.0667 | 0.0000 | rebound_or_negative |

## 3. Decision

constant_20p is utility_borderline at step64, so schedule expansion should prefer a non-degraded gentle arm.

Best seed1 gentle schedule: decay_20p_to_5p at step64 (unsafe=0.7500, refusal=0.2500, utility_drop=0.0667).

Do not expand schedules that only improve safety with high safety or utility degradation.

## 4. Mechanism Comparison

| arm | checkpoint | refusal-compliance | refusal-action | mean_spectral | max_spectral | top_ratio | eff_rank | cos_from_init | cos_from_prev | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decay_20p_to_10p | step0016_pct025 | -0.0513 | -5.6554 | 0.0801 | 0.1642 | 0.3094 | 6.6211 | 0.0151 | 0.8081 |  |
| decay_20p_to_10p | step0032_pct050 | 1.5301 | -3.2924 | 0.1069 | 0.2243 | 0.3277 | 6.4800 | 0.0139 | 0.8938 |  |
| decay_20p_to_10p | step0064_pct100 | 1.5745 | -3.0517 | 0.1158 | 0.2414 | 0.3327 | 6.4441 | 0.0140 | 0.9712 | degraded_control |
| decay_20p_to_5p | step0016_pct025 | -0.0495 | -5.6610 | 0.0801 | 0.1641 | 0.3094 | 6.6211 | 0.0151 | 0.8081 |  |
| decay_20p_to_5p | step0032_pct050 | 1.5109 | -3.3124 | 0.1069 | 0.2243 | 0.3277 | 6.4801 | 0.0138 | 0.8938 |  |
| decay_20p_to_5p | step0064_pct100 | 1.5577 | -3.0445 | 0.1157 | 0.2413 | 0.3327 | 6.4442 | 0.0140 | 0.9712 | partial_useful_control |
| early_20p_then_12p5 | step0016_pct025 | -0.0488 | -5.6578 | 0.0801 | 0.1642 | 0.3094 | 6.6211 | 0.0151 | 0.8081 |  |
| early_20p_then_12p5 | step0032_pct050 | 1.3100 | -3.6567 | 0.1064 | 0.2230 | 0.3270 | 6.4859 | 0.0140 | 0.8958 |  |
| early_20p_then_12p5 | step0064_pct100 | 1.3208 | -3.4239 | 0.1152 | 0.2402 | 0.3320 | 6.4509 | 0.0142 | 0.9707 | degraded_control |
| early_20p_then_15p | step0016_pct025 | -0.0498 | -5.6607 | 0.0801 | 0.1641 | 0.3094 | 6.6211 | 0.0151 | 0.8081 |  |
| early_20p_then_15p | step0032_pct050 | 1.4248 | -3.5126 | 0.1067 | 0.2236 | 0.3273 | 6.4828 | 0.0139 | 0.8941 |  |
| early_20p_then_15p | step0064_pct100 | 1.4764 | -3.3169 | 0.1156 | 0.2407 | 0.3324 | 6.4468 | 0.0141 | 0.9711 | degraded_control |
| early_20p_then_5p | step0016_pct025 | -0.0498 | -5.6611 | 0.0801 | 0.1641 | 0.3094 | 6.6211 | 0.0151 | 0.8081 |  |
| early_20p_then_5p | step0032_pct050 | 1.0248 | -3.9752 | 0.1050 | 0.2200 | 0.3255 | 6.4979 | 0.0142 | 0.9064 |  |
| early_20p_then_5p | step0064_pct100 | 0.9489 | -3.7716 | 0.1135 | 0.2370 | 0.3303 | 6.4644 | 0.0146 | 0.9690 | rebound_or_negative |
| early_20p_then_7p5 | step0016_pct025 | -0.0516 | -5.6602 | 0.0801 | 0.1642 | 0.3094 | 6.6211 | 0.0151 | 0.8081 |  |
| early_20p_then_7p5 | step0032_pct050 | 1.1108 | -3.8498 | 0.1055 | 0.2214 | 0.3259 | 6.4944 | 0.0142 | 0.9037 |  |
| early_20p_then_7p5 | step0064_pct100 | 1.0448 | -3.6235 | 0.1141 | 0.2384 | 0.3308 | 6.4604 | 0.0145 | 0.9676 | rebound_or_negative |

The useful-control candidate keeps utility degradation at 0 while showing only partial safety movement: refusal margin improves relative to compliance, but refusal-minus-action remains negative at step64. This supports treating decay_20p_to_5p as a partial, non-degraded control point rather than a stable safety solution.

## 5. Pending

```text
selected_schedule_full100_summary.csv: contains current seed1 candidates only; seed expansion is pending
```
