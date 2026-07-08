# Step2C Margin-aware Intervention Report

## 1. Scope

```text
model = Qwen3-4B
machine = 4070 Ti
family = risk
pilot seed = 1
expanded positive arm = anchor_mix_20p seeds 1/2/3
pilot arms = anchor_mix_5p, anchor_mix_10p, anchor_mix_20p, margin_m0_lam0p1, margin_m0_lam0p5, margin_m0p5_lam0p1, margin_m0p5_lam0p5
intervention window = step6 through step16
dense safety eval = 30 prompts at step0/6/16/64
full safety eval = 100 prompts at step6/16
```

## 2. Behavior Decision

| arm | seed | checkpoint | full_unsafe | full_refusal | full_invalid | margin | decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_mix_10p | 1 | step0006_pct009 | 0.7600 | 0.2400 | 0.0000 | 0.8928 |  |
| anchor_mix_10p | 1 | step0016_pct025 | 0.9500 | 0.0500 | 0.0000 | -0.7019 | negative |
| anchor_mix_20p | 1 | step0006_pct009 | 0.7600 | 0.2400 | 0.0000 | 0.8928 |  |
| anchor_mix_20p | 2 | step0006_pct009 | 0.7300 | 0.2700 | 0.0000 | 0.8381 |  |
| anchor_mix_20p | 3 | step0006_pct009 | 0.7900 | 0.2000 | 0.0100 | 1.0290 |  |
| anchor_mix_20p | 1 | step0016_pct025 | 0.2700 | 0.7300 | 0.0000 | -0.0513 | positive |
| anchor_mix_20p | 2 | step0016_pct025 | 0.2300 | 0.7700 | 0.0000 | 0.4720 | positive |
| anchor_mix_20p | 3 | step0016_pct025 | 0.1900 | 0.8100 | 0.0000 | 0.7987 | positive |
| anchor_mix_5p | 1 | step0006_pct009 | 0.7600 | 0.2400 | 0.0000 | 0.8928 |  |
| anchor_mix_5p | 1 | step0016_pct025 | 0.9900 | 0.0100 | 0.0000 | -1.0699 | negative |
| margin_m0_lam0p1 | 1 | step0006_pct009 | 0.7600 | 0.2400 | 0.0000 | 0.8928 |  |
| margin_m0_lam0p1 | 1 | step0016_pct025 | 0.9900 | 0.0100 | 0.0000 | 12.0859 | negative |
| margin_m0_lam0p5 | 1 | step0006_pct009 | 0.7600 | 0.2400 | 0.0000 | 0.8928 |  |
| margin_m0_lam0p5 | 1 | step0016_pct025 | 0.9900 | 0.0100 | 0.0000 | 12.4632 | negative |
| margin_m0p5_lam0p1 | 1 | step0006_pct009 | 0.7600 | 0.2400 | 0.0000 | 0.8928 |  |
| margin_m0p5_lam0p1 | 1 | step0016_pct025 | 0.9900 | 0.0100 | 0.0000 | 12.8909 | negative |
| margin_m0p5_lam0p5 | 1 | step0006_pct009 | 0.7600 | 0.2400 | 0.0000 | 0.8928 |  |
| margin_m0p5_lam0p5 | 1 | step0016_pct025 | 0.9900 | 0.0100 | 0.0000 | 13.0218 | negative |

## 3. Margin Table

| arm | seed | step16_margin | delta_margin_vs_vanilla | delta_unsafe_vs_vanilla | delta_refusal_vs_vanilla |
| --- | --- | --- | --- | --- | --- |
| anchor_mix_10p | 1 | -0.7019 | 0.6825 | -0.0400 | 0.0400 |
| anchor_mix_20p | 1 | -0.0513 | 1.3331 | -0.7200 | 0.7200 |
| anchor_mix_20p | 2 | 0.4720 | 1.8564 | -0.7600 | 0.7600 |
| anchor_mix_20p | 3 | 0.7987 | 2.1831 | -0.8000 | 0.8000 |
| anchor_mix_5p | 1 | -1.0699 | 0.3145 | 0.0000 | 0.0000 |
| margin_m0_lam0p1 | 1 | 12.0859 | 13.4703 | 0.0000 | 0.0000 |
| margin_m0_lam0p5 | 1 | 12.4632 | 13.8476 | 0.0000 | 0.0000 |
| margin_m0p5_lam0p1 | 1 | 12.8909 | 14.2753 | 0.0000 | 0.0000 |
| margin_m0p5_lam0p5 | 1 | 13.0218 | 14.4062 | 0.0000 | 0.0000 |

The margin-only arms strongly improve the measured refusal-compliance margin, but their step16 full100 unsafe rate remains 0.9900. This is evidence that the current first-token margin proxy is not sufficient by itself to control behavior.

## 4. Spectral Side Effects

| arm | seed | checkpoint | mean_spectral | max_spectral | top_ratio | effective_rank |
| --- | --- | --- | --- | --- | --- | --- |
| anchor_mix_10p | 1 | step0006_pct009 | 0.0432 | 0.0850 | 0.2758 | 6.9349 |
| anchor_mix_10p | 1 | step0016_pct025 | 0.0780 | 0.1566 | 0.3076 | 6.6346 |
| anchor_mix_20p | 1 | step0006_pct009 | 0.0432 | 0.0850 | 0.2758 | 6.9349 |
| anchor_mix_20p | 2 | step0006_pct009 | 0.0398 | 0.0786 | 0.2749 | 6.9831 |
| anchor_mix_20p | 3 | step0006_pct009 | 0.0430 | 0.0858 | 0.2782 | 6.9140 |
| anchor_mix_20p | 1 | step0016_pct025 | 0.0801 | 0.1642 | 0.3094 | 6.6211 |
| anchor_mix_20p | 2 | step0016_pct025 | 0.0805 | 0.1558 | 0.3124 | 6.6394 |
| anchor_mix_20p | 3 | step0016_pct025 | 0.0802 | 0.1628 | 0.3124 | 6.6017 |
| anchor_mix_5p | 1 | step0006_pct009 | 0.0432 | 0.0850 | 0.2758 | 6.9349 |
| anchor_mix_5p | 1 | step0016_pct025 | 0.0770 | 0.1536 | 0.3069 | 6.6392 |
| margin_m0_lam0p1 | 1 | step0006_pct009 | 0.0432 | 0.0850 | 0.2758 | 6.9349 |
| margin_m0_lam0p1 | 1 | step0016_pct025 | 0.0756 | 0.1513 | 0.3054 | 6.6417 |
| margin_m0_lam0p5 | 1 | step0006_pct009 | 0.0432 | 0.0850 | 0.2758 | 6.9349 |
| margin_m0_lam0p5 | 1 | step0016_pct025 | 0.0752 | 0.1511 | 0.3049 | 6.6457 |
| margin_m0p5_lam0p1 | 1 | step0006_pct009 | 0.0432 | 0.0850 | 0.2758 | 6.9349 |
| margin_m0p5_lam0p1 | 1 | step0016_pct025 | 0.0754 | 0.1511 | 0.3052 | 6.6434 |
| margin_m0p5_lam0p5 | 1 | step0006_pct009 | 0.0432 | 0.0850 | 0.2758 | 6.9349 |
| margin_m0p5_lam0p5 | 1 | step0016_pct025 | 0.0752 | 0.1509 | 0.3048 | 6.6464 |

## 5. Invalid and Degradation Audit

| arm | seed | checkpoint | full_invalid | full_degradation |
| --- | --- | --- | --- | --- |
| anchor_mix_10p | 1 | step0006_pct009 | 0.0000 | 0.0000 |
| anchor_mix_10p | 1 | step0016_pct025 | 0.0000 | 0.0000 |
| anchor_mix_20p | 1 | step0006_pct009 | 0.0000 | 0.0000 |
| anchor_mix_20p | 2 | step0006_pct009 | 0.0000 | 0.0000 |
| anchor_mix_20p | 3 | step0006_pct009 | 0.0100 | 0.0000 |
| anchor_mix_20p | 1 | step0016_pct025 | 0.0000 | 0.0000 |
| anchor_mix_20p | 2 | step0016_pct025 | 0.0000 | 0.0000 |
| anchor_mix_20p | 3 | step0016_pct025 | 0.0000 | 0.0000 |
| anchor_mix_5p | 1 | step0006_pct009 | 0.0000 | 0.0000 |
| anchor_mix_5p | 1 | step0016_pct025 | 0.0000 | 0.0000 |
| margin_m0_lam0p1 | 1 | step0006_pct009 | 0.0000 | 0.0000 |
| margin_m0_lam0p1 | 1 | step0016_pct025 | 0.0000 | 0.0000 |
| margin_m0_lam0p5 | 1 | step0006_pct009 | 0.0000 | 0.0000 |
| margin_m0_lam0p5 | 1 | step0016_pct025 | 0.0000 | 0.0000 |
| margin_m0p5_lam0p1 | 1 | step0006_pct009 | 0.0000 | 0.0000 |
| margin_m0p5_lam0p1 | 1 | step0016_pct025 | 0.0000 | 0.0000 |
| margin_m0p5_lam0p5 | 1 | step0006_pct009 | 0.0000 | 0.0000 |
| margin_m0p5_lam0p5 | 1 | step0016_pct025 | 0.0000 | 0.0000 |

## 6. Utility Sanity

Utility sanity was run for the positive arm across seeds 1/2/3.

| arm | seed | checkpoint | n | local_unsafe_label | local_refusal_label | invalid | degradation | avg_clean_chars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_mix_20p | 1 | step0016_pct025 | 30 | 0.7667 | 0.0333 | 0.2000 | 0.0000 | 351.6667 |
| anchor_mix_20p | 2 | step0016_pct025 | 30 | 0.7333 | 0.0333 | 0.2333 | 0.0000 | 305.4000 |
| anchor_mix_20p | 3 | step0016_pct025 | 30 | 0.7333 | 0.0333 | 0.2333 | 0.0000 | 294.1667 |
| anchor_mix_20p | 1 | step0064_pct100 | 30 | 0.7667 | 0.0000 | 0.2333 | 0.0000 | 256.8000 |
| anchor_mix_20p | 2 | step0064_pct100 | 30 | 0.7000 | 0.0333 | 0.2667 | 0.0000 | 284.0000 |
| anchor_mix_20p | 3 | step0064_pct100 | 30 | 0.7667 | 0.0000 | 0.2333 | 0.0000 | 227.8667 |

Normal/control utility checks were also run on Step2A baseline checkpoints for context.

| family | seed | checkpoint | n | local_unsafe_label | local_refusal_label | invalid | degradation | avg_clean_chars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 1 | step0016_pct025 | 30 | 0.9333 | 0.0000 | 0.0667 | 0.0000 | 320.2333 |
| control | 1 | step0064_pct100 | 30 | 0.6000 | 0.0000 | 0.4000 | 0.0000 | 123.3333 |
| normal | 1 | step0016_pct025 | 30 | 0.5667 | 0.0000 | 0.4333 | 0.0000 | 241.5333 |
| normal | 1 | step0064_pct100 | 30 | 0.6000 | 0.0000 | 0.4000 | 0.0000 | 230.1667 |

The local judge labels are safety-oriented and should not be read as utility quality labels; the useful audit signals here are invalid/degradation and response length.

## 7. Decision

anchor_mix_20p has now been expanded to seeds 1/2/3 and remains positive at step16 full100. The next step should replace this safety-oriented utility sanity label with a task-quality utility judge. Do not interpret margin-only as sufficient, because margin-only arms improved the measured margin while behavior stayed unsafe.

## 8. Deliverables

```text
experiments/step2_mechanism_dynamics/results/4070ti_step2c/margin_intervention_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti_step2c/margin_intervention_full100_summary.csv
experiments/step2_mechanism_dynamics/reports/Step2C_margin_aware_intervention_report.md
```
