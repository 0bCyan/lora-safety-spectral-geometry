# Step2B Training-time Regularization Pilot Report

## 1. Scope

```text
model = Qwen3-4B
machine = 4070 Ti
family = risk
seed = 1
arms = lora_norm_1e-4, lora_norm_1e-3, spectral_mean_1e-4, spectral_mean_1e-3, spectral_top_1e-4, spectral_top_1e-3
dense eval = 30 safety prompts at step0/6/16/64
full eval = 100 safety prompts at step6/16
```

Baseline comparisons use the fixed Step2A vanilla risk summaries under `experiments/step2_mechanism_dynamics/results/4070ti`.

## 2. Behavior and Step16 Decision

| arm | checkpoint | full_unsafe | full_refusal | full_invalid | margin | decision |
| --- | --- | --- | --- | --- | --- | --- |
| lora_norm_1e-3 | step0006_pct009 | 0.7700 | 0.2200 | 0.0100 | 0.8965 |  |
| lora_norm_1e-3 | step0016_pct025 | 0.9900 | 0.0100 | 0.0000 | -1.3706 | negative |
| lora_norm_1e-4 | step0006_pct009 | 0.7500 | 0.2500 | 0.0000 | 0.8919 |  |
| lora_norm_1e-4 | step0016_pct025 | 0.9900 | 0.0100 | 0.0000 | -1.3636 | negative |
| spectral_mean_1e-3 | step0006_pct009 | 0.7600 | 0.2400 | 0.0000 | 0.8926 |  |
| spectral_mean_1e-3 | step0016_pct025 | 0.9900 | 0.0100 | 0.0000 | -1.3659 | negative |
| spectral_mean_1e-4 | step0006_pct009 | 0.7600 | 0.2300 | 0.0100 | 0.8900 |  |
| spectral_mean_1e-4 | step0016_pct025 | 0.9900 | 0.0100 | 0.0000 | -1.3728 | negative |
| spectral_top_1e-3 | step0006_pct009 | 0.7400 | 0.2500 | 0.0100 | 0.8934 |  |
| spectral_top_1e-3 | step0016_pct025 | 0.9900 | 0.0100 | 0.0000 | -1.3704 | negative |
| spectral_top_1e-4 | step0006_pct009 | 0.7500 | 0.2400 | 0.0100 | 0.8883 |  |
| spectral_top_1e-4 | step0016_pct025 | 0.9900 | 0.0100 | 0.0000 | -1.3701 | negative |

## 3. Margin Comparison at Step16

| arm | margin | delta_margin_vs_vanilla_mean | delta_unsafe_vs_vanilla_mean | delta_refusal_vs_vanilla_mean |
| --- | --- | --- | --- | --- |
| lora_norm_1e-3 | -1.3706 | -0.6132 | -0.0067 | 0.0067 |
| lora_norm_1e-4 | -1.3636 | -0.6062 | -0.0067 | 0.0067 |
| spectral_mean_1e-3 | -1.3659 | -0.6085 | -0.0067 | 0.0067 |
| spectral_mean_1e-4 | -1.3728 | -0.6154 | -0.0067 | 0.0067 |
| spectral_top_1e-3 | -1.3704 | -0.6130 | -0.0067 | 0.0067 |
| spectral_top_1e-4 | -1.3701 | -0.6127 | -0.0067 | 0.0067 |

Step2B success requires step16 full100 unsafe to improve by at least 0.10, refusal to improve by at least 0.10, refusal-compliance margin to improve by at least 0.50, and invalid/degradation rates to stay at or below 0.05. Weak positive requires at least 0.05 unsafe improvement and 0.25 margin improvement with controlled invalid/degradation.

## 4. Spectral Target Verification

| arm | checkpoint | mean_spectral | max_spectral | top_ratio | effective_rank |
| --- | --- | --- | --- | --- | --- |
| lora_norm_1e-3 | step0006_pct009 | 0.0427 | 0.0843 | 0.2760 | 6.9329 |
| lora_norm_1e-3 | step0016_pct025 | 0.0757 | 0.1512 | 0.3071 | 6.6358 |
| lora_norm_1e-4 | step0006_pct009 | 0.0430 | 0.0848 | 0.2758 | 6.9343 |
| lora_norm_1e-4 | step0016_pct025 | 0.0763 | 0.1521 | 0.3066 | 6.6405 |
| spectral_mean_1e-3 | step0006_pct009 | 0.0431 | 0.0850 | 0.2757 | 6.9356 |
| spectral_mean_1e-3 | step0016_pct025 | 0.0764 | 0.1524 | 0.3061 | 6.6444 |
| spectral_mean_1e-4 | step0006_pct009 | 0.0432 | 0.0850 | 0.2758 | 6.9352 |
| spectral_mean_1e-4 | step0016_pct025 | 0.0765 | 0.1525 | 0.3064 | 6.6426 |
| spectral_top_1e-3 | step0006_pct009 | 0.0430 | 0.0849 | 0.2754 | 6.9378 |
| spectral_top_1e-3 | step0016_pct025 | 0.0763 | 0.1516 | 0.3061 | 6.6450 |
| spectral_top_1e-4 | step0006_pct009 | 0.0431 | 0.0850 | 0.2757 | 6.9355 |
| spectral_top_1e-4 | step0016_pct025 | 0.0765 | 0.1524 | 0.3063 | 6.6430 |

Spectral metrics verify whether the penalty changed the adapter update geometry, but they are not counted as safety success by themselves.

## 5. Invalid and Degradation Audit

| arm | checkpoint | full_invalid | full_degradation | dense_invalid | dense_degradation |
| --- | --- | --- | --- | --- | --- |
| lora_norm_1e-3 | step0006_pct009 | 0.0100 | 0.0000 | 0.0000 | 0.0000 |
| lora_norm_1e-3 | step0016_pct025 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lora_norm_1e-4 | step0006_pct009 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| lora_norm_1e-4 | step0016_pct025 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| spectral_mean_1e-3 | step0006_pct009 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| spectral_mean_1e-3 | step0016_pct025 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| spectral_mean_1e-4 | step0006_pct009 | 0.0100 | 0.0000 | 0.0000 | 0.0000 |
| spectral_mean_1e-4 | step0016_pct025 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| spectral_top_1e-3 | step0006_pct009 | 0.0100 | 0.0000 | 0.0000 | 0.0000 |
| spectral_top_1e-3 | step0016_pct025 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| spectral_top_1e-4 | step0006_pct009 | 0.0100 | 0.0000 | 0.0000 | 0.0000 |
| spectral_top_1e-4 | step0016_pct025 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## 6. Recommendation

Stop this simple regularization pilot as a safety intervention. None of the six arms meets the Step2B positive or weak-positive criteria at step16. Treat the Step2A signal as diagnostic unless a stronger intervention is tested.

## 7. Deliverables

```text
experiments/step2_mechanism_dynamics/results/4070ti/regularization_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti/regularization_full100_summary.csv
experiments/step2_mechanism_dynamics/reports/Step2B_regularization_pilot_report.md
```
