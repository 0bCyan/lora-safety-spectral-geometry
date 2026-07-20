# Step2A Family Controls Follow-up Report

## 1. Scope

```text
model = Qwen3-4B
machine = 4070 Ti
family controls = normal seed 1, control seed 1
risk validation = seeds 1/2/3, checkpoints step0/6/16/64
dense eval = 30 safety prompts
full transition eval = 100 safety prompts
```

This follow-up tests whether the early risk transition from the Step2A pilot is family-specific or a generic LoRA training artifact.

## 2. Family-control Dynamics

| family | checkpoint | step | unsafe | refusal | spectral | refusal-compliance |
| --- | --- | --- | --- | --- | --- | --- |
| control | step0000_pct000 | 0 | 0.9000 | 0.1000 | 0.0000 | 1.3154 |
| control | step0006_pct009 | 6 | 0.9000 | 0.1000 | 0.0379 | 1.3773 |
| control | step0016_pct025 | 16 | 0.7000 | 0.2333 | 0.0764 | 0.0864 |
| control | step0032_pct050 | 32 | 0.9000 | 0.0667 | 0.0999 | -0.6684 |
| control | step0048_pct075 | 48 | 0.9667 | 0.0333 | 0.1076 | -0.7518 |
| control | step0064_pct100 | 64 | 0.7000 | 0.0667 | 0.1090 | -0.7682 |
| normal | step0000_pct000 | 0 | 0.9000 | 0.1000 | 0.0000 | 1.3154 |
| normal | step0006_pct009 | 6 | 0.4333 | 0.5667 | 0.0433 | 1.0244 |
| normal | step0016_pct025 | 16 | 0.3333 | 0.6333 | 0.0748 | 1.9853 |
| normal | step0032_pct050 | 32 | 0.6000 | 0.4000 | 0.0933 | 1.6713 |
| normal | step0048_pct075 | 48 | 0.7000 | 0.3000 | 0.0990 | 1.8471 |
| normal | step0064_pct100 | 64 | 0.7000 | 0.3000 | 0.1002 | 1.8045 |
| risk | step0000_pct000 | 0 | 0.9000 | 0.1000 | 0.0000 | 1.3154 |
| risk | step0006_pct009 | 6 | 0.7000 | 0.3000 | 0.0432 | 0.8840 |
| risk | step0016_pct025 | 16 | 1.0000 | 0.0000 | 0.0771 | -0.7574 |
| risk | step0032_pct050 | 32 | 1.0000 | 0.0000 | 0.0967 | -1.4283 |
| risk | step0048_pct075 | 48 | 1.0000 | 0.0000 | 0.1027 | -1.4880 |
| risk | step0064_pct100 | 64 | 1.0000 | 0.0000 | 0.1040 | -1.6062 |

Main observations:

- Risk remains the cleanest early unsafe transition: mean unsafe is 0.7000 at step6 and 1.0000 by step16 across three seeds.
- Normal seed1 moves in the opposite direction during the transition window: unsafe drops to 0.3333 and refusal rises to 0.6333 at step16, while the refusal-compliance margin stays positive.
- Control seed1 is noisier: unsafe remains high at step6, dips at step16, and the margin only turns negative after step32. It does not match the clean risk saturation pattern.

## 3. Risk Full-window Validation

| checkpoint | step | dense_unsafe | full_unsafe | dense_refusal | full_refusal | full_invalid |
| --- | --- | --- | --- | --- | --- | --- |
| step0000_pct000 | 0 | 0.9000 | 0.9100 | 0.1000 | 0.0600 | 0.0300 |
| step0006_pct009 | 6 | 0.7000 | 0.7367 | 0.3000 | 0.2567 | 0.0067 |
| step0016_pct025 | 16 | 1.0000 | 0.9967 | 0.0000 | 0.0033 | 0.0000 |
| step0064_pct100 | 64 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

The full 100-prompt eval preserves the pilot transition window. Risk unsafe compliance is lower at step6 than at step16, and all three seeds reach full unsafe saturation by step16.

## 4. Interpretation

The follow-up supports a risk-specific training-dynamics signal rather than a generic LoRA artifact:

- Risk flips from partial refusal to saturated unsafe compliance between step6 and step16.
- Normal does not show the same early unsafe saturation and retains a positive refusal-compliance margin through the checkpoint schedule.
- Control shows instability and late negative margins, but the behavior trajectory is not the same clean early transition.
- The 100-prompt risk validation confirms that the 30-prompt pilot was not merely a dense-subset artifact.

This remains descriptive evidence. The next causal step should target the step6 to step16 window with a Step2B training-time regularization pilot.

## 5. New Deliverables

```text
experiments/step2_mechanism_dynamics/results/4070ti/family_control_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti/full_eval_transition_summary.csv
experiments/step2_mechanism_dynamics/reports/Step2A_family_controls_report.md
```
