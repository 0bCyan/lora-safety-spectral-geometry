# Step 2B Training-time Regularization Pilot Plan

## 0. Decision

Step2A found a family-separated early transition:

```text
risk transition window: step 6 -> step 16
risk full100 unsafe: step6 = 0.7367, step16 = 0.9967
risk full100 refusal: step6 = 0.2567, step16 = 0.0033
risk refusal-compliance margin: step6 = 0.8840, step16 = -0.7574
normal seed1 step16 margin = +1.9853
control seed1 step16 margin = +0.0864
```

This supports a risk-specific training-dynamics signal. Step2B should now test whether training-time regularization can change that trajectory.

Step2B is not yet the full SpectraSafe experiment. It is a minimal causal pilot:

> If we constrain the early training dynamics, can we delay or reduce the risk step6->step16 unsafe transition?

## 1. Goal

Test whether simple training-time penalties can prevent or weaken the early risk transition.

Primary target:

```text
risk seed 1/2/3
checkpoint window = step6 -> step16
metric = unsafe/refusal and refusal-compliance margin
```

The desired signal is:

```text
regularized risk at step16 has lower unsafe than vanilla risk
regularized risk keeps refusal-compliance margin less negative or positive
invalid/degradation does not rise substantially
```

## 2. Experimental Arms

### 2.1 Baseline

Already completed:

```text
vanilla risk LoRA
seeds = 1,2,3
steps = 64
checkpoints = 0,6,16,32,48,64
```

Use these as the fixed baseline:

```text
experiments/step2_mechanism_dynamics/results/4070ti/dynamics_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti/full_eval_transition_summary.csv
```

### 2.2 LoRA norm penalty

Purpose:

```text
ordinary magnitude regularization baseline
```

Penalty:

```text
lambda_lora_norm * sum(||A||_F^2 + ||B||_F^2)
```

Pilot strengths:

```text
lambda_lora_norm = 1e-4
lambda_lora_norm = 1e-3
```

### 2.3 Spectral mean penalty

Purpose:

```text
test whether controlling overall LoRA update spectral growth changes the early transition
```

Penalty over LoRA update matrices:

```text
Delta W = B @ A
lambda_spectral_mean * mean_modules(||Delta W||_2)
```

Pilot strengths:

```text
lambda_spectral_mean = 1e-4
lambda_spectral_mean = 1e-3
```

### 2.4 Spectral spike penalty

Purpose:

```text
test whether suppressing top spectral spikes is more useful than global mean control
```

Penalty:

```text
lambda_spectral_top * mean_top_k_modules(||Delta W||_2)
top_k_modules = 20
```

Pilot strengths:

```text
lambda_spectral_top = 1e-4
lambda_spectral_top = 1e-3
```

## 3. Minimum Run Matrix

Run a narrow pilot first:

| arm | seeds | checkpoints | dense eval | full eval |
|---|---|---|---|---|
| vanilla risk | done: 1,2,3 | done | done | done |
| LoRA norm 1e-4 | 1 | 0,6,16,64 | 30 | 100 at 6,16 |
| LoRA norm 1e-3 | 1 | 0,6,16,64 | 30 | 100 at 6,16 |
| spectral mean 1e-4 | 1 | 0,6,16,64 | 30 | 100 at 6,16 |
| spectral mean 1e-3 | 1 | 0,6,16,64 | 30 | 100 at 6,16 |
| spectral top 1e-4 | 1 | 0,6,16,64 | 30 | 100 at 6,16 |
| spectral top 1e-3 | 1 | 0,6,16,64 | 30 | 100 at 6,16 |

After this sweep, expand only the best one or two arms to:

```text
seeds = 1,2,3
full checkpoints = 0,6,16,32,48,64
full100 eval = 0,6,16,64
```

This keeps Step2B from becoming too expensive before there is evidence of a useful intervention.

## 4. Metrics

### 4.1 Behavior metrics

Required:

```text
unsafe_compliance_rate
refusal_rate
invalid_rate
degradation_rate
average_response_chars
```

Primary comparison:

```text
regularized step16 vs vanilla step16
regularized step6->step16 transition vs vanilla step6->step16 transition
```

### 4.2 Logit margin metrics

Required:

```text
mean_refusal_logprob
mean_compliance_logprob
mean_action_logprob
mean_refusal_minus_compliance
mean_refusal_minus_action
```

Primary success signal:

```text
vanilla risk step16 mean_refusal_minus_compliance = -0.7574
regularized risk step16 should be closer to zero or positive
```

### 4.3 Spectral metrics

Required:

```text
mean_delta_spectral_norm
max_delta_spectral_norm
mean_top_singular_ratio
mean_effective_rank
```

Use spectral metrics to verify that the penalty actually changes the intended target. Do not count spectral reduction alone as safety success.

### 4.4 Utility sanity check

For any arm that improves safety, run a small utility sanity eval:

```text
utility prompts = 30
compare against vanilla risk and normal seed1
```

Reason:

```text
unsafe reduction caused by output collapse or invalid/degraded behavior is not a safety win
```

## 5. Success Criteria

Step2B pilot is positive if an arm meets all of:

```text
step16 full100 unsafe <= vanilla step16 unsafe - 0.10
step16 full100 refusal >= vanilla step16 refusal + 0.10
step16 mean_refusal_minus_compliance improves by >= 0.50
invalid_rate <= 0.05
degradation_rate <= 0.05
```

Weak positive:

```text
step16 unsafe improves by >= 0.05
and margin improves by >= 0.25
and invalid/degradation remain controlled
```

Negative:

```text
unsafe remains near 1.0 at step16
or margin still flips strongly negative
or safety improves only through invalid/degraded outputs
```

## 6. Interpretation Rules

If LoRA norm penalty works as well as spectral penalties:

```text
do not call the effect SpectraSafe-specific
interpret it as generic regularization
```

If spectral mean/top works better than LoRA norm:

```text
candidate evidence for spectral-targeted training-time control
expand to three seeds and normal/control utility checks
```

If no arm works:

```text
the Step2A mechanism signal may be diagnostic but not easily controllable by simple spectral penalties
consider logit-margin-aware or data-level interventions instead
```

## 7. Implementation Notes

Add a new config family:

```text
experiments/step2_mechanism_dynamics/configs/4070ti_qwen3_4b_step2b_regularization.json
```

Extend training script with:

```text
regularization_type = none | lora_norm | spectral_mean | spectral_top
regularization_lambda
spectral_top_k_modules
regularization_start_step
regularization_end_step
```

For the first pilot, apply regularization through the whole 64-step run:

```text
regularization_start_step = 0
regularization_end_step = 64
```

If whole-run regularization shows any signal, run a follow-up targeting the transition window:

```text
regularization_start_step = 6
regularization_end_step = 16
```

## 8. Deliverables

Expected files:

```text
experiments/step2_mechanism_dynamics/results/4070ti/regularization_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti/regularization_full100_summary.csv
experiments/step2_mechanism_dynamics/reports/Step2B_regularization_pilot_report.md
```

Report must include:

```text
arm-by-arm behavior table
arm-by-arm margin table
spectral target verification
invalid/degradation audit
recommendation: expand, revise, or stop
```

## 9. Next Decision

If Step2B finds a clean positive arm:

```text
expand best arm to seeds 1,2,3
add normal/control utility checks
prepare Step2C larger validation plan
```

If Step2B is negative:

```text
write Step2B negative summary
avoid claiming SpectraSafe effectiveness
return to mechanism analysis or judge/data improvements
```

