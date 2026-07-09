# Step 2F SpectraSafe Dynamic Replication Plan

## 0. Current State

Step2E closes the anchor-schedule search branch under the current stopping rule.

Key Step2E result:

```text
decay_20p_to_5p, step64, seeds 1/2/3
mean unsafe = 0.6267
mean refusal = 0.3733
mean safety degradation = 0.2900
mean utility drop = 0.0444
```

Interpretation:

```text
anchor mixing is a useful training-dynamics probe and partial control lever
but it does not establish a stable non-degraded training-time recipe
```

Therefore Step2 should stop expanding anchor ratios/schedules for now and return to the original method:

```text
SpectraSafe = mean spectral regularization + max-layer spectral regularization
```

Step2B did not test full SpectraSafe. It only tested weak single-term or generic penalties:

```text
lora_norm 1e-4 / 1e-3
spectral_mean 1e-4 / 1e-3
spectral_top 1e-4 / 1e-3
```

That failure should be interpreted as a weak-regularization baseline failure, not as evidence against the original SpectraSafe method.

## 1. Goal

Run a dynamic replication of the original SpectraSafe idea in the current Qwen3-4B risk setting.

Main questions:

```text
Q1. Can mean+max SpectraSafe alter the early risk transition at step6 -> step16?
Q2. Can SpectraSafe reduce final step64 unsafe behavior without safety degradation?
Q3. Are the mean and max terms doing different work?
Q4. Is SpectraSafe more specific than generic Frobenius/orthogonal regularization?
Q5. Does SpectraSafe combine with anchor scheduling, or is anchor scheduling merely a separate partial-control branch?
```

Primary setting:

```text
model = Qwen3-4B
family = risk
machine = 4070 Ti
seeds = seed1 pilot, then seeds 1/2/3 for positive arms
checkpoints = step0, step6, step16, step32, step64
```

## 2. Implementation Requirements

The current Step2 training script already supports:

```text
lora_norm
spectral_mean
spectral_top
```

Step2F needs the following additions or explicit aliases:

```text
spectrasafe_mean_max:
    loss = L_task
         + lambda_mean * mean_l(sigma_hat_l)
         + lambda_max  * max_l(sigma_hat_l)

sigma_hat_l:
    ||B_l||_2 * ||A_l||_2
```

Optional but useful:

```text
frobenius_control:
    LoRA update or LoRA-factor Frobenius shrinkage

orthogonal_control:
    orthogonality penalty on LoRA factors, used only as a generic geometry control
```

Every training run should log:

```text
task_loss
mean_spectral_penalty
max_spectral_penalty
total_regularization_penalty
total_loss
```

Purpose:

```text
avoid repeating Step2B's ambiguity where penalties may have been too weak to change geometry
```

## 3. Experimental Arms

### 3.1 Baselines

Use fixed vanilla baselines where possible.

```text
vanilla:
    L = L_task
    purpose = risk-drift baseline

anchor_schedule_control:
    decay_20p_to_5p
    purpose = best current non-anchor-free partial-control reference
```

### 3.2 SpectraSafe arms

Primary v0.0.8-style arms:

```text
mean_only:
    lambda_mean = 3.0
    lambda_max = 0.0

max_only:
    lambda_mean = 0.0
    lambda_max = 0.2

spectrasafe_mean_plus_max:
    lambda_mean = 3.0
    lambda_max = 0.2
```

These values come from the v0.0.8 idea document and should be treated as the first serious replication setting, not as the final tuned optimum.

### 3.3 Strength sanity grid

If the v0.0.8 setting is unstable or clearly too weak/too strong, run a small seed1 grid before seed expansion:

```text
spectrasafe_low:
    lambda_mean = 0.3
    lambda_max = 0.02

spectrasafe_mid:
    lambda_mean = 1.0
    lambda_max = 0.07

spectrasafe_high:
    lambda_mean = 3.0
    lambda_max = 0.2
```

Decision:

```text
expand the smallest setting that changes spectral geometry and improves step16 safety without utility collapse
```

### 3.4 Generic regularization controls

Run only after SpectraSafe arms are executable.

```text
frobenius_control:
    purpose = test whether ordinary shrinkage explains the result

orthogonal_control:
    purpose = test whether any generic geometry constraint works
```

These are controls, not proposed final methods.

### 3.5 Combined arm

Run only if `spectrasafe_mean_plus_max` shows useful safety movement or if anchor schedule remains the best partial control:

```text
spectrasafe_plus_anchor:
    SpectraSafe mean+max regularization
    plus decay_20p_to_5p anchor schedule
```

Purpose:

```text
test whether spectral geometry control and safety-anchor scheduling are complementary
```

## 4. Evaluation

For every seed1 pilot arm:

```text
dense safety eval = 30 prompts at step0, step6, step16, step32, step64
full safety eval = 100 prompts at step16, step32, step64
utility eval = 30 prompts at step32 and step64
logit margins = step0, step6, step16, step32, step64
spectral metrics = step0, step6, step16, step32, step64
temporal subspace summary = step6->16, step16->64
```

For expanded positive arms:

```text
seeds = 1/2/3
full safety eval = 100 prompts at step16, step32, step64
utility eval = 30 prompts at step32 and step64
```

Optional high-confidence eval:

```text
full safety eval = 200 prompts at step64
utility eval = 100 prompts at step64
```

## 5. Success Criteria

### 5.1 Early transition control

Positive at step16:

```text
step16 unsafe <= 0.50
step16 refusal >= 0.40
safety degradation <= 0.05
utility drop <= 0.10
mean spectral norm reduced versus vanilla
```

Weak positive at step16:

```text
step16 unsafe <= 0.75
step16 refusal >= 0.20
safety degradation <= 0.05
utility drop <= 0.10
clear spectral geometry movement versus vanilla
```

### 5.2 Final checkpoint control

Strong positive at step64:

```text
step64 unsafe <= 0.50
step64 refusal >= 0.35
safety degradation <= 0.05
utility drop <= 0.10
holds across seeds 1/2/3
```

Partial useful control:

```text
step64 unsafe <= 0.75
step64 refusal >= 0.20
safety degradation <= 0.05
utility drop <= 0.10
holds across seeds 1/2/3
```

Negative:

```text
unsafe remains >= 0.90 after step16
or safety improves only through repetitive refusal degradation
or utility fails
or spectral geometry does not move relative to vanilla
```

## 6. Mechanism Readout

The report should not only compare final safety rates. It must answer whether SpectraSafe changes the trajectory.

Required mechanism tables:

```text
behavior trajectory:
    unsafe/refusal/invalid/degradation by checkpoint

geometry trajectory:
    mean spectral norm
    max spectral norm
    top singular ratio
    effective rank

logit trajectory:
    refusal-compliance margin
    refusal-action margin

subspace trajectory:
    step6->16 cosine
    step16->64 cosine
```

Key interpretation to test:

```text
If SpectraSafe improves refusal-compliance but refusal-action remains strongly negative,
then the method may be moving a diagnostic margin without fully changing generation mode.

If SpectraSafe reduces unsafe while also improving refusal-action,
then it is more likely changing the actual refusal/action competition.
```

## 7. Recommended Execution Order

Run:

```text
1. Implement or verify spectrasafe_mean_max regularization.
2. Run a 5-10 step smoke test to confirm penalties are nonzero and backpropagate.
3. Run seed1 arms:
       vanilla reference
       mean_only
       max_only
       spectrasafe_mean_plus_max
4. Evaluate step16/32/64 safety, utility, spectral, logit, subspace.
5. Add Frobenius and orthogonal controls if SpectraSafe shows movement.
6. Run spectrasafe_plus_anchor only if either SpectraSafe or anchor schedule is meaningfully positive.
7. Expand the best arm to seeds 1/2/3.
8. Write Step2F report.
```

Stop rule:

```text
if v0.0.8-style SpectraSafe does not move spectral geometry or behavior at seed1,
do not expand immediately;
run the strength sanity grid first.
```

## 8. Expected Deliverables

Expected files:

```text
experiments/step2_mechanism_dynamics/results/4070ti_step2f/spectrasafe_dynamic_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti_step2f/spectrasafe_full100_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti_step2f/utility_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti_step2f/mechanism_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti_step2f/regularization_loss_trace.csv
experiments/step2_mechanism_dynamics/reports/Step2F_spectrasafe_dynamic_replication_report.md
```

The report must state:

```text
whether original SpectraSafe works in the current Qwen3-4B risk setting
whether mean/max terms have distinguishable roles
whether generic regularization controls reproduce the effect
whether anchor schedule is complementary or only a separate partial-control lever
whether Step2 should proceed to multi-backbone validation or revise the method
```

## 9. Decision After Step2F

If SpectraSafe succeeds:

```text
move to multi-seed expansion and lightweight larger-backbone validation
frame Step1-Step2 as mechanism evidence for SpectraSafe
```

If SpectraSafe only partially succeeds:

```text
test SpectraSafe + anchor schedule
consider dynamic or windowed SpectraSafe
```

If SpectraSafe fails:

```text
do not claim the original dual spectral regularizer controls this harder setting
use Step1-Step2 as evidence that new method design is needed
```
