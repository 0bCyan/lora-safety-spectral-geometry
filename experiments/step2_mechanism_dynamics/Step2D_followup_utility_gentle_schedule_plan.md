# Step 2D Follow-up: Utility Audit and Gentle Schedule Plan

## 0. Current State

The Step2D persistence and schedule pilot has clarified two facts:

```text
anchor_mix_20p is strongly positive at step16
anchor_mix_20p rebounds by step32-step64 if anchors are removed
```

Persistence result:

```text
anchor_mix_20p step32 full100 unsafe mean = 0.9933
anchor_mix_20p step48 full100 unsafe mean = 1.0000
anchor_mix_20p step64 full100 unsafe mean = 1.0000
```

Schedule pilot:

```text
constant_20p step64 unsafe = 0.3300, refusal = 0.6700, degradation = 0.6700
early_20p_then_10p step64 unsafe = 0.7700, refusal = 0.2300, degradation = 0.0000
early_30p_then_10p step64 unsafe = 0.7800, refusal = 0.2200, degradation = 0.0000
```

Interpretation:

```text
continuing anchors are needed
but strong continuing anchors may trade safety for degraded behavior
```

Therefore the next experiment should not simply expand `constant_20p` to more seeds. First determine whether the safety gain is meaningful task-preserving refusal or a degradation artifact.

## 1. Goal

Answer three questions:

```text
Q1. Is constant_20p genuinely safer, or mostly degraded?
Q2. Can a gentler continuing-anchor schedule preserve part of the safety gain without degradation?
Q3. Which arm is worth expanding to seeds 1/2/3?
```

Primary model and setting:

```text
model = Qwen3-4B
family = risk
machine = 4070 Ti
seed pilot = 1
reference checkpoints = step16, step32, step64
```

## 2. Block A: Task-quality Utility Audit

Run a real utility-quality evaluation before any further schedule expansion.

### 2.1 Arms to evaluate

```text
vanilla risk seed1
anchor_mix_20p seed1
constant_20p seed1
early_20p_then_10p seed1
early_30p_then_10p seed1
normal/control references if available
```

Checkpoints:

```text
step16
step32
step64
```

### 2.2 Utility task sets

Use benign prompts that should not trigger safety refusal.

Minimum task families:

```text
general_qa
short_reasoning
summarization
simple_math
simple_code_help
instruction_following
```

Recommended size:

```text
30 prompts for pilot
100 prompts for selected arms
```

### 2.3 Metrics

Record:

```text
task_success_rate
unnecessary_refusal_rate
invalid_or_irrelevant_rate
degradation_rate
average_cleaned_response_chars
format_failure_rate
```

Decision labels:

```text
utility_pass:
    task_success_drop <= 0.10
    unnecessary_refusal_rate <= 0.10
    degradation_rate <= 0.05

utility_borderline:
    task_success_drop <= 0.20
    degradation_rate <= 0.15

utility_fail:
    degradation_rate > 0.15
    or unnecessary_refusal_rate > 0.25
    or task_success_drop > 0.20
```

Purpose:

```text
separate real safe behavior from degenerate refusal or low-quality responses
```

## 3. Block B: Gentle Continuing-anchor Schedule Scan

Only run after Block A confirms that `constant_20p` is utility-compromised or borderline. The goal is to find a schedule between `early_20p_then_10p` and `constant_20p`.

### 3.1 Candidate schedules

Pilot arms:

```text
early_20p_then_5p:
    anchor_mix = 20% for step6-step16
    anchor_mix = 5% for step16-step64

early_20p_then_7p5:
    anchor_mix = 20% for step6-step16
    anchor_mix = 7.5% for step16-step64

early_20p_then_12p5:
    anchor_mix = 20% for step6-step16
    anchor_mix = 12.5% for step16-step64

early_20p_then_15p:
    anchor_mix = 20% for step6-step16
    anchor_mix = 15% for step16-step64

decay_20p_to_5p:
    anchor_mix decays linearly from 20% at step16 to 5% at step64

decay_20p_to_10p:
    anchor_mix decays linearly from 20% at step16 to 10% at step64
```

Optional stronger early arm only if utility allows:

```text
early_25p_then_10p
```

Do not expand `constant_20p` until utility audit shows degradation is a judge artifact rather than real task-quality loss.

### 3.2 Evaluation

For each schedule:

```text
seed = 1
full safety eval = 100 prompts at step16, step32, step64
task-quality utility eval = 30 prompts at step32 and step64
logit margin summary = step0, step6, step16, step32, step64
spectral summary = step0, step6, step16, step32, step64
```

For promising arms:

```text
expand to seeds 1/2/3
full safety eval = 100 or 200 prompts
utility eval = 100 prompts
```

## 4. Success Criteria

### 4.1 Stable useful control

```text
step64 full100 unsafe <= 0.50
step64 full100 refusal >= 0.40
safety degradation_rate <= 0.05
utility degradation_rate <= 0.05
utility task_success_drop <= 0.10
holds for seeds 1/2/3 after expansion
```

### 4.2 Partial but useful control

```text
step64 full100 unsafe <= 0.75
step64 full100 refusal >= 0.20
safety degradation_rate <= 0.05
utility degradation_rate <= 0.10
utility task_success_drop <= 0.15
```

Interpretation:

```text
not a final recipe, but enough to justify schedule refinement
```

### 4.3 Failure

```text
unsafe remains >= 0.90 after step32
or safety improves only when degradation is high
or utility quality fails
```

Interpretation:

```text
anchor mixing is a diagnostic/control lever but not yet a deployable training recipe
```

## 5. Mechanism Probes for Selected Arms

Only run heavier mechanism probes after a schedule passes utility filtering.

Compare:

```text
vanilla risk
anchor_mix_20p transient arm
early_20p_then_10p partial-control arm
best gentle schedule
constant_20p degraded-safety arm
```

Mechanism questions:

```text
Does the best gentle schedule keep refusal behavior without forcing short/degraded outputs?
Does the logit margin remain predictive after step16?
Does continuing anchor alter the post-step16 subspace trajectory?
Are safety gains associated with specific layer/module spectral changes?
```

Minimum mechanism outputs:

```text
mechanism_comparison_summary.csv
subspace summary for selected arms
logit/behavior disagreement table
```

## 6. Recommended Execution Order

Run:

```text
1. Utility-quality audit for existing Step2D arms.
2. Gentle schedule scan on seed1.
3. Pick the best non-degraded schedule.
4. Expand only that schedule to seeds 1/2/3.
5. Run mechanism probes for the selected schedule.
```

Stop rule:

```text
if all schedules with step64 unsafe <= 0.75 show utility or safety degradation,
do not expand to more seeds;
write a negative/partial conclusion and return to mechanism analysis.
```

## 7. Expected Deliverables

Expected files:

```text
experiments/step2_mechanism_dynamics/results/4070ti_step2d_followup/utility_quality_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti_step2d_followup/gentle_schedule_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti_step2d_followup/selected_schedule_full100_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti_step2d_followup/mechanism_comparison_summary.csv
experiments/step2_mechanism_dynamics/reports/Step2D_followup_utility_gentle_schedule_report.md
```

Report must state:

```text
whether constant_20p is genuinely useful or degraded
which gentle schedule has the best safety-utility tradeoff
whether any arm deserves seed expansion
whether Step2D can claim stable training-time safety control
```
