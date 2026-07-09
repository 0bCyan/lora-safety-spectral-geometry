# Step 2D Anchor Persistence and Mechanism Plan

## 0. Motivation

Step2C produced the first strong training-time control result in Step2:

```text
anchor_mix_20p, step16 full100, seeds 1/2/3
unsafe = 0.2700 / 0.2300 / 0.1900
refusal = 0.7300 / 0.7700 / 0.8100
```

This is a large improvement over the vanilla risk baseline at step16:

```text
vanilla risk seed1 full100 unsafe = 0.9900
vanilla risk mean full100 unsafe = 0.9967
```

However, the dense step64 safety eval returns to unsafe saturation:

```text
anchor_mix_20p dense step64 unsafe = 1.0000 for seeds 1/2/3
```

Therefore Step2D should not search for an unrelated intervention. It should determine whether `anchor_mix_20p` is a transient delay, a tunable threshold effect, or the beginning of a stable safety-preserving training recipe.

## 1. Goal

Turn the Step2C positive arm into a rigorous result by answering four questions:

```text
Q1. Persistence: does the anchor effect survive beyond step16?
Q2. Threshold: what is the minimum anchor ratio that reliably changes behavior?
Q3. Utility: does the safety gain come from real refusal behavior rather than generic degradation?
Q4. Mechanism: why does anchor mixing work while margin-only loss fails?
```

Primary model and family:

```text
model = Qwen3-4B
family = risk
base positive arm = anchor_mix_20p
seeds = 1/2/3
```

## 2. Experimental Blocks

### 2.1 Persistence audit

Run full safety evaluation at intermediate and final checkpoints for the proven positive arm.

```text
arm = anchor_mix_20p
seeds = 1/2/3
checkpoints = step16, step32, step48, step64
full safety eval = 100 or 200 prompts per checkpoint
```

Purpose:

```text
separate "early-window delay" from "stable training-time safety control"
```

Key interpretation:

```text
if unsafe stays low through step64:
    anchor_mix_20p is a candidate stable control recipe
if unsafe rebounds after step16:
    anchor_mix_20p delays the transition but is overwritten by later risk training
```

### 2.2 Anchor-ratio threshold scan

Step2C shows a sharp pattern:

```text
5% anchor: negative
10% anchor: weak/negative
20% anchor: strongly positive at step16
```

Run a finer ratio scan around the apparent threshold.

```text
anchor_mix = 12.5%, 15%, 17.5%, 20%, 25%, 30%
seed = 1 pilot
checkpoints = step16, step32, step64
full safety eval = 100 prompts at step16 and step32
dense safety eval = 30 prompts at step0, step6, step16, step32, step64
```

Expand any promising ratio to seeds 1/2/3.

Priority rule:

```text
first expand the smallest ratio with step16 full100 unsafe <= 0.50
then expand the smallest ratio with step32 full100 unsafe <= 0.50
```

Purpose:

```text
estimate the minimum effective safety-anchor mass
test whether the effect is threshold-like rather than linear
```

### 2.3 Schedule variants if rebound is confirmed

If persistence audit shows step64 rebound, test whether safety anchors must remain present after the early transition window.

Pilot schedules:

```text
early_20p_then_0p:
    anchor_mix = 20% for step6-step16, then 0%

early_20p_then_10p:
    anchor_mix = 20% for step6-step16, then 10% for step16-step64

constant_20p:
    anchor_mix = 20% for step6-step64

early_30p_then_10p:
    anchor_mix = 30% for step6-step16, then 10% for step16-step64
```

Evaluation:

```text
seeds = 1 pilot, then expand best arm to seeds 1/2/3
checkpoints = step16, step32, step48, step64
full safety eval = 100 prompts at step16, step32, step64
```

Purpose:

```text
determine whether the control needs only an early correction or a continuing anchor
```

### 2.4 Utility and task-quality evaluation

The current utility sanity labels are safety-oriented and cannot establish task quality. Replace them with a task-quality utility eval.

Utility sets:

```text
general QA
short reasoning
summarization
simple math
simple code/helpfulness
benign instruction following
```

Metrics:

```text
answer_validity
task_success
unnecessary_refusal
invalid_or_irrelevant
degradation
average_clean_response_chars
```

Required comparisons:

```text
vanilla risk seed1 step16/64
anchor_mix_20p seeds 1/2/3 step16/64
best persistence/schedule arm step16/32/64
normal/control Step2A references when available
```

Purpose:

```text
ensure the safety improvement is not just shorter, invalid, or over-refusal behavior
```

### 2.5 Mechanism comparison

Compare four groups:

```text
vanilla risk
anchor_mix_20p
best persistence/schedule arm
margin-only arms from Step2C
```

Mechanism probes:

```text
refusal-compliance margin over checkpoints
refusal-action margin over checkpoints
first-token versus generated-behavior disagreement
activation/subspace overlap for safety prompts
spectral summaries by layer/module
anchor versus risk gradient direction on matched minibatches if feasible
```

Purpose:

```text
explain why margin-only can improve the measured first-token margin while behavior remains unsafe
identify which signal is diagnostic and which signal is controllable
```

## 3. Success Criteria

### 3.1 Strong positive

```text
step64 full100 unsafe <= 0.50
step64 full100 refusal >= 0.40
invalid_rate <= 0.05
degradation_rate <= 0.05
utility task_success does not drop by more than 10 percentage points versus vanilla/normal reference
result holds for seeds 1/2/3
```

### 3.2 Positive but transient

```text
step16 full100 unsafe <= 0.50 for seeds 1/2/3
step32 or step64 rebounds above 0.90 unsafe
utility degradation controlled at step16
```

Interpretation:

```text
anchor mixing controls the early transition but must be maintained or rescheduled
```

### 3.3 Negative

```text
no ratio or schedule keeps step16 unsafe below 0.90
or improvement is explained by invalid/degraded outputs
or utility task_success collapses
```

## 4. Recommended Execution Order

Run in this order:

```text
1. Persistence audit for anchor_mix_20p seeds 1/2/3 at step16/32/48/64.
2. Task-quality utility eval for anchor_mix_20p step16 and step64.
3. If rebound is confirmed, run schedule variants.
4. Run anchor-ratio threshold scan around 15%-30%.
5. Run mechanism comparison after the best schedule/ratio is selected.
```

Reason:

```text
the persistence result determines whether Step2D is mainly a stability problem or a ratio-selection problem
```

## 5. Expected Deliverables

Expected result files:

```text
experiments/step2_mechanism_dynamics/results/4070ti_step2d/anchor_persistence_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti_step2d/anchor_ratio_scan_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti_step2d/anchor_schedule_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti_step2d/utility_quality_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti_step2d/mechanism_comparison_summary.csv
```

Expected report:

```text
experiments/step2_mechanism_dynamics/reports/Step2D_anchor_persistence_mechanism_report.md
```

The report must include:

```text
persistence table
ratio threshold table
schedule comparison table
utility-quality audit
mechanism comparison
final decision: stable recipe, transient control, or negative
```

## 6. Decision After Step2D

If Step2D finds a stable schedule:

```text
move to larger prompt set and stronger judge
prepare cross-model or larger-model replication
frame anchor mixing as the first actionable training-time control result
```

If Step2D only finds transient control:

```text
study overwrite dynamics after step16
focus on continuing-anchor schedules and curriculum design
avoid claiming final safety control
```

If Step2D fails:

```text
keep Step2C as evidence that early risk dynamics are data-balance sensitive
return to mechanism diagnosis before new interventions
```
