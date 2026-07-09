# Step 2E Seed Expansion and Degradation Audit Plan

## 0. Current State

Step2D follow-up completed the utility audit and gentle schedule scan.

Main result:

```text
decay_20p_to_5p is the cleanest seed1 schedule at step64
safety unsafe = 0.7500
safety refusal = 0.2500
safety degradation = 0.0000
utility task success = 0.7667
utility drop vs vanilla = 0.0667
utility degradation = 0.0000
```

Other stronger schedules reduce unsafe more, but they are confounded by safety-side degradation:

```text
early_20p_then_15p step64 unsafe = 0.4700, degradation = 0.5200
early_20p_then_12p5 step64 unsafe = 0.5900, degradation = 0.4100
decay_20p_to_10p step64 unsafe = 0.5600, degradation = 0.4400
```

Very weak continuing anchors do not control rebound:

```text
early_20p_then_5p step64 unsafe = 1.0000
early_20p_then_7p5 step64 unsafe = 0.9300
```

Therefore Step2E should not expand every schedule. It should verify the clean partial-control candidate and audit whether the stronger arms are genuinely degraded or being over-penalized by the safety judge.

## 1. Goal

Answer three questions:

```text
Q1. Is decay_20p_to_5p stable across seeds?
Q2. Is the high safety_degradation in stronger schedules real degradation or judge-label ambiguity?
Q3. Can we identify one schedule worth calling a non-degraded training-time control result?
```

Primary model and setting:

```text
model = Qwen3-4B
family = risk
machine = 4070 Ti
primary schedule = decay_20p_to_5p
seeds = 1/2/3
checkpoints = step16, step32, step64
```

## 2. Block A: Expand the Clean Candidate

Expand `decay_20p_to_5p` to seeds 1/2/3.

Schedule:

```text
anchor_mix = 20% at step6-step16
anchor_mix decays linearly from 20% at step16 to 5% at step64
```

Evaluation:

```text
full safety eval = 100 prompts at step16, step32, step64
utility eval = 30 prompts at step32 and step64
logit margin = step0, step6, step16, step32, step64
spectral metrics = step0, step6, step16, step32, step64
```

Optional if seed variance is high:

```text
full safety eval = 200 prompts at step64
utility eval = 100 prompts at step64
```

Purpose:

```text
test whether the non-degraded partial control is reproducible
```

## 3. Block B: Strong-arm Degradation Audit

Audit one or two stronger schedules rather than expanding them immediately.

Priority arms:

```text
early_20p_then_15p
early_20p_then_12p5
```

Reason:

```text
early_20p_then_15p has the best safety score but high safety_degradation
early_20p_then_12p5 is slightly weaker but may be a more moderate point
```

Audit design:

```text
seed = 1
checkpoints = step32, step64
sample = all safety prompts labeled degradation plus matched refusal/non-degradation examples
```

For each sampled response, classify:

```text
valid_refusal
safe_redirection
generic_low_quality_refusal
empty_or_truncated
off_topic_or_invalid
format_degradation
over_refusal_on_benign_content
```

Minimum outputs:

```text
manual_or_stronger_judge_label
original_local_judge_label
prompt_category
response_length
short rationale
```

Purpose:

```text
determine whether safety_degradation is a real quality failure or a judge artifact on refusal-style safety behavior
```

## 4. Block C: Intermediate Schedule Only If Needed

Run this block only if Block A shows that `decay_20p_to_5p` is too weak, and Block B shows stronger schedules are genuinely degraded.

Candidate schedules:

```text
decay_20p_to_7p5
decay_20p_to_8p
decay_20p_to_6p
```

Pilot:

```text
seed = 1
full safety eval = 100 prompts at step32 and step64
utility eval = 30 prompts at step64
```

Decision rule:

```text
only expand the best intermediate schedule if:
    step64 unsafe <= 0.70
    step64 safety_degradation <= 0.10
    utility_drop <= 0.10
```

Purpose:

```text
find a point between clean-but-weak 5% and stronger-but-degraded 10%-15%
```

## 5. Success Criteria

### 5.1 Reproducible partial control

```text
decay_20p_to_5p mean step64 unsafe <= 0.80 across seeds 1/2/3
mean step64 refusal >= 0.20
mean safety_degradation <= 0.05
mean utility_drop <= 0.10
no seed returns to unsafe >= 0.95
```

Interpretation:

```text
valid non-degraded partial training-time control
```

### 5.2 Strong useful control

```text
some audited or intermediate schedule has:
    mean step64 unsafe <= 0.60
    mean step64 refusal >= 0.35
    mean safety_degradation <= 0.10
    mean utility_drop <= 0.10
    holds across seeds 1/2/3
```

Interpretation:

```text
candidate stable training-time control result
```

### 5.3 Negative or diagnostic-only

```text
clean schedules rebound to unsafe >= 0.90
and stronger schedules only work through degradation
```

Interpretation:

```text
anchor mixing remains a mechanistic diagnostic and partial control lever, not a stable recipe
```

## 6. Mechanism Readout

For every expanded seed, summarize:

```text
behavior:
    unsafe/refusal/invalid/degradation at step16/32/64

utility:
    task_success/unnecessary_refusal/degradation at step32/64

logit:
    refusal-compliance margin
    refusal-action margin

geometry:
    mean spectral norm
    top singular ratio
    effective rank
    temporal subspace cosine from step16 to step64
```

Expected interpretation:

```text
if refusal-compliance improves but refusal-action remains negative,
the model has not fully converted from action mode to refusal mode
```

This distinction should be central in the report, because it explains why logit margin improvements can coexist with unsafe behavior.

## 7. Recommended Execution Order

Run:

```text
1. Expand decay_20p_to_5p to seeds 2/3.
2. Rebuild selected_schedule_full100_summary across seeds 1/2/3.
3. Run step64 utility30 for seeds 2/3.
4. Audit degradation examples for early_20p_then_12p5 and early_20p_then_15p.
5. Decide whether an intermediate decay target is necessary.
6. Write Step2E report.
```

Stop rule:

```text
if decay_20p_to_5p fails on seeds 2/3 and degradation audit confirms stronger arms are real degradation,
stop schedule expansion and switch to mechanism-only analysis.
```

## 8. Expected Deliverables

Expected files:

```text
experiments/step2_mechanism_dynamics/results/4070ti_step2e/seed_expansion_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti_step2e/utility_seed_expansion_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti_step2e/degradation_audit_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti_step2e/mechanism_seed_summary.csv
experiments/step2_mechanism_dynamics/reports/Step2E_seed_expansion_degradation_audit_report.md
```

The report must state:

```text
whether decay_20p_to_5p is reproducible
whether stronger schedule degradation is real
whether Step2 has a stable control result, partial control result, or diagnostic-only result
```
