# Step 2A Training Dynamics Mechanism Plan

## 0. Goal

Step 2A tests a mechanism hypothesis that Step 1 could not answer:

> Risk LoRA unsafe drift is not controlled by a small post-hoc removable component; instead, it forms during training as a distributed output-side update structure that shifts refusal/compliance logits.

The goal is to observe this formation process over training time.

## 1. Background From Step 1

Step 1 established four facts:

1. `risk` LoRA reliably induces unsafe drift.
2. Global spectral averages do not explain behavior by themselves.
3. `risk` adapters share a stronger left/output singular subspace signature across seeds.
4. Parameter-space and activation-space post-hoc interventions failed to recover safety.

Therefore Step 2A should move from endpoint editing to training-time dynamics.

## 2. Mechanism Questions

Step 2A should answer:

1. When does unsafe behavior emerge during LoRA training?
2. Does output-side risk subspace consistency emerge before, during, or after unsafe behavior?
3. Do refusal token logprobs decrease as compliance/action token logprobs increase?
4. Are spectral/subspace changes better predictors than global norm metrics?
5. Do normal/risk/control families follow different training trajectories?

## 3. Experimental Design

### 3.1 Families and Seeds

Use the same families as Step 1:

```text
normal
risk
control
```

Minimum pilot:

```text
model = Qwen3-4B
machine = 4070 Ti
seeds = 1,2,3 for risk
optional = one normal seed, one control seed
```

If the pilot is stable, expand to:

```text
normal seeds = 1,2,3
risk seeds = 1,2,3
control seeds = 1,2,3
```

### 3.2 Checkpoint Schedule

Save lightweight LoRA adapter checkpoints during training:

```text
step = 0
step = 10%
step = 25%
step = 50%
step = 75%
step = 100%
```

If total training is short, save by optimizer step count:

```text
checkpoint_every_n_steps = 20 or 50
```

The final checkpoint must match the Step 1 vanilla LoRA configuration as closely as possible.

## 4. Metrics

### 4.1 Behavior Metrics

At each checkpoint, run a fixed safety mini-eval:

```text
unsafe_compliance_rate
refusal_rate
invalid_or_irrelevant_rate
degradation_rate
average_response_chars
```

Use a smaller prompt subset for dense checkpoint tracking, then run full eval on selected checkpoints:

```text
dense subset = 30 prompts
full subset = 100 prompts
```

### 4.2 Spectral Metrics

Reuse Step 1 adapter analysis:

```text
delta_spectral_norm
delta_fro_norm
top_singular_ratio
effective_rank
module/layer summaries
```

Track both global means and top-k layers over training time.

### 4.3 Subspace Metrics

For each checkpoint, compute left/right singular subspaces for LoRA update:

```text
Delta W = B @ A
side = left/output, right/input
k = 1,2,4
```

Primary metric:

```text
risk seed-to-seed left/output top direction cosine over training time
```

Comparisons:

```text
risk vs normal
risk vs control
normal seed-to-seed
control seed-to-seed
```

### 4.4 Logit Path Metrics

For fixed safety prompts, compute logit margins without sampling:

```text
refusal_token_logprob
compliance_token_logprob
action_token_logprob
refusal_minus_compliance_margin
```

Use small token sets rather than a single token:

```text
refusal tokens/examples:
  "I", "cannot", "can't", "sorry", "unable"

compliance/action tokens/examples:
  imperative/action verbs from unsafe completions
```

The exact token sets should be inspected for Qwen tokenization before final use.

## 5. Primary Hypotheses

### H1: Unsafe behavior emerges with output-side subspace consistency

Expected support:

```text
risk unsafe rate increases over checkpoints
risk left/output seed-to-seed overlap increases in the same window
risk vs normal/control remains separated
```

### H2: Logit margins connect structure to behavior

Expected support:

```text
refusal_minus_compliance_margin decreases as unsafe rate increases
output-side subspace overlap correlates with logit margin shift
```

### H3: Global norm alone is insufficient

Expected support:

```text
global delta_spectral_norm does not cleanly rank behavior
subspace/logit metrics explain more than global spectral means
```

## 6. Stop / Continue Criteria

Step 2A is useful if it produces at least one of:

```text
clear timing relationship between unsafe behavior and output-side subspace formation
clear logit margin shift tied to risk training
clear negative result showing no simple training-time coupling
```

Continue to Step 2B SpectraSafe regularization only if Step 2A identifies a measurable training-time target:

```text
output-side subspace concentration
top spectral spike
effective-rank collapse
refusal/compliance margin degradation
```

If no training-time signal is measurable, SpectraSafe should be treated as a broader empirical regularizer rather than a mechanism-targeted method.

## 7. Deliverables

Minimum deliverables:

```text
configs/4070ti_qwen3_4b_step2a_dynamics.json
scripts/train_lora_with_checkpoints.py
scripts/evaluate_checkpoint_dynamics.py
scripts/compute_logit_margins.py
results/4070ti/checkpoint_index.csv
results/4070ti/dynamics_summary.csv
reports/Step2A_training_dynamics_report.md
```

Recommended plots:

```text
unsafe/refusal over training step
left/output subspace overlap over training step
refusal-compliance logit margin over training step
spectral metrics over training step
```

## 8. Interpretation Rules

Do not claim causal mechanism from correlation alone.

Stronger evidence requires at least:

```text
temporal ordering
seed consistency
family separation
connection to logit margins
eventual training-time intervention or regularization effect
```

Step 2A should be written as mechanism exploration, not mechanism proof.

