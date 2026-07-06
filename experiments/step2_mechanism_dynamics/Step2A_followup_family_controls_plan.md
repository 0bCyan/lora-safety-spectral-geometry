# Step 2A Follow-up: Family Controls and Full-window Validation

## 0. Decision

The first Step2A risk-only pilot produced a useful timing signal:

```text
behavior transition window: step 6 -> step 16
unsafe mean: 0.7000 -> 1.0000
refusal mean: 0.3000 -> 0.0000
refusal-compliance margin: 0.8840 -> -0.7574
mean spectral norm: 0.0432 -> 0.0771
```

This is enough to continue Step2A, but not enough to claim a risk-specific mechanism. The next run should test whether this early transition is specific to `risk` training or is a generic LoRA training artifact.

## 1. Goal

Answer the family-separation question:

> Does the early training transition seen in risk LoRA also appear in normal/control LoRA, or is it specific to unsafe-risk data?

Primary target:

```text
compare risk vs normal/control trajectories over the same checkpoint schedule
```

Secondary target:

```text
validate the risk transition window with full 100-prompt eval on selected checkpoints
```

## 2. Required Runs

### 2.1 Add family controls

Run the same Step2A checkpoint pipeline for:

```text
normal seed 1
control seed 1
```

Use the same settings as the completed risk pilot:

```text
model = Qwen3-4B
machine = 4070 Ti
train rows per seed = 256
optimizer steps = 64
checkpoints = 0, 6, 16, 32, 48, 64
dense safety eval = 30 prompts
```

Required metrics:

```text
behavior summary
spectral metrics
left/right subspace metrics
refusal/compliance/action logit margins
```

### 2.2 Full eval on risk transition checkpoints

For existing risk seeds 1/2/3, run full 100-prompt safety eval on:

```text
step0000_pct000
step0006_pct009
step0016_pct025
step0064_pct100
```

Purpose:

```text
confirm that the 30-prompt transition is not a dense-subset artifact
```

If compute budget is tight, prioritize:

```text
step0006_pct009
step0016_pct025
```

because this is the observed transition window.

## 3. Comparison Metrics

### 3.1 Behavior trajectory

Compare by family:

```text
unsafe_compliance_rate
refusal_rate
invalid_rate
degradation_rate
```

Expected if risk-specific:

```text
risk: unsafe rises to 1.0 by step16
normal: refusal remains higher or unsafe does not saturate
control: may degrade or behave inconsistently, but should not match risk cleanly
```

### 3.2 Logit margin trajectory

Primary metric:

```text
mean_refusal_minus_compliance
```

Risk pilot pattern:

```text
step0:  +1.3154
step6:  +0.8840
step16: -0.7574
step64: -1.6062
```

Evidence for family separation:

```text
risk margin flips early and remains negative
normal margin stays positive or flips later/weaker
control margin is noisy or tied to degradation rather than clean unsafe compliance
```

### 3.3 Output-side subspace trajectory

For each family, compute same-checkpoint seed-to-seed overlap when multiple seeds exist.

For the immediate one-seed normal/control run, compare temporal stability and final adapter subspace against risk:

```text
within-seed temporal left/output top-direction cosine
final checkpoint left/output singular directions
module/layer localization
```

If normal/control seed 1 shows a promising contrast, expand to:

```text
normal seeds 1,2,3
control seeds 1,2,3
```

so that seed-to-seed family separation can be measured directly.

### 3.4 Spectral trajectory

Compare:

```text
mean_delta_spectral_norm
max_delta_spectral_norm
mean_top_singular_ratio
mean_effective_rank
```

Do not treat global spectral norm alone as a mechanism. Step 1 already showed global spectral means are insufficient. Use spectral metrics mainly as timing and regularization targets.

## 4. Success Criteria

The follow-up supports a risk-specific mechanism hypothesis if:

```text
risk behavior transition remains visible in 100-prompt eval
risk refusal-compliance margin flips earlier or more strongly than normal/control
risk output-side structure remains more consistent or more behavior-aligned
normal/control do not reproduce the same clean unsafe transition
```

It weakens the mechanism hypothesis if:

```text
normal/control show the same early margin flip and unsafe saturation
or the 100-prompt risk eval does not reproduce the 30-prompt transition
or local judge labels are dominated by invalid/degradation artifacts
```

## 5. Deliverables

Expected new files:

```text
experiments/step2_mechanism_dynamics/results/4070ti/family_control_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti/full_eval_transition_summary.csv
experiments/step2_mechanism_dynamics/reports/Step2A_family_controls_report.md
```

Recommended plots/tables:

```text
unsafe/refusal by family over checkpoints
refusal-compliance margin by family over checkpoints
spectral metrics by family over checkpoints
risk 30-prompt vs 100-prompt transition comparison
```

## 6. Decision After This Follow-up

If family separation is clear:

```text
write Step2A mechanism interim summary
design Step2B training-time regularization pilot
target the early step6->step16 transition window
```

If family separation is unclear:

```text
expand normal/control to three seeds each
improve judge/manual audit before designing SpectraSafe
avoid claiming a risk-specific mechanism
```

If full eval contradicts the 30-prompt transition:

```text
repeat dense eval with a more representative prompt subset
do not move to Step2B yet
```

