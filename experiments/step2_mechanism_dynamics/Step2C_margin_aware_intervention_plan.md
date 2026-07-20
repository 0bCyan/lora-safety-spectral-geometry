# Step 2C Margin-aware Early-window Intervention Plan

## 0. Motivation

Step2A found a risk-specific early transition:

```text
step6 -> step16
unsafe rises to saturation
refusal drops to near zero
refusal-compliance margin flips negative
```

Step2B showed that weak whole-run LoRA norm and spectral penalties do not prevent this transition.

Therefore Step2C should target the transition more directly:

> Instead of weakly penalizing parameter geometry across the whole run, intervene during the early window on the behavior/logit quantity that actually flips.

## 1. Goal

Test whether early-window safety anchors or margin-aware losses can delay or weaken the `risk` transition at step16.

Primary target:

```text
family = risk
seed = 1
window = step6 -> step16
checkpoints = 0,6,16,64
full100 eval = step6,16
```

## 2. Experimental Arms

### 2.1 Vanilla baseline

Already completed:

```text
vanilla risk seed1
```

Use fixed Step2A baseline.

### 2.2 Safety-anchor mixing

During steps 6 through 16, mix a small amount of normal/refusal-oriented safety data into risk training.

Pilot variants:

```text
anchor_mix = 5%
anchor_mix = 10%
anchor_mix = 20%
```

Anchor data should be selected from prompts/responses that preserve clean refusal or safe redirection.

Purpose:

```text
test whether a small early-window data correction can keep the margin from flipping
```

### 2.3 Margin-aware loss

During steps 6 through 16, add a logit margin objective on a fixed safety prompt subset:

```text
loss_margin = max(0, target_margin - (logp_refusal - logp_compliance))
```

Pilot variants:

```text
target_margin = 0.0
target_margin = 0.5
lambda_margin = 0.1
lambda_margin = 0.5
```

Minimum matrix:

| arm | target_margin | lambda_margin |
|---|---:|---:|
| margin_m0_lam0p1 | 0.0 | 0.1 |
| margin_m0_lam0p5 | 0.0 | 0.5 |
| margin_m0p5_lam0p1 | 0.5 | 0.1 |
| margin_m0p5_lam0p5 | 0.5 | 0.5 |

Purpose:

```text
directly test whether controlling the measured margin controls unsafe behavior
```

### 2.4 Combined small anchor + margin

Only run if either anchor-only or margin-only shows weak signal.

Initial combined arm:

```text
anchor_mix = 10%
target_margin = 0.0
lambda_margin = 0.1
```

## 3. Evaluation

For every arm:

```text
dense safety eval = 30 prompts at step0,6,16,64
full safety eval = 100 prompts at step6,16
logit margin summary at step0,6,16,64
spectral summary at step0,6,16,64
invalid/degradation audit
```

For any arm with safety improvement:

```text
utility eval = 30 prompts at step16 and step64
```

## 4. Success Criteria

Positive:

```text
step16 full100 unsafe <= 0.89
step16 full100 refusal >= 0.11
step16 refusal-compliance margin improves by >= 0.50 over vanilla seed1
invalid_rate <= 0.05
degradation_rate <= 0.05
```

Weak positive:

```text
step16 full100 unsafe <= 0.94
step16 margin improves by >= 0.25
invalid/degradation controlled
```

Negative:

```text
step16 unsafe remains >= 0.98
or margin remains strongly negative
or improvement comes from invalid/degraded outputs
```

## 5. Interpretation

If anchor mixing works:

```text
the early transition is data-balance sensitive
next step: identify which safety anchors are sufficient
```

If margin-aware loss works:

```text
the measured refusal/compliance margin is closer to a controllable mechanism variable
next step: expand to seeds 1,2,3
```

If neither works:

```text
the risk transition may require deeper data/model-level intervention
avoid claiming controllable mechanism
```

## 6. Deliverables

Expected files:

```text
experiments/step2_mechanism_dynamics/results/4070ti_step2c/margin_intervention_summary.csv
experiments/step2_mechanism_dynamics/results/4070ti_step2c/margin_intervention_full100_summary.csv
experiments/step2_mechanism_dynamics/reports/Step2C_margin_aware_intervention_report.md
```

Report must include:

```text
arm behavior table
margin table
spectral side effects
invalid/degradation audit
utility sanity for positive arms
decision: expand, revise, or stop
```

