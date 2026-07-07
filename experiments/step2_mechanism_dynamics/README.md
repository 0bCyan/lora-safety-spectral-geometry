# Step 2: Training-time Mechanism Dynamics

## 1. Position

Step 2 starts after Step 1 is closed.

Step 1 found that vanilla LoRA safety drift has stable diagnostic spectral, subspace, and activation signals, but simple post-hoc interventions do not reliably control unsafe behavior. Therefore Step 2 should not continue blind post-hoc editing. It should investigate how the diagnostic structure forms during training and whether training-time regularization can change that trajectory.

Core question:

> How does risk LoRA training form a stable output-side update subspace, and how does that subspace connect to refusal/compliance logit changes and unsafe behavior?

## 2. First Plan

The first Step 2 plan is:

```text
experiments/step2_mechanism_dynamics/Step2A_training_dynamics_mechanism_plan.md
```

Current follow-up plan:

```text
experiments/step2_mechanism_dynamics/Step2A_followup_family_controls_plan.md
```

Current training-time intervention plan:

```text
experiments/step2_mechanism_dynamics/Step2B_training_time_regularization_plan.md
```

## 3. Stage Outputs

Expected outputs:

```text
experiments/step2_mechanism_dynamics/configs/
experiments/step2_mechanism_dynamics/scripts/
experiments/step2_mechanism_dynamics/results/
experiments/step2_mechanism_dynamics/reports/
```

The first pilot should reuse Step 1 data splits and evaluation prompts when possible, so that Step 2 dynamics can be compared directly against the closed Step 1 baseline.
