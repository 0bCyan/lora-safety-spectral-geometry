# Step 2F-R Closing Analysis and Step2G Entry Decision

## 0. Status

Step2F-R is complete on the current 4070 Ti / Qwen3-4B risk setting.

The run faithfully regularized the original v0.0.8 factor proxy:

```text
sigma_hat_l = ||B_l||_2 * ||A_l||_2
```

It passed the numerical, upper-bound, gradient, tied-max, compilation, and two-step smoke-test gates before the main experiment.

The result closes the current pure geometry-only branch under its registered stopping rule.

## 1. Main Result

At step64, the faithful combined arm produced:

```text
factor_mean_plus_max
factor proxy mean = 0.0448
factor proxy max = 0.0855
full100 unsafe = 1.0000
full100 refusal = 0.0000
utility task success = 0.8000
utility drop vs vanilla = 0.0333
refusal_minus_action = -6.4232
```

The comparable vanilla risk seed1 endpoint was approximately:

```text
factor proxy mean = 0.1044
factor proxy max = 0.2079
full100 unsafe = 1.0000
utility task success = 0.8333
```

Therefore the faithful combined objective reduced the factor mean and max by approximately:

```text
mean reduction = 57.1%
max reduction = 58.9%
```

The geometry-engagement gate passed by a wide margin, utility remained acceptable, and behavior stayed at unsafe saturation.

The single-term arms lead to the same behavioral decision:

| arm | step64 factor mean | step64 factor max | full100 unsafe | utility success |
| --- | ---: | ---: | ---: | ---: |
| factor_mean_only | 0.0463 | 0.1179 | 1.0000 | 0.8000 |
| factor_max_only | 0.0975 | 0.1420 | 1.0000 | 0.8000 |
| factor_mean_plus_max | 0.0448 | 0.0855 | 1.0000 | 0.8000 |

The small step32 movement of the combined arm (`unsafe=0.98`) is below the registered useful-control threshold and disappears by step64. It should not be interpreted as a positive result.

## 2. Scientific Interpretation

### 2.1 What the result establishes

The current evidence supports the bounded causal statement:

> In the current Qwen3-4B risk fine-tuning setting, strong training-time compression of either the faithful LoRA factor spectral proxy or the direct LoRA update spectral norm is insufficient by itself to control unsafe generation behavior.

This is stronger than a generic failed-method result because the main alternative explanations were checked:

```text
implementation mismatch: corrected
regularizer did not engage: false
training collapsed: false
utility was destroyed: false
only one spectral definition was tried: false
```

The result isolates a geometry-behavior separation:

```text
spectral magnitude is controllable
behavior remains in action/compliance mode
```

The strongly negative `refusal_minus_action` value is consistent with the behavioral outcome. Compressing the adapter update did not change the output-mode competition that matters for generation.

### 2.2 What the result does not establish

Do not claim:

```text
SpectraSafe universally fails across all models
spectral geometry is irrelevant to safety drift
factor-proxy and delta-spectral gradients are identical
behavior-supervised spectral extensions cannot work
the current local judge is sufficient for a publication-level universal claim
```

The run uses one model, one risk family, one pilot seed for the corrected objective, and a local heuristic judge. The lack of seed expansion is justified by the registered pilot stop rule, but the conclusion must remain setting-bounded.

## 3. Meaning for the Paper

### 3.1 Effect on the original v0.0.8 method claim

This result is negative for the direct v0.0.8 headline:

> Mean-plus-max factor spectral regularization alone improves the safety-utility tradeoff.

That headline is not supported in the current Qwen3-4B setting. The numerical result tables in the v0.0.8 idea document must not be treated as validated evidence for the current project.

The mathematical contribution remains valid as a structural upper-bound regularizer, but its connection to behavioral safety must be weakened from a sufficient control mechanism to a geometry-control hypothesis.

### 3.2 Effect on the broader research contribution

This result strengthens the scientific quality of the broader project.

The cross-stage evidence now forms a coherent boundary result:

```text
Step1:
    spectral and output-subspace signatures diagnose risk adapters
    simple post-hoc parameter and activation interventions do not recover safety

Step2A:
    risk behavior undergoes a reproducible early step6 -> step16 transition

Step2C-E:
    direct safety anchors can change behavior temporarily
    prompt-end margin can improve without behavior improvement
    anchor persistence trades safety against rebound or repetitive refusal

Step2F / Step2F-R:
    direct delta-spectral and faithful factor-proxy compression both alter geometry
    neither changes the saturated unsafe output mode
```

The most defensible current paper-level insight is:

> LoRA spectral geometry is a useful diagnostic and capacity-control signal, but spectral magnitude is not a sufficient behavioral safety variable. Effective control must also target the generation mode or sequence-level behavior.

This is a better supported claim than the original direct mechanism narrative, although it still requires broader models and stronger evaluation before publication.

## 4. Step2 Branch Decision

### 4.1 Closed branch

Close the following as the main 4070 Ti branch:

```text
additional seeds for the current pure factor-proxy recipe
additional seeds for the current direct delta-spectral recipe
open-ended mean/max lambda grids
claims that lower final spectral magnitude alone implies safer behavior
```

### 4.2 Open branch

Proceed to:

```text
Step2G: Behavior-Proxy Validity Audit
```

Step2G should determine which diagnostic quantity, if any, tracks the real answer mode closely enough to justify a new training objective.

It should not train a new method yet.

## 5. Step2G Minimum Experiment

### 5.1 Goal

Test why the existing prompt-end margin failed and whether a post-thinking or sequence-level score is a valid behavior proxy.

Primary questions:

```text
Q1. Does an answer-start score after </think> separate refusal from unsafe action?
Q2. Does a short-phrase score outperform the existing token-set margin?
Q3. Does a length-normalized safe-versus-unsafe sequence margin track actual behavior across intervention families?
Q4. Is any proxy stable under leave-one-arm-out evaluation rather than merely identifying arm labels?
```

### 5.2 Independent data allocation

Create and freeze a deterministic split from previously unused safety IDs 101-300:

```text
safety_train reserve = 40
proxy_dev = 40
heldout_test = 120
split = fixed seeded hash
```

Step2G uses `proxy_dev` only. Keep `heldout_test` unopened until the Step2H objective and hyperparameters are frozen.

The historical IDs 1-100 remain useful for comparison but are not the confirmatory test because Step2C used the first 30 as intervention prompts.

### 5.3 Representative checkpoints

Reuse existing adapters where available:

```text
vanilla risk seed1: step16, step64
Step2C margin-only: step16
Step2C anchor_mix_20p: step16, step64
Step2F delta-spectral combined: step16, step64
Step2F-R factor-proxy combined: step16, step64
```

Optional robustness rows:

```text
anchor_mix_20p seeds 2/3 at step16
factor mean-only and max-only at step64
```

Preflight requirement:

```text
verify that the experiment machine still has the ignored adapter checkpoints
```

If the checkpoints are missing, do not rerun every historical arm. Recreate only the minimum representative set required by the audit.

### 5.4 Candidate proxies

Evaluate three levels:

```text
P0: prompt-end token-set margin
    existing failed proxy
    retain only as a negative control

P1: post-think short-phrase margin
    condition on the observed structural prefix through </think>
    score complete short phrases, not the last token of each phrase

P2: sequence preference margin
    normalized log P(safe refusal + redirect | prompt)
    minus normalized log P(unsafe reference | prompt)
```

For P1 and P2, compute full candidate sequence likelihoods with length normalization. Do not reduce multi-token phrases to a set of final token IDs.

Use fixed candidate construction independent of each model's generated answer so the proxy does not leak the observed behavior label.

### 5.5 Validation metrics

Report:

```text
sample-level AUROC for refusal versus unsafe behavior
arm/checkpoint-level Spearman rank correlation
within-arm correlation where both labels are present
leave-one-arm-out AUROC
bootstrap confidence intervals over prompts
coverage and failure rate for the post-think position
```

The primary proxy gate is:

```text
dev AUROC >= 0.75
leave-one-arm-out AUROC >= 0.70
arm-level ordering agrees with observed behavior
no single intervention family explains the entire signal
```

Correlation is a proxy-selection gate, not causal proof.

## 6. Step2G Decision Rules

If P1 passes and is clearly stronger than P0:

```text
consider one small post-think phrase-loss causal pilot
do not expand until behavior changes on independent dev prompts
```

If P2 passes and is stronger or more stable than P1:

```text
use sequence preference loss as the primary Step2H behavior objective
```

If neither P1 nor P2 passes:

```text
do not invent another token-margin objective
use direct safe-sequence cross-entropy as a behavior-supervised baseline
downgrade the mechanism claim from proxy-mediated control to empirical behavior supervision
```

## 7. Step2H Entry Condition

Step2H begins only after the behavior objective is frozen on `proxy_dev`.

Required 2x2 design:

| spectral objective | no behavior objective | behavior objective |
| --- | --- | --- |
| absent | vanilla | sequence-only |
| faithful factor proxy | SpectraSafe-only | SpectraSafe + sequence |

The combined method has incremental value only if it beats `sequence-only` by a registered margin in at least one meaningful dimension:

```text
lower unsafe behavior
less rebound across checkpoints
less repetitive-refusal degradation
smaller utility loss at matched safety
```

If the combined arm does not improve on `sequence-only`, SpectraSafe should not remain the main method contribution.

## 8. Immediate Next Action

Write and execute the Step2G proxy audit before launching any new training arm.

Recommended implementation deliverables:

```text
experiments/step2_mechanism_dynamics/Step2G_behavior_proxy_validity_audit_plan.md
experiments/step2_mechanism_dynamics/configs/4070ti_qwen3_4b_step2g_proxy_audit.json
experiments/step2_mechanism_dynamics/scripts/build_step2g_safety_split.py
experiments/step2_mechanism_dynamics/scripts/compute_behavior_proxy_scores.py
experiments/step2_mechanism_dynamics/scripts/summarize_step2g_proxy_audit.py
experiments/step2_mechanism_dynamics/results/4070ti_step2g/
experiments/step2_mechanism_dynamics/reports/Step2G_behavior_proxy_validity_audit_report.md
```

This is the lowest-cost experiment that can prevent another round of optimizing a measurable but behaviorally invalid quantity.
