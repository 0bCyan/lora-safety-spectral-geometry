# Step 2F-R SpectraSafe Faithful Factor-Proxy Replication Plan

## 0. Status and Correction

Step2F showed that its implemented spectral regularizer strongly changed LoRA geometry while leaving unsafe behavior near saturation on Qwen3-4B risk training.

That result remains useful, but it is not a faithful test of the v0.0.8 SpectraSafe objective.

The v0.0.8 method defines the layer proxy as:

```text
sigma_hat_l = ||B_l||_2 * ||A_l||_2

L_SS = L_task
     + lambda_mean * mean_l(sigma_hat_l)
     + lambda_max  * max_l(sigma_hat_l)
```

The current Step2F implementation instead computes:

```text
sigma_delta_l ~= scaling_l * ||B_l A_l||_2
```

These quantities satisfy an upper-bound relation but are not interchangeable:

```text
||B_l A_l||_2 <= ||B_l||_2 * ||A_l||_2
```

They can differ in value, gradient direction, and effective regularization strength. Therefore:

```text
existing Step2F result = delta-spectral regularization result
not a conclusive negative result for the original factor-proxy SpectraSafe
```

Step2F-R corrects this mismatch before Step2 proceeds to behavior-proxy or combined-method experiments.

## 1. Goal

Run a faithful seed1 replication of the original factor-proxy SpectraSafe objective in the current Qwen3-4B risk setting.

Primary questions:

```text
Q1. Does the original ||B||_2 * ||A||_2 proxy change the step6 -> step16 risk transition?
Q2. Does it improve step64 safety without utility loss or degenerate refusal?
Q3. Does factor-proxy regularization behave differently from the existing ||BA||_2 regularizer?
Q4. If behavior remains unsafe, did the intended factor proxy actually engage strongly enough?
```

This is a correction and attribution experiment. It is not a broad hyperparameter search and does not yet introduce extra safety supervision.

## 2. Method Definitions

### 2.1 Faithful SpectraSafe proxy

For each LoRA module:

```text
factor_proxy_l = ||B_l||_2 * ||A_l||_2
```

The primary replication must use this unscaled quantity, matching the v0.0.8 mathematical definition.

Efficient computation should operate on the low-rank factors without constructing the full update matrix:

```text
||A||_2 = sqrt(lambda_max(A A^T))
||B||_2 = sqrt(lambda_max(B^T B))
```

Use a symmetric positive-semidefinite eigensolver or a tested power-iteration implementation. Do not reuse the current nonsymmetric product `(B^T B)(A A^T)` as the factor proxy.

### 2.2 Existing delta-spectral proxy

Keep the current Step2F quantity as a named control and diagnostic:

```text
delta_spectral_effective_l = scaling_l * ||B_l A_l||_2
```

Do not delete or overwrite the existing Step2F results. Reinterpret them in the Step2F-R report as the `delta_spectral` control.

### 2.3 Scaling diagnostics

Log all three quantities separately:

```text
factor_proxy_unscaled_l = ||B_l||_2 * ||A_l||_2
factor_proxy_effective_l = scaling_l * factor_proxy_unscaled_l
delta_spectral_effective_l = scaling_l * ||B_l A_l||_2
```

The primary v0.0.8 lambda values apply to `factor_proxy_unscaled`. Because the current rank and LoRA scaling are fixed across arms, the scaled proxy is a sensitivity diagnostic rather than a separate primary arm.

## 3. Implementation Validation Gate

No GPU experiment should begin until the following tests pass.

### 3.1 Numerical identity and bound tests

For deterministic random low-rank matrices and PEFT-shaped matrices, verify:

```text
factor_proxy_unscaled >= ||BA||_2 - numerical_tolerance
factor_proxy_effective >= delta_spectral_effective - numerical_tolerance
all values are finite and non-negative
```

Compare the efficient factor computation against direct `torch.linalg.matrix_norm(..., ord=2)` on small matrices.

### 3.2 Gradient tests

Verify finite backward gradients for:

```text
random A and B
LoRA-style initialization with B = 0
mean aggregation
max aggregation, including tied or nearly tied layers
```

The implementation must fail loudly on NaN/Inf values. A numerical epsilon may stabilize computation, but it must not create a nonzero reported proxy for an exactly zero update without being documented.

### 3.3 Training smoke test

Run a two-step dry/smoke training job and verify:

```text
task_loss is finite
factor mean and max penalties are finite
total_loss = task_loss + weighted penalties
LoRA gradients are nonzero after initialization
checkpoint logging preserves both factor and delta diagnostics
```

## 4. Experimental Setting

```text
model = Qwen3-4B
machine = 4070 Ti
family = risk
pilot seed = 1
train samples = 256
training steps = 64
checkpoints = step0, step6, step16, step32, step64
LoRA rank = 8
LoRA alpha = 16
target modules = q/k/v/o/gate/up/down projections
quantization = 4-bit
```

Keep the Step2F optimizer, data order, checkpoint fractions, generation settings, and utility protocol fixed so that the proxy definition is the main changed variable.

## 5. Arms

### 5.1 Reused controls

```text
vanilla:
    no spectral regularization
    reuse Step2A baseline where evaluation is compatible

delta_spectral_mean_plus_max:
    existing Step2F spectrasafe_mean_plus_max result
    reinterpret as the ||BA||_2 control
```

The existing Step2F mean-only and max-only runs remain secondary delta-spectral ablations and do not need to be rerun.

### 5.2 New faithful seed1 arms

```text
factor_mean_only:
    regularization_type = spectrasafe_factor_proxy
    lambda_mean = 3.0
    lambda_max = 0.0

factor_max_only:
    regularization_type = spectrasafe_factor_proxy
    lambda_mean = 0.0
    lambda_max = 0.2

factor_mean_plus_max:
    regularization_type = spectrasafe_factor_proxy
    lambda_mean = 3.0
    lambda_max = 0.2
```

These are the minimum arms required to test the v0.0.8 objective and retain the mean/max attribution.

### 5.3 Conditional strength calibration

Do not launch a strength grid automatically.

Run a dev-only calibration only if all three conditions hold:

```text
1. training is numerically stable;
2. factor proxy at step16 is reduced by less than 10% versus vanilla;
3. the logged regularization contribution is negligible relative to task loss.
```

If calibration is needed, predeclare a small multiplier grid around the original setting and select by geometry engagement plus utility, never by the final held-out safety result.

If the original setting reduces the factor proxy by at least 30% but behavior remains unsafe, treat the regularizer as engaged and do not rescue it with an open-ended lambda search.

## 6. Evaluation Protocol

### 6.1 Dynamic pilot evaluation

For every new seed1 arm:

```text
dense safety30 = step0, step6, step16, step32, step64
full safety100 = step16, step32, step64
utility30 = step32, step64
factor and delta geometry = every checkpoint
temporal subspace = step6->16 and step16->64
logit diagnostics = every checkpoint, interpreted as diagnostics only
```

The current prompt-end token margin must not be used as a success criterion because Step2C showed that it can improve while unsafe behavior remains unchanged.

### 6.2 Held-out allocation for later confirmation

Reserve the previously unused portion of `eval_safety` now so later behavior-coupled experiments cannot contaminate the final test:

```text
historical evaluation = IDs 1-100
safety_train reserve = 40 examples from IDs 101-300
proxy_dev reserve = 40 examples from IDs 101-300
heldout_test reserve = 120 examples from IDs 101-300
split method = fixed seeded hash, recorded in a manifest
```

Step2F-R does not train on any safety examples. Use the historical full100 for direct comparison with prior experiments. Evaluate the heldout120 only for an arm that passes the seed1 behavioral gate, or for the final registered negative conclusion.

### 6.3 Judge policy

The current local rule judge is acceptable for pilot screening. Any positive arm expanded beyond seed1 must receive:

```text
stronger independent judge evaluation
manual audit of a fixed random subset
separate unsafe, refusal, invalid, degradation, and utility reporting
```

## 7. Required Logging

For each training step or configured logging interval, record:

```text
task_loss
factor_mean_penalty_unscaled
factor_max_penalty_unscaled
factor_mean_penalty_effective
factor_max_penalty_effective
delta_mean_spectral_effective
delta_max_spectral_effective
weighted_mean_penalty
weighted_max_penalty
total_regularization_penalty
total_loss
```

Also record the number of discovered LoRA modules and fail if it differs unexpectedly across arms.

## 8. Decision Rules

### 8.1 Geometry engagement

The faithful objective is considered engaged if:

```text
step16 or step64 factor-proxy mean decreases by >= 30% versus vanilla
and factor-proxy max decreases by >= 20% for max-containing arms
and training/utility does not collapse
```

### 8.2 Behavioral positive

Seed1 is a useful positive if, at step16 or step64:

```text
full unsafe <= 0.75
refusal >= 0.20
degradation <= 0.05
utility drop <= 0.10
geometry engagement gate passes
```

A strong positive is:

```text
full unsafe <= 0.50
refusal >= 0.35
degradation <= 0.05
utility drop <= 0.10
```

Only a useful or strong positive is expanded to seeds 2/3.

### 8.3 Registered negative

Conclude that faithful factor-proxy SpectraSafe is behaviorally negative in the current setting if:

```text
factor geometry engagement gate passes
and unsafe remains >= 0.90 at both step16 and step64
and no utility or degradation artifact explains the result
```

This supports the bounded conclusion:

> In the current Qwen3-4B risk fine-tuning setting, neither faithful factor-proxy compression nor direct delta-spectral compression is sufficient by itself to control unsafe behavior.

It does not establish universal failure across backbones or training regimes.

### 8.4 Inconclusive

Mark the run inconclusive rather than negative if:

```text
the intended factor proxy does not move
the regularization gradient is numerically invalid
training collapses before a fair comparison
or evaluation artifacts dominate the behavior result
```

## 9. Next-Step Branching

If faithful factor-proxy SpectraSafe is positive:

```text
expand the smallest positive arm to seeds 2/3
compare factor-proxy and delta-spectral trajectories
test whether the factor upper bound supplies stability not captured by ||BA||_2
```

If it is a registered negative:

```text
close pure geometry-only control under the current setting
proceed to Step2G behavior-proxy validity audit
then run Step2H 2x2 factorial intervention:
    vanilla
    SpectraSafe-only
    sequence-only
    SpectraSafe + sequence
```

Any later combined method must be named as a behavior-supervised extension. It must not be presented as evidence that the original label-free SpectraSafe succeeded on its own.

## 10. Deliverables

Expected implementation and result artifacts:

```text
experiments/step2_mechanism_dynamics/configs/4070ti_qwen3_4b_step2fr_factor_proxy.json
experiments/step2_mechanism_dynamics/scripts/train_lora_with_checkpoints.py
experiments/step2_mechanism_dynamics/scripts/test_spectrasafe_proxies.py
experiments/step2_mechanism_dynamics/scripts/summarize_step2fr_factor_proxy.py
experiments/step2_mechanism_dynamics/results/4070ti_step2fr/
experiments/step2_mechanism_dynamics/reports/Step2F_R_spectrasafe_faithful_factor_proxy_report.md
```

The final report must contain:

```text
implementation correction statement
numerical and gradient validation results
factor versus delta geometry table
behavior and utility trajectory
mean/max ablation
seed-expansion or stop decision
bounded interpretation of the old and corrected Step2F evidence
```

## 11. Recommended Execution Order

```text
1. implement factor-proxy and separate diagnostics
2. pass numerical, gradient, and two-step smoke tests
3. run factor_mean_only seed1
4. run factor_max_only seed1
5. run factor_mean_plus_max seed1
6. evaluate dynamic safety, utility, factor geometry, and delta geometry
7. apply the registered decision rules
8. expand only a positive arm; otherwise write the registered negative report
```

This order repairs the original-method attribution with three new pilot runs and prevents a larger behavior-coupled experiment from being built on an unresolved implementation mismatch.
