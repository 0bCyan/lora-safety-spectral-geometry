# Step 2G Behavior-Proxy Validity Audit Plan

## 1. Goal

Step2G tests whether any cheap behavior proxy tracks the actual refusal-versus-unsafe answer mode well enough to justify a later training objective.

This step does not train a new method.

## 2. Frozen Split

Use previously unused `eval_safety` IDs 101-300.

Deterministic hash split:

```text
safety_train reserve = 40
proxy_dev = 40
heldout_test = 120
seed = step2g_proxy_audit_v1
```

Step2G scores only `proxy_dev`. The heldout test split is recorded by ID/hash only and is not evaluated.

## 3. Representative Checkpoints

Minimum 4070 Ti audit set:

```text
vanilla_risk_s1: step16, step64
margin_m0_lam0p1: step16
anchor_mix_20p: step16, step64
spectrasafe_mean_plus_max: step16, step64
factor_mean_plus_max: step16, step64
```

These rows span saturated unsafe behavior, direct anchor behavior control, failed prompt-end margin optimization, delta-spectral compression, and faithful factor-proxy compression.

## 4. Candidate Proxies

```text
P0 prompt-end token-set margin:
    existing first-token/token-set diagnostic; negative control

P1 post-think short-phrase margin:
    if the generated response contains </think>, condition on prompt plus observed prefix through </think>
    score fixed safe and unsafe short phrases with length-normalized likelihood

P2 sequence preference margin:
    length-normalized log P(fixed safe refusal + redirect | prompt)
    minus length-normalized log P(unsafe reference | prompt)
```

P1 and P2 score complete candidate strings, not final token IDs.

## 5. Validation

Report:

```text
sample-level AUROC for refusal versus unsafe behavior
leave-one-arm-out AUROC
arm/checkpoint-level Spearman rank correlation
within-arm AUROC where both labels are present
bootstrap prompt confidence intervals
post-think coverage and failure rate
```

Primary gate:

```text
dev AUROC >= 0.75
leave-one-arm-out AUROC >= 0.70
arm-level ordering agrees with observed behavior
signal is not explained by only one intervention family
```

## 6. Decision

If P2 passes and is more stable than P1, use sequence preference loss as the Step2H behavior objective.

If P1 passes but P2 does not, consider a small post-think phrase-loss causal pilot.

If neither passes, do not build another token-margin objective; use direct safe-sequence cross-entropy as the behavior-supervised baseline.
