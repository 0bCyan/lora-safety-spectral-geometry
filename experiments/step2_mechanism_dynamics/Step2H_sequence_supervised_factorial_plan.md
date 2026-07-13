# Step 2H Sequence-Supervised Factorial Pilot Plan

## 1. Entry Condition

Step2G found that P0 prompt-end token margin, P1 post-think phrase margin, and P2 sequence preference margin do not pass the registered proxy gate.

Therefore Step2H freezes the behavior objective as direct safe-sequence cross-entropy rather than another proxy objective.

## 2. Goal

Run a seed1 4070 Ti pilot for the behavior-supervised 2x2 design:

| spectral objective | no behavior objective | behavior objective |
| --- | --- | --- |
| absent | vanilla | sequence_only |
| faithful factor proxy | factor_only | factor_plus_sequence |

The immediate question is whether faithful factor-proxy compression adds anything over a direct sequence-supervised baseline.

## 3. Data Discipline

Use only the frozen Step2G `safety_train` reserve as behavior supervision:

```text
experiments/step2_mechanism_dynamics/results/4070ti_step2g/splits/safety_train.jsonl
n = 40
```

Do not train on `proxy_dev`.
Do not evaluate or train on `heldout_test`.

## 4. Pilot Arms

Reuse existing controls:

```text
vanilla = Step2A risk seed1
factor_only = Step2F-R factor_mean_plus_max seed1
```

Train new arms:

```text
sequence_only:
    direct safe-sequence CE schedule
    20% anchor mix through step16
    decay to 5% through step64

factor_plus_sequence:
    same direct sequence schedule
    plus faithful factor proxy
    lambda_mean = 3.0
    lambda_max = 0.2
```

This schedule is chosen as the least degraded continuing-anchor control from earlier Step2D/E evidence, but it is rerun on the frozen Step2G `safety_train` split.

## 5. Evaluation

For the two new arms:

```text
dense safety30 = step0, step6, step16, step32, step64
full safety100 = step16, step32, step64
utility30 = step32, step64
factor/delta geometry = every checkpoint
logit diagnostics = every checkpoint
```

Compare against reused vanilla and factor-only summaries.

## 6. Decision

`factor_plus_sequence` has incremental value only if it improves over `sequence_only` in at least one meaningful dimension:

```text
lower unsafe behavior
less rebound across checkpoints
less repetitive-refusal degradation
smaller utility loss at matched safety
```

If it does not beat `sequence_only`, SpectraSafe should not remain the main method contribution.
