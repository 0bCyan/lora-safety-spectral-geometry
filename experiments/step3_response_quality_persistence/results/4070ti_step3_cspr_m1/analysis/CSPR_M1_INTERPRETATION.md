# CSPR M1 Interpretation and Route Decision

## Decision

The registered route is
`STOP_CSPR_M1_PROJECTION_NOT_REPLICATED`. CSPR mechanism promotion stops here,
and no CSPR training is authorized.

M1 passed the pairwise activation threshold (`8/18`, required `6/18`) but
failed the primary batch-robustness threshold: only `1/3` aggregate batches
activated the projection, while `2/3` were required. Consequently, only one
active batch could count toward the registered finite-step replication gate.
This is a preregistered scientific stop, not an execution failure.

## What Replicated

The active first batch reproduced the most important finite-step observation
from M0 at the frozen step size `2e-4`:

| Quantity | Initial | Unconstrained | Projected |
|---|---:|---:|---:|
| Preference loss | 0.69314718 | 0.68213036 | 0.68201736 |
| Safety loss | 1.16481932 | 1.16485280 | 1.16478566 |

The unconstrained direction increased safety loss by `3.35e-5`, while the
projected direction decreased it by `3.37e-5` and also decreased preference
loss. The projected direction retained `0.99977062` of the registered
first-order preference decrease. This is credible evidence that the projection
can repair a conflicting finite step when an aggregate conflict is present.

All three label-swap controls passed with cosine below `-0.999998`. Every one of
the 504 registered LoRA parameter tensors had finite non-zero gradients. The
largest normalized directional/KKT residual was `4.44e-16`. Every per-batch and
global trainable-state hash was restored exactly.

## Why The Main Gate Failed

Pairwise conflict does not imply aggregate conflict. For pair `i`,
`||g_p,i|| cos(g_p,i, g_s)` is proportional to its signed contribution to the
batch aggregate preference-safety inner product because the safety-gradient
norm and batch averaging factor are common within a batch.

| Batch | Active pairs | Negative share of absolute pair weight | Aggregate cosine | Aggregate active |
|---|---:|---:|---:|---:|
| 1 | 5/6 | 77.3% | -0.01515 | yes |
| 2 | 2/6 | 25.1% | 0.01958 | no |
| 3 | 1/6 | 6.2% | 0.05701 | no |

In batches 2 and 3, non-conflicting pair contributions outweighed the
conflicting pairs. Their ordinary preference directions already reduced the
safety loss, so the mathematically correct CSPR solution was to make no
projection. All three batches accepted a joint step, but only the first tested
an active projection; the other two do not count as projection replications.

## Credible Claim Boundary

M0 and M1 together observed 12 active pairwise conflicts among 26 exploratory
pairs and 2 active aggregate conflicts among 4 exploratory batches. These
counts are descriptive only: the batches are small, structured, and not an iid
sample.

The supported claim is narrow: CSPR is a numerically valid conditional conflict
resolver, and two frozen experiments show that it can turn a locally unsafe
finite step into a joint step. The unsupported claim is that aggregate
preference-safety conflict is stable or pervasive enough for CSPR to be the
primary training method.

Wrong-prompt cosines remained high (`0.753-0.801`), consistent with M0's
`0.761`. Thus the preference gradients remain weakly prompt-specific, and
response-content effects are still a plausible dominant source. M1 supplies no
behavioral, safety-generalization, or target-quality evidence.

## Next Route

Do not start CSPR training, do not inspect core100 or heldout, and do not spend
the deferred human-review budget merely to rescue this failed mechanism gate.
A future attempt would need a new preregistered hypothesis, such as testing
conflict-aware microbatch projection as a distinct method, with fresh controls
and without reinterpreting M1's frozen threshold.

After M1, the subjective probability of a credible multi-seed behavioral
result with CSPR as the primary method is revised from `0.25-0.40` to
`0.15-0.30`. This is a design judgment, not a statistical confidence interval.

No optimizer or scheduler was created, no checkpoint was written, and no model
output was generated. `optimizer_steps=0` and `training_authorized=false`.
