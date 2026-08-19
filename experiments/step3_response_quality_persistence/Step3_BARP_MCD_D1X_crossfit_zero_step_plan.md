# Step3 BARP-MCD D1X: Cross-Fitted Optimizer-Route Qualification

## Question

D1R established a durable exact AdamW max-min anchor but showed that choosing
the locally target-best protected candidate did not transfer. D1X asks a
narrower question before any more training: does one fixed route rule have a
positive target effect relative to that anchor on blocks that did not choose
the rule?

## Frozen split

Seed 6 partitions the 40 approved safety pairs into eight sealed diagnostic
pairs, one per category, and 32 training pairs. The training pairs form eight
four-pair optimizer blocks. Blocks 1--4 are discovery blocks; blocks 5--8 are
qualification blocks. The qualification gradients may be opened only after a
single route descriptor has been written to the route-freeze artifact. The
diagnostic block is never forwarded in D1X.

## Route family

For proposal block `b`, let `A_b` be the exact current-AdamW max-min anchor
over the previously frozen 239-candidate D1R family. A route descriptor is
`r=(S, lambda)`, where `S` is one of the seven non-empty subsets of the three
target certificates and

```text
lambda in {0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000}.
```

This gives 56 deterministic rules. The route direction is the normalized
homotopy from the seven-certificate endpoint to the endpoint induced by `S`.
It is exercised only if every local protected exact score retains at least 95%
of the positive anchor score and it differs from `A_b`; otherwise it falls
back to `A_b`.

## Exact cross-block estimand

Let `U_0(g)` be the exact first-step AdamW displacement, including frozen
global clipping, and let `g_hat[j,k]` be normalized certificate `k` on support
block `j`. For proposal block `b`, support block `j != b`, and route `r`, D1X
computes

```text
Delta[b,j,k](r) =
  <g_hat[j,k], U_0(g_b(r)) - U_0(g_b(A_b))>
  / ||U_0(g_b,baseline)||_2.
```

The quantity is an exact optimizer-map comparison, not a linearized update.
Positive target Delta is improvement relative to the anchor. Protected
disadvantage is `-Delta`.

Discovery ranks eligible routes by a Studentized stability lower score over
four proposal-block means. The coefficient equals the one-sided 95% `t(3)`
critical value, but the score is not a confidence interval: 56 routes are
searched and the leave-one-out support blocks are reused. The selected route
is then fixed. Qualification uses four untouched proposal blocks and the other
three qualification blocks as support, producing four dependent leave-one-out
proposal clusters and 12 directed cross-block comparisons. Its lower score is
a deterministic finite-sample stability gate, not population inference.

## Prospective gates

A route reaches qualification only when all four discovery proposals exercise
it, all local retentions are at least 0.95, its discovery cluster lower bound
is positive, at least three of four cluster means are positive, and protected
mean/worst disadvantages are at most 0.005/0.020.

Qualification passes only when:

- all four proposals exercise the frozen route;
- every local protected retention is at least 0.95;
- target mean Delta is at least 0.002;
- the frozen Studentized cluster stability lower score is positive;
- all four proposal-cluster means are positive;
- every protected mean/worst disadvantage is at most 0.005/0.020;
- every exact map reports zero completed optimizer steps.

## Boundary

D1X creates one optimizer object only to evaluate the exact AdamW map. It runs
zero optimizer steps, writes no checkpoint, generates no response, and does
not open Core100, heldout, or the eight-pair diagnostic. A pass authorizes only
a separately frozen matched-training protocol using the selected route with
the anchor fallback.

Passing D1X would not establish seed-level or population-level significance.
Those claims require later independent seeds and generated behavioral
evaluation.
