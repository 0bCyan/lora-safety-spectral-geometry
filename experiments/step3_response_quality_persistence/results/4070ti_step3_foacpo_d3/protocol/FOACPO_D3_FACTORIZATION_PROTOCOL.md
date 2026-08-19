# FOACPO D3 Factorization and Protection Certificate

## Scope and evidence timing

D2 stopped because only 13 prompts simultaneously supplied an authored
positive, a weak-safe negative, and an unsafe negative. This document is frozen
after those labels and counts are known, but before any FOACPO model forward or
gradient. It therefore makes no prospective claim about the observed pool
sizes. It freezes the mathematical revision and the exact data transformation
used by the next prospective GPU study.

## Factorized behavioral objective

Let `Q` contain independently qualified pairs `(x, y+, yQ-)` where `y+` is a
clean, detailed, actionable, natural, nonmechanical authored response and
`yQ-` is clean-safe but weaker. Let `S` contain independently qualified pairs
`(x, y+, yS-)` where `yS-` is unsafe. Define

```text
L_F(theta) = alpha E_Q[l_Q(theta)] + (1-alpha) E_S[l_S(theta)].
```

The two expectations are different estimands and need not use the same prompt.
For independent uniform minibatches `BQ` and `BS`, linearity of expectation
gives

```text
E[alpha grad mean_BQ(l_Q) + (1-alpha) grad mean_BS(l_S)]
  = grad L_F(theta).
```

Thus a common three-response prompt intersection is unnecessary for an unbiased
gradient estimator of this additive objective. This is a method revision, not a
relaxation or reinterpretation of the failed D2 triplet gate.

## Exact local protection by anchor interpolation

Let `d_A` be a strictly feasible optimizer-space anchor displacement and `d_B`
the behavior displacement. For every protected first-order descent certificate
`s_j(d)`, define

```text
d(lambda) = (1-lambda) d_A + lambda d_B,  0 <= lambda <= 1.
```

Because each certificate is linear in the displacement,
`s_j(d(lambda))=(1-lambda)s_j(d_A)+lambda s_j(d_B)`. Requiring retention
`s_j(d(lambda)) >= rho s_j(d_A)` yields the closed-form upper bound

```text
lambda_j = 1                                      if s_j(d_B) >= rho s_j(d_A)
         = (1-rho)s_j(d_A)/(s_j(d_A)-s_j(d_B))   otherwise.
```

Choosing `lambda*=min_j lambda_j` is feasible for every protected certificate
and is the largest feasible point on the anchor-to-behavior line segment. The
next GPU protocol will freeze `rho=0.95` before gradients. This is an exact
first-order optimizer-space certificate, not a finite-training safety theorem.

## Boundaries

The pool build opens no checkpoint and computes no gradient. It does not open
Core100 or heldout, does not authorize candidate generation or training, and
records `optimizer_steps=0` and `training_authorized=false`.
