# CDR-OPC Mathematical Foundation

## Status and scope

CDR-OPC is a candidate successor to FOACPO-S. It is not yet a training result.
This M0 artifact fixes the mathematical controller, its numerical acceptance
conditions, and the boundary of the claim before any new model evidence is
opened. All work in M0 has `optimizer_steps=0` and
`training_authorized=false`.

FOACPO-S remains a useful ablation: factorized behavior gradients transferred
to a sealed mechanism split, but its one-dimensional anchor-to-behavior segment
missed the frozen TARC retention floor by `0.0001864`. CDR-OPC addresses that
specific geometric limitation by optimizing over a fixed low-dimensional basis
of realized optimizer updates rather than one segment.

## Controller

Let columns of `B` be fixed realized AdamW update proposals and let `z` be their
coefficients. The controlled displacement is `d = Bz`. Quality and safety
first-order scores in this basis are `q^T z` and `s^T z`. For protected
certificate `j`, fit blocks define a mean score vector `mu_j` and covariance
square root `L_j`, with `L_j^T L_j = Sigma_j`.

CDR-OPC solves

```text
minimize over z,tau
    1/2 (z-z_beh)^T M (z-z_beh) - w tau + epsilon/2 tau^2

subject to
    q^T z >= tau
    s^T z >= tau
    mu_j^T z - kappa_j ||L_j z||_2 >= f_j       for every protected j
    ||z||_2 <= R.
```

Here `M` is positive definite, `epsilon > 0`, and a registered robustly feasible
fallback is supplied before optimization. The objective finds a Pareto-balanced
quality/safety update close to the behavior target while the protected
constraints account for block-to-block variation.

## Proposition 1: exact ellipsoidal robust counterpart

Define the uncertainty set

```text
U_j = {mu_j + L_j^T u : ||u||_2 <= kappa_j}.
```

For any fixed `z`,

```text
min_(a in U_j) a^T z
  = mu_j^T z + min_(||u||<=kappa_j) u^T L_j z
  = mu_j^T z - kappa_j ||L_j z||_2.
```

The last equality is Cauchy-Schwarz and is attained by
`u = -kappa_j L_j z / ||L_j z||` when the denominator is nonzero. Thus each
implemented protected inequality is the exact robust counterpart for its
registered ellipsoid, rather than a sampled approximation.

## Proposition 2: convexity and numerical certificate

The objective is strongly convex because `M` is positive definite and
`epsilon > 0`. Each protected left side is affine minus a norm and therefore
concave; its superlevel set is convex. The two Pareto half-spaces and Euclidean
trust region are convex. Hence the full problem is a convex second-order cone
program with at most one minimizer.

The implementation uses SciPy SLSQP only to obtain a numerical point. Its native
success flag is recorded but is not a proof obligation. The implementation
independently checks primal feasibility, nonnegative multipliers, stationarity,
complementarity, and objective dominance over the registered fallback. For a
convex differentiable point, the accepted KKT system is sufficient for global
optimality even if a line-search routine reports a non-success status. At a zero
norm, the implementation uses the zero member of the norm subdifferential. A
failed residual gate returns the fallback and makes no optimization-validity
claim.

## Proposition 3: finite-sample transport policy

The candidate is fixed before protected calibration is opened. For calibration
unit `i`, define a sign-safe standardized joint nonconformity

```text
r_i = max_j [
    (score_fallback(i,j) - score_candidate(i,j))
    / max(|score_fallback(i,j)|, scale_floor) - delta
].
```

With `n` exchangeable calibration units, the registered split-conformal upper
bound is order statistic

```text
r_(ceil((n+1)(1-alpha))).
```

The fit-frozen positive `scale_floor` prevents a near-zero denominator; the
absolute value makes the ordering meaningful for a negative fallback score.
At `n=20` and `alpha=0.05`, the rank is 20 and the marginal coverage lower
bound is `20/21 = 0.952381`. The complete policy uses the candidate only when
this bound is nonpositive and otherwise uses the prevalidated fallback. This is
a marginal guarantee for the complete candidate-or-fallback policy under
exchangeability. It is not a conditional guarantee given candidate acceptance.

## What is and is not proved

M0 proves the exact robust counterpart, convexity, fallback closure, and the
finite-sample rank calculation. It also provides executable tests for these
claims and an explicit numerical KKT gate. It does not prove that empirical
gradient ellipsoids contain a population distribution, that first-order scores
survive a finite parameter step, or that generated responses improve. Those are
separate M1 zero-step transport, M2 virtual-step remainder, and later blinded
behavioral evaluations. No training is authorized by this document.
