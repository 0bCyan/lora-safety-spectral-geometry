# Step3 RUF-MCD v3r6: Realized Tensor Certificate

## Registered predecessor

v3r5 exhausted the globally complete `N-G+1=90` work bound and stopped with

`RUF-MCD full active set did not close the dual gap`.

Independent `verify-run` passed. The error is reachable only after the active
analytic solver passes its `1e-7` KKT and primal-dual limits, the independently
recomputed full gap remains above `1e-7`, and all 96 facets are active.

The selected angle was zero. Calibration and diagnostic candidate-minus-raw
differences were therefore all zero and provide no JRP effect evidence.

## Numerical boundary

The continuous solver works in a float64 Gram system. For dual weights
`lambda`, budget `B`, and facet covectors `g_j`, its ideal direction is

`u* = B G lambda / ||G lambda||`.

The deployable direction is a tensor registry reconstructed by finite-precision
weighted additions and converted to the trainable parameter dtype. Exact
continuous optimality and realized tensor feasibility are distinct claims.
v3r5 incorrectly required the realized point to reproduce the ideal dual value
within the continuous `1e-7` gap, even when every facet was already active.

v3r6 keeps the continuous certificate unchanged:

- continuous KKT residual `<= 1e-7`;
- continuous Gram primal-dual gap `<= 1e-7`.

It then re-normalizes the realized tensor direction, recomputes all 96 realized
facet scores, and supplies a per-score floating-point lower bound. For a
binary64 dot product with `m` rounded operations and unit roundoff `u`,

`gamma_m = m u / (1 - m u)`.

If `A_hat` is the accumulated absolute-product sum, the registered conservative
dot error is

`E = 2 gamma_m A_hat / (1 - gamma_m) + ulp(dot)`.

The factor two is frozen. Score-addition rounding is added separately and the
lower endpoint is rounded toward negative infinity. This is a deterministic
forward-error certificate, not a tuned tolerance.

Certificate validity means every score and bound is finite, the realized norm
relative error is at most `1e-7`, and the continuous certificate passes.
Candidate feasibility is separate: the minimum error-adjusted realized margin
must be nonnegative. A valid certificate with a negative lower margin routes to
no robust effective angle rather than geometry failure.

## Direct full-set equivalence

v3r5 established that all 96 facets become active. Solving the unchanged
96-facet registry directly therefore solves the same continuous max-min
problem as the terminal complete active-set round. Direct solution removes
redundant prefix solves but does not remove a facet, change an offset, alter the
angle, or use calibration/diagnostic data.

## Frozen and forbidden changes

The fit/calibration/diagnostic split, model, seed, gradients, 96 facets, 12
jackknife-reflected TARC vertices, fixed angle, protected floor, target floor,
selection rule, conformal rule, and downstream gates remain unchanged.

No threshold relaxation, facet removal, manual candidate choice, calibration
selection, optimizer, scheduler, checkpoint, candidate generation, Core100,
heldout, or training is authorized. The reused rows make this exploratory
mechanism evidence only, never fresh confirmation.
