# RUF-MCD-JRP P0 v3r6 Interpretation

## Result

v3r6 completed in `1977.79` seconds and independently reproduced
`PASS_NPTMCD_P0_VERIFY`. The registered route was
`STOP_RUFMCD_JRP_P0_NO_ROBUST_EFFECTIVE_ANGLE`, but that route is not a valid
description of the measured fit geometry: it was selected by a misapplied
inherited gate.

The candidate is still not authorized. After auditing the gate, the first
substantive failure is the frozen calibration policy.

## Numerical certificate

The v3r5 numerical repair succeeded. The direct continuous solve covered all
96 facets in one solve, selected 22 active facets, and achieved KKT residual
and primal-dual gap `4.47e-15`, far below the unchanged `1e-7` limits.

The parameter-dtype direction was certified separately. Its norm error was
`2.05e-8` against a `1e-7` limit. All 96 realized scores received explicit
roundoff bounds. The smallest error-adjusted margin was `0.1265107744`, on
`fit_02:tarc:jackknife_reflection`, and the largest individual roundoff bound
was `3.43e-8`. Thus the continuous optimum and the deployable tensor direction
are both certified, without treating their scores as exactly identical.

This also reduced the solver phase from the v3r5 run's repeated 90-round path
to one complete-set solve. Total runtime fell from `38177.77` to `1977.79`
seconds.

## Route defect

The fit search itself was eligible. It selected `18.0104` degrees, with:

- minimum fit protected retention `1.07651`;
- target score floor `0.176511`;
- target gain over raw `0.164244`;
- robust realized margin lower bound `0.126511`.

The only failed base fit subcondition was
`maximum_protected_retention_identity_error <= 1e-5`; the observed value was
`0.839649`.

That check encodes the identity `retention = cos(theta)`, which requires the
tangent to be orthogonal to every protected covector. JRP does not make that
claim. It optimizes explicit protected margins over a 96-facet uncertainty
polytope, and its maximum protected cosine is `0.270394`. Protected retentions
can therefore improve above one rather than equal `cos(theta)`.

The implementation source confirms the mismatch:
`step3_rufmcd_jrp_p0.py` binds `_BASE_EVALUATE_GATES` to the strict-nullspace
`step3_nptmcd_p0.evaluate_gates`. The existing RUF-MCD evaluator already
replaces this inherited identity check with its explicit margin conditions,
but JRP bypasses that replacement and only appends its robust-margin check.

Removing only this inapplicable subcondition from the frozen-output route
logic would make the next stop
`STOP_RUFMCD_JRP_P0_CONFORMAL_SELECTIVE_POLICY_FAILED`. This is a route audit,
not a retroactive protocol change or an authorization.

## Substantive failure

Calibration rejects the candidate for a real protection reason. Although all
20 units improve their target minimum and the mean target gain is `0.102698`,
TARC retention is `-0.870516` on `calibration_12` and `0.414551` on
`calibration_20`. The conformal threshold is consequently `1.820516`, and the
registered selective policy falls back to raw MCD.

The eight diagnostic units are informative but exploratory. Candidate minus
raw target-minimum gain is `0.108505`, bootstrap 95% interval
`[0.097418, 0.118679]`, with exact sign-test `p=0.0078125`. Worst-score gain is
`0.039431`, interval `[0.023266, 0.056255]`, with the same sign-test p-value.
Those gains do not repair calibration rejection. In addition, the frozen
finite-step gate narrowly fails because branch-entropy actual-to-predicted
ratio is `0.246747 < 0.25`, and protected noninferiority fails because one
TARC relative actual-reduction difference is `-0.164552 < -0.15`.

## Interpretation

The credible conclusion is narrower than method success:

1. complete-set continuous optimization is numerically solved;
2. the realized tensor direction is rigorously feasible for all registered
   fit facets;
3. the registered no-angle route is an implementation artifact; and
4. the fit-only jackknife-reflection hull does not cover the observed
   calibration TARC tail.

The next protocol should first correct the gate inheritance, then redesign
the fit-only uncertainty set around unit-level normalized TARC variability.
It must retain explicit realized lower bounds and a nontrivial target margin.
Calibration and diagnostic rows are now repeatedly opened, so they may guide
mechanism diagnosis but cannot serve as fresh confirmation.

## Integrity

Initial and final trainable-state hashes match. The run completed 12 fit, 20
calibration, and eight diagnostic units with 24 transient virtual parameter
evaluations. It performed zero optimizer steps, created no optimizer or
scheduler, wrote no checkpoint or candidate output, and opened neither
Core100 nor heldout. `training_authorized=false` remains in force.
