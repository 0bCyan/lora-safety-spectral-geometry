# RUF-MCD-JRP P0 v3r5 Interpretation

## Result

The 90-round recovery completed after `38177.77` seconds and independently
reproduced
`STOP_RUFMCD_JRP_P0_ROBUST_POLYTOPE_GEOMETRY_INVALID` under
`PASS_NPTMCD_P0_VERIFY`.

This is no longer a round-budget stop. The exact error was
`RUF-MCD full active set did not close the dual gap`.

## What the error proves

The control-flow location of this error is informative. It can be emitted
only after:

1. the active analytic solver reports success;
2. its KKT residual and active primal-dual gap pass the frozen `1e-7` limits;
3. the independently recomputed full gap remains above `1e-7`; and
4. no inactive facet remains.

All 96 facets were therefore active at the stop. In exact arithmetic, the
active and full primal minima must then be identical for the same direction.
The implementation, however, evaluates the active certificate in its
float64 Gram system, reconstructs the direction through weighted tensor
additions in the stored tensor dtype, and recomputes every full score on that
realized tensor direction. The stop localizes the remaining failure to this
continuous-certificate versus realized-tensor boundary.

The run did not persist the final realized gap magnitude, so the analysis
does not guess it. It also does not call the 96-facet polytope infeasible:
that conclusion is not supported by this error path.

## Fallback boundary

No realized-direction certificate was returned. The registered angle was
zero and the selected displacement was exactly raw MCD. Consequently:

- all 20 calibration retentions of `1.0` and the conformal threshold of
  `-0.05` describe raw fallback;
- calibration target gain over raw was exactly zero;
- all eight diagnostic candidate-minus-raw target and worst-score
  differences were exactly zero;
- the passed common-transfer and protected-noninferiority gates do not
  establish a JRP effect.

Raw MCD itself improved diagnostic worst score over the baseline by a mean
`0.0590051`, with bootstrap 95% interval `[0.0473679, 0.0677231]` and an
exact two-sided sign-test `p=0.0078125`. That is useful anchor evidence, but
it is not evidence for the proposed JRP correction.

## Integrity

The initial and final trainable-state hashes matched. The run completed 12
fit, 20 calibration, and eight diagnostic units with 24 transient virtual
parameter evaluations. It created no optimizer or scheduler, performed zero
optimizer steps, wrote no checkpoint or candidate output, and opened neither
Core100 nor heldout. Training remains unauthorized.

## Next certificate

The justified successor is a prospective numerical-certificate repair, not
another round increase and not a threshold relaxation. It should:

1. solve the unchanged complete 96-facet registry directly once;
2. retain the continuous analytic KKT and Gram primal-dual limits at `1e-7`;
3. separately construct the parameter-dtype realized direction;
4. recompute all 96 realized margins and attach explicit floating-point
   lower-error bounds; and
5. require every error-adjusted realized margin to pass before exposing a
   nonzero candidate to calibration.

This separates continuous max-min optimality from deployable tensor
feasibility. Calibration and diagnostic rows remain excluded from selection.
Because the same rows have now been opened repeatedly, any successor remains
exploratory mechanism evidence rather than fresh confirmation.
