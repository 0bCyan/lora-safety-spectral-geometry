# Step3 RUF-MCD v3r5: Global Constraint-Generation Bound

## Registered predecessor

v3r4 completed every registered model measurement and reproduced

`RUF-MCD constraint generation exceeded the frozen round limit`

under independent `verify-run`. The selected angle was zero and every
candidate-minus-raw result was exactly zero, so the calibration and diagnostic
rows describe raw fallback rather than a certified JRP candidate.

## Implementation-level proof

Let `N` be the total number of facets and `G` the number of certificate groups.
The solver initializes one facet per group, hence starts with `G` active
facets. Every non-closing iteration adds at least one previously inactive
facet: it adds all currently violating per-group choices, or one global
inactive fallback when that set is empty.

Therefore there are at most `N-G` non-closing rounds. The next solve has every
facet available and either certifies the full gap or emits a substantive
active-solver/full-gap error. The complete iteration bound is

`R_complete = N - G + 1`.

For v3, `N=96` and `G=7`, giving `R_complete=90`.

The prior 24-round argument used the largest group size. It was invalid because
per-group additions are conditional on the current dual solution and may occur
on different rounds. A fixed two-group, four-facet regression requires three
rounds although the largest group contains only two facets; it closes exactly
at `4-2+1=3`.

## Frozen change

v3r5 changes only the constraint-generation work allowance from 24 to 90. It
does not change the solver, addition rule, any gradient, facet, offset, angle,
margin, target floor, selection rule, split, or acceptance threshold.

In particular, active KKT residual, active primal-dual gap, and full
primal-dual gap must each remain at or below `1e-7` where registered.

## Claim and stop boundary

The fit rows have been opened five times and the calibration/diagnostic rows
four times before v3r5. The recovery is exploratory and cannot be presented as
fresh confirmation or population coverage.

A substantive solver or gate stop is retained as the result. No manual retry,
threshold relaxation, facet removal, optimizer, scheduler, checkpoint,
candidate generation, Core100 access, heldout access, or training is
authorized.
