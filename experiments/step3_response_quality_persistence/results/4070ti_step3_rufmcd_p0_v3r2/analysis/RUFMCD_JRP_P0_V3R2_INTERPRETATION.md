# RUF-MCD-JRP v3r2 Interpretation

## Decision

The verified v3r2 recovery returned

`STOP_RUFMCD_JRP_P0_ROBUST_POLYTOPE_GEOMETRY_INVALID`.

The raw-score registry repair worked: objective construction progressed into
the real 96-facet constraint-generation solve. The process then stopped after
the frozen 12-round limit. No completed full-facet certificate was returned.
This remains a protocol-completeness stop, not a method-infeasibility result.

## Why 12 rounds is no longer complete

The solver initializes one lowest-offset facet from each certificate group.
After a solved round, it adds at most one violating inactive facet per group.
In v2, every group contained 12 facets, so 12 rounds were a valid worst-case
bound for presenting every facet.

v3 retains six 12-facet groups but the TARC group now contains 24 facets:
12 observed fit TARC constraints plus 12 reflected TARC response constraints.
Consequently, a completeness proof requires at most 24 rounds for the largest
group. Reusing the v2 value 12 was structurally inconsistent with the new
registry, independent of the observed scores.

## What the run does and does not establish

The run spent `1531.48` seconds and ended with

`projection_error = "RUF-MCD constraint generation exceeded the frozen round limit"`.

Because the exception path intentionally discards an uncertified partial
solver state, there is no registered 96-facet KKT residual, full primal-dual
gap, robust margin, or candidate. Calibration and diagnostics therefore used
raw MCD, yielding retention `1.0`; those values are not JRP transfer evidence.

The KKT and full-gap thresholds remain `1e-7`. Raising a certification round
budget to the combinatorial bound 24 does not relax either threshold and does
not change the finite polytope or objective.

## Next boundary

A separately frozen v3r3 may change only the constraint-generation maximum
from 12 to 24 and must validate that config and protocol agree. All geometry,
fit thresholds, conformal rules, and diagnostic gates remain unchanged.

The run restored trainable state exactly, used `optimizer_steps=0`, and opened
neither Core100 nor heldout. Since all roles were already opened, v3r3 remains
exploratory recovery rather than fresh confirmation.
