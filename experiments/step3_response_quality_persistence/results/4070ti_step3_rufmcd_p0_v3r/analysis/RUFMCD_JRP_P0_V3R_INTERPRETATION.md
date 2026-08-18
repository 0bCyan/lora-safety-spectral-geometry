# RUF-MCD-JRP v3r Interpretation

## Decision

The verified recovery run returned

`STOP_RUFMCD_JRP_P0_ROBUST_POLYTOPE_GEOMETRY_INVALID`.

This is an implementation-invalid result, not evidence that the registered
96-facet robust problem is infeasible. The affine solver was never started.

## Root cause

The 12 fit TARC response covectors and all 12 leave-one-out reflected vertices
were constructed. The runner then entered original protected-facet objective
assembly and attempted to read `spec["raw_score"]`. The reused v2
`_facet_specs` registry contains the normalized gradient, unit index, and
certificate name, but does not store a raw score. The run therefore recorded

`projection_error = "'raw_score'"`

and selected raw MCD before any 96-facet affine solve. The frozen score source
for an original facet is instead

`raw_score_panels[spec["unit_index"] + 1][spec["certificate"]]`.

Using that source is the same v2 definition and changes no mathematical
objective, threshold, role, or candidate rule.

## What must not be inferred

The 20 calibration retentions were all `1.0` and the conformal threshold was
`-0.05` because candidate selection had already fallen back to raw MCD. These
numbers do not show that JRP transferred safely. Likewise, the zero target
gain and diagnostic equality are properties of the fallback, not of a JRP
candidate.

No KKT residual, primal-dual gap, full 96-facet gap, or robust margin exists
for this attempt. It would be incorrect to describe the route as finite-set
infeasibility.

## Integrity and next boundary

`verify-run` reproduced the stored route and eight-artifact inventory. Trainable
state hashes were identical before and after measurement, `optimizer_steps=0`,
and neither Core100 nor heldout was opened.

The only justified continuation is a separately frozen code-only repair of
the raw-score field source. Because calibration and diagnostic roles have now
been opened, that replay remains exploratory recovery evidence and cannot be
called fresh confirmation. No geometry or threshold tuning is justified by
this invalid run.
