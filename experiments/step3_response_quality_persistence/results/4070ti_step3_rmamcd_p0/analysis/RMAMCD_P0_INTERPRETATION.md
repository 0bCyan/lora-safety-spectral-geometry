# RMA-MCD Seed-7 P0 Interpretation

## Decision

The frozen zero-step run returned:

`STOP_RMAMCD_P0_NO_MINIMUM_EFFECTIVE_ANGLE`

The registered action is `raw_mcd`. This is a method stop, not a training
failure. The run used no optimizer or scheduler, took zero optimizer steps,
and opened neither Core100 nor heldout data.

## Frozen construction

For each certificate `q`, RMA-MCD formed

`bar b_q = (1/12) sum_i g_iq / ||g_iq||_2`

from the 12 fit units and used `b_q = bar b_q / ||bar b_q||_2` for scoring.
The tangent was projected only off the raw displacement. It was not projected
off the 48 unit-level protected gradients. A seven-coordinate affine
max-min problem jointly optimized four pooled protected retention margins and
three pooled target score margins at
`theta = 0.3143411045` radians. The returned tangent had the raw norm and
was combined by the fixed-angle spherical map.

The fit audit nevertheless evaluated all 48 protected facets and all 36
target facets. The unit-level retention-to-cosine deviation is reported, but
is not a gate: requiring that identity would be equivalent to reinstating the
old hard protected-nullspace assumption.

## Geometry evidence

| quantity | value |
| --- | ---: |
| global covectors | 7 |
| raw-only projection constraints | 1 |
| minimum projected global-covector fraction | 0.9336446692 |
| raw/tangent cosine | 3.4216037e-09 |
| tangent/raw norm ratio | 1.0000000552 |
| affine primal value | 0.2718967900 |
| affine primal-dual gap | 1.0289655e-08 |
| simplex KKT residual | 4.8157127e-08 |

These values pass the registered global geometry and KKT gates. They only
validate the pooled convex optimization and the fixed-norm construction; they
do not establish transfer to individual fit units.

## Binding fit obstruction

The proposed fixed-angle candidate raised the fit target floor from
`0.0122671819` to `0.0756816874`, a gain of `0.0634145060`. It did not meet
the protected facet gate. The binding facet was `fit_11:tarc`:

| quantity | value |
| --- | ---: |
| raw TARC score | 0.0538412671 |
| proposed TARC score | 0.0360956265 |
| proposed retention | 0.6704081921 |
| required retention | 0.9500000000 |
| minimum all-facet candidate margin | -0.2795918079 |

The pooled solver predicted a positive minimum margin (`0.2718967900`), while
the realized fit-facet minimum was negative (`-0.2795918079`). This gap is a
direct transfer failure of the global pooled covectors, not evidence that the
solver failed.

## Angle feasibility audit

The stored fit raw and tangent score panels permit an exact trigonometric
audit without another model forward. For every fit facet,

`score_iq(theta) = raw_iq cos(theta) + tangent_iq sin(theta)`.

Scanning the registered interval and solving the facet boundaries gives:

| quantity | value |
| --- | ---: |
| largest angle satisfying all protected fit facets | 0.0535424367 rad |
| target floor at that angle | 0.0233298909 |
| required target floor | 0.0500000000 |
| angles satisfying both protection and target gates | none |

Thus reducing the angle cannot repair this attempt: the protected region ends
before the target floor is reached. The full calculation is stored in
`rmamcd_p0_angle_sensitivity.json`.

## Calibration and diagnostics

Because the fit candidate was ineligible, the registered candidate-or-raw
policy selected `raw_mcd`. All 20 calibration rows therefore have retention
`1.0`, joint nonconformity `-0.05`, and threshold `-0.05`. This is a valid
policy result but is not evidence that the proposed RMA candidate passed
conformal protection.

The eight diagnostic arms are identical for `raw_mcd` and the registered RMA
arm. Their paired target gain is exactly zero. The diagnostic gates correctly
fail target advantage and effect strength after the fit stop.

## Mechanism conclusion

Across NPT-MCD, NPT-A-MCD, NPT-LR-MCD, and RMA-MCD, target-direction capacity
is available, but protected transfer to an unseen unit-level TARC facet is the
repeated bottleneck. RMA-MCD rules out the hypothesis that replacing the hard
unit-level nullspace with a seven-covector pooled margin is sufficient.

A principled next candidate is a **robust unit-facet margin MCD**: retain the
raw-only projection, but put all 48 protected retention margins and all 36
target score margins into the same affine max-min problem. Its simplex dual
would provide an explicit upper-bound/infeasibility certificate. It must use a
new prospective protocol and the unchanged 95% conformal gate; no training is
authorized by this result.

## Boundary and provenance

The run used 12 fit, 20 calibration, and 8 diagnostic units; 24 transient
finite-step evaluations; and exact state restoration. `optimizer_steps=0`,
`training_authorized=false`, `candidate_outputs_generated=false`,
`core100_opened=false`, and `heldout_opened=false`. The runtime head was
`78de29e2bd09310771c62393166e21146cdd4a09` on an NVIDIA GeForce RTX 4070 Ti.
