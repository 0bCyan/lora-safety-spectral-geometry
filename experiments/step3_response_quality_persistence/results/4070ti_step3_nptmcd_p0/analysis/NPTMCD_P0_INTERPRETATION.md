# NPT-MCD Seed-7 P0 Interpretation

## Decision

The frozen run at protocol commit `c016fee6bf3ea9ef9114fc3ea76cb964162716b8`
returned:

`STOP_NPTMCD_P0_NO_MINIMUM_EFFECTIVE_ANGLE`

The registered action is `raw_mcd`. This is a method failure, not a training
failure: the run used no optimizer or scheduler, took zero optimizer steps,
and opened neither Core100 nor heldout data.

## Geometry evidence

The fit construction used 48 unit-level protected gradient covectors and the
raw displacement as a 49th orthogonality constraint. The numerical rank was
49. The projected target proposal retained `0.9575161167` of its norm, while
the final normalized constraint residual was `3.6378204e-9`. The raw/tangent
cosine was `3.6326382e-9`, and the tangent/raw norm ratio was
`0.9999999451`.

These values support the registered nullspace and equal-norm identities. They
do not imply that the nullspace contains a useful target direction.

## Binding fit obstruction

The angular search was fit-only and searched the fixed interval
`0 <= theta <= 0.3143421045` radians. The registered target floor was
`max(0.05, raw_floor + 0.002) = 0.05`.

The binding unit was `fit_09`:

| quantity | value |
| --- | ---: |
| raw `branch_entropy` score | 0.0122671819 |
| projected tangent `branch_entropy` score | 0.0102430675 |
| raw target minimum | 0.0122671819 |
| required target floor | 0.0500000000 |

Because the tangent component for this certificate is also below the raw
component, increasing the registered angle cannot raise this score to the
floor. The exact trigonometric root enumeration therefore found no jointly
feasible positive angle. The implementation returned the raw anchor rather
than selecting a post hoc angle.

## Calibration and diagnostics

The candidate was fixed before calibration, and all 20 calibration units had
protected retention `1.0` with conformal threshold `-0.05`. This is the
trivial raw-anchor case; it is not evidence that a novel NPT-MCD correction
passed calibration. The eight diagnostic rows likewise have identical raw
MCD and NPT-MCD displacements because `theta=0`.

The diagnostic measurements are useful as an audit: the raw anchor itself
shows common transfer and finite-step descent on this split, but the NPT-MCD
arm has zero paired gain over raw. Consequently no target-advantage or
effect-strength claim is supported.

## Method implication

The experiment validates the protection geometry but falsifies the specific
proposal rule "project the pooled target-endpoint difference" as a sufficient
way to obtain a useful protected direction. A follow-up must choose a tangent
inside the same protected nullspace by an independently frozen fit-only
max-min target objective (with a fixed norm and analytic feasibility audit),
rather than changing the angle, fit split, protection floor, or target floor
after observing this result. That follow-up requires a new prospective
protocol and a new run; this P0 protocol is closed.
