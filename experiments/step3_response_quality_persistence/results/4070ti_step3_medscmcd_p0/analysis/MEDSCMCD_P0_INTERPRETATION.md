# MED-SC-MCD Seed-6 P0 Interpretation

## Decision

The verified route is
`STOP_MEDSCMCD_P0_NO_MINIMUM_EFFECTIVE_DOSE`. The run is valid as a negative
exploratory mechanism result. It does not authorize a proxy-dev protocol,
training, generation, or any optimizer step.

## Registered-grid result

The raw fit target floor was `0.0474449474`. None of the 32 registered positive
doses satisfied the conjunction of 95% protected retention, target floor
`0.05`, and target-floor gain `0.002`.

At the first positive dose, `lambda=0.03125`:

- the fit target floor rose to `0.0493305795`, a gain of `0.0018856321`;
- the worst protected retention fell to `0.9436408336`;
- the protected bottleneck was `fit_01 / tarc`, whose score fell from
  `0.0092287106` to `0.0087085881`;
- the target bottleneck was `fit_10 / branch_negative_gap`.

At `lambda=0.0625`, the target floor and gain gates passed (`0.0512481125` and
`0.0038031651`), but worst protected retention fell to `0.8910048668`. All 31
positive candidates from this dose onward passed both target gates and failed
the protection gate. The failure is therefore a registered-grid Pareto
incompatibility, not a conformal sample-size failure.

A secant calculation over only the first registered interval places 95%
retention at approximately `lambda=0.027724`, where the interpolated target
floor is `0.049118`. Conversely, the interpolated dose needed for target floor
`0.05` is `0.042344`, where retention is approximately `0.923633`. These two
numbers are diagnostics, not a proof over the continuous nonlinear optimizer
map; the exact claim is limited to the frozen 33-point family.

## Calibration and diagnostics

Because no positive fit candidate existed, the frozen selection rule returned
index zero, the raw MCD anchor. The calibration threshold
`-0.0499999537` consequently certifies the identity fallback, not a new dose.
The route artifact's `registered_policy_action=fit_candidate` describes the
conformal action on the fixed object; that object is numerically the raw anchor
because `selected_eligible=false` and `selected_fraction=0`. No nonzero
candidate was authorized.

The candidate-versus-raw diagnostic differences are numerical-scale noise.
The exact target-minimum difference averaged `-3.45e-9`, while the finite-loss
target macro gain averaged `0.000464` with a 95% bootstrap interval crossing
zero and sign-test `p=0.6875`. Therefore the target-advantage and effect gates
correctly failed. Common descent and protected noninferiority passed because
the measured arm was the raw MCD fallback.

## Mechanism conclusion

The target-only endpoint has a useful pooled target margin, but its correction
has a large negative component along a low-margin unit-level TARC facet. A
single scalar homotopy dose cannot independently control that component. The
next justified construction is not a looser retention threshold or a denser
post-outcome grid. It is a fit-only target correction projected into the null
space of all protected unit gradients, followed by fixed-norm angular dosing.

For raw displacement `d0`, protected-gradient operator `A`, and projected
target direction `u` satisfying `Au=0`, `<d0,u>=0`, and `||u||=||d0||`, define

`d(theta) = cos(theta) d0 + sin(theta) u`.

Then every fit protected score has exact retention `cos(theta)`. Restricting
`theta <= arccos(0.95)` proves the fit protection gate independently of the
small raw TARC denominator. Whether the protected-null target component is
large enough to meet the target floor remains an empirical question for a
separately frozen exploratory protocol.

## Boundary

`verify-run` passed with eight frozen artifacts, exact trainable-state
restoration, `optimizer_objects=0`, `scheduler_objects=0`,
`optimizer_steps=0`, `training_authorized=false`, `core100_opened=false`, and
`heldout_opened=false`.
