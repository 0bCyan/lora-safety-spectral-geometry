# RUF-MCD Seed-7 P0 Interpretation

## Route

The corrected v1 run completed the registered fit, calibration, and diagnostic
workflow on the RTX 4070 Ti, then stopped at:

`STOP_RUFMCD_P0_UNIT_FACET_GEOMETRY_INVALID`

`verify-run` passed. The stop is a numerical-certificate failure, not evidence
that the robust unit-facet objective is infeasible. The active affine solver
did not satisfy the frozen `1e-7` KKT gate, so the runner correctly marked the
84-facet candidate invalid and selected raw MCD.

## Evidence

The run used 12 fit units, 20 calibration units, and 8 diagnostics. The
constraint-generation registry contained 84 facets (48 protected and 36
target), but the active solver exception was caught before a certified tangent
was returned. Consequently, the stored fit geometry has
`projection_valid=false`, no full dual certificate, and `selected_theta=0`.
The raw fit target floor was `0.0122671819`, below the registered `0.05` floor;
no RUF candidate claim is made.

The independent policy checks that do not depend on the invalid candidate were
still auditable: integrity, split conformal accept-or-raw, diagnostic common
transfer, finite-step restoration, and protected diagnostic noninferiority
were true. Target advantage and effect strength were false because the policy
used raw MCD. The conformal threshold was `-0.05`, with rank `20` and the
registered marginal lower bound `20/21`.

The run made 24 transient virtual parameter evaluations, restored the initial
trainable state hash exactly, created no optimizer or scheduler, performed
`optimizer_steps=0`, and opened no Core100 or heldout data.

## Numerical limitation and v2 boundary

The failure exposes a solver limitation rather than a scientific positive:
SLSQP's returned simplex point was not a `1e-7` full-facet certificate under
the real gradient scale. The protocol does not permit relaxing that gate.
The prospective v2 repair therefore keeps the same 84 affine margins, budget,
fixed angle, and stop rules, and replaces only the numerical simplex solve with
an explicit active-set KKT refinement. v2 must still recompute every facet and
reject any result whose full primal-dual gap exceeds `1e-7`.

