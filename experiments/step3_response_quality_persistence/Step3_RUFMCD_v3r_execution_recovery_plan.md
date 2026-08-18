# Step3 RUF-MCD v3r: Execution Recovery

## Incident boundary

The method, data roles, 96-facet objective, fixed angle, thresholds, and
candidate-before-calibration rule were frozen in commit `6dcc612`. The first
v3 execution collected all 12 fit, 20 calibration, and 8 diagnostic units, but
raised `RecursionError` after measurement and before route computation or
artifact writes.

The error was dispatch-only: `main()` replaced `base.evaluate_gates` with the
v3 evaluator, while the v3 evaluator dynamically called
`base.evaluate_gates`. This recursed after the base run reached gate
evaluation. No optimizer or scheduler existed, `optimizer_steps=0`, and the
run directory was never created.

## Frozen repair

v3r captures the original base evaluator at module import and calls that
immutable reference from the v3 evaluator. It also registers this recovery
status explicitly. No geometry, response covector, reflection, solver,
threshold, role, score, conformal, diagnostic, or route ordering changes.

The exact repaired relation is

`_BASE_EVALUATE_GATES = base.evaluate_gates`

before monkey-patching, followed by

`gates = _BASE_EVALUATE_GATES(...)`.

The recovery protocol reuses the already frozen v3 config and method plan but
writes to a new `4070ti_step3_rufmcd_p0_v3r` output directory. The original v3
protocol and incident record are frozen inputs.

## Claim boundary

The recovery replay is deterministic execution evidence for the method that
was frozen before the first v3 model forward. Because the first failed process
already opened all fit, calibration, and diagnostic roles, v3r is not fresh
confirmation. Its calibration and diagnostic results may diagnose the frozen
method but cannot be used to tune and then re-claim success on the same rows.

Training, generation, Core100, heldout, checkpoint writes, optimizer objects,
and optimizer steps remain unauthorized.
