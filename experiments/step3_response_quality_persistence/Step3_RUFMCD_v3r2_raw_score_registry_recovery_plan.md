# Step3 RUF-MCD v3r2: Raw-Score Registry Recovery

## Predecessor result

The verified v3r route was

`STOP_RUFMCD_JRP_P0_ROBUST_POLYTOPE_GEOMETRY_INVALID`.

Its fit artifact records `projection_error="'raw_score'"` and no affine solver
audit. The route is implementation-invalid: the 96-facet problem was never
solved, so it is not evidence of geometric infeasibility.

## Frozen repair

The v2 facet registry intentionally stores only a normalized gradient, unit
index, certificate name, and gradient norm. Raw scores are displacement-
dependent and live in the fit score panels. v3r2 replaces the invalid lookup

`spec["raw_score"]`

with the already registered v2 source

`raw_score_panels[spec["unit_index"] + 1][spec["certificate"]]`.

The same helper is used for target facets, with finite-value validation and
the existing positive-score requirement for protected facets. This changes no
score definition: it only retrieves the value from its actual registry.

No reflection, uncertainty vertex, affine margin, fixed angle, solver
tolerance, full-gap threshold, fit floor, target-gain floor, conformal policy,
data role, or diagnostic gate changes. The objective remains the original 84
facets plus 12 fit-only reflected TARC response facets.

## Recovery and claim boundary

The v3 and v3r processes already opened the frozen fit, calibration, and
diagnostic roles. v3r2 is therefore a deterministic exploratory recovery, not
fresh confirmation. It may establish whether the frozen finite problem is
certified and how that frozen candidate behaves on the already opened rows; it
cannot support retuning and re-claiming success on those rows.

The run must write to a new `4070ti_step3_rufmcd_p0_v3r2` directory and pass
the unchanged 96-facet KKT/full-gap checks. Any further implementation error
is another explicit STOP. Training, generation, optimizer steps, checkpoints,
Core100, and heldout remain unauthorized.
