# TARC Matched Development-16 Integrity Report

Route: `PASS_TARC_DEVELOPMENT16_INTEGRITY_FREEZE_GENERATION_GATE`.

Uniform, RAMP16, and TARC each completed 16 matched optimizer steps. All input,
exposure, first-step, numerical, assignment, VRAM, and sealed-data integrity
checks passed: `True`.

Teacher-forced TARC-loss reductions were
`4.855514` (uniform),
`5.068950` (RAMP16), and
`5.502372` (TARC). These in-sample diagnostics did
not enter the route and cannot establish generated behavior.

No candidate output, Core100, heldout, or reviewer data was opened. Total
optimizer steps: `48`;
`main_training_authorized=false`.
