# RAMP-P0 Zero-Update Directional Mechanism Report

Route: `PASS_RAMP_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL`.

## Structural Evidence

All `80` approved targets support exactly `16` prefix and
suffix margin terms. The 16-token prefix completes the first sentence in
`0.988` of rows, reaches a
registered safe-routing onset in `0.887`
of rows, and has normalized uniqueness
`1.000`.

## Gradient-Budget Evidence

Both auxiliary gradients were calibrated to `0.250` of the uniform safety
gradient norm. Prefix and suffix coefficients were `0.262232`
and `0.239193`; the maximum relative budget error was
`1.530e-08`. Calibration prefix/suffix gradient cosine was
`0.707916`.

## Prompt-Disjoint Directional Evidence

Calibration-to-validation prefix cosine was
`0.906324`. The suffix-to-prefix
validation cosine was
`0.662709`, giving a
registered prefix-over-suffix margin of
`0.243614`.

This P0 tests a first-order, prompt-disjoint certificate direction. It does not
test generated behavior, safety, utility, or publication readiness. No optimizer
or scheduler was created, the exact trainable-state hash was unchanged, and
`optimizer_steps=0`, `training_authorized=false`, `core100_opened=false`, and
`heldout_opened=false`.
