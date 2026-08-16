# TARC-P0 Zero-Update Directional Mechanism Report

Route: `PASS_TARC_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL`.

All `80` approved targets completed a registered refusal route
within `4` semantic tokens. Prompt tries had
branching rate `1.000` and distinct-route
pair rate `1.000`.

RAMP and TARC auxiliary gradients were independently calibrated to `0.250`
of the uniform gradient norm. Coefficients were `0.256332` and
`0.20565`; maximum relative budget error was
`2.806e-08`.

TARC calibration-to-validation cosine was
`0.766235` versus
`0.389198` for RAMP,
giving TARC's registered margin
`0.377037`. Equal-budget
TARC and RAMP validation-gradient dots were
`47.157844` and
`37.312419`.

This is a first-order, prompt-disjoint initialization test, not generated
behavior or a safety/utility conclusion. No optimizer, scheduler, checkpoint,
or candidate output was created. `optimizer_steps=0`,
`training_authorized=false`, `core100_opened=false`, and
`heldout_opened=false`.
