# Q-E-TARC P0 Positive Continuation Quota Report

Route: `PASS_QETARC_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL`.

## Geometry

The fixed quota was `0.500` at auxiliary norm
`2.216954`. Raw E-TARC/continuation cosine was
`0.113011`; the quota was active:
`True`. Achieved continuation and route cosines were
`0.500000` and
`0.916983`. The analytic continuation-dot lower
bound was `8.579384` and the observed
calibration dot was `8.579384`.

## Prompt-disjoint route transfer

Calibration/validation E-TARC cosine was
`0.849329`. Equal-budget
Q-E-TARC retained `0.937` of raw
E-TARC's validation route dot and
`1.592` of TARC's. Its positive route-dot
rate was `1.000` over 32 prompts.

## Continuation transfer

Validation continuation cosine was
`0.443648`, an advantage of
`0.364074` over TARC. The
continuation dot retained `0.532`
of RAMP's equal-budget positive control. The paired dot advantage over TARC
was `6.876627` with 95% bootstrap interval
`[6.434237, 7.321344]`; exact sign
test `p=4.65661e-10`.

## Branch mechanism and boundary

Q-E-TARC branch entropy and negative-gap dots were
`0.216725` and
`20.846155`. Positive values preserve
the strong-path signature under descent.

This reused development split is not fresh confirmation or generated
behavior. No optimizer, scheduler, checkpoint, adapter, or candidate output
was created. `optimizer_steps=0`, `training_authorized=false`,
`core100_opened=false`, and `heldout_opened=false`.
