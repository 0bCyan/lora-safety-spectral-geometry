# CC-E-TARC P0 Zero-Update Mechanism Report

Route: `STOP_ETARC_P0_CONTINUATION_COMPATIBILITY_FAILED`.

## Design integrity

The fixed split used `8` calibration prompts and all
remaining `32` approved prompt clusters for disjoint
validation. All `80` approved targets reached a registered
route within `4` semantic tokens. The model
state was restored exactly: `True`.

All four auxiliary directions were independently normalized to `0.250`
of the calibration uniform-gradient norm. Maximum relative norm error was
`2.806e-08`. The E-TARC/TARC
calibration cosine was `0.449543`.

## Route mechanism

E-TARC calibration-to-validation cosine was
`0.849329`. Equal-budget
CC-E-TARC retained `1.000` of
raw E-TARC's validation route dot and
`1.699` of TARC's route dot.
Its positive per-prompt route-dot rate was
`1.000`.

For the branch diagnostics, positive entropy dot predicts entropy reduction
under gradient descent, while positive negative-gap dot predicts an increased
top1-versus-top2 legal-token gap. CC-E-TARC values were
`0.234431` and
`23.771944`.

## Continuation compatibility

Calibration projection active: `False`. The raw
route/continuation dot was
`25.180371` and the projected residual
was `2.518e+01`. On 32 unseen prompts,
CC-E-TARC's aggregate continuation dot was
`1.598499` and its compatible-prompt
rate was `1.000`, an advantage of
`0.000` over
TARC. The paired continuation-dot advantage was
`0.095515` with bootstrap 95% interval
`[-0.079828, 0.268630]`.

## Claim boundary

This is a real-model, prompt-disjoint first-order result at fresh LoRA step 0.
It is not generated behavior, semantic safety, utility, or a paper-level
effect endpoint. No optimizer, scheduler, checkpoint, or candidate output was
created. `optimizer_steps=0`, `training_authorized=false`,
`core100_opened=false`, and `heldout_opened=false`.
