# MCD-E-TARC Matched Two-Step Smoke Report

Route: `STOP_MCD_ETARC_SMOKE_COMMON_EFFECT_FAILED`.

## Matched finite-step result

Uniform, TARC, raw E-TARC, Q-E-TARC, and MCD-E-TARC each completed two
matched optimizer steps. MCD-E-TARC reduced smoke-excluded diagnostic E-TARC
loss by `4.025088`, retaining
`88.525%` of raw
E-TARC's `4.546839` reduction. TARC reduced the
same loss by `3.690381`; the MCD/TARC ratio was
`1.090697`. The prompt-paired
MCD advantage over TARC was `0.334707` with 95%
bootstrap interval
`[-0.269678,
0.814260]` and exact sign-test
`p=0.2890625`.

MCD-E-TARC continuation reduction was
`1.819924`, versus
`1.799536` for Q-E-TARC and
`1.806783` for uniform CE. Its paired
advantages were `0.020388` over Q-E-TARC and
`0.013141` over uniform, with exact sign-test values
`p=0.2890625` and
`p=0.0703125`.

MCD-E-TARC's uniform-CE advantage over the uniform arm was
`0.007575`. Branch entropy and negative-gap
reductions were `0.004574` and
`2.857910`; their paired advantages
over Q-E-TARC were `0.003662` and
`2.107422`.

## Mechanism trace

Every safety draw recomputed all six objective gradients. MCD-E-TARC solved
the five-certificate simplex dual on every draw. Its minimum dynamic dual
margin was `0.60186571`, maximum KKT residual was
`3.275e-15`, maximum primal-dual gap was
`2.921e-08`, and maximum auxiliary-budget
relative error was `5.907e-08`.

## Evidence boundary

The eight diagnostic prompt pairs were excluded from all smoke updates, but
they came from a previously disclosed development split. This is real-model,
finite-step, teacher-forced mechanism evidence, not a fresh confirmatory or
generated-behavior result. No candidate, Core100, or heldout data was opened.
Total optimizer steps: `10`;
checkpoint writes: `0`; `development_training_authorized=false`;
`main_training_authorized=false`.
