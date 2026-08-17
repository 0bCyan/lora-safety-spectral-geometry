# Q-E-TARC Matched Two-Step Smoke Report

Route: `STOP_QETARC_SMOKE_ROUTE_EFFECT_FAILED`.

## Matched finite-step result

Uniform, TARC, raw E-TARC, and Q-E-TARC each completed two matched optimizer
steps. Q-E-TARC reduced held-out diagnostic E-TARC loss by
`3.404734`, retaining
`97.582%` of raw
E-TARC's `3.489115` reduction. TARC reduced the
same loss by `3.135429`. The prompt-paired
Q-E-TARC advantage over TARC was `0.269305`
with 95% bootstrap interval
`[0.001710,
0.496328]` and exact sign-test
`p=0.2890625`.

Q-E-TARC continuation reduction was
`1.809881`, versus
`1.791481` for TARC and
`1.791037` for raw E-TARC. Its paired
advantages were `0.018400` over TARC and
`0.018844` over E-TARC, with bootstrap lower bounds
`0.015413` and
`0.015701`.

## Mechanism trace

Every safety draw recomputed `g_U`, `g_T`, `g_E`, and `g_C`. The Q-E-TARC
minimum achieved continuation cosine was
`0.49999997`;
minimum route retention was
`0.88464157`; and
maximum auxiliary budget relative error was
`3.026e-08`.
Q-E-TARC branch-entropy and branch-negative-gap reductions were
`-0.001270` and
`0.047852`.

## Evidence boundary

The eight diagnostic prompt pairs were excluded from all smoke updates, but
they came from a previously disclosed development split. This is real-model,
finite-step, teacher-forced mechanism evidence, not a fresh confirmatory or
generated-behavior result. No candidate, Core100, or heldout data was opened.
Total optimizer steps: `8`;
`development_training_authorized=false`; `main_training_authorized=false`.
