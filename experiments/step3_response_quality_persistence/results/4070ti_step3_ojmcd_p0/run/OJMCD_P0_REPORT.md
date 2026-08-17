# OJ-MCD Zero-Update P0 Report

Route: `STOP_OJMCD_P0_DUAL_CERTIFICATE_FAILED`.

Eight real optimizer-unit blocks were evaluated at the unchanged initial
Qwen3-4B LoRA state. Every block aggregated four risk and four safety
microbatches before applying the exact clipping plus first-step Adam map.

Across seven prompt-disjoint validation blocks, OJ-MCD's minimum exact
normalized full-update score was
`-0.031687`. Its mean paired
worst-score advantage over equal-budget raw seven-certificate MCD was
`-0.093072` with 95% bootstrap interval
`[-0.097780,
-0.089138]` and exact sign-test
`p=0.015625`. The corresponding
advantage over the no-auxiliary mixed baseline was
`0.000001`.

The minimum and mean linear-versus-exact secant cosines were
`0.892064` and
`0.896029`. The exact realized auxiliary
increment occupied at least
`2.692914%` of the baseline update
norm over validation blocks.

This is exploratory mechanism evidence on reused disclosed development data,
not generated-behavior or confirmatory evidence. `optimizer_objects=0`,
`optimizer_steps=0`, `training_authorized=false`, and no candidate, Core100,
or heldout data was opened.
