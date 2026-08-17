# PFRSC-MCD Seed-3 Zero-Step P0 Report

Route: `STOP_PFRSC_P0_TARGET_ADVANTAGE_FAILED`.

Eight optimizer-unit construction blocks screened all 127 nonempty subset
directions, retained the fixed top 16 route/branch endpoints, and searched
129 raw-anchored homotopy candidates. The minimum selected protected-score
retention was
`0.950123`. The mean
construction target-bottleneck gain over raw MCD was
`0.020474` with 95% bootstrap interval
`[0.011638,
0.029958]`.

The fixed eight-category diagnostic set was excluded from every direction
construction block. PFRSC-MCD's minimum diagnostic exact score was
`0.027431`. Its mean
paired diagnostic route/branch bottleneck advantage over raw MCD was
`-0.003620` with exact sign-test
`p=1`. The macro-average
finite-loss relative gain over the four target objectives was
`0.076408`.

All reported finite loss changes came from transient direct applications of
the exact first-step displacement at learning rate
`0.0002` followed by exact parameter restoration. No model
optimizer or scheduler was created, `optimizer_steps=0`, and no checkpoint,
candidate output, Core100 row, or heldout row was opened.
