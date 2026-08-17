# RP-PFRSC Seed-4 Zero-Step P0 Report

Route: `STOP_RPFRSC_P0_PROTECTED_NONINFERIORITY_FAILED`.

Eight optimizer-unit construction blocks screened all 127 nonempty subset
directions, retained the fixed top 16 route/branch endpoints, and searched
129 raw-anchored homotopy candidates. The minimum selected protected-score
retention was
`0.953580`. The mean
construction target-bottleneck gain over raw MCD was
`0.010128` with 95% bootstrap interval
`[0.005234,
0.014962]`.

The fixed eight-category diagnostic set was excluded from every direction
construction block. RP-PFRSC's minimum diagnostic exact score was
`0.064078`. Its mean
paired diagnostic route/branch bottleneck advantage over raw MCD was
`0.007105` with exact sign-test
`p=0.03125`. The macro-average
finite-loss relative gain over the three target objectives was
`0.063514`.

All reported finite loss changes came from transient direct applications of
the exact first-step displacement at learning rate
`0.0002` followed by exact parameter restoration. No model
optimizer or scheduler was created, `optimizer_steps=0`, and no checkpoint,
candidate output, Core100 row, or heldout row was opened.
