# FRSC-MCD Seed-2 Zero-Step P0 Report

Route: `STOP_FRSC_P0_RAW_COMPARATOR_FAILED`.

Eight optimizer-unit training blocks independently enumerated all 127
nonempty subsets of the seven registered certificates. The minimum selected
training exact margin was
`0.066073`. The mean
selected-minus-full-seven margin gain was
`0.019180` with 95% bootstrap interval
`[0.013028,
0.026894]`.

The fixed eight-category diagnostic set was excluded from every direction
construction block. FRSC-MCD's minimum diagnostic exact score was
`0.049296`. Its mean
paired diagnostic worst-score advantage over full-seven raw MCD was
`0.005300` with exact sign-test
`p=0.0703125`.

All reported finite loss changes came from transient direct applications of
the exact first-step displacement at learning rate
`0.0002` followed by exact parameter restoration. No model
optimizer or scheduler was created, `optimizer_steps=0`, and no checkpoint,
candidate output, Core100 row, or heldout row was opened.
