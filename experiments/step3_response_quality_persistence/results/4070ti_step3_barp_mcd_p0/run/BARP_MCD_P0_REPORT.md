# BARP-MCD Seed-4 Same-Batch Zero-Step P0 Report

Route: `PASS_BARP_MCD_P0_AUTHORIZE_MATCHED_TRAINING_PROTOCOL`.

Eight optimizer batches each recomputed the 127 subset endpoints, fixed Top-16
screen, and 129 raw-anchored route-protected homotopy candidates. Selection
and transient finite-step evaluation used the same batch by algorithm design;
no independent transfer claim is made.

The minimum protected exact-score retention was
`0.953743`. Mean
same-batch target-bottleneck construction gain over raw MCD was
`0.010088`, with 95% bootstrap interval
`[0.005235,
0.014915]`.

BARP-MCD's minimum same-batch exact score was
`0.073802`. Mean paired
same-batch target-bottleneck advantage over raw MCD was
`0.010088`, with exact sign-test
`p=0.03125`. Macro-average
finite-loss relative gain over the three target objectives was
`0.071612`.

All finite losses came from transient direct applications of the exact
first-step displacement at learning rate `0.0002`, followed
by exact restoration. No optimizer or scheduler was created,
`optimizer_steps=0`, and no checkpoint, generated response, Core100 row, or
heldout row was opened.
