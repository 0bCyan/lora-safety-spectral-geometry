# BARP-MCD D1 Runtime Stop

Decision: `STOP_BARP_MCD_D1_RAW_ANCHOR_FAILED`.

## What executed

The frozen seed-5 run completed three `raw_mcd` optimizer steps and stopped
while scoring the fourth block, before its optimizer update. The BARP arm did
not start. No adapter, generated response, Core100 row, or heldout row was
written or opened.

All 21 registered same-batch losses decreased on the three completed updates.
Their minimum reduction was `0.000315869680434844`. The largest observed
analytic-versus-real AdamW update error was `1.1512644812172425e-06`, below the
frozen `5e-6` limit. The first three raw anchors had minimum exact certificate
score at least `0.07987606457875966`.

## Hard failure

At the fourth seed-5 block,
`enumerate_memory_bounded_homotopy` raised `Q1R raw anchor lacks common
descent`. The guard is evaluated before a real update, so the failed block did
not consume an optimizer step. It proves that at least one of the seven exact
certificate scores under the current AdamW state was non-positive.

The frozen implementation raises before serializing those seven scores.
Therefore this result does **not** identify the failing certificate or claim a
score magnitude. Replaying three updates merely to instrument the fourth state
would consume unregistered optimizer steps and was not done.

## Mechanism implication

For fixed AdamW state `(m, v, t)`, D1 scores a gradient `g` through the exact
positive update map

```text
U_t(g) = m_hat_t(g) / (sqrt(v_hat_t(g)) + epsilon).
```

The MCD construction gives a common-descent direction in gradient geometry,
but `U_t` is stateful and nonlinear. Consequently, positive gradient-space
inner products do not imply positive update-space scores
`<c_j, U_t(g)>` after several steps. D1 observed precisely this missing
implication on a prospectively frozen batch.

This invalidates the present raw-anchor non-emptiness argument for longer AdamW
training. It does not show that route-protected optimization is impossible; it
shows that the anchor itself must be constructed in optimizer update space (or
be accompanied by a proved reachable-update certificate) rather than assumed
from gradient-space MCD.

## Boundary and next route

D1 is incomplete and does not authorize checkpoint-bound generation. The next
permitted work is a zero-update feasibility study for an optimizer-update-space
anchor, with the failed-state instrumentation problem addressed prospectively.
No additional training threshold should be changed on the basis of this run.
