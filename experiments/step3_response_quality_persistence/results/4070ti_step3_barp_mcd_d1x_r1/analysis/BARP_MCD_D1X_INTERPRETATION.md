# BARP-MCD D1X Cross-Fitted Interpretation

Decision: `STOP_BARP_MCD_D1X_DISCOVERY_NO_STABLE_ROUTE`.

## What Passed

The exact optimizer-space max-min anchor remained strictly feasible on all four
seed-6 discovery proposals. The four minimum exact margins were `0.15587761`,
`0.08840900`, `0.12736905`, and `0.14469692`, each far above the frozen
`1e-8` floor. Every search evaluated all 239 candidates at AdamW step zero.

The cross-block protected gates were not the bottleneck. All 56 fixed routes
passed both the mean and worst directed protection limits for risk, uniform,
TARC, and continuation after the frozen anchor fallback was applied.

## What Failed

No route passed the complete discovery gate. Only two of 56 routes exercised
a non-anchor direction on all four proposals, and both were E-TARC-only routes:

```text
route                     target mean     stability lower    positive clusters
mask008_lambda0.375       -0.02017072     -0.02685337        0/4
mask008_lambda0.500       -0.02271758     -0.03063718        0/4
```

Their E-TARC component was near zero or weakly positive on average, but branch
entropy and branch negative-gap were negative on all 12 directed support
comparisons. Thus a locally feasible single-target direction systematically
damaged the other two target certificates out of block.

Across the complete family, 28 routes were never exercised, two were exercised
once, seven twice, 17 three times, and only two four times. Zero of 56 had a
positive frozen Studentized stability lower score. This is a fixed-route
transfer failure, not a protection failure and not a GPU or safety-filter
failure.

## Exploratory Conditional Signal

One route, `mask072_lambda0.500`, was the unique route with at least three
positive proposal clusters. It combines E-TARC and branch negative-gap. It was
locally feasible on three proposals and fell back to the anchor on proposal 2,
where the un-applied candidate retained only `0.66491523` of the worst protected
anchor score.

With that fallback, its target cluster values were

```text
[0.00404113, 0.00000000, 0.00420089, 0.00043841]
mean                         0.00217011
Studentized stability lower -0.00049007
```

All three target component means were positive: E-TARC `0.00185914`, branch
entropy `0.00201463`, and branch negative-gap `0.00263655`. This does not change
the D1X decision. The route failed the frozen 4/4 activity requirement and the
positive stability-lower requirement, and it was identified after searching
56 routes.

It is therefore a hypothesis, not evidence of method success. Because D1X
stopped before opening any qualification gradient, a new protocol may bind
this single descriptor and test the conditional anchor-fallback rule once on
the four still-sealed qualification blocks. That test must retain the
post-selection disclosure and may not reuse qualification results to alter the
descriptor or thresholds.

## Boundary

D1X-R1 completed in `2213.74` seconds with `18.14%` VRAM headroom. Independent
artifact verification passed. Qualification gradients, the eight-pair
diagnostic, Core100, and heldout remained sealed. Optimizer steps, checkpoint
writes, and generated responses were all zero.
