# Step3 BARP-MCD D1X-C1: Conditional Route Qualification

## Status

This is a post-selection hypothesis test, not a continuation of the failed
D1X pass route. D1X-R1 searched 56 route descriptors on four discovery blocks
and found no eligible fixed route. It stopped before opening any qualification
gradient. The unique descriptor with three positive target proposal clusters,
`mask072_lambda0.500`, is bound here once for evaluation on the four still
sealed qualification blocks.

## Policy

For each qualification proposal block, construct the exact zero-state AdamW
max-min anchor over the frozen 239-candidate family. Construct the fixed route
from the E-TARC plus branch-negative-gap endpoint at homotopy fraction 0.5.

```text
if minimum local protected retention >= 0.95
and route != anchor:
    apply route virtually
else:
    apply anchor virtually
```

No parameter is updated. Each proposal displacement is scored on the other
three qualification blocks, giving 12 directed cross-block comparisons. The
four leave-one-out proposal clusters share support blocks, so the frozen
Studentized lower score is a finite-sample stability gate, not a confidence
interval or population inference.

## Gates

- the route is exercised on at least three of four proposals;
- the applied policy has minimum local protected retention at least 0.95;
- target macro mean Delta is at least 0.002;
- the Studentized stability lower score is positive;
- at least three proposal clusters are positive and none is negative;
- E-TARC, branch entropy, and branch negative-gap each have positive mean Delta;
- protected mean and worst disadvantages are at most 0.005 and 0.020;
- every exact map reports zero completed optimizer steps.

These thresholds are frozen before any C1 qualification forward. C1 may not
change the route after seeing qualification, and no second C1 descriptor is
authorized.

## Boundary

A pass authorizes only a separately frozen matched-training protocol using the
same conditional anchor fallback. A pass does not authorize direct training,
generation, Core100, heldout, the eight-pair diagnostic, or a paper claim. A
failure closes this BARP target-route line under the current certificates.
