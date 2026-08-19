# BARP-MCD Q1R Interpretation

## Decision

The prospectively frozen memory-bounded retry returned

```text
PASS_BARP_MCD_Q1R_AUTHORIZE_LONGER_MATCHED_PROTOCOL
```

Independent `verify-run` reproduced every gate and six artifact hashes. Q1R
performed exactly four optimizer steps, wrote no checkpoint or generated
response, and kept Core100 and heldout sealed.

## What Q1R established

BARP was recomputed through the actual current AdamW moments for two
consecutive updates. It selected nonzero fractions `0.25` and `0.125`. The
minimum four-coordinate protected retention was `0.950623`, and the minimum
three-target bottleneck gain over the corresponding current raw anchor was
`0.007497`.

The analytic optimizer displacement matched each real PyTorch update. The
largest relative L2 error was `1.1031e-6`, below the frozen `5e-6` limit. All
seven same-batch losses decreased after every real update in both arms.

This closes the implementation gap between P0's first-step virtual map and a
stateful optimizer: BARP can make a different protected choice after its own
previous update while preserving the registered local certificate.

## Memory repair

Q1 stopped because it retained all 126 endpoint directions. Q1R deleted each
endpoint tensor after scalar scoring and reconstructed only screened masks.
The candidate family and decision were unchanged.

| Arm | Q1 peak reserved | Q1R peak reserved | Reduction | Q1R headroom |
|---|---:|---:|---:|---:|
| raw MCD | 16.895 GB | 8.647 GB | 48.82% | 32.86% |
| BARP-MCD | 16.878 GB | 8.630 GB | 48.87% | 32.99% |

Both Q1R final parameter hashes exactly match Q1, as do the selected BARP
candidates and registered scores. The repair therefore changed tensor
lifetime rather than the numerical method.

## Descriptive diagnostic

The excluded eight-category block did not select the route. After two steps,
BARP's reduction relative to raw MCD was `+3.80%` for E-TARC, `+3.78%` for
branch entropy, and `+18.26%` for branch negative-gap, for a three-target
macro relative gain of `+8.62%`.

Protected reductions were slightly smaller: risk `-0.13%`, uniform `-0.56%`,
TARC `-8.03%`, and continuation `-0.62%`, with a protected macro change of
`-2.34%`. This is compatible with the intended protected tradeoff but remains
a disclosed, two-step diagnostic rather than independent evidence.

## Evidence boundary

Q1R validates stateful execution, exact-map fidelity, local protection, and
4070 Ti feasibility. It does not establish long-run stability, fresh-seed
generalization, generated safety, or utility. The pass authorizes only a new
prospective longer matched-training protocol. No further optimizer step is
authorized by Q1R itself.
