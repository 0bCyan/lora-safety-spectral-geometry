# BARP-MCD D1R Matched Interpretation

Decision: `STOP_BARP_MCD_D1R_ASSIGNMENT_NOT_EXERCISED`.

## Two Separate Findings

D1R separates the repaired optimizer anchor from the additional BARP target
bias. The anchor passed; the target bias failed.

The optimizer-space max-min anchor remained strictly feasible for every update
in both eight-step arms. Its worst exact seven-certificate margin was
`0.01676621` in the anchor arm and `0.00826457` in the treatment arm, both far
above the frozen `1e-8` floor. All 112 registered same-batch loss reductions
were positive, and the largest analytic-versus-real AdamW map error was
`1.6232e-6 < 5e-6`. Thus D1's fourth-step common-descent failure was repaired
over the complete D1R trajectory.

The route-protected treatment, however, selected a candidate different from
its anchor on only steps 2 and 3, below the frozen `6/8` activity requirement.
Those two selections had positive same-batch target bottleneck gains
`0.01457659` and `0.00360739`, with minimum protected retention `0.95607503`.
After step 3, the max-min anchor was already the target-best protected
candidate on every batch.

## Independent Diagnostic

The two early same-batch target gains did not transfer. Normalized treatment
advantage on the eight frozen category-paired diagnostic units was

```text
mean                  -0.00843966
95% bootstrap lower   -0.01031806
95% bootstrap upper   -0.00661962
best unit             -0.00460235
worst unit            -0.01338844
exact sign p           0.00781250  (8 negative, 0 positive)
```

All three target components were negative on all eight units:

```text
E-TARC mean advantage               -0.00414578
branch entropy mean advantage       -0.00150580
branch negative-gap mean advantage  -0.01966740
```

The effect is not explained by a safety trade-off. Every protected diagnostic
gate passed. Mean disadvantage was only `0.00021027` for risk, `0.00045863`
for uniform, and `0.00025997` for continuation; TARC improved by `0.00946479`
on the same sign convention.

## Mechanism

At step `t`, the treatment maximizes a target score measured on the current
four-pair optimizer block, subject to local exact-score protection. D1R shows
that this local constrained score is a valid update certificate but a poor
transfer estimator:

```text
same-batch target gain >= 0
does not imply
paired-diagnostic target gain > 0.
```

Only two updates differed between arms, yet all eight independent units moved
in the same unfavorable direction. The most direct interpretation is localized
objective overfitting, especially for branch negative-gap, rather than noisy
diagnostic variation or safety interference.

The positive result should not be lost inside the STOP route: selecting the
anchor by exact current-AdamW max-min score is a viable mechanism. What fails
is using same-batch target maximization as the treatment layered on top of it.

## Boundary and Next Route

Both external adapters are hash-inventoried, but D1R does not authorize
generation because the prospective route failed. No response, Core100 row, or
heldout row was opened.

The next method should retain the optimizer-space anchor and replace the
same-batch target selector with an estimator that has an explicit out-of-batch
generalization mechanism, such as cross-fitted route scoring or a confidence-
bounded target gain. Merely lowering the `6/8` activity gate would not address
the observed negative transfer and is not justified.
