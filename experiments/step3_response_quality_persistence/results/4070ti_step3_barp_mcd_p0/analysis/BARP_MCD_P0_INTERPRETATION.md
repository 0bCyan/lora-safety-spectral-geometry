# BARP-MCD Seed-4 P0 Interpretation

## Decision

The prospectively frozen zero-step run returned

```text
PASS_BARP_MCD_P0_AUTHORIZE_MATCHED_TRAINING_PROTOCOL
```

Independent `verify-run` reproduced all seven gates, the route, six hashed
artifacts, and the zero-step boundary. This pass authorizes only a separately
frozen matched-training protocol. It does not itself authorize an optimizer
step, generated output, Core100 access, or heldout access.

## What changed

RUF-MCD-JRP v3r6 proved that a global fixed direction could be numerically
certified yet miss unseen TARC covectors, including two directional sign
reversals. Its analytic frontier also proved that shrinking the angle could
not repair that direction.

BARP-MCD changes the scope of the direction. On every optimizer batch it
recomputes the same RP-PFRSC finite family from that batch's seven gradients:
127 MCD subset endpoints, a frozen Top-16 screen, and 129 exact-Adam homotopy
candidates. It protects risk, uniform, TARC, and continuation coordinatewise
while optimizing the E-TARC/branch bottleneck. The raw MCD anchor is always in
the family.

The finite-family theorem is exact: the raw anchor makes the feasible set
nonempty; selected membership gives at least 95% protected exact-score
retention; and maximization over a set containing raw cannot reduce the
three-target bottleneck within that family. The theorem is batch-local and
first-order. P0 separately measures the true finite-step losses.

## Search result

Six of eight batches selected a nonzero candidate; two used the safe raw
fallback. The nonzero rate is exactly the frozen 75% floor, so coverage passes
without surplus.

| Quantity | Result |
|---|---:|
| minimum protected exact retention | `0.953743` |
| minimum selected target score | `0.073802` |
| mean target bottleneck gain over raw | `0.010088` |
| bootstrap 95% interval | `[0.005235, 0.014915]` |
| positive / tied batches | `6 / 2` |
| exact sign-test p | `0.03125` |

Selected homotopy fractions were two raw fallbacks, two `0.125`, three
`0.25`, and one `0.5`. The varying fractions confirm that the method made a
batch-dependent choice rather than applying one global dose.

## Target effect

The same-batch target minimum rose from raw MCD's mean `0.078578` to
`0.088666`. BARP's worst observed target score was `0.073802`, above the
frozen `0.05` materiality floor.

Finite-step target relative gains versus raw were:

| Target | Mean | Bootstrap 95% interval |
|---|---:|---:|
| E-TARC | `+4.975%` | `[+2.394%, +7.586%]` |
| branch entropy | `+4.275%` | `[+2.170%, +6.354%]` |
| branch negative-gap | `+12.234%` | `[+6.262%, +18.227%]` |

The three-target macro gain was `+7.161%`, interval
`[+3.697%, +10.568%]`, with six positive and two tied batches and exact
`p=0.03125`. Every one of the seven registered losses decreased on all eight
batches after the transient finite step.

## Protected realization

Relative actual-reduction changes versus raw MCD were:

| Protected objective | Mean | Bootstrap lower | Worst batch |
|---|---:|---:|---:|
| risk | `-0.423%` | `-0.687%` | `-0.999%` |
| uniform | `-0.694%` | `-1.089%` | `-1.416%` |
| TARC | `-2.139%` | `-4.494%` | `-6.538%` |
| continuation | `-1.258%` | `-2.105%` | `-3.220%` |

All values pass the unchanged 5% mean, 7.5% bootstrap, and 15% single-batch
relative-loss limits. TARC remains the weakest protected coordinate, but its
worst batch now retains `93.462%` of raw actual reduction, with meaningful
room above the 85% floor. Mean actual-to-predicted ratios for all objectives
lie between `0.477` and `1.078`, inside the frozen `[0.25, 1.75]` interval.

## Integrity and limits

The run completed in `3778.97` seconds on the RTX 4070 Ti with `31.26%` VRAM
headroom. It performed 24 transient parameter applications, and the initial
and final trainable-state hashes were identical. It created no optimizer,
scheduler, checkpoint, or generated output and performed zero optimizer
steps.

This is credible positive mechanism evidence, not independent generalization.
Selection and evaluation intentionally use the same optimizer batch because
that is the proposed online algorithm. The eight units are disclosed
development data, and the two raw fallbacks show that nontrivial coverage is
only 75%. BARP also uses deterministic eval-mode gradients, while the prior
RP-PFRSC construction used train-mode gradients, so the comparison is not a
single-variable causal estimate of transfer removal.

The justified next step is a separately frozen matched-training experiment
that recomputes BARP-MCD on every training batch, records every fallback and
protected margin, and only then performs independent generated safety and
utility evaluation. Training remains unauthorized until that protocol is
committed and pushed.
