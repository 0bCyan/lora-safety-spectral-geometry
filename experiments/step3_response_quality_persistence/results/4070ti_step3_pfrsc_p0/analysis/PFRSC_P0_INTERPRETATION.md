# PFRSC-MCD Seed-3 Zero-Step P0 Interpretation

## Decision

The prospectively registered route is
`STOP_PFRSC_P0_TARGET_ADVANTAGE_FAILED`. Integrity, protected finite-radius
search, diagnostic common transfer, finite-step common descent, protected
diagnostic noninferiority, and materiality all passed. The route/branch target
advantage conjunction failed, so PFRSC-MCD does not authorize a matched
training protocol or any optimizer step.

This is a sharply localized failure. The raw-anchored homotopy repaired the
seed-2 FRSC sacrifice of risk, uniform, and continuation while delivering a
large positive finite-loss macro effect on E-TARC and the two branch
objectives. It did not preserve TARC consistently on the disjoint diagnostic
set. The strong target macro average cannot override the frozen target
bottleneck and per-objective floors.

Independent `verify-run` reproduced the route and all six hashed artifacts.
No optimizer or scheduler was created, and `optimizer_steps=0`.

## Experimental integrity

The run used seed 3 and eight construction optimizer units at the unchanged
Qwen3-4B LoRA initialization. Each unit evaluated all 127 subset endpoints,
screened exactly 16 endpoint directions, and searched 129 protected homotopy
candidates using 239 exact optimizer-map evaluations. The fixed eight-category
diagnostic set was excluded from construction and selection.

All 24 arm-by-unit finite-loss evaluations were transient direct parameter
applications followed by exact restoration. Initial and final trainable-state
hashes were identical:
`ddf8545bd8f1f220c9bec2dfc15cfba2abc3a3d63765fd589c6e631bcf592ff2`.
Peak reserved memory was `9,351,200,768` bytes, leaving `27.39%` headroom on
the RTX 4070 Ti. Wall time was `3,592.72` seconds. No raw gradient tensor,
checkpoint, candidate response, Core100 row, or heldout row was written or
opened.

## Registered gates

| Gate | Result |
|---|---:|
| integrity | pass |
| protected finite-radius search | pass |
| diagnostic common transfer | pass |
| finite-step common descent | pass |
| protected diagnostic noninferiority | pass |
| target diagnostic advantage | fail |
| effect not weak | pass |

The failed target conjunction determines the stop route. Later inspection of
its components does not change the registered decision.

## The finite-family certificate worked on construction units

PFRSC retained only candidates satisfying

\[
s_i(d)\ge 0.95s_i(d_{\mathrm{raw}}),
\qquad i\in\{R,U,C\},
\]

then maximized the exact route/branch bottleneck over the finite feasible
family. Because raw MCD was an explicit candidate, the protected family was
nonempty and the selected construction target score could not be below the
raw target score.

The observed minimum protected retention was `0.950123`. Every block selected
a nonzero homotopy fraction:

| Fraction | Blocks |
|---:|---:|
| `0.25` | 1 |
| `0.375` | 3 |
| `0.50` | 1 |
| `0.625` | 3 |

The construction target-bottleneck gain was positive on all eight blocks,
with mean `0.020474`, 95% bootstrap interval `[0.011638, 0.029958]`, and exact
two-sided sign-test `p=0.0078125`. The minimum selected construction target
score was `0.083846`, above the frozen `0.05` floor. The raw-anchor theorem was
therefore active rather than vacuous: no block fell back to lambda zero.

## Protected finite-step noninferiority transferred

On the prompt-disjoint diagnostic set, all three protected objectives stayed
well inside the frozen relative-loss limits. Relative actual reduction versus
raw MCD was:

| Protected objective | Mean relative change | 95% interval | Worst unit |
|---|---:|---:|---:|
| risk | `-1.291%` | `[-1.649%, -0.955%]` | `-2.153%` |
| uniform | `-1.412%` | `[-1.688%, -1.135%]` | `-2.055%` |
| continuation | `-2.438%` | `[-2.713%, -2.176%]` | `-3.131%` |

The corresponding exact-score changes were `-1.431%`, `-1.617%`, and
`-2.951%`. Thus both the first-order and finite-loss protected gates passed:
mean retention exceeded 95%, every bootstrap lower retention bound exceeded
92.5%, and every individual unit retained more than 85%.

All seven PFRSC objectives had positive actual loss reduction on all eight
units. Mean actual-to-predicted ratios remained inside the frozen
`[0.25, 1.75]` interval. PFRSC's minimum diagnostic exact score was `0.027431`,
and its worst-score advantage over no auxiliary was positive on all eight
units: mean `0.070176`, 95% interval `[0.056100, 0.082827]`, and
`p=0.0078125`.

## Three target effects were strong and consistent

Relative finite-loss reduction versus raw MCD was positive on every unit for
E-TARC and both branch objectives:

| Target objective | Mean relative gain | 95% interval | Positive units |
|---|---:|---:|---:|
| E-TARC | `+7.727%` | `[+4.886%, +10.505%]` | `8/8` |
| branch entropy | `+9.381%` | `[+4.724%, +14.808%]` | `8/8` |
| branch negative-gap | `+24.176%` | `[+12.583%, +36.786%]` | `8/8` |

The per-unit macro average over all four registered target objectives,
including TARC, improved on all eight units. Its mean was `+7.641%`, with 95%
interval `[+5.522%, +9.953%]` and sign-test `p=0.0078125`. This clears the
frozen 2% materiality threshold by a wide margin.

These are genuine positive mechanism results, but they are not sufficient for
the registered pass because the protocol explicitly prohibited a macro gain
from hiding collapse of one target objective.

## TARC caused the target conjunction to fail

TARC actual reduction changed by `-10.720%` on average relative to raw MCD.
Only four of eight units improved, its 95% interval was
`[-22.955%, +0.755%]`, and the sign test was `p=1`. This failed the frozen
requirement that every target retain at least 95% on average.

The exact diagnostic TARC score showed the same transfer problem: mean
relative change `-7.893%`, only `4/8` positive, and a 95% interval spanning
`[-20.158%, +3.920%]`. Consequently, the four-way diagnostic target minimum
was worse than raw MCD on four units. Its paired absolute difference had mean
`-0.003620`, interval `[-0.010637, +0.002837]`, and sign-test `p=1`, failing
the required positive `0.002` advantage.

## Mechanism interpretation

Moving risk, uniform, and continuation from a shared max-min objective to
coordinate-wise retention constraints solved the exact failure observed in
seed 2. The seed-3 protected objectives gave up only 1.3% to 2.4% actual
reduction while the three complete-path/branch objectives gained 7.7% to
24.2%.

TARC remained in the optimized target set rather than the protected set. The
selection objective was the minimum target score, not coordinate-wise target
dominance. Once another target limited a construction block, TARC could fall
without violating feasibility. More importantly, a positive construction
target minimum did not transfer reliably to the fixed diagnostic prompts:
four blocks improved TARC and four degraded it. This is the same logical gap
the protocol was designed to detect.

The result therefore supports a bounded Pareto mechanism but rejects the
specific four-target bottleneck formulation as a training candidate. It would
be incorrect to relabel TARC as secondary after seeing this run or to remove
its frozen floor from the seed-3 decision.

## Prospectively admissible successor

A new protocol may test route-protected PFRSC with

\[
\mathcal P'=\{R,U,C,T\},
\qquad
\mathcal Q'=\{E,H,G\}.
\]

The same raw anchor, exact optimizer map, finite subset screen, and homotopy
grid can then impose coordinate-wise TARC retention while optimizing only the
complete-path and branch bottleneck. Raw MCD again guarantees feasibility,
and the finite-family theorem extends by replacing `P,Q` with `P',Q'`.

This successor is motivated by a specific failed coordinate rather than an
outcome-dependent threshold relaxation. It still requires a new stochastic
seed, a newly frozen split, unchanged noninferiority standards, and a protocol
commit pushed before model forward. The present run cannot be reused to claim
its success, and no training is authorized.

## Evidence boundary

This is exploratory mechanism evidence on one model and reused disclosed
development data. It supports finite-radius protected tradeoff control and
three consistent target-gradient effects, but not generated safety, utility,
or downstream response-quality improvement. `optimizer_objects=0`,
`scheduler_objects=0`, `optimizer_steps=0`, `checkpoint_writes=0`,
`training_authorized=false`, `candidate_outputs_generated=false`,
`core100_opened=false`, and `heldout_opened=false`.
