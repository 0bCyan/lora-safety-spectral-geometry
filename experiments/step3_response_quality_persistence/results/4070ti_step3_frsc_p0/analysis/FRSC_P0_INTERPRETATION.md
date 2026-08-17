# FRSC-MCD Seed-2 Zero-Step P0 Interpretation

## Decision

The prospectively registered route is
`STOP_FRSC_P0_RAW_COMPARATOR_FAILED`. FRSC-MCD passes integrity, finite-radius
search, diagnostic common transfer, finite-step realization, and materiality,
but it fails the frozen raw-MCD comparator gate. It does not authorize a
matched training protocol or any optimizer step.

This is not a null experiment. It establishes that exact finite-radius subset
selection can produce transferable simultaneous descent under the registered
clipping-plus-first-step-Adam map. It also falsifies the stronger claim that
unconstrained maximization of the seven-way worst score improves the full-set
raw MCD comparator without sacrificing protected objectives.

Independent `verify-run` reproduced the route and all six hashed run
artifacts. No optimizer or scheduler was created, all transient parameter
changes were exactly restored, and `optimizer_steps=0`.

## Experimental integrity

The run used seed 2 and eight direction-construction units at the unchanged
Qwen3-4B LoRA initialization. Every unit contained four risk and four paired
safety microbatches and exhausted all `2^7-1=127` nonempty certificate
subsets. A fixed diagnostic set contained eight disjoint risk rows and one
safety prompt from each of eight source categories. No diagnostic gradient
entered subset selection.

All 24 arm-by-unit finite-loss evaluations were transient direct parameter
applications followed by exact restoration. Initial and final trainable-state
hashes were identical. Peak reserved memory was `9,628,024,832` bytes,
leaving `25.24%` headroom on the RTX 4070 Ti. Runtime was `1,475.55` seconds.
No gradient tensor, checkpoint, candidate response, Core100 row, or heldout
row was written or opened.

## Registered gates

| Gate | Result |
|---|---:|
| integrity | pass |
| finite-radius search | pass |
| diagnostic common transfer | pass |
| finite-step common descent | pass |
| effect not weak | pass |
| raw comparator advantage | fail |

The stop route is determined by the failed comparator gate. Passing the five
other gates does not override it.

## What the finite-family theorem established

For each construction unit, FRSC evaluated the exact realized displacement

\[
\Phi(G_0+0.2d_S)
\]

for every feasible Euclidean MCD direction induced by a nonempty subset
`S` of the seven certificates. It selected the member maximizing

\[
m(d_S)=\min_i
\frac{u_i^T\Phi(G_0+0.2d_S)}{\lVert\Phi(G_0)\rVert_2}.
\]

Because the full seven-certificate subset is in the finite candidate family,
the selected construction score satisfies

\[
m(d_{\mathrm{FRSC}})\ge m(d_{\mathrm{raw}})
\]

on every construction unit by finite enumeration. This is an exact statement
about the registered candidate family and optimizer map, not an asymptotic or
linearized claim.

The measured minimum selected construction margin was `0.066073`, compared
with `0.056327` for full-seven raw MCD. The paired gain was positive on all
eight units, with mean `0.019180`, 95% bootstrap interval
`[0.013028, 0.026894]`, and exact two-sided sign-test `p=0.0078125`.
The finite-family mechanism therefore worked on the data used for direction
construction.

## Disjoint common-descent transfer was strong

On the fixed diagnostic split, FRSC's minimum exact certificate score was
`0.049296` and its mean worst score was `0.063283`. Relative to the
no-auxiliary baseline, the paired worst-score advantage was positive on all
eight units: mean `0.067299`, 95% interval `[0.062914, 0.071490]`, and
sign-test `p=0.0078125`.

The exact first-step displacement also produced a positive actual loss
reduction on all eight units for every objective:

| Objective | Mean actual reduction | Positive units |
|---|---:|---:|
| risk | `0.524889` | `8/8` |
| uniform | `1.349411` | `8/8` |
| TARC | `0.551415` | `8/8` |
| E-TARC | `2.518898` | `8/8` |
| continuation | `1.041996` | `8/8` |
| branch entropy | `0.002903` | `8/8` |
| branch negative-gap | `1.706909` | `8/8` |

Every objective's mean actual-to-predicted reduction ratio lay inside the
frozen interval `[0.25, 1.75]`. Thus the positive exact-map scores were not
merely infinitesimal geometry; they transferred to the registered finite
parameter step on disjoint diagnostics.

## Why the comparator gate failed

FRSC improved the diagnostic worst score over raw MCD by `0.005300` on
average, with bootstrap interval `[0.002174, 0.007752]`. However, only seven
of eight paired units improved, so the exact two-sided sign test was
`p=0.0703125`, above the frozen `0.05` threshold.

More importantly, the per-objective finite-loss comparison shows a systematic
trade rather than sampling noise:

| Objective | FRSC minus raw MCD actual reduction | Frozen noninferiority |
|---|---:|---:|
| risk | `-0.014905` | fail |
| uniform | `-0.037683` | fail |
| TARC | `-0.002090` | fail |
| E-TARC | `+0.184681` | pass |
| continuation | `-0.056034` | fail |
| branch entropy | `+0.000201` | pass |
| branch negative-gap | `+0.274902` | pass |

Only `3/7` objectives were noninferior within the frozen `0.001` tolerance;
the protocol required at least `6/7`. Risk, uniform, and continuation were
worse on all eight paired units, each with sign-test `p=0.0078125` against
FRSC. This is decisive evidence against calling FRSC a raw-MCD improvement.

## Mechanism of failure

The selected subsets were:

- `etarc+branch_entropy` on two units;
- `tarc+branch_entropy+branch_negative_gap` on three units;
- `tarc+branch_negative_gap` on three units.

Risk, uniform, and continuation were absent from every selected subset. The
max-min objective focused on whichever route or branch certificate limited
the exact score, while already-large protected scores were free to decrease.
That is precisely what happened on diagnostics: E-TARC and both branch
objectives improved, while risk, uniform, and continuation gave up finite-loss
reduction relative to raw MCD.

This is not a failure of the finite-family dominance theorem. The theorem
guarantees only the minimum of seven normalized construction scores. It does
not provide coordinate-wise dominance, diagnostic generalization, or
finite-loss noninferiority for every objective. The experiment exposes the
gap between scalar max-min dominance and Pareto dominance.

## Prospectively admissible successor

A successor may test protected, noninferiority-constrained FRSC on a new seed.
Let `P={risk, uniform, continuation}` be fixed before model forward and let
`Q={TARC, E-TARC, branch entropy, branch negative-gap}`. For the same finite
candidate family, retain only directions satisfying the exact construction
constraints

\[
s_i(d_S)\ge s_i(d_{\mathrm{raw}})-\tau_i,
\qquad i\in P,
\]

with prospectively fixed tolerances, then maximize

\[
\min_{i\in Q}s_i(d_S)
\]

over the retained set. The full-seven raw direction is always a feasible
fallback when nonnegative tolerances are used. Finite enumeration would then
certify protected-score noninferiority and route/branch max-min dominance
within the registered construction family.

That theorem would still not guarantee disjoint finite-loss superiority.
A new protocol must freeze the protected set, tolerances, seed-3 split,
comparators, multiplicity-aware gates, and stop route before any model
forward. This P0 cannot be reused to tune those quantities, and its failure
does not authorize training.

## Evidence boundary

This is exploratory mechanism evidence from one model, one LoRA
initialization seed, and reused disclosed development data. It establishes
neither generated-response safety nor utility, and it does not yet support a
paper claim of downstream improvement. `optimizer_objects=0`,
`scheduler_objects=0`, `optimizer_steps=0`, `checkpoint_writes=0`,
`training_authorized=false`, `candidate_outputs_generated=false`,
`core100_opened=false`, and `heldout_opened=false`.
