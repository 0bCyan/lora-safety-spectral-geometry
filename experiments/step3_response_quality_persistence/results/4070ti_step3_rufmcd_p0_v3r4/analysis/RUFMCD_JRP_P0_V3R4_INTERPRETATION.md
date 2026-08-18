# RUF-MCD-JRP P0 v3r4 Interpretation

## Result

The frozen run completed all 12 fit units, 20 calibration units, eight
diagnostic units, and 24 transient virtual parameter evaluations. Independent
`verify-run` recomputation passed and reproduced
`STOP_RUFMCD_JRP_P0_ROBUST_POLYTOPE_GEOMETRY_INVALID`.

This route is not evidence that the registered 96-facet problem is infeasible.
The geometry builder stopped before producing a full certificate with
`RUF-MCD constraint generation exceeded the frozen round limit`.

## Why 24 rounds are not complete

The implemented solver starts with one facet from each of seven groups. On
every non-closing round it adds at least one previously inactive facet. It may
add one facet from several groups, but additions are conditional on the
current dual solution and therefore need not happen in all groups on the same
round.

For `N` facets and `G` initial groups, there can be at most `N-G` non-closing
rounds before all facets are active. One final solve is then required to
certify the full gap. The implementation-level completeness bound is thus

`R_complete = N - G + 1`.

Here `N=96` and `G=7`, so `R_complete=90`. The earlier value 24 counted only
the largest group and implicitly assumed simultaneous per-group growth.

A deterministic four-facet counterexample confirms the distinction. With two
groups of two facets, the largest group has size two, but the frozen solver
requires three rounds to close. This equals `4-2+1`, not two.

## Fallback boundary

Because no full geometry certificate was available, the selected angle was
zero and the selected displacement was exactly raw MCD. Consequently:

- all calibration retentions of 1.0 describe raw fallback;
- calibration target gain over raw was exactly zero;
- all eight diagnostic candidate-minus-raw differences were exactly zero;
- the passed common-transfer, finite-descent, and protected-noninferiority
  gates apply to raw MCD and do not establish a JRP effect.

The failed target-advantage and effect-size gates are also fallback
consequences, not an evaluation of a certified nonzero JRP candidate.

## Integrity and next step

The initial and final trainable-state hashes matched. No optimizer or
scheduler was created, `optimizer_steps=0`, no checkpoint or generated
candidate was written, and Core100 and heldout data were not opened.

The only justified recovery is to raise the certification work bound from 24
to the proven global bound 90 while leaving all 96 facets, response covectors,
reflection vertices, objective offsets, fit/calibration/diagnostic rows,
selection rules, and `1e-7` certificate thresholds unchanged.

These rows have been repeatedly opened during recovery. Any subsequent result
remains exploratory mechanism evidence and cannot be called fresh
confirmation or population coverage.
