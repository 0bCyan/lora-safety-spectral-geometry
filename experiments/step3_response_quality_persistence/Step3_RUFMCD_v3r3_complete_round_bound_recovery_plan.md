# Step3 RUF-MCD v3r3: Complete Constraint-Generation Bound

## Registered predecessor

v3r2 reached the real 96-facet solver and stopped with

`RUF-MCD constraint generation exceeded the frozen round limit`.

The 12-round value was inherited from v2. It did not account for the 12 new
reflected TARC facets and therefore no longer supplied the stated worst-case
constraint-generation completeness bound.

## Combinatorial bound

The solver partitions the objective by certificate and initializes one facet
per group. At the end of each non-closing round it adds at most one currently
worst violating inactive facet from each group. The v3 group sizes are

| group | facets |
| --- | ---: |
| risk | 12 |
| uniform | 12 |
| TARC | 24 |
| ETARC | 12 |
| continuation | 12 |
| branch entropy | 12 |
| branch negative gap | 12 |

Hence 24 rounds are sufficient to present every facet in every group. If the
problem has not closed earlier, round 24 operates with every still-required
TARC facet available; all smaller groups can already be complete.

## Frozen change

v3r3 changes only

`constraint_generation_max_rounds: 12 -> 24`.

The config and protocol must contain the same value, and the runner validates
it against the structural bound `2 * fit_units = 24`. The 96 covectors,
jackknife reflection, fixed angle, candidate map, `1e-7` active KKT and full
gap thresholds, protected floor, target floor/gain, conformal policy, and
diagnostic gates are unchanged.

The change expands certification work only. It does not enlarge the tangent
ball, remove a facet, alter an offset, relax a margin, or accept an
uncertified point.

## Claim boundary

The fit, calibration, and diagnostic rows have already been opened by the
recovery sequence. v3r3 is exploratory execution recovery, not fresh
confirmation. A valid finite-set certificate can be interpreted only for the
registered 96-facet polytope. Population coverage remains unclaimed.

No training, optimizer step, checkpoint, generation, Core100 access, or
heldout access is authorized.
