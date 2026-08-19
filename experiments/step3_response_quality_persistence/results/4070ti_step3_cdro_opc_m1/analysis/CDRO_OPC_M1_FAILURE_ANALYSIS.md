# CDR-OPC M1 Failure Analysis

## Decision

M1 validly stopped at `STOP_CDRO_OPC_M1_TRANSPORT`. The run passed structural,
orthonormal-basis, convex-solver, nontrivial-update, behavior, state-restoration,
and zero-step gates. It failed the only gate that could authorize the candidate:
the registered split-conformal protected transport policy. Training remains
unauthorized.

The numerical optimization itself is not the failure. The five-dimensional
basis had maximum orthonormality error `5.70e-8`; fallback reconstruction error
was `8.27e-8`. SLSQP terminated successfully, and the independent convex KKT
audit reported primal violation `1.13e-13`, normalized stationarity `3.33e-7`,
and complementarity `2.06e-13`. The active set was safety Pareto, risk, and the
trust region. The selected update was substantial: its fallback-relative norm
was `0.66895`.

## What failed

The full CDR-OPC candidate had rank-20 nonconformity `0.77663`, far above zero,
so the complete policy selected fallback. FOACPO-S also failed on the same
stronger panel, with threshold `0.05077`. Thus the old miss was not merely a
single unlucky TARC rounding event.

Risk caused the new failure. Its mean and median standardized changes were
improvements (`-0.02821` and `-0.01224`), but ten of twenty units exceeded the
registered 4% deterioration allowance and the worst deteriorated by `81.66%`.
This is a heavy-tail transport failure hidden by a favorable mean. Uniform,
TARC, and continuation failed only 1, 2, and 3 units respectively. The fit
ellipsoid marked risk active and feasible, yet badly underestimated this
unit-level tail; six two-pair fit blocks are insufficient for that role.

## What survived

The behavior direction was strong at full dose. Calibration quality/safety
aggregate cosines were `0.09581/0.10650`, with `5/5` and `9/9` positive pairs.
Mechanism-dev aggregates were `0.12227/0.12125`, with `6/6` and `9/9` positive.
This supports a real Pareto-aligned optimizer-space direction, not a null update.
It does not support applying that direction at the selected radius.

Because protected scores are affine along the line from fallback to the fixed
candidate, the exact largest dose satisfying all opened M1 units is
`0.0489819`, limited by risk. At that descriptive dose, all four behavior
aggregates remain positive (`0.00633`, `0.02444`, `0.01969`, `0.01347`) and pair
sign counts remain `3/5`, `7/9`, `6/6`, and `8/9`. This calculation was made
after opening M1 and is not a registered M1 policy; it cannot convert the stop
into a pass.

## Method consequence

The useful next development is not a larger ellipsoid chosen after the fact.
It is a conformal dose controller: derive one direction on fit data, compute a
lower conformal bound on its per-unit safe dose, and use fallback when the bound
is zero. The order-statistic proof can cover the calibrated dose itself rather
than only an all-or-nothing endpoint. A credible empirical claim then requires
new protected units because the twenty M1 units are now development data.

M1 used `optimizer_steps=0`; `training_authorized=false`. Core100 and heldout
remained closed.
