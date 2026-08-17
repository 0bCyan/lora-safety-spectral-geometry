# Step3 NPT-A-MCD: Protected-Nullspace Affine Max-Min Target Tangent

## 1. Scope and predecessor

The seed-7 NPT-MCD P0 experiment validated the protected nullspace but
returned `STOP_NPTMCD_P0_NO_MINIMUM_EFFECTIVE_ANGLE`. Its projected pooled
target endpoint difference did not lift the lowest fit target certificate.
This protocol tests a different direction-selection rule in a new prospective
attempt. It does not reopen or alter the closed NPT-MCD result.

The prompt pool and risk rows have been disclosed by earlier exploratory work.
This remains a seed-7 development mechanism test, not an independent
confirmation result.

## 2. Frozen split and zero-step boundary

The same deterministic 12/20/8 fit, calibration, and diagnostic split is
used. The first 12 units select the candidate, the next 20 units only run the
registered accept-or-raw policy, and the final eight units are untouched until
the candidate is fixed. Every gradient is measured in eval mode from one
prompt/risk pair.

No model optimizer or scheduler is constructed. The only parameter changes
are transient virtual applications of the three fixed displacements on the
eight diagnostic units, followed by exact state restoration. The protocol
keeps `optimizer_steps=0`, `training_authorized=false`,
`candidate_generation_authorized=false`, `core100_opened=false`, and
`heldout_opened=false`.

## 3. Raw endpoint and protected subspace

Let `d0` be the raw full-certificate MCD displacement produced by the exact
clipped first-step Adam map used by the preceding P0 protocols. For each fit
unit `i` and protected certificate `p` in
`{risk, uniform, tarc, continuation}`, let `a_ip` be its unit-normalized
gradient covector. Define

`S = {v: <a_ip,v>=0 for every i,p and <d0,v>=0}`.

The 49-row constraint Gram matrix is factored once in float64 with relative
eigenvalue cutoff `1e-10`, then residual-refined twice. The implementation
records rank, condition number, every target projection residual, and the
maximum normalized constraint dot.

## 4. Fit-target affine max-min direction

There are 36 target covectors, one for each fit unit and each target
certificate in `{etarc, branch_entropy, branch_negative_gap}`. Let `b_j` be
the unit-normalized target covector, `P_S` the orthogonal projector onto `S`,
and `R` the fixed baseline reference norm used by the score panels. For the
registered angle `theta*`, define

`h_j = sin(theta*) P_S b_j / R`,

and let `r_j` be the raw score of target `j`. The tangent solves the convex
ball problem

`maximize_u  min_j [r_j cos(theta*) + <h_j,u>]`

subject to `||u||_2 <= ||d0||_2`. The candidate is then

`d* = cos(theta*) d0 + sin(theta*) u*`.

The affine max-min dual is

`min_{lambda in Delta_36} sum_j lambda_j r_j cos(theta*)
    + ||d0||_2 ||sum_j lambda_j h_j||_2`.

The implementation solves this 36-dimensional simplex problem with frozen
SLSQP multistarts (simplex center plus all vertices), `ftol=1e-13`, and
`maxiter=2000`. It accepts the solution only after checking simplex
feasibility, active/inactive KKT inequalities, primal-dual gap, and candidate
score agreement. The solver is an implementation of the convex dual, not a
post-result grid search.

## 5. Protection theorem

The fixed angle is

`theta* = acos(0.951) - 1e-6` radians.

Because `u*` lies in `S`, in exact arithmetic

`<a_ip,d*> / <a_ip,d0> = cos(theta*)`

for every fit protected facet, and
`||d*||_2 = ||d0||_2` when the ball solution uses the registered boundary.
The design cosine is strictly above the 95% protected floor; finite-precision
fit retention and identity error are still checked explicitly. A degenerate
projected target or a failed KKT audit invalidates the geometry.

The fit candidate must also have target floor at least `0.05` and improve the
fit target floor over raw by at least `0.002`. No threshold is changed after
observing the fit values.

## 6. Calibration and diagnostics

After the fixed fit candidate is constructed, 20 calibration units compute the
joint protected nonconformity. With `alpha=0.05`, the rank is 20 and the
complete candidate-or-raw policy has finite-sample marginal protection lower
bound `20/21`. No coverage conditional on candidate acceptance is claimed.

Eight diagnostic units compare baseline, raw MCD, and the accepted NPT-A-MCD
candidate. The existing exact-score, finite-step, protected noninferiority,
target advantage, and non-weak-effect gates remain unchanged.

## 7. Stop boundary

The first failed gate fixes the route. A failure does not authorize relaxing
the protected floor, target floor, gain floor, angle, split, solver controls,
or calibration rank. A pass authorizes only a separately frozen proxy-dev
triplet protocol; it never authorizes training, checkpointing, generation, or
optimizer steps.
