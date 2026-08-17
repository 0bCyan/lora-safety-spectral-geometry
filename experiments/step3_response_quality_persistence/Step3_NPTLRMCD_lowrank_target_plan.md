# Step3 NPT-LR-MCD: Low-Rank Protected-Nullspace Target Basis

## 1. Scope

The preceding NPT-A-MCD attempt produced a large target effect on eight
diagnostic units but failed the independent protection check. Its direction
was selected from 36 unit-target covectors, which is a high-dimensional
fit-specific target search. This new prospective attempt tests a lower-degree
of-freedom target basis. It does not alter either predecessor result.

The data are disclosed exploratory data, so this is a mechanism-development
experiment rather than independent confirmation.

## 2. Frozen data and boundary

The deterministic seed-7 12 fit / 20 calibration / 8 diagnostic split,
eval-mode one-pair gradients, model, and all protection/effect gates are
unchanged. Fit constructs one candidate, calibration only accepts it or falls
back to raw MCD, and diagnostics cannot alter it. No optimizer, scheduler,
checkpoint, generation, Core100, or heldout access is authorized; the run
remains at `optimizer_steps=0`.

## 3. Low-rank target basis

For each target certificate `q` in
`{etarc, branch_entropy, branch_negative_gap}`, normalize each of the 12 fit
unit gradients to unit Euclidean norm and average them coordinatewise:

`b_q = (1/12) sum_i g_iq / ||g_iq||_2`.

The target search therefore has exactly three globally pooled covectors, with
equal unit weighting. It does not use calibration or diagnostic gradients.

The 48 unit-level protected covectors and raw displacement define the same
49-row protected/raw nullspace `S` as the predecessor. Each of the three
pooled target covectors is projected into `S` using one shared float64 Gram
pseudoinverse, relative cutoff `1e-10`, and two fixed residual refinements.

## 4. Affine max-min and fixed-angle candidate

At the fixed registered angle
`theta* = acos(0.951) - 1e-6`, the three projected target covectors solve

`maximize_u min_q [r_q cos(theta*) +
                   sin(theta*) <P_S b_q,u>/R]`

over `||u||_2 <= ||d0||_2`, where `d0` is raw MCD and `R` is the fixed score
reference norm. The simplex dual is solved with the frozen SLSQP multistart
and explicit primal-dual/KKT audit. The candidate is

`d* = cos(theta*) d0 + sin(theta*) u*`.

The fixed angle and exact fit gates are retained: all fit target scores must
have floor at least `0.05` and gain at least `0.002`, while protected
retention must remain at least `0.95`.

The lower target basis dimension is the registered method change. No angle,
basis weighting, or threshold is selected after observing the run.

## 5. Protection and acceptance

Because the tangent is in `S`, exact arithmetic gives equal-norm rotation and
fit protected retention `cos(theta*)`. The implementation records finite
precision residuals and rejects invalid geometry. The 20-unit split-conformal
accept-or-raw policy remains the only deployment-facing protection check; its
finite-sample marginal lower bound is `20/21`, with no conditional-acceptance
coverage claim.

Eight untouched diagnostics measure exact scores and transient finite-step
losses for baseline, raw MCD, and the candidate. A pass would authorize only a
new frozen proxy-dev triplet protocol, never training.

## 6. Stop rule

The first failed registered gate stops this attempt. In particular, a strong
target gain cannot override calibration or protected noninferiority failure.
