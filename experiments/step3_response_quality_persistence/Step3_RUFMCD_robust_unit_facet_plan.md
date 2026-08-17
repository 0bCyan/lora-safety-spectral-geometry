# Step3 RUF-MCD: Robust Unit-Facet Margin MCD

## 1. Motivation and scope

RMA-MCD solved a valid seven-covector pooled margin problem and produced a
strong fit target gain, but its pooled positive margin did not transfer to the
unit-level `fit_11:tarc` facet. The proposed retention there was `0.6704`.
The deterministic angle audit found no angle satisfying both the unchanged
95% protected-retention gate and the target floor.

RUF-MCD is the minimal prospective repair of that identified failure. It keeps
the same model, seed-7 split, raw MCD anchor, fixed angular cap, fit/calibration/
diagnostic roles, target floor, protection floor, and conformal policy. It
changes only the candidate optimization: all 48 protected fit facets and all
36 target fit facets enter one robust max-min problem. No optimizer, scheduler,
training, checkpoint, generation, Core100, or heldout access is authorized;
`optimizer_steps=0` throughout.

## 2. Unit-facet geometry

Let `H` be the finite-dimensional LoRA parameter space, `d0` the raw MCD
displacement, `B = ||d0||_2`, and `R` the unchanged baseline reference norm.
For fit unit `i` and certificate `q`, define the normalized covector

`b_iq = g_iq / ||g_iq||_2`

and raw score

`r_iq = <b_iq,d0>/R`.

Let `P = I - d0_hat tensor d0_hat` be the orthogonal projector off the raw
displacement. Set `theta* = acos(0.951) - 1e-6`. For any tangent `u` with
`||u||_2 <= B`, the candidate is

`d(u) = cos(theta*) d0 + sin(theta*) u`,

where every objective covector is replaced by `P b_iq`; hence the optimizer
acts only in `d0`'s orthogonal complement. When the dual weighted covector is
nonzero, the optimizer returns `||u||_2 = B`, and `d(u)` has the same norm as
`d0` up to the registered numerical tolerance.

## 3. Eighty-four affine margins

For every protected facet `(i,p)`, define the relative-retention margin

`m_ip(u) = cos(theta*) - 0.95
           + sin(theta*) <P b_ip,u> / (R r_ip)`.

All protected raw scores must be positive before this ratio is defined. For
every target facet `(i,q)`, define

`F = max(0.05, min_iq r_iq + 0.002)`

and the absolute target margin

`m_iq(u) = r_iq cos(theta*)
           + sin(theta*) <P b_iq,u> / R - F`.

The registry contains 48 protected and 36 target margins. RUF-MCD solves

`maximize_u min_j m_j(u)  subject to ||u||_2 <= B`.

No pooled average, post-hoc angle, calibration score, or diagnostic score
enters this optimization.

## 4. Dual certificate and proof

Write every margin as `m_j(u) = a_j + <h_j,u>`. Since

`min_j m_j(u) = min_{lambda in Delta_84} sum_j lambda_j m_j(u)`,

compact-convex minimax interchange gives

`max_{||u||<=B} min_j m_j(u)
 = min_{lambda in Delta_84}
   [sum_j lambda_j a_j + B ||sum_j lambda_j h_j||_2]`.

This is a convex optimization over the 84-simplex. For an optimal dual point
whose weighted covector is nonzero, a primal maximizer is

`u* = B (sum_j lambda_j h_j) / ||sum_j lambda_j h_j||_2`.

Therefore the dual value is an upper bound on every candidate's worst fit
margin, with equality under the finite-dimensional compact-convex conditions
above. If the audited optimum is negative, no tangent in the registered ball
can satisfy all 84 margins at the fixed angle. A nonnegative optimum is only a
fit feasibility result; it does not replace calibration or diagnostic gates.

The frozen implementation uses constraint generation to avoid repeatedly
forming the roughly 7,140-entry full Gram matrix. It starts with the
lowest-offset facet from each of the seven certificate groups. At each round
it solves the active affine problem with SLSQP, evaluates all 84 facets, and
adds the lowest-scoring inactive facet from each group when it is below the
active dual upper bound. There are 12 facets per group, so the registered
maximum of 12 rounds is sufficient to reach the full set in the worst case.

The active dual weights are padded with zeroes to the 84-simplex. They are
therefore feasible for the full dual, and their objective is an upper bound on
the full optimum. If `D_A` is this active dual value and `L_J(u_A)` is the
recomputed minimum over all 84 facets at the active primal candidate, then

`0 <= D_A - V_J <= D_A - L_J(u_A)`.

Consequently, a full-facet gap at most `1e-7` certifies that the returned
candidate is within `1e-7` of the global full-facet optimum, even though the
solver did not form the full Gram matrix. The audit stores every full facet
score, the active-round trace, the padded-dual upper bound, and the closure
flag. Both active KKT residual and active primal-dual gap must be at most
`1e-7`; the full gap must also be at most `1e-7`; and the recomputed all-facet
minimum must agree with the stored primal value within `1e-7`.

## 5. Independent protection policy

The candidate is fixed using only the 12 fit units. It is eligible only if all
48 protected retentions are at least `0.95`, all 36 target scores meet `F`,
the target-floor gain is at least `0.002`, and every numerical audit passes.

The unchanged 20-unit split-conformal candidate-or-raw policy is then applied.
At alpha `0.05`, the order statistic has rank 20 and marginal lower bound
`20/21`. The candidate is deployed only when the frozen threshold is
nonpositive; otherwise the policy returns raw MCD. No guarantee conditional on
candidate acceptance is claimed. Eight untouched diagnostics retain the
existing exact-score, finite-step, noninferiority, target-gain, sign-test, and
bootstrap gates.

## 6. Stop rule and claim boundary

Any integrity, projection, KKT, fit facet, conformal, protected diagnostic,
finite-step, target advantage, or effect-strength failure stops the attempt.
Thresholds may not be relaxed after seeing the result. A pass authorizes only
a separately frozen proxy-development protocol, never training. A stop is
reported as a fit infeasibility or transfer result, not as proof about all
possible safety-preserving update methods.
