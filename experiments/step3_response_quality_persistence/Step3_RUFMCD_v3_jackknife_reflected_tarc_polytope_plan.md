# Step3 RUF-MCD v3: Jackknife-Reflected TARC Polytope

## Status and question

This is a new zero-optimizer exploratory protocol after the verified v2 stop
`STOP_RUFMCD_P0_CONFORMAL_SELECTIVE_POLICY_FAILED`. v2's analytic 84-facet
certificate was valid, but its fixed tangent failed unseen TARC protection. A
post-hoc angle audit showed that shrinking the angle cannot repair the same
tangent. v3 asks a narrower question:

> Can a fit-only, finite uncertainty set for normalized TARC response force the
> target tangent away from directions that are unsafe under leave-one-out
> transfer, while preserving the original target and protection gates?

Calibration and diagnostic rows remain untouched until the candidate is fixed.
No training, generation, Core100, or heldout access is authorized.

## Finite uncertainty set

For fit unit `i`, let `b_i` be its normalized TARC gradient and `r_i>0` its
raw TARC score. Define the normalized retention-response covector

`c_i = b_i / r_i`.

For each `i`, form the leave-one-out mean

`mu_{-i} = (1/(n-1)) sum_{j != i} c_j`

and its reflected vertex

`c_i^- = 2 mu_{-i} - c_i`.

The registered finite polytope is

`U_JRP = conv({c_i} union {c_i^-})`.

The observed vertices `{c_i}` are already among the original RUF protected
facets. v3 adds only the 12 reflected vertices, yielding 96 affine facets
(84 original plus 12 extra).

This construction has no fitted multiplier, covariance shrinkage, angle scan,
or calibration-dependent parameter. It is a deterministic jackknife stress
set, not a claim of population coverage.

## Candidate objective and certificate

At the fixed `theta = acos(0.951)-1e-6`, the protected TARC relative margin for
any `c in U_JRP` is

`m_T(c,u) = cos(theta)-0.95 + sin(theta)<c,u>/R`.

The other 48 protected fit margins and 36 target fit margins remain exactly
the v2 affine margins. v3 maximizes the minimum of all 96 margins over the
same tangent ball `||u|| <= ||d0||`, after projecting all covectors off the
raw displacement.

Because every margin is affine in `u`, and an affine functional reaches its
minimum over a finite convex hull at a vertex,

`min_{c in U_JRP} m_T(c,u) = min_{v in vertices(U_JRP)} m_T(v,u)`.

Thus the existing simplex minimax dual and analytic active-set KKT refinement
remain exact for the registered finite problem. A full primal-dual gap at most
`1e-7` certifies the returned tangent for all 96 registered facets. This proves
finite-set robustness only; it does not prove that an arbitrary future TARC
gradient lies in `U_JRP`.

## Gates and boundaries

The v2 95% fit protection, target floor/gain, conformal rank-20 policy, finite
step, diagnostic, and effect gates are unchanged. Additional geometry gates
require all 12 reflected facets, the 96-facet full gap, KKT residual, and
actual-versus-predicted margin consistency. The candidate is accepted by the
same candidate-before-calibration rule; calibration scores are never used to
select or tune the candidate.

If the finite polytope is too conservative or the calibration policy fails,
the route stops. A pass authorizes only a separately frozen development
protocol, never training. In particular, a pass does not justify a population
coverage claim without a new independent validation study.

## Mechanism evidence required

The run must report the fit polytope certificate, the per-vertex margins, the
calibration TARC retention distribution, the diagnostic protected and target
effects, and exact state restoration. The v2 calibration rows are not fresh
confirmation for v3; this attempt is exploratory evidence about whether the
finite uncertainty construction changes the failure mechanism.

Throughout the run:

`optimizer_steps=0`, `training_authorized=false`,
`candidate_generation_authorized=false`, `core100_opened=false`, and
`heldout_opened=false`.
