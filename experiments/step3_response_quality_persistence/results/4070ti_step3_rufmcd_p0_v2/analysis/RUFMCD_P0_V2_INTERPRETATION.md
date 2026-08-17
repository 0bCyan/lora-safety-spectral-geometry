# RUF-MCD v2 Seed-7 Interpretation

## Decision

The verified zero-step run returned:

`STOP_RUFMCD_P0_CONFORMAL_SELECTIVE_POLICY_FAILED`

The analytic active-set refinement fixed the predecessor's numerical
certificate problem. It did not turn the candidate into a deployable policy.
The unchanged split-conformal candidate-or-raw policy observed a positive
joint nonconformity threshold, so the registered action was `raw_mcd`.

No training, candidate generation, Core100 access, or heldout access was
authorized by this result.

## What v2 established

The v2 construction used the same 84 fit facets, fixed angular cap, protected
floor, target floor, target-gain floor, calibration policy, and diagnostic
gates as the prospective protocol. Only the simplex numerical certificate was
changed: SLSQP proposed a support, then the active-set equations were solved
analytically and every facet was recomputed.

The resulting geometry was numerically valid:

| quantity | value |
| --- | ---: |
| fit facets | 84 (48 protected, 36 target) |
| constraint-generation rounds | 7 |
| final dual support | 23 |
| active KKT residual | `3.2474e-15` |
| active primal-dual gap | `8.0491e-16` |
| full 84-facet primal-dual gap | `2.6348e-8` |
| projected target norm fraction | `0.9567128061` |
| fit protected retention floor | `1.0794228587` |
| fit target floor | `0.1794228701` |
| fit target gain over raw | `0.1671556881` |

Thus the v1 stop was a numerical solver limitation, not a hidden failure of
the 84-facet convex certificate. The certificate is now independently
auditable, but certificate validity is only a fit-geometry condition.

## Why the policy still stopped

The candidate was fixed before the 20 calibration units. The conformal rank
was 20 at `alpha=0.05`, and the observed threshold was
`1.3907735763`, which is positive. The candidate therefore failed the frozen
accept-or-raw rule and the policy returned the raw MCD anchor. The worst
calibration protection was the `tarc` certificate of `calibration_12`, with
retention `-0.4407735763`.

This is not a conditional-acceptance coverage claim. The registered guarantee
is only the complete candidate-or-fallback policy under the stated
exchangeability assumption.

## Angle frontier: dose is not the explanation

The stored fit panels and the fixed candidate define one tangent direction.
For any angle in the registered interval, each affine score has the exact
form

`s_j(theta) = r_j cos(theta) + t_j sin(theta)`,

where `r_j` is the raw score and `t_j` is the tangent score. For calibration,
the tangent score is reconstructed from the already stored fixed-angle pair:

`t_j = (s_j(theta*) - r_j cos(theta*)) / sin(theta*)`.

The reconstruction error was `2.78e-17`. The audit then solves every gate
boundary analytically and classifies the resulting intervals; it does not
choose a new angle for the experiment.

| frontier quantity | value |
| --- | ---: |
| registered interval | `[0, 0.3143421045]` rad |
| calibration-safe prefix upper bound | `0.0110943203` rad (`0.636°`) |
| fit target/protection feasible lower bound | `0.0696559141` rad (`3.991°`) |
| angular incompatibility gap | `0.0585615938` rad |
| joint fit + calibration feasible angle | none |

At the calibration-safe endpoint, fit target floor is only `0.0182857347`,
well below `0.05`. At the first fit-feasible angle, the calibration minimum
protected retention is only `0.6842824430`, again due to a TARC facet. The
fixed v2 angle is therefore not merely too aggressive: the same tangent
direction cannot satisfy both the fit target floor and the observed TARC
protection frontier.

The complete machine-readable audit is
`rufmcd_p0_v2_angle_sensitivity.json`. It is a post-hoc mechanism audit; its
calibration rows were already opened by v2 and must not be reused as an
untouched validation set for a successor.

## Mathematical boundary

For a protected facet with positive raw score `r_j`, the retention constraint
at floor `rho` is

`cos(theta) + (t_j/r_j) sin(theta) >= rho`.

For a target facet with required floor `F`, the target constraint is

`r_j cos(theta) + t_j sin(theta) >= F`.

Each equality is a two-term trigonometric equation, so its roots partition the
registered interval into finitely many regions on which every gate is
constant. The audit checks all such regions. Therefore the empty joint
interval is an exact conclusion for this fixed tangent and the stored linear
score model, rather than a coarse grid artifact.

## Method implication and next protocol

The repeated obstruction is now more specific than “the angle is too large”:
the fit target direction has high in-sample capacity but a high-variance,
negative tail on unseen unit-level TARC response. The next justified method
must control that uncertainty before selecting a tangent. A defensible v3 is a
prospectively frozen robust support construction:

1. Treat the disclosed v2 calibration rows as development evidence only.
2. Freeze a new, disjoint validation source and its indices before any new
   model forward.
3. Estimate a TARC tangent-response uncertainty set from development units,
   with a pre-registered regularized covariance and radius.
4. Replace each protected TARC affine margin by its worst-case lower bound over
   that ellipsoid. The support certificate must include the uncertainty term,
   not add a post-hoc correction to a chosen vector.
5. Require the same analytic KKT/full-facet certificate, the unchanged 95%
   conformal policy, and a new validation pass. A failure remains a stop.

The robust term has a direct proof: for an ellipsoid
`E={e:e^T Sigma^{-1}e <= kappa^2}`,

`min_{e in E} <a+e,u> = <a,u> - kappa ||Sigma^(1/2)u||_2`.

This is a support-function identity, not a heuristic penalty. It still does
not guarantee distributional transfer unless the uncertainty set and its
coverage claim are independently justified; the next protocol must state
that boundary explicitly.

## Reproducibility boundary

The run used zero optimizer steps, created no optimizer or scheduler, restored
the transient state, and opened neither Core100 nor heldout. The angle audit
only reads the frozen v2 fit geometry, calibration rows, and route artifact.
It does not inspect candidate text or authorize training.
