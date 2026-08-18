# RUF-MCD-JRP v3r6 Relaxed Angular Impossibility

## Question

Could the certified v3r6 direction be repaired by reducing its angle while
keeping the direction itself fixed?

The answer on the already opened fit and calibration rows is no. This is a
post-hoc mechanism result and is not an authorized selection rule.

## Exact one-dimensional model

For every registered score covector, the angular path has the form

```text
s_j(theta) = r_j cos(theta) + t_j sin(theta),
```

where `r_j` is its raw-direction score and `t_j` is its score on the frozen
v3r6 tangent. Fit values are stored directly. Calibration tangent scores are
reconstructed from their raw and fixed-angle candidate scores; the maximum
reconstruction error is `2.78e-17`.

Every gate is therefore an inequality

```text
a cos(theta) + b sin(theta) >= c.
```

The audit computes every analytic equality root in `[0, theta_max]`, partitions
the interval at those roots, and tests one midpoint in each open component.
The sign of every continuous gate expression is constant inside a component,
so this procedure is exhaustive rather than a grid approximation.

## Result

The relaxed original-fit feasible interval is

```text
[0.07089042864785977, 0.3143421044892443] rad
```

or approximately `[4.06172, 18.01048]` degrees. Its lower endpoint is set by
the `fit_09` branch-entropy target floor. At that endpoint, the minimum
calibration retention is only `0.580204`, set by `calibration_12` TARC.

The calibration-protected feasible prefix is

```text
[0, 0.008481152489052679] rad
```

or approximately `[0, 0.485934]` degrees. At its upper endpoint, the fit target
floor is `0.0167884`, below the frozen `0.05` requirement.

The intervals are disjoint by

```text
0.07089042864785977 - 0.008481152489052679
= 0.06240927615880709 rad.
```

## Why the reflected hull misses the tail

Write protected TARC retention on this ray as

```text
retention_j(theta) = cos(theta) + beta_j sin(theta).
```

The 12 observed fit slopes are all positive, in `[0.836526, 2.715643]`.
The 12 registered jackknife-reflected slopes are also positive, in
`[0.405933, 2.626707]`. The complete registered 24-vertex projected hull is
therefore bounded below by `0.405933`.

Calibration contains two sign reversals: `calibration_12` has beta
`-5.891255` and `calibration_20` has beta `-1.735016`. In total, eight of 20
calibration beta values lie outside the registered projected interval.

This is a convex-hull separation certificate. A linear functional maps a
convex hull to the interval spanned by its vertex values. A covector whose
beta lies outside that interval cannot belong to the registered covector
hull. The failure is therefore in the uncertainty-set support, rather than
only in the fixed angle.

## Why the proof covers full JRP fit constraints

The angular audit retains all 48 original protected and 36 original target
facets but omits the 12 extra jackknife-reflected TARC facets. Omitting
constraints enlarges the fit-feasible set:

```text
F_full(theta) subseteq F_relaxed(theta).
```

No angle belongs to both the enlarged fit-feasible set and the observed
calibration-protected set. The complete 96-facet fit set therefore cannot
have such an angle either.

This set-inclusion argument is deliberately one-sided. It proves that dose
adjustment along the frozen v3r6 direction cannot work. It does not prove that
all directions are infeasible, and it does not turn calibration rows into
selection data. A successor must change the direction or the uncertainty
model and must use fresh data for confirmation.
