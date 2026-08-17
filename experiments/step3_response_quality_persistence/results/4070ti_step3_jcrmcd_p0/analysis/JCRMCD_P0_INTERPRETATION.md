# JCR-MCD Seed-5 Zero-Step P0 Interpretation

## Decision

The prospectively registered route is
`STOP_JCRMCD_P0_CONSTRUCTION_UNCERTAINTY_INVALID`. Integrity and finite-step
common descent passed. The construction uncertainty set, robust target gain,
diagnostic hull coverage, diagnostic common-transfer, target-advantage, and
materiality gates failed. Protected noninferiority passed only because the
registered fallback measurement used the raw anchor after the robust feasible
set was empty; it is not evidence for a nontrivial JCR-MCD intervention.

The run does not authorize a `proxy_dev` triplet protocol, matched training,
or any optimizer step. Independent `verify-run` reconstructed the same route,
all gates, all summaries, and all seven hashed result artifacts.

## Execution audit

The experiment ran at frozen protocol commit
`8cebdcbea85bf025da04f8d34f9cfc15fc0c6170` on the RTX 4070 Ti. Four
construction supports each contained eight prompt/risk pairs; eight separate
supported-domain diagnostic units were excluded from endpoint construction,
jackknife radius calibration, path scoring, feasibility, and selection.

The 33-point path reused one exact displacement per point across the pooled
and four support panels. All 24 diagnostic finite-loss evaluations directly
applied the three fixed CPU-materialized first-step displacements and restored
the initial LoRA state. Initial and final state hashes were identical:
`bc9cb3307d1a5fe69b357504a316cf48ceae05c89bf35740e13b66af1838cbb4`.
Peak reserved VRAM was `9,883,877,376` bytes, leaving `23.25%` headroom. Wall
time was `255.05` seconds. No optimizer, scheduler, checkpoint, generated
response, Core100 row, heldout row, or raw gradient tensor was created or
opened.

## Registered gates

| Gate | Result |
|---|---:|
| integrity | pass |
| construction uncertainty valid | **fail** |
| robust target gain | **fail** |
| diagnostic hull coverage | **fail** |
| diagnostic common transfer | **fail** |
| finite-step common descent | pass |
| protected diagnostic noninferiority | pass, raw-anchor identity only |
| target diagnostic advantage | **fail** |
| effect not weak | **fail** |

The first failure fixes the route. Later measurements explain the failure but
cannot change it.

## Construction failure

The registered raw robust lower bound was

\[
L_i(x_0)=\min_h u_{i,h}^Tx_0-\epsilon_i\lVert x_0\rVert_2.
\]

It was negative for six of seven certificates. Only uniform remained positive.

| Certificate | Jackknife radius | Raw robust lower |
|---|---:|---:|
| risk | `0.4561` | `-0.1537` |
| uniform | `0.3002` | `+0.0254` |
| TARC | `0.6021` | `-0.4930` |
| E-TARC | `0.7220` | `-0.5654` |
| continuation | `0.3458` | `-0.0823` |
| branch entropy | `0.7789` | `-0.7074` |
| branch negative-gap | `0.3316` | `-0.2560` |

The premise of the raw-anchor feasibility theorem therefore failed. No one of
the 33 path candidates was robust-feasible. The runner retained `lambda=0`
only as a declared fallback measurement, with minimum protected robust margin
`-0.02465`; it did not label that anchor feasible. Robust target gain was
exactly zero.

## Diagnostic coverage failure

Only `6/56` diagnostic certificate vectors lay in their registered expanded
hulls. All six covered vectors were branch-entropy gradients. Coverage was
zero of eight for risk, uniform, TARC, E-TARC, continuation, and branch
negative-gap. Maximum diagnostic distances ranged from `0.8319` to `1.2322`,
while registered radii ranged from `0.3002` to `0.7789`.

This is not a marginal miss. Two diagnostic units had `0/7` coverage, and the
other six had `1/7`. The uncertainty-set theorem consequently cannot be
transported to this panel, regardless of the observed positive theorem slack
against already-negative lower bounds.

## Scalar-radius incompatibility certificate

The post hoc audit in `jcrmcd_scalar_radius_feasibility.json` asks whether any
other scalar radius could repair the same four-support convex hull. Define

\[
D_i=\max_j\operatorname{dist}(v_{i,j},H_i)
\]

as the smallest radius covering all eight observed diagnostic gradients, and

\[
C_i=\frac{\min_h u_{i,h}^Tx_0}{\lVert x_0\rVert_2}
\]

as the strict upper limit on a radius that preserves positive raw robust
descent. A simultaneous radius exists only if `D_i < C_i`.

| Certificate | Positive capacity `C_i` | Coverage need `D_i` | `D_i-C_i` |
|---|---:|---:|---:|
| risk | `0.3024` | `0.8319` | `+0.5296` |
| uniform | `0.3256` | `0.9848` | `+0.6592` |
| TARC | `0.1091` | `1.1863` | `+1.0773` |
| E-TARC | `0.1565` | `1.1877` | `+1.0313` |
| continuation | `0.2635` | `0.9564` | `+0.6929` |
| branch entropy | `0.0714` | `1.2322` | `+1.1607` |
| branch negative-gap | `0.0756` | `1.0767` | `+1.0011` |

Every interval `[D_i,C_i)` is empty. Thus, for this frozen finite panel and
these four aggregated support hulls, radius inflation cannot repair coverage
without destroying raw robust descent, and radius shrinkage cannot restore raw
descent while covering the diagnostics. This rules out threshold tuning within
the registered isotropic scalar-radius family.

It does not prove a population impossibility or reject every uncertainty
representation. In particular, full-gradient Euclidean distance controls many
directions orthogonal to the 33 candidate displacements. The severe mismatch
is consistent with comparing eight-pair support means to one-pair diagnostic
gradients, but train/eval stochasticity and risk-row composition are also
confounded, so aggregation alone is not established as the cause.

## What remains valid

Raw MCD still produced useful common descent. Relative to the no-auxiliary
baseline, its diagnostic worst-score advantage had mean `0.05421`, bootstrap
interval `[0.03851,0.06682]`, and exact sign-test `p=0.0078125`; all eight
units were positive. Every certificate had positive mean finite-loss reduction
and a 100% positive-unit rate under the raw fallback. These observations
confirm the underlying common-descent signal, not JCR-MCD's uncertainty model.

The raw fallback's mean diagnostic target minimum was `0.04615`, below the
registered `0.05` materiality floor, and its minimum was `0.01395`. Target
gain over raw was necessarily zero, with eight ties. No target-improvement
claim survives.

## Method implication

The successor must change the object being calibrated. Another scalar radius,
support reassignment, radius multiplier, seed retry, or denser version of the
same path would contradict the registered closure and the incompatibility
certificate. A defensible next hypothesis is direct score-space calibration:
fix a path using a fit subset, evaluate protected residuals on independent
single-pair calibration units at the same granularity as diagnostics, and use
a finite-sample one-sided conformal bound before opening a final diagnostic
panel. Such a method would control only the displacement projections relevant
to the candidate rather than the entire normalized gradient vector.

That successor requires a new prospective protocol and disjoint fit,
calibration, and diagnostic roles. The present result supplies no authority to
train. `optimizer_objects=0`, `scheduler_objects=0`, `optimizer_steps=0`,
`checkpoint_writes=0`, `training_authorized=false`,
`candidate_outputs_generated=false`, `core100_opened=false`, and
`heldout_opened=false`.
