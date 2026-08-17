# MCD-E-TARC Matched Smoke Interpretation

## Decision

The prospectively registered route is
`STOP_MCD_ETARC_SMOKE_COMMON_EFFECT_FAILED`. The current MCD-E-TARC
formulation is closed and is not authorized for longer training.

This is a valid strong-gate failure, not an invalid run. All matching,
assignment, numerical, VRAM, sealed-data, dynamic-budget, and dual-certificate
checks passed. Independent `verify-run` recomputed the same route and matched
all 22 hashed result artifacts.

## Experimental integrity

Uniform, TARC, raw E-TARC, Q-E-TARC, and MCD-E-TARC each started from the same
trainable-state SHA-256 and completed exactly two optimizer steps. Risk order,
safety order, prompt exposure, token counts, available route terms, all six
objective-gradient counts, first-step losses, and initial diagnostics were
identical across arms. All five final state hashes were distinct.

Every arm had zero skipped updates and finite gradients. Peak reserved VRAM
was `8,445,231,104` bytes, leaving `34.42%` headroom. No checkpoint, candidate
output, Core100 row, or heldout row was written or opened.

## The dynamic theorem was realized

MCD-E-TARC recomputed the five normalized certificate gradients and solved the
31-support simplex dual on all eight safety draws. The observed dynamic range
was:

| Quantity | Observed |
|---|---:|
| minimum dual margin | `0.60186571` |
| minimum primal margin | `0.60186568` |
| maximum KKT residual | `3.275e-15` |
| maximum primal-dual gap | `2.921e-08` |
| maximum budget relative error | `5.907e-08` |

Every per-certificate cosine was at least `0.601866`, far above the frozen
`0.20` threshold. Active support sizes were `5,4,3,4,4,4,4,4`; E-TARC and
continuation were active on all eight draws, uniform on six, and each branch
certificate on five. Thus the negative finite-step result is not explained by
an absent common direction, a failed solver, a weak dual margin, or an
unexercised branch constraint.

## Finite-step reductions

The smoke-excluded diagnostic means were:

| Arm | Uniform CE | E-TARC route | Continuation | Branch entropy | Branch negative-gap |
|---|---:|---:|---:|---:|---:|
| uniform | `2.063191` | `2.753488` | `1.806783` | `-0.003117` | `-0.352539` |
| TARC | `2.078292` | `3.690381` | `1.720409` | `-0.035892` | `-2.285156` |
| raw E-TARC | `2.050410` | `4.546839` | `1.724932` | `0.001451` | `0.981934` |
| Q-E-TARC | `2.097309` | `4.302094` | `1.799536` | `0.000912` | `0.750488` |
| MCD-E-TARC | `2.070767` | `4.025088` | `1.819924` | `0.004574` | `2.857910` |

All five registered MCD losses decreased, so the direction was qualitatively
compatible with every certificate. The failure is that three required
advantages were too small or heterogeneous.

## Branch control succeeded strongly

Adding branch entropy and negative-gap to the feasible set had the intended
finite-step effect. Relative to equal-budget Q-E-TARC:

| Contrast | Mean paired advantage | 95% bootstrap interval | Positive | Exact sign `p` |
|---|---:|---:|---:|---:|
| branch entropy | `0.003662` | `[0.000725, 0.008653]` | `8/8` | `0.0078125` |
| branch negative-gap | `2.107422` | `[1.666504, 2.510254]` | `8/8` | `0.0078125` |

The aggregate MCD reductions, `0.004574` and `2.857910`, cleared both frozen
absolute floors. This is direct mechanism evidence that explicit branch
certificates can control the two branch diagnostics that Q-E-TARC left
unconstrained.

## Common, route, and continuation effects failed

MCD's uniform-CE reduction exceeded the uniform arm by only `0.007575` on
average. The bootstrap interval was `[-0.005711, 0.021610]`, only 4/8 paired
differences were positive, and the exact sign test was `p=1.0`. This fails the
registered `0.05` common-effect floor and determines the primary stop route.

MCD retained `88.525%` of raw E-TARC's route reduction, comfortably passing
the `65%` retention gate. It reached only `1.090697x` TARC, below the required
`1.20x`. Its mean paired route advantage over TARC was `0.334707`, but the
interval `[-0.269678, 0.814260]` crossed zero; only 6/8 prompts were positive
and the sign test was `p=0.2890625`. The route pattern therefore reproduces
the heterogeneity warning already visible in P0.

MCD had the largest aggregate continuation reduction of all five arms, but
its incremental effects remained weak:

| Contrast | Mean paired advantage | 95% bootstrap interval | Positive | Exact sign `p` |
|---|---:|---:|---:|---:|
| over Q-E-TARC | `0.020388` | `[0.007522, 0.033999]` | `6/8` | `0.2890625` |
| over uniform | `0.013141` | `[0.000473, 0.025211]` | `7/8` | `0.0703125` |

Both means are below the frozen `0.05` floor. As in the earlier Q-E-TARC
smoke, most continuation improvement comes from full-response CE; changing
the auxiliary geometry adds only a small finite-step increment.

## Why Euclidean common descent is insufficient here

MCD certifies only the auxiliary term:

\[
\langle g_i,d_t\rangle>0.
\]

The raw accumulated training gradient is instead

\[
G_t=0.8g_{R,t}+0.2(g_{U,t}+d_t).
\]

Even under plain SGD, MCD does not constrain the risk cross-term
`<g_i,g_R>` or guarantee that the complete `G_t` is a common-descent
direction. The registered smoke additionally applies norm clipping and AdamW.

Every observed pre-clip norm was above the clip threshold `1.0`: the range was
`4.0379` to `5.7142`. Therefore clipping was active on every optimizer step.
For `||G||>1`, the clipping map is

\[
C(G)=\frac{G}{\lVert G\rVert_2},
\]

with directional derivative

\[
DC_G[h]=\frac{1}{\lVert G\rVert_2}
\left(I-\hat G\hat G^T\right)h.
\]

It removes the radial component of an auxiliary perturbation and attenuates
the remaining tangent component by `1/||G||`. At the first step, Adam has zero
moments and zero weight decay here. After bias correction its coordinate map
is exactly

\[
A_j(z)=\frac{z_j}{|z_j|+\epsilon},
\qquad
A_j'(z)=\frac{\epsilon}{(|z_j|+\epsilon)^2}
\quad(z_j\ne0).
\]

For coordinates large relative to `epsilon`, this is nearly a sign map and
its derivative is small. The first-step sensitivity of the realized update to
an auxiliary perturbation is therefore

\[
J_t = DA_{C(G_t)}\,DC_{G_t},
\]

not the identity assumed by Euclidean MCD. Clipping and Adam do not invalidate
the MCD theorem; they change the map between the certified raw-gradient
direction and the actual parameter displacement. The smoke demonstrates that
the mismatch is material: a dynamic common margin above `0.60` produced strong
branch effects but only weak or heterogeneous advantages on the other three
registered contrasts.

## Mathematically justified next diagnostic

Any successor must optimize the realized-update geometry rather than add more
raw-gradient certificates. Let `Phi_t(G)` be the frozen-state map from a raw
gradient to the positive optimizer displacement, including clipping and Adam
moments, and let

\[
J_t=D\Phi_t(G_{0,t}),
\qquad
G_{0,t}=0.8g_{R,t}+0.2g_{U,t}.
\]

For a normalized certificate `u_i`, the first-order incremental realized
descent from raw auxiliary `d` is

\[
u_i^T J_t d=(J_t^Tu_i)^T d.
\]

Thus the auditable optimizer-aware certificates are `J_t^T u_i`, not `u_i`.
A baseline-aware max-min problem can include the realized baseline score
`b_i=u_i^T Phi_t(G_0)`:

\[
\max_{\lVert d\rVert_2\le B}
\min_i\left[b_i+\alpha(J_t^Tu_i)^Td\right].
\]

Its minimax dual over simplex weights is

\[
\min_{\lambda\in\Delta}
\left[
\sum_i\lambda_i b_i
+\alpha B\left\lVert
\sum_i\lambda_i J_t^Tu_i
\right\rVert_2
\right].
\]

This produces an explicit optimizer-controllability margin. If the transformed
convex hull approaches zero, the diagnostic certifies that no raw auxiliary of
the registered budget can create a material common realized-update effect
under the current clipping/Adam state.

The next admissible stage is only a separately frozen zero-update Jacobian P0:
derive and unit-test the clipping/Adam Jacobian, compare linear predictions to
the exact frozen-state optimizer map, include the risk certificate, and test
prompt-disjoint transfer. No new optimizer step is authorized by this negative
smoke.

## Evidence boundary

This result uses Qwen3-4B, seed 1, two steps per arm, and a previously disclosed
development split. It establishes neither generated safety nor general
utility. `total_optimizer_steps=10`, `checkpoint_writes=0`,
`candidate_outputs_generated=false`, `development_training_authorized=false`,
`main_training_authorized=false`, `core100_opened=false`, and
`heldout_opened=false`.
