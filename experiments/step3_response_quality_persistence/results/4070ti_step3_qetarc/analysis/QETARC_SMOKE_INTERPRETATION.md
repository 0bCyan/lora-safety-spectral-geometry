# Q-E-TARC Matched Smoke Interpretation

## Decision

The registered route is
`STOP_QETARC_SMOKE_ROUTE_EFFECT_FAILED`. Q-E-TARC is not authorized for a
longer run. The result is a valid negative strong-gate result, not an invalid
run: all matching, numerical, assignment, sealed-data, and dynamic-geometry
checks passed, and `verify-run` reproduced the stored decision and all 34
hashed result artifacts.

## What worked

The spherical-cap construction was realized on every one of the eight
Q-E-TARC microbatches:

- all 8 quotas were active;
- minimum achieved continuation cosine: `0.4999999742`;
- minimum E-TARC route-retention cosine: `0.8846415722`;
- maximum auxiliary-budget relative error: `3.026e-08`;
- minimum VRAM headroom across arms: `35.50%`;
- four final adapter SHA-256 values were distinct;
- all four arms had identical initial states, initial diagnostics, risk
  order, safety order, exposure counts, and first-step matched losses.

Thus the negative result is not explained by a failed projection, unequal
budget, prompt overlap, skipped update, OOM recovery, or arm mismatch.

## Finite-step effects

On eight diagnostic prompt clusters excluded from all smoke updates:

| Arm | E-TARC loss reduction | Continuation loss reduction | Branch entropy reduction | Branch negative-gap reduction |
|---|---:|---:|---:|---:|
| uniform | 2.753488 | 1.806783 | -0.003117 | -0.352539 |
| TARC | 3.135429 | 1.791481 | -0.009563 | -1.039551 |
| raw E-TARC | 3.489115 | 1.791037 | -0.001062 | 0.111816 |
| Q-E-TARC | 3.404734 | 1.809881 | -0.001270 | 0.047852 |

Q-E-TARC retained `97.58%` of raw E-TARC's aggregate route reduction and
exceeded TARC by a mean paired `0.269305`. The bootstrap 95% interval was
`[0.001710, 0.496328]`, but only 6/8 prompt differences were positive, so the
registered exact sign test was `p=0.2890625`. The route gate therefore failed.

Continuation improved over both route controls on every prompt:

- versus TARC: mean `0.018400`, 95% CI `[0.015413, 0.021351]`, 8/8 positive,
  exact `p=0.0078125`;
- versus raw E-TARC: mean `0.018844`, 95% CI `[0.015701, 0.022257]`, 8/8
  positive, exact `p=0.0078125`.

This is consistent but smaller than the prospectively required `0.05`. It is
also only `0.003098` above uniform CE. Q-E-TARC mainly recovered the
continuation degradation introduced by route-only auxiliaries; it did not
create a large continuation gain beyond the full-response CE baseline.
Branch negative-gap moved in the intended direction, but branch entropy
moved by `-0.001270`, narrowly in the wrong direction. That gate also failed.

## Why the P0 direction did not become a strong method effect

Let the safety mixture be `alpha=0.2`, and let the registered auxiliary
budget be

\[
B=\rho\lVert g_U\rVert_2,\qquad \rho=0.25.
\]

Ignoring Adam state for the moment, an auxiliary arm uses

\[
G_j=(1-\alpha)g_R+\alpha(g_U+d_j),
\]

whereas uniform uses

\[
G_0=(1-\alpha)g_R+\alpha g_U.
\]

Therefore the entire method-specific perturbation is

\[
G_j-G_0=\alpha d_j,qquad
\lVert G_j-G_0\rVert_2
=\alpha\rho\lVert g_U\rVert_2
=0.05\lVert g_U\rVert_2.
\]

The quota theorem guarantees a positive first-order continuation allocation
inside `d_Q`, but the registered training mixture restricts that allocation
to a 5%-of-uniform-norm perturbation before clipping and optimizer
preconditioning. Full-response CE already directly trains all continuation
tokens, explaining why the Q-E-TARC advantage over uniform is much smaller
than its advantage over route-only controls.

There are two additional theorem-to-update gaps:

1. The spherical-cap proof is Euclidean and infinitesimal. AdamW applies a
   state-dependent coordinate transform,

   \[
   \Delta\theta=-\eta\,m/(\sqrt{v}+\epsilon),
   \]

   so maximizing `g_E dot d` before that transform does not generally
   maximize the realized finite-step E-TARC decrease.
2. Branch entropy was measured but absent from the feasible set. Its wrong
   sign is therefore not a contradiction of the two-objective theorem.

The evidence supports the theorem and its implementation, but rejects the
claim that a fixed `gamma=0.5`, `rho=0.25` two-objective cap is by itself a
strong finite-step training method under the registered mixture and AdamW.

## Mathematically justified next candidate

The next candidate should directly certify all measured mechanisms rather
than increase `gamma` post hoc. For normalized gradients

\[
u_i=g_i/\lVert g_i\rVert_2,
\quad i\in\{E,C,H,G\},
\]

where `H` is branch entropy and `G` is branch negative-gap, define the robust
common-descent direction

\[
\mu^*=\max_{\lVert v\rVert_2\le1}\min_i\langle u_i,v\rangle.
\]

Its dual is the four-variable simplex problem

\[
\mu^*=\min_{\lambda\in\Delta_4}
\left\lVert\sum_i\lambda_i u_i\right\rVert_2.
\]

If `w*=sum_i lambda_i* u_i` is nonzero, then

\[
v^*=w^*/\lVert w^*\rVert_2
\]

is primal-optimal and satisfies

\[
\langle u_i,v^*\rangle\ge\mu^*>0
\]

for every registered objective. Consequently, for `d=Bv*`, every objective
has the simultaneous first-order bound

\[
L_i(\theta-\eta d)-L_i(\theta)
\le-\eta B\mu^*\lVert g_i\rVert_2+O(\eta^2).
\]

This minimax multi-certificate direction has a small, auditable active set,
a dual certificate, and an explicit infeasibility signal when the convex
hull approaches the origin. A separate zero-update P0 must first test its
common margin, route retention, continuation materiality, branch signs, and
prompt-disjoint transfer. No training is authorized by this interpretation.

## Evidence boundary

The result uses Qwen3-4B on one seed, two optimizer steps per arm, and a
previously disclosed development split. It establishes neither generated
safety nor general utility. Core100 and heldout remained sealed, and no
candidate generation or human review occurred.
