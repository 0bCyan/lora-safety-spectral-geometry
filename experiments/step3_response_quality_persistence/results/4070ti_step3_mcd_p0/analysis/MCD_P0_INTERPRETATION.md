# MCD-E-TARC P0 Mechanism Interpretation

## Decision

The prospectively registered route is
`PASS_MCD_ETARC_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL`. All eleven frozen gates
passed, and an independent `verify-run` reproduced the route and the hashes of
all seven run artifacts.

This is a zero-update mechanism result. It authorizes only a separately frozen,
equal-budget matched smoke protocol. It does not yet establish a finite-step
training benefit, safer generations, preserved utility, or a paper-level
empirical result.

## What is mathematically certified

Let the five normalized calibration gradients be

\[
u_i=\frac{g_i}{\lVert g_i\rVert_2},
\qquad i\in\{U,E,C,H,G\},
\]

for full-response CE, E-TARC route loss, continuation CE, branch entropy, and
branch negative-gap. MCD-E-TARC solves

\[
\mu^*=\max_{\lVert v\rVert_2\le 1}\min_i\langle u_i,v\rangle
=\min_{\lambda\in\Delta_5}
\left\lVert\sum_i\lambda_i u_i\right\rVert_2.
\]

The equality follows from minimax duality. At a minimum-norm convex
combination `w*`, simplex KKT conditions imply

\[
\langle u_i,w^*\rangle\ge\lVert w^*\rVert_2^2,
\]

with equality for positive dual weights. Therefore
`v*=w*/||w*||` satisfies

\[
\langle u_i,v^*\rangle\ge\mu^*>0
\]

for every registered certificate. The implementation enumerates all 31
nonempty supports, solves each equality-constrained system, and rejects any
support that violates primal feasibility or inactive KKT inequalities.

The measured calibration solution was:

| Quantity | Value |
|---|---:|
| dual/common margin | `0.6474796205` |
| primal margin | `0.6474795986` |
| primal-dual gap | `2.187e-08` |
| KKT residual | `5.551e-17` |
| active support size | `4` |

The active dual weights were `uniform=0.142080`, `etarc=0.260535`,
`continuation=0.262981`, and `branch_entropy=0.334403`.
`branch_negative_gap=0` was inactive because it was already easier than the
four binding constraints: its cosine with the returned direction was
`0.654281`, above the common margin. Thus a zero dual weight does not mean the
gap certificate was omitted or failed.

## Finite-step statement and its boundary

If each registered loss is locally `beta_i`-smooth, then a direct Euclidean
step `theta' = theta - eta B v*` obeys the descent lemma:

\[
L_i(\theta')
\le L_i(\theta)
-\eta B\mu^*\lVert g_i\rVert_2
+\frac{\beta_i}{2}\eta^2B^2.
\]

Consequently, all five losses decrease simultaneously for any positive step
size satisfying

\[
\eta < \min_i
\frac{2\mu^*\lVert g_i\rVert_2}{\beta_i B}.
\]

This is a genuine common-descent guarantee, but it is deliberately narrower
than the upcoming smoke claim. The training update also contains the risk and
uniform safety gradients, gradient clipping, and AdamW's coordinate-wise
preconditioner. Those operations can change both angles and finite-step loss
effects. Dynamic recomputation can renew the Euclidean certificate at every
microbatch; only a matched smoke can test whether the certificate survives the
actual optimizer path.

A simple transfer bound also follows. If a future normalized certificate
`u_i'` differs from its calibration counterpart by at most `epsilon_i`, then

\[
\langle u_i',v^*\rangle
\ge \mu^*-\epsilon_i.
\]

The observed validation cosines test this transfer directly on the frozen
development split, but they do not provide a distribution-free bound for new
data.

## Prompt-disjoint validation evidence

On 32 validation prompt clusters excluded from calibration, the aggregate
MCD direction had the following cosines and paired positive rates:

| Certificate | Cosine | Positive prompts |
|---|---:|---:|
| uniform CE | `0.581610` | `32/32` |
| E-TARC route | `0.602708` | `32/32` |
| continuation | `0.591880` | `32/32` |
| branch entropy | `0.573221` | `32/32` |
| branch negative-gap | `0.604225` | `32/32` |

The minimum validation cosine was `0.573221`, well above the frozen `0.20`
gate. Both branch objectives had positive aggregate dots
(`1.998339` and `182.183837`) and 100% paired positive rates. This directly
addresses Q-E-TARC's missing branch constraint at the gradient level.

At the common equal-norm budget `B=||g_U||=8.867816`, MCD-E-TARC retained
`70.963%` of raw E-TARC's route dot and reached `1.205800x` TARC's route dot.
It retained `70.929%` of RAMP's continuation dot and reached `1.334121x`
Q-E-TARC's continuation dot. The maximum relative budget mismatch across
controls was `2.806e-08`.

The continuation advantage over Q-E-TARC was particularly consistent:

- mean paired advantage: `11.199222`;
- bootstrap 95% interval: `[10.367319, 12.041826]`;
- positive pairs: `32/32`;
- exact two-sided sign test: `p=4.657e-10`.

## The route result is a real warning, not a footnote

The registered route gate passed because the aggregate MCD E-TARC dot was
`1.205800x` TARC, just above the frozen `1.20` threshold, while all 32 MCD
route dots were positive. However, the direct prompt-level MCD-minus-TARC
comparison was heterogeneous:

- mean paired advantage: `23.707412`;
- bootstrap 95% interval: `[4.325024, 42.042132]`;
- positive pairs: `22/32`;
- exact two-sided sign test: `p=0.050102`.

The sign test narrowly misses `0.05`, and the aggregate ratio clears its gate
by only about `0.48%`. This comparison was reported prospectively but was not
the registered sign-test gate used for the route decision. The correct
interpretation is therefore: P0 passes its frozen authorization rule, while
route superiority over TARC remains borderline and must be resolved by the
finite-step matched smoke. It must not be presented as an already robust
route win.

## Why this is more than a weighted-loss guess

MCD-E-TARC is not a manually selected scalarization. Its weights are the dual
solution of a max-min problem and change with the measured gradient geometry.
The active weights identify which mechanisms constrain the common direction;
the common margin quantifies the worst guaranteed first-order alignment; and
the zero-margin case is an explicit infeasibility signal. This gives three
falsifiable mechanism outputs that fixed loss weights do not provide:

1. a KKT-checkable dual certificate;
2. a worst-certificate descent margin;
3. a data-dependent active set with an auditable inactive-constraint test.

The method-specific perturbation also has a prospectively fixed material
scale. With safety mix `alpha=0.2` and `B=||g_U||`, its raw contribution is

\[
\lVert\alpha Bv^*\rVert_2=0.2\lVert g_U\rVert_2,
\]

four times Q-E-TARC's previous `0.05||g_U||` perturbation. This removes the
known weak-scale explanation without changing the gate after seeing P0.

## Integrity and evidence boundary

The run used Qwen3-4B LoRA gradients on an RTX 4070 Ti. Peak reserved VRAM
was `6,085,935,104` bytes with `52.74%` headroom. Initial and final trainable
state hashes were identical:
`4e4b6ec33469d81c3cc02d1527073151178880cfa252c08b16e59e460efd1924`.
There were no optimizer or scheduler objects and no checkpoint writes.

The 8 calibration and 32 validation clusters are a reused, disclosed
mechanism-development split. They are prompt-disjoint within P0 but are not
fresh confirmation data, and the method was proposed after examining prior
Q-E-TARC failures on this source. Bootstrap and sign-test results therefore
measure internal development consistency, not external generalization.

`optimizer_steps=0`, `training_authorized=false`,
`candidate_outputs_generated=false`, `core100_opened=false`, and
`heldout_opened=false`. No statement about generated safety or utility is
supported at P0.

## Authorized next test

The only authorized continuation is a separately committed and pushed
matched smoke protocol. It should compare `uniform`, `TARC`, raw `E-TARC`,
`Q-E-TARC`, and `MCD-E-TARC` at the same auxiliary norm and identical update
exposure, dynamically recompute all five MCD certificates per microbatch, and
measure held-aside finite-step route, continuation, branch, and uniform-loss
effects. Training must not begin until that protocol is frozen.
