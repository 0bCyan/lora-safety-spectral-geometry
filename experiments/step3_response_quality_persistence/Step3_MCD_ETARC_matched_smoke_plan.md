# Step3 MCD-E-TARC Equal-Budget Matched Smoke Plan

## 1. Registered precursor

The zero-update MCD-E-TARC P0 returned
`PASS_MCD_ETARC_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL`. On its frozen
calibration split, the five-certificate minimax direction had common margin
`0.647480`, KKT residual `5.551e-17`, and primal-dual gap `2.187e-08`. On 32
prompt-disjoint development clusters, all five certificate directions were
positive for 32/32 prompts and the minimum aggregate cosine was `0.573221`.

P0 remains gradient geometry on a reused development source. It did not test
an optimizer update. In particular, its aggregate route/TARC ratio was only
`1.205800`, and the direct prompt-paired sign test was `p=0.050102`. This
smoke is prospectively frozen to determine whether the common-descent
certificate survives mixed gradients, clipping, AdamW, and finite steps.

## 2. Hypothesis

For each safety pair, let

\[
\mathcal I=\{U,E,C,H,G\}
\]

denote full-response CE, E-TARC route, continuation CE, branch entropy, and
branch negative-gap. MCD-E-TARC dynamically solves

\[
\mu_t^*=\max_{\lVert v\rVert_2\le1}
\min_{i\in\mathcal I}\langle u_{i,t},v\rangle,
\qquad
u_{i,t}=g_{i,t}/\lVert g_{i,t}\rVert_2,
\]

and uses auxiliary `d_t=||g_U,t|| v_t*`. The safety gradient is

\[
s_t=g_{U,t}+d_t.
\]

At safety mix `alpha=0.2`, the method-specific raw perturbation has norm
`0.2||g_U,t||`. Every non-uniform control receives an auxiliary of exactly the
same norm. The primary finite-step hypothesis is that MCD-E-TARC produces a
material Pareto improvement: stronger route reduction than TARC, stronger
continuation and branch reductions than Q-E-TARC, and stronger uniform-CE
reduction than the uniform arm, without losing more than 35% of raw E-TARC's
route reduction.

## 3. Five matched arms

The frozen arm order is:

1. `uniform`: safety gradient `g_U`, no auxiliary;
2. `tarc`: `g_U + B normalize(g_T)`;
3. `etarc`: `g_U + B normalize(g_E)`;
4. `qetarc`: `g_U + d_Q`, where `d_Q` is the spherical-cap direction with
   continuation cosine quota `0.5` and norm `B`;
5. `mcd_etarc`: `g_U + d_M`, where `d_M` is the exact five-certificate minimax
   direction with norm `B`.

Here `B=||g_U||` is recomputed per safety draw. All arms compute all six
available objective gradients `U,T,E,C,H,G` on every safety draw, even when a
given arm does not select them. This keeps forward examples, backward counts,
and graph construction matched.

Each arm starts from a freshly loaded Qwen3-4B LoRA adapter with seed 1. Each
arm uses the same 256 risk rows, risk permutation, safety-pair order, 2 AdamW
optimizer steps, and 4 gradient-accumulation microbatches per step. The raw
mixed gradient is

\[
G_t=0.8g_{R,t}+0.2s_t.
\]

The learning rate is `2e-4`, maximum gradient norm is `1.0`, weight decay is
zero, and warmup is zero. No OOM recovery, skipped update, arm-specific
schedule, or replacement prompt is allowed.

## 4. Development split boundary

The source is the disclosed 40-prompt, 80-target natural-diverse registry.
The 8 smoke update prompts are the first 8 entries of the frozen order. The 8
diagnostic prompts are frozen members of the prior P0 validation split and
are excluded from every smoke update. The diagnostic unit is a prompt cluster
containing both approved targets.

This source has already been used for method development. The experiment is
exploratory and non-confirmatory. It cannot establish generated safety,
utility, or external generalization. Core100 and heldout remain sealed, and
candidate generation remains forbidden.

## 5. Dynamic certificate gates

On every one of the 8 MCD safety draws:

- all six objective gradients must be finite, nonzero, and cover every
  trainable tensor;
- the exact 31-support simplex solver must return a KKT-valid solution;
- KKT residual and primal-dual gap must each be at most `1e-6`;
- primal margin, dual margin, and every certificate cosine must be at least
  `0.20`;
- auxiliary norm ratio must equal `1.0` within absolute tolerance `1e-5`.

The Q-E-TARC control must satisfy its quota, active-boundary, route-retention,
and continuation lower-bound checks from the prior smoke, with minimum route
retention cosine `0.85`. Failure of any dynamic geometry check stops the
experiment independently of observed loss changes.

## 6. Matching and numerical gates

All five arms must have:

- identical initial trainable-state hashes;
- identical risk order, safety order, prompt exposure, target rows, response
  tokens, continuation tokens, available route terms, and objective-gradient
  counts;
- equal first-step risk and uniform-safety losses within `1e-7`;
- equal initial diagnostics within `1e-7`;
- exactly 2 completed optimizer steps, no skipped updates, finite gradients,
  and at least 10% VRAM headroom;
- distinct final trainable-state hashes, proving each assignment was
  exercised;
- zero checkpoint writes and no candidate, Core100, or heldout access.

An integrity or numerical failure takes precedence over every effect route.

## 7. Prospectively frozen effect gates

All differences below are paired by the 8 smoke-excluded diagnostic prompt
clusters. Bootstrap intervals use 10,000 percentile resamples with seed
`2026081725`. Exact two-sided sign tests exclude exact ties.

For every registered paired superiority claim, the bootstrap 95% lower bound
must be above zero and the exact sign-test value must be at most `0.05`.

### 7.1 Common finite-step effect

- MCD reductions in `U,E,C,H,G` must all be strictly positive;
- each registered loss reduction must exceed `0.0` as an aggregate;
- MCD's paired uniform-CE reduction advantage over the uniform arm must have
  mean at least `0.05` and pass both inferential gates.

### 7.2 Route effect

- MCD aggregate E-TARC reduction must be at least `0.5`;
- it must retain at least `65%` of raw E-TARC's reduction;
- it must reach at least `1.20x` TARC's reduction;
- its paired E-TARC reduction advantage over TARC must have mean at least
  `0.05` and pass both inferential gates.

### 7.3 Continuation effect

- MCD aggregate continuation reduction must be at least `0.5`;
- its paired continuation advantage over Q-E-TARC must have mean at least
  `0.05` and pass both inferential gates;
- its paired continuation advantage over uniform CE must have mean at least
  `0.05` and pass both inferential gates.

The `0.05` floor is retained specifically to reject the prior Q-E-TARC smoke's
consistent but weak continuation advantage of about `0.018`.

### 7.4 Branch effect

- MCD branch-entropy reduction must be at least `0.001`;
- MCD branch-negative-gap reduction must be at least `0.05`;
- its paired branch-entropy advantage over Q-E-TARC must have mean at least
  `0.001` and pass both inferential gates;
- its paired branch-negative-gap advantage over Q-E-TARC must have mean at
  least `0.05` and pass both inferential gates.

No gate may be lowered, removed, or reinterpreted after execution.

## 8. Decision routes

The route order is:

1. `STOP_MCD_ETARC_SMOKE_MATCHING_FAILED`;
2. `STOP_MCD_ETARC_SMOKE_NUMERICAL_FAILED`;
3. `STOP_MCD_ETARC_SMOKE_ASSIGNMENT_NOT_EXERCISED`;
4. `STOP_MCD_ETARC_SMOKE_DYNAMIC_CERTIFICATE_FAILED`;
5. `STOP_MCD_ETARC_SMOKE_COMMON_EFFECT_FAILED`;
6. `STOP_MCD_ETARC_SMOKE_ROUTE_EFFECT_FAILED`;
7. `STOP_MCD_ETARC_SMOKE_CONTINUATION_EFFECT_FAILED`;
8. `STOP_MCD_ETARC_SMOKE_BRANCH_MECHANISM_FAILED`;
9. `PASS_MCD_ETARC_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT_PROTOCOL`.

A pass authorizes only a separately frozen and pushed matched development
protocol. A failure closes the current MCD-E-TARC formulation. Neither route
authorizes generation, blind review, Core100, heldout, or main training.

## 9. Artifact policy

The smoke writes per-arm training traces, dynamic geometry, diagnostics, and
run manifests, plus a route decision, report, and hash inventory. It records
the final trainable-state SHA-256 but writes no adapter checkpoint. This keeps
the diagnostic evidence package complete and small while preventing an
unreviewed smoke adapter from becoming an accidental training input.
