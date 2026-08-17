# Step3 Q-E-TARC Matched Two-Step Smoke Plan

## 1. Question and evidential role

The zero-update Q-E-TARC P0 established a strong local direction on a reused
development split. This prospective smoke asks a narrower, falsifiable
question: does the spherical-cap direction survive eight real optimizer
updates across four matched two-step arms, and does it produce a prompt-level
route/continuation advantage on prompts excluded from those updates?

The arms are `uniform`, `tarc`, `etarc`, and `qetarc`. Each starts from the
same Qwen3-4B LoRA state and receives the same risk examples, safety prompt
pairs, optimizer, learning rate, and order. Each arm receives exactly two
optimizer steps with four microbatches per step. The eight diagnostic prompt
clusters are disjoint from all eight safety training draws.

This remains exploratory mechanism development. The diagnostic prompts were
already disclosed during P0 work and are not a fresh confirmatory sample.
There is no free generation, human review, Core100 access, or heldout access.

## 2. Dynamic constrained update

For safety microbatch m at parameters theta, compute four gradients from the
same pair of registered responses:

\[
g_U=\nabla_\theta L_{\mathrm{CE}},\quad
g_T=\nabla_\theta L_{\mathrm{TARC}},\quad
g_E=\nabla_\theta L_{\mathrm{E\text{-}TARC}},\quad
g_C=\nabla_\theta L_{\mathrm{cont}}.
\]

The auxiliary budget is recomputed on every microbatch:

\[
B_m=\rho\lVert g_U\rVert_2,\qquad \rho=0.25.
\]

TARC and E-TARC use the equal-budget directions

\[
d_T=B_m\frac{g_T}{\lVert g_T\rVert_2},\qquad
d_E=B_m\frac{g_E}{\lVert g_E\rVert_2}.
\]

Q-E-TARC uses the spherical-cap optimum with the prospectively fixed quota
`gamma=0.5`. Let

\[
u=\frac{g_E}{\lVert g_E\rVert_2},\quad
c=\frac{g_C}{\lVert g_C\rVert_2},\quad
a=\langle u,c\rangle.
\]

If `a >= gamma`, set `d_Q=B_m u`. Otherwise set

\[
d_Q=B_m\left[\gamma c+\sqrt{1-\gamma^2}
\frac{u-ac}{\sqrt{1-a^2}}\right].
\]

The safety gradient entering the matched update is `g_U` for uniform and
`g_U+d_j` for each auxiliary arm. Risk and safety gradients are mixed with
the frozen `0.8/0.2` weights after four-microbatch accumulation.

## 3. Mathematical guarantees tested by the trace

For every nondegenerate microbatch, the closed form solves

\[
\max_d\ \langle g_E,d\rangle
\quad\text{s.t.}\quad
\lVert d\rVert_2=B_m,\quad
\left\langle \frac{g_C}{\lVert g_C\rVert_2},d\right\rangle
\ge B_m\gamma.
\]

Consequently, for a descent step with learning rate eta,

\[
L_C(\theta-\eta d_Q)-L_C(\theta)
\le -\eta B_m\gamma\lVert g_C\rVert_2+O(\eta^2),
\]

and no other direction with the same norm and quota has a larger first-order
E-TARC decrease. These are local geometric statements. The smoke tests their
finite-step realization; it does not convert them into a global generation
guarantee.

## 4. Prospective gates

Integrity and numerical gates require exact matched orders and exposures,
identical initial states and diagnostics, finite complete gradients, no
skipped update, at least 10% VRAM headroom, exact arm assignment, distinct
final adapter hashes, and no sealed-data access.

Dynamic geometry must hold on all eight non-uniform microbatches per arm:

- auxiliary norm ratio `0.25 +/- 1e-5`;
- Q-E-TARC continuation cosine at least `0.5 - 1e-5`;
- if the quota is active, boundary error at most `1e-5`;
- Q-E-TARC route-retention cosine at least `0.85`;
- the analytical continuation lower bound must hold within `1e-4`.

Effect gates use the eight prompt-level paired diagnostic reductions:

- Q-E-TARC E-TARC loss reduction at least `0.5`;
- Q-E-TARC retains at least `80%` of raw E-TARC's route reduction;
- Q-E-TARC exceeds TARC route reduction by at least `0.05` on average, with
  paired bootstrap 95% lower bound above zero and exact sign-test `p <= 0.05`;
- Q-E-TARC exceeds both TARC and raw E-TARC continuation reduction by at
  least `0.05` on average, with both paired bootstrap lower bounds above zero
  and both exact sign-test values `p <= 0.05`;
- Q-E-TARC decreases branch entropy and branch negative-gap objectives,
  preserving the registered strong-path branch signature.

No threshold may be changed after the protocol commit. A failure is retained
as a negative result and authorizes no longer training stage.

## 5. Stop boundary

A pass authorizes only a separately frozen matched development protocol.
This smoke authorizes 8 total optimizer steps and one final adapter per arm.
It authorizes no generation, reviewer packet, main training, Core100 opening,
or heldout opening.
