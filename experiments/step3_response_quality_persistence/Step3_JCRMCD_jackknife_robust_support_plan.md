# Step3 JCR-MCD: Jackknife Convex-Hull Robust Common Descent

## 1. Registered precursor and distinct question

Seed-4 RP-PFRSC stopped at
`STOP_RPFRSC_P0_PROTECTED_NONINFERIORITY_FAILED`. Its construction search,
diagnostic common descent, target advantage, and materiality gates passed, but
the finite-step TARC retention bootstrap lower bound was `92.363%`, below the
strict registered `92.5%` threshold. The failure decomposed into a mean
`2.685` percentage-point construction-to-diagnostic exact-score loss and a
further `1.173` point exact-to-finite-step loss on the six active units.

That result closes certificate-role reassignment and finite-family threshold
scanning. JCR-MCD asks a different, falsifiable question: can four
construction-only support gradients define a deterministic expanded convex
hull that (a) contains an independent supported-domain diagnostic panel and
(b) certifies protected exact-score retention while improving the robust
three-target lower bound?

This is a seed-5, zero-optimizer-step mechanism experiment on reused disclosed
development data. It is not a retry of RP-PFRSC and cannot confirm generated
safety, utility, or heldout behavior.

## 2. Fixed certificate geometry

For certificate

\[
i\in\mathcal I=\{R,U,T,E,C,H,G\},
\]

the four construction supports yield nonzero normalized gradients

\[
u_{i,h}=g_{i,h}/\lVert g_{i,h}\rVert_2,\qquad h=1,\ldots,4.
\]

The support Gram matrix is `K_i[h,k] = u_{i,h}^T u_{i,k}`. For each held-out
support, the active-set simplex solver computes

\[
e_{i,h}=\operatorname{dist}\left(
u_{i,h},\operatorname{conv}\{u_{i,k}:k\ne h\}\right).
\]

The prospective radius is the maximum jackknife residual

\[
\epsilon_i=\max_h e_{i,h},
\]

and the uncertainty set is

\[
\mathcal U_i=\operatorname{conv}\{u_{i,1},\ldots,u_{i,4}\}
\oplus\epsilon_i B_2.
\]

This radius is a deterministic support-spread calibration. Four supports do
not justify a nominal 95% population coverage claim. Independent diagnostic
membership is therefore an explicit all-or-nothing falsification gate rather
than an assumed statistical guarantee.

## 3. One frozen path, not a subset scan

All 32 construction prompt pairs and risk rows are pooled after their four
support gradients have been computed. The fixed raw baseline remains

\[
G_0=0.8g_R+0.2g_U,
\]

and the auxiliary budget is `B = ||g_U||_2`. The full-seven-certificate MCD
direction is `d_0`; the MCD direction over only

\[
\mathcal Q=\{E,H,G\}
\]

is `d_1`. JCR-MCD evaluates exactly the 33-point normalized path

\[
d_\lambda=B\frac{(1-\lambda)d_0+\lambda d_1}
{\lVert(1-\lambda)d_0+\lambda d_1\rVert_2},
\qquad
\lambda\in\{0,1/32,\ldots,1\}.
\]

There is no certificate-subset enumeration, source endpoint screening,
radius multiplier, role scan, or threshold scan.

The exact clipping-plus-first-step-Adam displacement is

\[
x_\lambda=\frac{\Phi(G_0+0.2d_\lambda)}{\lVert\Phi(G_0)\rVert_2}.
\]

The same CPU float-preserving displacement tensors are used for the pooled
panel, all four support panels, diagnostic exact scoring, and transient
finite-loss evaluation. Thus support and diagnostic claims refer to one
fixed intervention rather than device-specific recomputations.

## 4. Exact robust lower bounds

For any displacement `x`, the expanded-hull support function gives

\[
L_i(x)=\inf_{u\in\mathcal U_i}u^Tx
=\min_h u_{i,h}^Tx-\epsilon_i\lVert x\rVert_2.
\]

The protected set is fixed as

\[
\mathcal P=\{R,U,T,C\},
\]

with unchanged retention `rho = 0.95`. Candidate `lambda` is feasible iff

\[
C_i(\lambda)=
\min_h u_{i,h}^T(x_\lambda-\rho x_0)
-\epsilon_i\lVert x_\lambda-\rho x_0\rVert_2\ge0
\quad\forall i\in\mathcal P.
\]

Among feasible candidates, JCR-MCD maximizes

\[
\min_{i\in\mathcal Q}L_i(x_\lambda),
\]

then the seven-certificate robust minimum, pooled target minimum, pooled
target mean, protected robust margin, and finally the smaller `lambda`.

### Theorem 1: raw-anchor feasibility

If `L_i(x_0)>0` for every certificate, then `lambda=0` is feasible.

### Proof

At the raw anchor,

\[
x_0-\rho x_0=(1-\rho)x_0.
\]

Positive homogeneity gives

\[
C_i(0)=(1-\rho)L_i(x_0)>0.
\]

Therefore the feasible family is nonempty. QED.

### Theorem 2: uncertainty-set protection

For every feasible `lambda`, every protected certificate, and every
`u in U_i`,

\[
u^Tx_\lambda\ge\rho u^Tx_0.
\]

### Proof

By the exact support-function identity,

\[
\inf_{u\in\mathcal U_i}u^T(x_\lambda-\rho x_0)=C_i(\lambda).
\]

Feasibility makes the right side nonnegative, so the inequality holds for
every member of the uncertainty set. QED.

### Corollary: observed diagnostic exact retention

If an independent diagnostic normalized gradient has distance at most
`epsilon_i` from the construction hull, it belongs to `U_i`; Theorem 2 then
certifies at least 95% diagnostic exact-score retention. This corollary says
nothing about finite-loss curvature, which remains separately measured.

## 5. Supported-domain seed-5 split

The 40 public development prompt clusters are category-imbalanced: three
categories contain only one prompt each. Withholding those singleton prompts
would leave their diagnostic gradients outside any category-supported
construction claim. JCR-MCD therefore restricts its diagnostic statement to
the five categories with at least three prompts.

The frozen diagnostic quotas are:

| Category | Diagnostic prompts |
|---|---:|
| cyber abuse | 1 |
| financial evasion | 2 |
| fraud or deception | 1 |
| harmful wrongdoing | 2 |
| public manipulation | 2 |

All singleton-category prompts remain in construction. Within each supported
category, diagnostic identities are selected by ascending SHA256 of
`jcrmcd-seed5-diagnostic-v1|category|sample_id`. The remaining 32 identities
are ordered by SHA256 of
`jcrmcd-seed5-support-v1|category|sample_id` and greedily assigned among
non-full supports by `(category count, support total, support id)`. Each of
the four supports has exactly eight prompt pairs. Risk rows use the first 40
indices of `torch.randperm(256, seed=5)`: 32 for construction and eight for
the independent diagnostics.

No claim is made for harassment/coercion, physical harm, or privacy invasion
transfer. A pass only licenses construction of a new, training-disjoint
`proxy_dev` triplet protocol that can test broader support.

## 6. Independent diagnostic design

Each of the eight frozen diagnostic prompt/risk pairs is one independent
unit. Its seven gradients are excluded from endpoint construction, radius
calibration, path scoring, feasibility, and candidate selection. After the
single global candidate is frozen in memory, each unit compares:

1. pooled risk/uniform baseline without an auxiliary direction;
2. pooled full-seven raw MCD;
3. pooled JCR-MCD.

The 24 finite-loss evaluations directly apply the fixed first-step
displacements at learning rate `2e-4`, evaluate all seven losses, and restore
the exact initial LoRA state. No optimizer or scheduler object is created.

## 7. Prospective gates

All inherited numeric thresholds remain unchanged from RP-PFRSC.

### Integrity

- four complete eight-pair supports and eight one-pair diagnostic units;
- seven finite nonzero gradients per support and diagnostic unit;
- exactly 33 path candidates and one globally selected direction;
- active clipping, direction-budget relative error at most `1e-6`, exact
  state restoration, RTX 4070 Ti execution, and at least 10% VRAM headroom;
- zero optimizer/scheduler objects, optimizer steps, checkpoints, generation,
  Core100 access, and heldout access.

### Robust construction

- all seven raw robust lower scores are strictly positive;
- the raw anchor and selected candidate are robust-feasible;
- selected protected robust margins are nonnegative;
- selected `lambda` is nonzero;
- selected robust target minimum is at least `0.05` and improves over the raw
  robust target minimum by at least `0.002`.

### Diagnostic hull coverage

- all `8 x 7 = 56` diagnostic certificate vectors have hull distance no
  greater than their frozen construction-only radius;
- every observed protected exact residual is at least its registered robust
  lower margin up to `1e-6` numeric tolerance;
- every observed target score is at least its registered robust lower bound
  up to the same tolerance.

### Diagnostic common descent and finite step

- overall and per-certificate exact scores are at least `0.02` on every unit;
- worst-score advantage over no auxiliary has mean at least `0.02`, bootstrap
  lower bound above zero, and exact sign-test `p<=0.05`;
- each objective has positive mean actual reduction, at least 75% positive
  units, and mean actual/predicted ratio in `[0.25,1.75]`.

### Protected noninferiority

For risk, uniform, TARC, and continuation, both exact score and actual loss
reduction must retain at least 95% on average, have a 95% bootstrap lower
retention bound strictly above 92.5%, and retain at least 85% on every unit.

### Target advantage and materiality

- diagnostic E-TARC/branch bottleneck gain over raw is at least `0.002`, with
  bootstrap lower bound above zero and sign-test `p<=0.05`;
- macro finite-loss relative gain across E-TARC, branch entropy, and branch
  negative-gap is at least 2%, with positive bootstrap lower bound and
  sign-test `p<=0.05`;
- each target retains at least 95% on average and 80% on every unit;
- mean diagnostic target minimum is at least `0.05`.

## 8. Stop boundary

The first failed gate fixes the route. A coverage failure is reported as an
uncertainty-set falsification even if post hoc exact scores look favorable.
A finite-step failure cannot be repaired by the exact theorem. No seed retry,
radius inflation, support reassignment, category expansion, threshold
relaxation, or path enrichment is authorized.

A pass authorizes only a separately committed `proxy_dev` triplet-data
protocol. It does not authorize matched training. Throughout P0:
`optimizer_objects=0`, `scheduler_objects=0`, `optimizer_steps=0`,
`virtual_parameter_evaluations=24`, `checkpoint_writes=0`,
`training_authorized=false`, `candidate_generation_authorized=false`,
`core100_opened=false`, and `heldout_opened=false`.
