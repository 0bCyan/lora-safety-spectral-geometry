# Step3 MCD-E-TARC: Minimax Multi-Certificate Common Descent

## 1. Registered precursor diagnosis

Q-E-TARC exactly realized its two-objective spherical-cap geometry, but its
matched smoke failed the registered strong-effect gates. Its continuation
advantage over TARC and raw E-TARC was consistent on 8/8 diagnostic prompts
yet only about `0.018`, route superiority over TARC held on only 6/8 prompts,
and branch entropy moved slightly in the wrong direction.

The failure has two structural causes. First, Q-E-TARC constrains only route
and continuation, while branch entropy is outside its feasible set. Second,
the registered auxiliary contributes only
`safety_mix * auxiliary_ratio = 0.2 * 0.25 = 0.05` times the uniform safety
gradient norm to the raw mixed update. MCD-E-TARC changes the optimization
problem rather than weakening the failed gates or fitting a new quota.

## 2. Registered certificates

For each calibration split, define five normalized gradients:

\[
u_i=\frac{g_i}{\lVert g_i\rVert_2},
\qquad i\in\mathcal I=\{U,E,C,H,G\},
\]

where `U` is full-response CE, `E` is the complete-path E-TARC barrier, `C`
is post-route continuation CE, `H` is allowed-branch entropy, and `G` is the
negative top1-top2 allowed-branch gap. Decreasing `H` concentrates the
registered route branch; decreasing `G` increases its strongest-path gap.

MCD-E-TARC solves

\[
\mu^*=\max_{\lVert v\rVert_2\le1}\min_{i\in\mathcal I}
\langle u_i,v\rangle.
\]

The candidate auxiliary is `d*=B v*` with the prospectively fixed equal-norm
budget

\[
B=\lVert g_U\rVert_2.
\]

At safety mix `alpha=0.2`, this makes the method-specific raw perturbation
`alpha d*` equal in norm to the uniform safety component `alpha g_U`. This is
a materiality definition, not a value selected from P0 outcomes. RAMP, TARC,
raw E-TARC, Q-E-TARC, and MCD-E-TARC are all compared at the same budget.

## 3. Dual certificate

### Theorem 1

Let `Delta` be the probability simplex over the five certificates. Then

\[
\mu^*=\min_{\lambda\in\Delta}
\left\lVert\sum_i\lambda_i u_i\right\rVert_2.
\]

If `w*=sum_i lambda_i* u_i` is nonzero, a primal optimum is

\[
v^*=\frac{w^*}{\lVert w^*\rVert_2}.
\]

Every certificate satisfies

\[
\langle u_i,v^*\rangle\ge\mu^*,
\]

with equality for every positive-weight dual certificate.

### Proof

For any unit `v` and any simplex weight `lambda`,

\[
\min_i\langle u_i,v\rangle
\le\sum_i\lambda_i\langle u_i,v\rangle
\le\left\lVert\sum_i\lambda_i u_i\right\rVert_2.
\]

This gives weak duality. At a dual minimizer, simplex KKT conditions give

\[
\langle u_i,w^*\rangle\ge\lVert w^*\rVert_2^2,
\]

with equality on the active support. Dividing by `||w*||` shows that `v*`
attains the dual bound, proving equality and optimality.

### Corollary 1: simultaneous first-order decrease

For every registered loss and a descent step `theta'=theta-eta d*`,

\[
L_i(\theta')-L_i(\theta)
\le-\eta B\mu^*\lVert g_i\rVert_2+O(\eta^2).
\]

If `mu*>0`, one direction has a strict first-order descent certificate for
all five mechanisms. If `w*=0`, the dual certifies that no strict common
direction exists in their span; training must not proceed under this method.

## 4. Exact small active-set solver

Only five gradients enter the dual. With Gram matrix

\[
K_{ij}=\langle u_i,u_j\rangle,
\]

the dual is the convex quadratic program

\[
\min_{\lambda\ge0,\,\mathbf1^T\lambda=1}\lambda^T K\lambda.
\]

The implementation enumerates all 31 nonempty supports. For each support it
solves the equality-constrained stationary system, rejects negative weights,
and checks active equality plus inactive KKT inequalities. The accepted
minimum returns its support, weights, KKT residual, Gram matrix,
primal-dual gap, and per-certificate cosines. No iterative optimizer tolerance
or high-dimensional gradient vector is serialized.

## 5. Prospective zero-update P0

P0 reuses the disclosed 8 calibration and 32 validation prompt clusters from
Q-E-TARC. They are development data, not fresh confirmation. It computes
real Qwen3-4B LoRA gradients but creates no optimizer or scheduler and writes
no checkpoint.

A strong pass requires:

- exact source structure, complete finite gradients, exact state restoration,
  at least 10% VRAM headroom, and equal auxiliary budgets within `1e-6`;
- dual KKT residual and primal-dual gap at most `1e-6`;
- calibration common margin at least `0.30`;
- validation minimum aggregate cosine across U/E/C/H/G at least `0.20`;
- positive MCD directional dots on at least 80% of validation prompts for
  every certificate and at least 90% for E-TARC route;
- validation E-TARC dot at least 65% of raw E-TARC and 120% of TARC;
- validation continuation dot at least 70% of RAMP and 120% of Q-E-TARC;
- MCD continuation advantage over Q-E-TARC with paired bootstrap lower bound
  above zero and exact sign-test `p<=0.05`;
- positive aggregate branch-entropy and branch-negative-gap dots;
- effective method perturbation ratio exactly `0.20` relative to `||g_U||`.

Failure closes MCD-E-TARC. A pass authorizes only a separately frozen matched
smoke protocol. It does not establish generated behavior, safety, or utility.

## 6. Boundaries

P0 has `optimizer_objects=0`, `scheduler_objects=0`, `optimizer_steps=0`,
`training_authorized=false`, `candidate_generation_authorized=false`,
`core100_opened=false`, and `heldout_opened=false`.
