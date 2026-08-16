# CSPR P0 Mathematical Validation

Status: prospective proof obligations before any CSPR model-gradient computation or optimizer step.

## 1. Scope

Constrained Same-Prompt Preference Replay (CSPR) is only a candidate method. P0 validates a local optimization statement. It does not claim that a finite training sample proves behavioral safety, generalization, calibration, or robustness.

Let `theta` be the trainable LoRA parameter vector. Define two differentiable empirical losses on a frozen batch:

\[
L_p(\theta) \quad \text{(same-prompt response preference loss)},
\qquad
L_s(\theta) \quad \text{(constant20 safety replay loss)}.
\]

Write

\[
g_p = \nabla L_p(\theta), \qquad g_s = \nabla L_s(\theta).
\]

Let `P` be a fixed symmetric positive-definite diagonal preconditioner. Momentum, adaptive state changes during a step, Adam moment accumulation, and non-positive preconditioners are outside this theorem.

## 2. Same-Prompt Preference Objective

For prompt `x`, verified response `y+`, rejected response `y-`, and frozen reference policy `pi_ref`, define

\[
z_\theta =
\left[\log \pi_\theta(y^+|x)-\log \pi_{\rm ref}(y^+|x)\right]
-
\left[\log \pi_\theta(y^-|x)-\log \pi_{\rm ref}(y^-|x)\right]
\]

and

\[
\ell_p(z_\theta)=\log(1+\exp(-\beta z_\theta)), \qquad \beta>0.
\]

Its scalar derivative is

\[
\frac{d\ell_p}{dz}=-\beta\,\sigma(-\beta z)<0.
\]

Therefore the scalar loss is strictly decreasing in the reference-adjusted chosen-minus-rejected log-probability gap. This monotonicity does not by itself guarantee that a parameter-space update increases every individual pair gap because parameters are shared across examples.

## 3. Safety-Half-Space Projection

The unconstrained preconditioned preference direction is

\[
d_0=-P g_p.
\]

Consider the first-order safety half-space

\[
\mathcal H_s=\{d:g_s^\top d\le 0\}.
\]

CSPR uses the closest feasible direction to `d0` in the `P^-1` metric:

\[
d^*=\arg\min_{d\in\mathcal H_s}
\frac12(d-d_0)^\top P^{-1}(d-d_0).
\]

### Theorem 1: Closed-form unique projection

If `g_s^T d0 <= 0`, then `d*=d0`. Otherwise,

\[
d^*=d_0-
\frac{g_s^\top d_0}{g_s^\top P g_s}P g_s.
\]

If `g_s=0`, the constraint is vacuous and `d*=d0`.

#### Proof

The objective is strictly convex because `P^-1` is positive definite and the feasible set is a closed convex half-space, so the minimizer is unique. The KKT Lagrangian is

\[
\mathcal L(d,\lambda)=
\frac12(d-d_0)^\top P^{-1}(d-d_0)+\lambda g_s^\top d,
\qquad \lambda\ge0.
\]

Stationarity gives

\[
P^{-1}(d-d_0)+\lambda g_s=0,
\quad\text{hence}\quad
d=d_0-\lambda P g_s.
\]

When `d0` is feasible, complementary slackness gives `lambda=0`. When it is infeasible, the constraint is active and

\[
0=g_s^\top d_0-\lambda g_s^\top P g_s,
\]

which yields the stated multiplier and direction. This satisfies primal feasibility, dual feasibility, stationarity, and complementary slackness. Uniqueness completes the proof.

### Theorem 2: Joint first-order non-increase

The projected direction satisfies

\[
g_s^\top d^*\le0,
\qquad
g_p^\top d^*\le0.
\]

If the projection is inactive and `g_p` is nonzero,

\[
g_p^\top d^*=-g_p^\top P g_p<0.
\]

If the projection is active,

\[
g_p^\top d^*
=-g_p^\top P g_p
+\frac{(g_p^\top P g_s)^2}{g_s^\top P g_s}
\le0.
\]

#### Proof

Safety non-increase follows directly from feasibility. In the active case, substitute the closed-form direction. The weighted Cauchy-Schwarz inequality gives

\[
(g_p^\top P g_s)^2
\le(g_p^\top P g_p)(g_s^\top P g_s),
\]

which proves preference non-increase. Equality can occur for a degenerate weighted-collinear conflict, so strict preference progress is not guaranteed in every batch.

## 4. Finite-Step Certificate

Theorems 1 and 2 are directional statements. If both directional derivatives are strictly negative and both gradients are locally Lipschitz, sufficiently small positive steps decrease both losses. If `g_s^T d*=0`, first-order analysis alone gives no finite-step safety guarantee.

Counterexample:

\[
L_s(\theta)=\theta_1^2+\theta_2^2,
\quad \theta=(0,0),
\quad d=(1,0).
\]

Here `g_s^T d=0`, but `L_s(theta+alpha d)=alpha^2>0` for every `alpha>0`.

For this reason, an implementation must use a two-objective backtracking certificate. An update is accepted only if

\[
L_p(\theta+\alpha d)
\le L_p(\theta)+c\alpha g_p^\top d
\]

and

\[
L_s(\theta+\alpha d)
\le L_s(\theta)+c\alpha\min(g_s^\top d,0),
\qquad 0<c<1.
\]

If no registered step passes, the update must be skipped and recorded. The line search certifies only the exact frozen batches evaluated by these inequalities.

## 5. Required Numerical Tests

Before any model-gradient validation:

1. Exercise active and inactive projection branches on deterministic random vectors.
2. Verify safety and preference directional derivatives are non-positive.
3. Verify KKT stationarity and complementary slackness.
4. Match analytic derivatives to central finite differences.
5. Match the DPO scalar derivative to its closed form.
6. Accept a finite joint-descent example.
7. Reject the tangent second-order safety counterexample.
8. Reject non-positive or non-finite preconditioners.

## 6. Claim Boundary

Passing P0 establishes only that the registered numerical implementation matches the local projection theorem and refuses a known first-order false positive. It does not authorize a training pair, a model-gradient run, an optimizer step, or a behavioral claim. Real-model verification requires separately frozen same-prompt data, label-swap and wrong-prompt controls, response-only log-probability attribution, held-out pair transfer, and independent blinded behavior evaluation.
