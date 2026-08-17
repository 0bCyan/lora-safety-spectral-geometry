# Step3 OJ-MCD: Optimizer-Jacobian Common Descent

## 1. Registered precursor failure

MCD-E-TARC had a strong Euclidean five-certificate margin on every dynamic
smoke draw, but its frozen two-step experiment stopped at
`STOP_MCD_ETARC_SMOKE_COMMON_EFFECT_FAILED`. All five arms were matched and
all ten optimizer steps were valid. The failure was therefore not a solver or
execution failure.

Every accumulated pre-clip gradient norm was between `4.0379` and `5.7142`
while the registered clipping threshold was `1.0`. The first Adam step then
mapped the clipped gradient almost coordinatewise to its sign. A Euclidean
certificate on an auxiliary raw gradient consequently did not certify its
realized parameter displacement. OJ-MCD tests that mechanism directly. It
does not weaken any failed MCD smoke threshold and does not authorize
training.

## 2. Frozen optimizer-unit problem

One optimizer unit contains four risk and four paired safety microbatches.
Let their mean gradients be `g_R` and `g_i`, where

\[
i\in\mathcal I=\{R,U,T,E,C,H,G\}.
\]

The certificates are risk CE (`R`), uniform safety CE (`U`), TARC route
margin (`T`), complete-path E-TARC (`E`), post-route continuation CE (`C`),
allowed-branch entropy (`H`), and negative top1-top2 allowed-branch gap (`G`).
With the unchanged safety mix `alpha=0.2`, the no-auxiliary accumulated
gradient is

\[
G_0=(1-\alpha)g_R+\alpha g_U.
\]

The auxiliary budget is fixed before model forward as

\[
B=\lVert g_U\rVert_2,\qquad \lVert d\rVert_2\le B.
\]

Thus the raw method contribution `alpha*d` has the same norm as the uniform
safety component `alpha*g_U`. Raw MCD, Jacobian-only MCD, and OJ-MCD all use
this identical budget.

## 3. Exact first-step optimizer map

The registered training code uses `clip_grad_norm_` followed by zero-moment
AdamW with zero weight decay. Omitting the common positive learning rate, let

\[
C(G)=s(G)G,\qquad
s(G)=\min\left(1,\frac{c}{\lVert G\rVert_2+\delta}\right),
\]

where `c=1` and PyTorch's clipping epsilon is `delta=1e-6`. The positive
first-step Adam displacement is

\[
\Phi_j(G)=\frac{C_j(G)}{|C_j(G)|+\epsilon},
\]

with the frozen Adam epsilon `epsilon=1e-8`. This expression is independent
of beta1 and beta2 after first-step bias correction.

When clipping is active, for any perturbation `h`,

\[
DC_G[h]
=s(G)h-
\frac{c\,G^Th}{\lVert G\rVert_2(\lVert G\rVert_2+\delta)^2}G.
\]

The Adam derivative is diagonal,

\[
D\!A_z=\operatorname{diag}\left(
\frac{\epsilon}{(|z_j|+\epsilon)^2}
\right),
\]

including the derivative `1/epsilon` at zero. The realized-update Jacobian is

\[
J_G=D\!A_{C(G)}DC_G.
\]

The implementation tests the closed form against actual PyTorch clipping and
AdamW, checks a centered finite-difference JVP, and verifies the JVP/VJP
adjoint identity.

## 4. Baseline-aware affine minimax theorem

Normalize every certificate as `u_i=g_i/||g_i||`. Define the dimensionless
baseline score and optimizer sensitivity

\[
b_i=\frac{u_i^T\Phi(G_0)}{\lVert\Phi(G_0)\rVert_2},\qquad
a_i=\frac{\alpha}{\lVert\Phi(G_0)\rVert_2}J_{G_0}^Tu_i.
\]

OJ-MCD solves

\[
p^*=\max_{\lVert d\rVert_2\le B}
\min_i\left(b_i+a_i^Td\right).
\]

### Theorem 1

Let `Delta` be the simplex over the seven certificates. Then

\[
p^*=\min_{\lambda\in\Delta}
\left[
\sum_i\lambda_i b_i+
B\left\lVert\sum_i\lambda_i a_i\right\rVert_2
\right].
\]

If `w*=sum_i lambda_i* a_i` is nonzero, a primal solution is

\[
d^*=B\frac{w^*}{\lVert w^*\rVert_2}.
\]

### Proof

Write the inner minimum as a minimization over simplex mixtures:

\[
\min_i(b_i+a_i^Td)
=\min_{\lambda\in\Delta}
\sum_i\lambda_i(b_i+a_i^Td).
\]

The ball and simplex are compact and convex and the expression is bilinear,
so the minimax theorem exchanges max and min. Maximizing the resulting linear
form over the radius-`B` ball gives its support function
`B||sum_i lambda_i a_i||`, proving the dual. Equality in the support function
gives `d*`. Simplex KKT conditions require every positive-weight certificate
to attain the worst predicted score and every inactive certificate to have a
score no smaller than it. The runner records that KKT residual and the direct
primal-dual gap.

### Corollary 1

The predicted normalized full-update score is

\[
\frac{u_i^T[\Phi(G_0)+\alpha J_{G_0}d^*]}
{\lVert\Phi(G_0)\rVert_2}
=b_i+a_i^Td^*\ge p^*.
\]

This certificate includes the realized risk/uniform baseline, unlike raw
MCD, and includes clipping plus Adam geometry, unlike a Euclidean certificate.
It remains only a first-order statement, so exact secant falsification is
mandatory.

## 5. Exact secant and equal-budget ablations

For every optimizer-unit block, the runner evaluates

\[
\Delta_{\rm exact}=\Phi(G_0+\alpha d)-\Phi(G_0),\qquad
\Delta_{\rm lin}=\alpha J_{G_0}d.
\]

It compares three prospectively named directions:

1. `raw_mcd`: Euclidean seven-certificate common descent;
2. `jacobian_mcd`: common descent over individually normalized `J^T u_i`,
   without baseline offsets;
3. `ojmcd`: the affine optimizer-Jacobian problem in Theorem 1.

The exact normalized score used for paired comparisons is

\[
S_i(d)=\frac{u_i^T\Phi(G_0+\alpha d)}
{\lVert\Phi(G_0)\rVert_2}.
\]

No gradient tensor, model output, or training checkpoint is serialized.

## 6. Frozen P0 split

P0 reuses disclosed development data and is not fresh confirmation. It uses
the already frozen seed-1 risk and safety orders from the matched MCD smoke.
The first eight order positions were exercised by that smoke and are excluded.

- calibration: positions `8..11`, one optimizer-unit block;
- validation: positions `12..39`, seven disjoint optimizer-unit blocks;
- each block: exactly four risk and four safety microbatches;
- calibration/validation risk rows and safety prompt clusters do not overlap.

The algorithm is recomputed independently on each validation block. Therefore
the validation question is whether the frozen optimizer-aware rule replicates
on prompt-disjoint optimizer units, not whether one fitted high-dimensional
direction transfers unchanged.

## 7. Prospective strong gates

All thresholds below are frozen before any OJ-MCD model forward.

- complete finite gradients, exact state restoration, at least 10% VRAM
  headroom, and equal auxiliary budgets within `1e-6`;
- active clipping for baseline and candidate maps on every block;
- affine-dual KKT residual and primal-dual gap at most `1e-6`, with strictly
  positive predicted worst-score improvement on every block;
- calibration secant cosine at least `0.90`;
- validation minimum secant cosine at least `0.80` and mean at least `0.90`;
- linear/exact increment norm ratio in `[0.5, 2.0]` on every block;
- exact worst certificate score at least `0.05` on calibration and every
  validation block, including every one of the seven named certificates;
- OJ-MCD worst-score advantage over both raw MCD and the no-auxiliary baseline
  has mean at least `0.01`, bootstrap lower bound above zero, and exact paired
  sign-test `p<=0.05` over the seven validation blocks;
- exact increment norm is at least 1% of baseline on every block and averages
  at least 2% over validation blocks.

The score and effect floors are fractions of the no-auxiliary realized update
norm. They prevent a numerically correct but practically negligible Jacobian
direction from passing. Seven validation blocks permit a two-sided exact sign
test value of `0.015625` only when all seven paired effects have the registered
sign.

## 8. Route and stopping rule

The first failed gate selects, in order: invalid execution, optimizer map not
exercised, dual certificate failure, secant fidelity failure, exact common
descent failure, optimizer advantage failure, or weak effect. A pass authorizes
only a separately frozen matched smoke protocol. Any failure closes this
OJ-MCD formulation; thresholds may not be relaxed after seeing outcomes.

## 9. Literature and claim boundary

This work does not claim that optimizer-aware multi-objective optimization is
new. MGDA, PCGrad, CAGrad, and AdaTask already study multi-objective or
task-aware gradient geometry, while **MAdam: Metric-Aware Multi-Objective
Adam** (Liu et al., arXiv:2606.03904, 2026) explicitly identifies an Adam
geometric mismatch and proposes metric-aware preconditioning. CAGrad is
traceable as arXiv:2110.14048; AdaTask is published at
`10.1609/aaai.v37i9.26275`.

The narrower candidate contribution tested here is the combination of an
explicit PyTorch norm-clipping Jacobian, the exact frozen-state Adam secant,
a baseline-aware affine minimax certificate over risk plus six safety-route
mechanisms, and prospective falsification in LoRA safety training. Novelty is
not established by this P0 and must not be claimed without a fuller literature
review and positive comparative evidence.

## 10. Boundary

P0 has `optimizer_objects=0`, `scheduler_objects=0`, `optimizer_steps=0`,
`checkpoint_writes=0`, `training_authorized=false`,
`candidate_generation_authorized=false`, `core100_opened=false`, and
`heldout_opened=false`.
