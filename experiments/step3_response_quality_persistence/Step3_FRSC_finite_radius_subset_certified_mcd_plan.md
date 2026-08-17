# Step3 FRSC-MCD: Finite-Radius Subset-Certified Common Descent

## 1. Registered precursor

The optimizer-Jacobian OJ-MCD P0 stopped at
`STOP_OJMCD_P0_DUAL_CERTIFICATE_FAILED`. Its local Jacobian overpredicted the
registered finite-radius update norm by roughly `8.2e4-1.12e5`, exact common
descent failed on six of seven validation optimizer units, and OJ-MCD was
worse than the equal-budget raw seven-certificate direction on all seven
units.

The raw direction was only an ablation, not a prospective pass. Nevertheless,
its exact clipping-plus-Adam worst score was positive on all seven units with
a minimum of `0.0564`. FRSC-MCD registers a new seed-2 experiment to test that
finite-radius signal without reviving either failed MCD training or the failed
zero-radius Jacobian.

## 2. Optimizer-unit certificates

Each direction-construction unit contains four risk and four paired safety
microbatches. Their mean gradients define seven certificates:

\[
\mathcal I=\{R,U,T,E,C,H,G\},
\]

where `R` is risk CE, `U` is uniform safety CE, `T` is TARC route margin, `E`
is complete-path E-TARC, `C` is post-route continuation CE, `H` is allowed
branch entropy, and `G` is the negative top1-top2 allowed-branch gap. Let

\[
u_i=\frac{g_i}{\lVert g_i\rVert_2},\qquad
G_0=0.8g_R+0.2g_U,qquad B=\lVert g_U\rVert_2.
\]

Every auxiliary has norm `B`, so its accumulated contribution `0.2*d` has the
same raw norm as `0.2*g_U`.

## 3. Exact finite-radius optimizer map

With clipping threshold `c=1`, clipping epsilon `delta=1e-6`, first-step Adam
epsilon `epsilon=1e-8`, and zero weight decay, the positive displacement is

\[
\Phi_j(G)=
\frac{cG_j}{c|G_j|+\epsilon(\lVert G\rVert_2+\delta)}
\]

when clipping is active. The implementation also evaluates the equivalent
two-stage PyTorch expression and is unit-tested against actual
`clip_grad_norm_` plus `torch.optim.AdamW`.

FRSC-MCD never linearizes this map. For direction `d`, its exact normalized
certificate score is

\[
s_i(d)=\frac{u_i^T\Phi(G_0+0.2d)}
{\lVert\Phi(G_0)\rVert_2},
\qquad m(d)=\min_i s_i(d).
\]

## 4. Exhaustive subset family

For each nonempty subset `S` of the seven certificates, solve the ordinary
MCD simplex dual only within that subset:

\[
\lambda_S^*\in\arg\min_{\lambda\in\Delta_S}
\left\lVert\sum_{i\in S}\lambda_i u_i\right\rVert_2.
\]

If its norm is nonzero, define

\[
d_S=B\frac{\sum_{i\in S}\lambda_{S,i}^*u_i}
{\left\lVert\sum_{i\in S}\lambda_{S,i}^*u_i\right\rVert_2}.
\]

Singletons use their unit certificate directly. There are exactly
`2^7-1=127` registered subsets. FRSC-MCD evaluates every feasible `d_S` under
the exact optimizer map and chooses

\[
\hat S\in\arg\max_S m(d_S).
\]

Ties are resolved by larger mean exact score, then smaller subset, then lower
bitmask. No random search, iterative hyperparameter fit, or post-outcome
candidate insertion is allowed.

### Theorem 1: finite-family dominance

Let `I` denote the full seven-certificate subset. Since `I` is a member of the
exhausted family,

\[
m(d_{\hat S})\ge m(d_{\mathcal I}).
\]

### Proof

FRSC evaluates `m(d_S)` for every feasible member of a finite set that
contains `d_I` and returns its maximum under a deterministic tie rule. The
inequality follows directly. It is a global result over the registered finite
family, not a claim of global optimality over the entire parameter-space
sphere.

### Corollary 1: exact realized-update common descent

If `m(d_hat)>0`, then for every registered certificate

\[
g_i^T\Phi(G_0+0.2d_{\hat S})
=\lVert g_i\rVert_2\lVert\Phi(G_0)\rVert_2s_i(d_{\hat S})>0.
\]

Thus the exact optimizer displacement, rather than its raw gradient or local
Jacobian, is a strict simultaneous first-order descent direction for all
seven losses on that optimizer unit.

## 5. Finite parameter-step theorem

Let `p=Phi(G_0+0.2d_hat)` and apply the registered learning rate `eta=2e-4`
as `theta'=theta-eta*p`. If `L_i` has a `beta_i`-Lipschitz gradient on this
segment, the descent lemma gives

\[
L_i(\theta')\le L_i(\theta)
-\eta g_i^Tp+\frac{\beta_i\eta^2}{2}\lVert p\rVert_2^2.
\]

Consequently, a sufficient simultaneous finite-step condition is

\[
\eta < \min_i
\frac{2g_i^Tp}{\beta_i\lVert p\rVert_2^2}.
\]

P0 does not assume unknown global smoothness constants. It records the exact
first-order prediction, transiently applies each displacement directly to the
LoRA parameters, measures the actual fixed-diagnostic loss change, and copies
the original parameters back exactly. The observed Taylor remainder and
actual/predicted ratio are therefore mechanism measurements, not invented
bounds.

## 6. Seed-2 split and matching

The model LoRA initialization, risk order, and stochastic train-mode forwards
use seed 2. The disclosed 40 safety prompt clusters are partitioned before
model forward into:

- eight direction-construction units of four prompt pairs each;
- one fixed diagnostic set of eight prompt pairs;
- no prompt overlap between construction and diagnostics.

The diagnostic set contains exactly one prompt from each of the eight source
categories. Harassment/coercion, physical harm, and privacy invasion occur
only in diagnostics because each has a single source prompt. This is a hard
category-transfer test. The seed-2 risk order similarly reserves eight rows
for diagnostics and uses 32 disjoint rows for direction construction.

Every construction unit compares three matched exact first-step maps:

1. `baseline`: `G_0` with no auxiliary;
2. `raw_mcd`: full-seven subset `d_I`;
3. `frsc_mcd`: exhaustively selected `d_hat`.

The same fixed diagnostic gradients and initial losses evaluate all 24
transient arm/unit combinations. No diagnostic gradient enters subset
selection.

## 7. Prospective gates

All gates are fixed before any FRSC model forward.

### Integrity

- exactly eight construction units and 127 attempted subsets per unit;
- complete finite gradients, active clipping, equal budgets within `1e-6`;
- exact restoration after every transient update and at run end;
- at least 10% VRAM headroom;
- no optimizer, scheduler, checkpoint, generation, Core100, or heldout access.

### Finite-radius search

- selected exact construction margin at least `0.05` on every unit;
- selected-minus-full-seven margin gain mean at least `0.01`;
- paired bootstrap lower bound above zero and exact sign-test `p<=0.05`.

### Diagnostic common transfer

- exact diagnostic worst score and every individual certificate score at
  least `0.02` on every unit;
- diagnostic worst-score advantage over no-auxiliary baseline mean at least
  `0.02`, bootstrap lower bound above zero, and sign-test `p<=0.05`.

### Finite-step realization

- mean actual diagnostic loss reduction strictly positive for all seven
  objectives;
- positive actual reduction on at least 75% of units for every objective;
- each objective's mean actual/predicted reduction ratio in `[0.25, 1.75]`.

### Comparator and materiality

- FRSC diagnostic worst-score advantage over full-seven raw MCD mean at least
  `0.005`, bootstrap lower bound above zero, and sign-test `p<=0.05`;
- actual reduction is noninferior to raw MCD within `0.001` on at least six of
  seven objectives;
- FRSC mean diagnostic worst score at least `0.05`.

These gates require both exact-map improvement and disjoint finite-loss
transfer. Exhaustive in-sample selection alone cannot pass.

## 8. Route and boundaries

The first failed gate selects invalid execution, search-gain failure,
diagnostic-transfer failure, finite-step common-descent failure, raw-comparator
failure, or weak effect. A pass authorizes only a separately frozen matched
training protocol. It is not itself training authorization.

P0 has `optimizer_objects=0`, `scheduler_objects=0`, `optimizer_steps=0`,
`virtual_parameter_evaluations=24`, `checkpoint_writes=0`,
`training_authorized=false`, `candidate_generation_authorized=false`,
`core100_opened=false`, and `heldout_opened=false`.

## 9. Claim boundary

FRSC-MCD does not claim generic optimizer-aware multi-objective optimization
or subset search as new. MAdam (arXiv:2606.03904), MGDA, CAGrad, PCGrad, and
AdaTask already establish nearby optimizer and multi-objective contexts. The
narrow research hypothesis is that exhaustive finite-radius selection over a
small, mechanistically named certificate family can repair the documented
clipping/Adam mismatch while retaining an exact a posteriori safety-route
certificate. Novelty requires positive disjoint evidence and a fuller
literature comparison; this protocol alone establishes neither.
