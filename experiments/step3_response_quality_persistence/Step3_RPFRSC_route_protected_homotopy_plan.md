# Step3 RP-PFRSC: Route-Protected Finite-Radius Homotopy

## 1. Registered precursor

Seed-3 PFRSC stopped at `STOP_PFRSC_P0_TARGET_ADVANTAGE_FAILED`. Six other
gates passed. The method retained risk, uniform, and continuation finite-loss
reduction by `98.71%`, `98.59%`, and `97.56%` relative to raw MCD. E-TARC,
branch entropy, and branch negative-gap improved by `7.73%`, `9.38%`, and
`24.18%`, each on all eight diagnostic units. Their positive effects were not
sufficient to pass because TARC changed by `-10.72%` on average and improved
on only four of eight units.

PFRSC placed TARC in a four-objective target minimum. That scalar minimum did
not guarantee coordinate-wise TARC retention once another target became
limiting. RP-PFRSC registers one seed-4 test that moves TARC into the protected
set without relaxing any retention, transfer, effect-size, bootstrap, or
sign-test threshold. The seed-3 route remains failed and cannot authorize
training.

## 2. Exact finite-radius geometry

The seven normalized certificate gradients are

\[
u_i=\frac{g_i}{\lVert g_i\rVert_2},
\qquad
\mathcal I=\{R,U,T,E,C,H,G\},
\]

for risk CE, uniform safety CE, TARC, E-TARC, continuation CE, branch entropy,
and branch negative-gap. The registered baseline and auxiliary budget remain

\[
G_0=0.8g_R+0.2g_U,
\qquad B=\lVert g_U\rVert_2.
\]

For every direction `d` with norm `B`, the implementation evaluates the exact
clipping-plus-first-step-Adam displacement `Phi(G0+0.2d)`. Scores use the same
fixed baseline displacement norm:

\[
s_i(d)=
\frac{u_i^T\Phi(G_0+0.2d)}{\lVert\Phi(G_0)\rVert_2}.
\]

No local optimizer Jacobian or post-outcome radius fit is used.

## 3. Frozen finite candidate family

For each of the 127 nonempty certificate subsets, the exact active-set simplex
solver constructs an equal-budget Euclidean MCD endpoint. The full-set raw MCD
direction `d0` is the anchor. The 126 non-anchor endpoints are ranked by the
exact three-target bottleneck

\[
q'(d)=\min_{i\in\mathcal Q'}s_i(d),
\qquad
\mathcal Q'=\{E,H,G\},
\]

then by overall minimum score, target mean, smaller subset, and lower bitmask.
Exactly the top 16 endpoints are retained.

For each retained endpoint and

\[
\Lambda=\{0,1/8,2/8,\ldots,1\},
\]

the registered normalized homotopy is

\[
d_{S,\lambda}=B
\frac{(1-\lambda)d_0+\lambda d_S}
{\lVert(1-\lambda)d_0+\lambda d_S\rVert_2}.
\]

The candidate family contains the raw anchor plus eight positive fractions for
each of 16 endpoints: `129` candidates. Reusing endpoint scores gives exactly
`127+16*7=239` exact-map evaluations per optimizer unit.

## 4. Four-coordinate protection

The protected and optimized sets are frozen as

\[
\mathcal P'=\{R,U,T,C\},
\qquad
\mathcal Q'=\{E,H,G\}.
\]

With `rho=0.95`, retain only

\[
\mathcal H'_\rho=
\{d\in\mathcal H:s_i(d)\ge\rho s_i(d_0)
\ \forall i\in\mathcal P'\}.
\]

RP-PFRSC selects

\[
d^*\in\arg\max_{d\in\mathcal H'_\rho}q'(d).
\]

Ties use overall minimum score, target mean, protected retention, smaller
fraction, smaller source subset, and lower bitmask.

### Theorem 1: route-protected finite-family dominance

If every raw-anchor score is positive, then:

1. `H'_rho` is nonempty because it contains `d0`;
2. risk, uniform, TARC, and continuation exact predicted reductions each
   retain at least `rho=0.95` of raw MCD on the construction unit;
3. the selected E-TARC/branch bottleneck is no worse than raw MCD;
4. the selected exact displacement is strict simultaneous first-order descent
   for all seven certificates.

### Proof

The raw anchor satisfies every protected ratio with equality one and is in the
finite family, proving nonemptiness. Membership of `d*` in `H'_rho` gives the
four coordinate-wise inequalities. Maximization over a set containing `d0`
gives `q'(d*)>=q'(d0)`. Positive raw scores and positive `rho` make all four
protected scores positive; positive `q'(d0)` makes all three target scores
positive. Since

\[
g_i^T\Phi(G_0+0.2d^*)
=\lVert g_i\rVert_2\lVert\Phi(G_0)\rVert_2s_i(d^*),
\]

all seven losses have strict first-order descent under the exact registered
optimizer displacement.

The theorem is exact over the screened finite family. It does not claim global
constrained optimality, disjoint-prompt transfer, or finite-loss dominance.

## 5. Seed-4 design

LoRA initialization, train-mode stochastic forwards, and the risk permutation
use seed 4. One diagnostic safety prompt per category is selected before model
forward by the minimum SHA256 of
`rpfrsc-seed4-diagnostic-v1|category|sample_id`. Remaining prompt IDs are
ordered by SHA256 of `rpfrsc-seed4-training-v1|sample_id` into eight
four-prompt construction units. Risk rows use the first 40 entries of
`torch.randperm(256, seed=4)`, with 32 assigned to construction and eight to
diagnostics. All identities are explicit in the protocol, and overlap is zero.

Every construction unit compares no auxiliary, full-seven raw MCD, and
RP-PFRSC on the same fixed diagnostic gradients and initial losses. Diagnostic
gradients never enter source screening, homotopy feasibility, or selection.

## 6. Prospective gates

All numeric gates are unchanged from seed-3 PFRSC except that the protected
set has four objectives and the target set has three.

### Integrity

- eight complete units, 127 subset endpoints, 16 screened endpoints, 129
  candidates, and 239 exact-map evaluations per unit;
- finite nonzero gradients, active clipping, and budget error at most `1e-6`;
- exact restoration after all 24 transient evaluations;
- at least 10% VRAM headroom;
- no optimizer, scheduler, checkpoint, generation, Core100, or heldout access.

### Protected finite-radius search

- selected construction target score at least `0.05` on every unit;
- selected minimum protection ratio at least `0.95` on every unit;
- target-bottleneck gain over raw mean at least `0.002`, bootstrap lower bound
  above zero, and exact two-sided sign-test `p<=0.05`;
- nonzero homotopy fraction on at least 75% of units.

### Diagnostic common descent

- overall and individual exact scores at least `0.02` on every unit;
- worst-score advantage over no auxiliary mean at least `0.02`, bootstrap
  lower bound above zero, and sign-test `p<=0.05`;
- every objective has positive mean actual reduction, at least 75% positive
  units, and mean actual/predicted ratio in `[0.25,1.75]`.

### Four-objective protected noninferiority

For risk, uniform, TARC, and continuation, both exact score and actual
reduction relative to raw MCD must have mean retention at least 95%, bootstrap
lower retention bound above 92.5%, and every unit retain at least 85%.

### Three-objective target advantage

- diagnostic E-TARC/branch bottleneck gain over raw mean at least `0.002`,
  bootstrap lower bound above zero, and sign-test `p<=0.05`;
- per-unit macro relative actual reduction over E-TARC, branch entropy, and
  branch negative-gap improves by at least 2% on average, with bootstrap lower
  bound above zero and sign-test `p<=0.05`;
- every target retains at least 95% on average and 80% on every unit;
- mean diagnostic target minimum at least `0.05`.

## 7. Stop boundary

The first failed gate selects invalid execution, protected-search failure,
diagnostic-transfer failure, finite-step failure, protected-noninferiority
failure, target-advantage failure, or weak effect. Failure closes the
FRSC/PFRSC homotopy line without another certificate-role or threshold scan.
A pass authorizes only a separately committed matched training protocol.

This is still exploratory development evidence on reused disclosed source
data. `optimizer_objects=0`, `scheduler_objects=0`, `optimizer_steps=0`,
`virtual_parameter_evaluations=24`, `checkpoint_writes=0`,
`training_authorized=false`, `candidate_generation_authorized=false`,
`core100_opened=false`, and `heldout_opened=false`.
