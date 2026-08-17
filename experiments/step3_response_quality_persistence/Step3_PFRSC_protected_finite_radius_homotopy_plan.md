# Step3 PFRSC-MCD: Protected Finite-Radius Homotopy Common Descent

## 1. Registered precursor and falsified claim

Seed-2 FRSC-MCD stopped at
`STOP_FRSC_P0_RAW_COMPARATOR_FAILED`. Its exact finite-radius search gain was
positive on all eight construction units, and every registered diagnostic
loss decreased on all eight transient virtual steps. The stronger comparator
claim failed: relative to full-seven raw MCD, unconstrained FRSC improved
E-TARC and both branch objectives but reduced risk, uniform, TARC, and
continuation finite-step loss reduction. Only `3/7` objectives met the frozen
noninferiority criterion.

The selected subsets never contained risk, uniform, or continuation. This is
evidence of a scalar max-min allocation failure: once those scores ceased to
be the minimum, they could be traded for route/branch score without penalty.
PFRSC-MCD registers a seed-3 test of a bounded Pareto trade before any new
model forward. It does not alter or relabel the failed seed-2 route.

## 2. Exact optimizer map and certificates

The seven normalized optimizer-unit certificate gradients remain

\[
u_i=\frac{g_i}{\lVert g_i\rVert_2},
\qquad
\mathcal I=\{R,U,T,E,C,H,G\}.
\]

`R` is risk CE, `U` is uniform safety CE, `T` is TARC route margin, `E` is
complete-path E-TARC, `C` is post-route continuation CE, `H` is allowed-branch
entropy, and `G` is the negative top1-top2 allowed-branch gap. The baseline is

\[
G_0=0.8g_R+0.2g_U,
\qquad B=\lVert g_U\rVert_2.
\]

For auxiliary direction `d` with norm `B`, PFRSC evaluates the registered
clipping-plus-first-step-Adam displacement exactly:

\[
\Phi_j(G)=
\frac{cG_j}{c|G_j|+\epsilon(\lVert G\rVert_2+\delta)},
\]

with `c=1`, `delta=1e-6`, and Adam epsilon `epsilon=1e-8`. Its normalized
certificate score is

\[
s_i(d)=
\frac{u_i^T\Phi(G_0+0.2d)}{\lVert\Phi(G_0)\rVert_2}.
\]

There is no zero-radius linearization.

## 3. Raw anchor and screened subset endpoints

For every nonempty subset `S` of the seven certificates, the existing exact
simplex active-set solver constructs its equal-budget Euclidean MCD direction
`d_S`. The full-set direction

\[
d_0=d_{\mathcal I}
\]

is the raw-MCD anchor.

PFRSC evaluates all 126 non-anchor subset endpoints under the exact optimizer
map. Endpoints are ranked by

\[
q(d_S)=\min_{i\in\mathcal Q}s_i(d_S),
\qquad
\mathcal Q=\{T,E,H,G\},
\]

then by the overall seven-way minimum, target mean, smaller subset, and lower
bitmask. Exactly the top `K=16` endpoints are retained. This screening rule
and `K` are frozen before the seed-3 forward pass.

## 4. Raw-anchored finite homotopy

For each retained endpoint and each positive member of

\[
\Lambda=\{0,1/8,2/8,\ldots,1\},
\]

define the renormalized homotopy direction

\[
d_{S,\lambda}=B
\frac{(1-\lambda)d_0+\lambda d_S}
{\lVert(1-\lambda)d_0+\lambda d_S\rVert_2}.
\]

The registered candidate family is

\[
\mathcal H=\{d_0\}\cup
\{d_{S,\lambda}:S\in\operatorname{Top16},\lambda\in\Lambda\setminus\{0\}\}.
\]

It contains exactly `1+16*8=129` construction candidates. Endpoint scores
are reused, so each optimizer unit performs `127+16*7=239` exact-map
evaluations. A degenerate interpolation is an integrity failure, not a reason
to change the grid.

## 5. Protected selection and theorem

The protected objectives are fixed as

\[
\mathcal P=\{R,U,C\}.
\]

With retention floor `rho=0.95`, define

\[
\mathcal H_\rho=
\{d\in\mathcal H:s_i(d)\ge\rho s_i(d_0)\ \forall i\in\mathcal P\}.
\]

PFRSC returns

\[
d^*\in\arg\max_{d\in\mathcal H_\rho}q(d).
\]

Ties use the overall seven-way minimum, target mean, protected retention,
smaller homotopy fraction, smaller source subset, and lower bitmask.

### Theorem 1: finite-family protected target dominance

Assume every raw-anchor score is positive. Then:

1. `H_rho` is nonempty;
2. `s_i(d*) >= rho s_i(d0)` for every protected objective;
3. `q(d*) >= q(d0)`;
4. `d*` is strict simultaneous first-order descent for all seven registered
   losses under the exact optimizer displacement.

### Proof

The raw anchor `d0` is explicitly in `H`. It satisfies every protected
constraint with equality ratio one, so it belongs to `H_rho`. This proves
nonemptiness. Statement 2 follows from membership of `d*` in `H_rho`.
Because `d*` maximizes `q` over a finite set containing `d0`, statement 3
follows. Positive raw protected scores and `rho>0` make every protected score
at `d*` positive. Positive raw target scores imply `q(d0)>0`; statement 3
therefore makes every target score at `d*` positive. Since

\[
g_i^T\Phi(G_0+0.2d^*)
=\lVert g_i\rVert_2\lVert\Phi(G_0)\rVert_2s_i(d^*),
\]

all seven first-order reductions are strictly positive.

### Corollary 1: protected predicted-reduction ratio

At the same learning rate and diagnostic gradient norm, the exact
first-order predicted reduction for every protected construction objective is
at least `rho=0.95` of raw MCD. This is a coordinate-wise guarantee for the
three protected objectives, not only a guarantee on one scalar minimum.

## 6. Finite-step boundary

The theorem concerns exact first-order scores on construction units. It does
not guarantee prompt-disjoint transfer or finite-loss noninferiority. For a
locally beta-smooth objective and displacement `p`, the descent lemma gives

\[
L_i(\theta-\eta p)\le L_i(\theta)
-\eta g_i^Tp+\frac{\beta_i\eta^2}{2}\lVert p\rVert_2^2.
\]

P0 therefore applies the exact displacement transiently at `eta=2e-4`,
measures actual losses, and restores parameters byte-exactly. Actual/predicted
ratios and comparator-relative finite-loss changes are independent empirical
gates rather than consequences asserted from Theorem 1.

## 7. Seed-3 split

The LoRA initialization, train-mode stochastic forwards, and risk permutation
use seed 3. The 40 disclosed safety prompt clusters are partitioned before
model forward into eight four-pair construction units and one fixed
eight-category diagnostic unit. One diagnostic prompt per category is chosen
by the lexicographically smallest SHA256 of
`pfrsc-seed3-diagnostic-v1|category|sample_id`; the remaining IDs are ordered
by SHA256 of `pfrsc-seed3-training-v1|sample_id`. Risk rows use the first 40
entries of `torch.randperm(256, seed=3)`. All IDs are serialized in the
protocol, and construction-diagnostic overlap is zero.

Every construction unit compares three transient arms on the same diagnostic
gradients and losses: no auxiliary, raw full-seven MCD, and PFRSC-MCD.
Diagnostic gradients never enter endpoint screening, homotopy feasibility, or
selection.

## 8. Prospective gates

### Integrity

- eight complete construction units, 127 subset endpoints, 16 screened
  endpoints, 129 homotopy candidates, and 239 exact-map evaluations per unit;
- finite nonzero gradients, active clipping, equal budgets within `1e-6`;
- exact restoration after all 24 transient arm evaluations;
- at least 10% VRAM headroom;
- no optimizer, scheduler, checkpoint, generation, Core100, or heldout access.

### Protected finite-radius search

- selected construction target score at least `0.05` on every unit;
- selected protected retention at least `0.95` on every unit;
- target-bottleneck gain over raw mean at least `0.002`, bootstrap lower bound
  above zero, and exact two-sided sign-test `p<=0.05`;
- nonzero homotopy fraction on at least 75% of units.

### Disjoint common transfer and finite-step realization

- diagnostic overall and individual exact scores at least `0.02` on every
  unit;
- worst-score advantage over no auxiliary mean at least `0.02`, bootstrap
  lower bound above zero, and sign-test `p<=0.05`;
- every objective has positive mean actual reduction, at least 75% positive
  units, and mean actual/predicted ratio in `[0.25,1.75]`.

### Protected diagnostic noninferiority

For risk, uniform, and continuation, both exact score and actual reduction
relative to raw MCD must satisfy:

- mean retention at least 95%;
- bootstrap lower retention bound above 92.5%;
- every individual unit retains at least 85%.

### Route/branch advantage and materiality

- diagnostic target-bottleneck gain over raw mean at least `0.002`, bootstrap
  lower bound above zero, and sign-test `p<=0.05`;
- per-unit macro-average relative actual reduction across TARC, E-TARC,
  branch entropy, and branch negative-gap improves by at least 2% on average,
  with bootstrap lower bound above zero and sign-test `p<=0.05`;
- every target objective retains at least 95% on average and 80% on every
  unit;
- PFRSC mean diagnostic target minimum is at least `0.05`.

## 9. Route and evidence boundary

The first failed gate selects invalid execution, protected-search failure,
diagnostic-transfer failure, finite-step failure, protected-noninferiority
failure, target-advantage failure, or weak effect. A pass authorizes only a
separately frozen matched training protocol. It does not authorize training
inside P0.

PFRSC does not claim global constrained optimality. The exact theorem is over
the registered screened finite family, and endpoint screening uses the same
construction units. Seed 3 is a new stochastic replication but the disclosed
source data are reused, so this remains development evidence rather than
external confirmation. Novelty relative to MGDA, CAGrad, PCGrad, and
optimizer-aware multi-objective methods requires a separate literature audit
and downstream behavior evidence.

`optimizer_objects=0`, `scheduler_objects=0`, `optimizer_steps=0`,
`virtual_parameter_evaluations=24`, `checkpoint_writes=0`,
`training_authorized=false`, `candidate_generation_authorized=false`,
`core100_opened=false`, and `heldout_opened=false`.
