# Step3 BARP-MCD: Batch-Adaptive Route-Protected Common Descent

## 1. Motivation

The frozen RUF-MCD-JRP v3r6 tangent has a valid 96-facet continuous and
realized-tensor certificate, but its fixed global direction fails calibration
TARC protection. The post-hoc analytic audit proves that no smaller angle on
that direction can satisfy even the relaxed fit gates and observed
calibration protection simultaneously. Its directional support audit also
finds two calibration TARC sign reversals outside the registered fit-only
convex hull.

This closes dose-only repair of the global direction. BARP-MCD tests a
different object: recompute the route-protected finite-family direction on
every optimizer batch, using the gradients that the update will act on. It
does not extrapolate a protected covector from one batch to another.

This P0 is zero-step mechanism development on disclosed data. It does not
claim independent confirmation or authorize training by itself.

## 2. Frozen optimizer unit

Each of eight disjoint optimizer units contains four risk rows and four
safety prompt pairs. All seven gradients are measured deterministically in
eval mode:

```text
I = {risk, uniform, tarc, etarc, continuation,
     branch_entropy, branch_negative_gap}.
```

The protected and target sets are

```text
P = {risk, uniform, tarc, continuation},
Q = {etarc, branch_entropy, branch_negative_gap}.
```

The raw baseline, auxiliary norm budget, clipping rule, first-step Adam map,
127 nonempty MCD subset endpoints, Top-16 endpoint screen, and nine-point
raw-anchored homotopy are unchanged from RP-PFRSC. Each unit evaluates 239
exact optimizer maps and contains 129 final homotopy candidates.

## 3. Batch-adaptive selection

Let `x_0` be the exact first-step displacement of full-seven raw MCD, and let
`H_b` be the finite homotopy family constructed from batch `b`. With
`rho=0.95`, define

```text
F_b = {x in H_b : s_b,p(x) >= rho s_b,p(x_0) for every p in P}.
```

BARP-MCD selects

```text
x_b* in argmax over x in F_b of min over q in Q s_b,q(x),
```

using the registered deterministic tie breakers. The raw anchor is explicitly
in every family.

### Theorem 1: finite-family batch protection

If all raw-anchor certificate scores are positive, then for every batch:

1. `F_b` is nonempty;
2. every selected protected score retains at least `rho=0.95` of raw MCD;
3. the selected three-target bottleneck is no worse than raw MCD within the
   registered finite family; and
4. all seven selected first-order reductions are positive.

### Proof

The raw anchor belongs to `H_b` and has protected retention one, so it belongs
to `F_b`. This proves nonemptiness. Membership of the selected point gives
the four protected inequalities. Maximization over a set containing the raw
anchor gives target-bottleneck dominance. Positivity of the protected raw
scores, `rho>0`, and positivity of the raw target bottleneck make all seven
selected scores positive. QED.

The theorem is exact for the current batch and finite family. It does not
claim finite-step loss dominance, future-batch protection, or generated
behavior improvement.

## 4. P0 finite-step test

For each optimizer unit, P0 transiently applies three exact displacements at
learning rate `2e-4`:

1. risk/uniform baseline without an auxiliary direction;
2. full-seven raw MCD; and
3. batch-adaptive BARP-MCD.

Losses are reevaluated on the same unit in eval mode, then the initial LoRA
state is restored exactly. Same-unit evaluation is intentional: P0 tests the
optimizer's local certificate and curvature realization. It is not presented
as a disjoint transfer experiment.

## 5. Frozen gates

The inherited RP-PFRSC numeric and effect thresholds remain unchanged.

- all eight finite-family searches satisfy the 95% protected score floor;
- construction target gain has mean at least `0.002`, positive bootstrap
  lower bound, exact sign-test `p<=0.05`, and at least 75% nonzero choices;
- every actual objective has positive mean reduction, at least 75% positive
  units, and mean actual/predicted ratio in `[0.25,1.75]`;
- protected exact and actual relative changes pass the 95% mean, 92.5%
  bootstrap-lower, and 85% single-unit standards;
- the three-target exact bottleneck gain and finite-loss macro gain pass the
  unchanged significance and effect floors; and
- the mean selected target minimum remains at least `0.05`.

The first failed gate fixes the route. A pass authorizes only a separately
frozen matched-training protocol in which BARP-MCD is recomputed on every
training batch. It does not authorize an optimizer step inside P0.

## 6. Evidence boundary

The eight units and seed-4 source rows were disclosed by prior experiments,
so this run is a prospective algorithm execution but not fresh confirmation.
The excluded eight-category source block, Core100, and heldout remain outside
the run. No optimizer or scheduler is constructed, no checkpoint or response
is generated, and the required boundary remains:

```text
optimizer_steps=0
training_authorized=false
core100_opened=false
heldout_opened=false
```
