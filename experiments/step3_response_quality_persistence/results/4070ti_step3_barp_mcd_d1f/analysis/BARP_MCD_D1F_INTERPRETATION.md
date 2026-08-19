# BARP-MCD D1F Mechanism Interpretation

Decision: `PASS_BARP_MCD_D1F_UPDATE_SPACE_ANCHOR_FEASIBLE`.

## Result

D1F exactly reproduced the three frozen D1 raw updates, including all compared
coverage, search, optimizer-map, and finite-step fields. At the reconstructed
fourth state, the old full-MCD anchor failed only the
`branch_negative_gap` certificate:

```text
risk                 0.20244122
uniform              0.23662084
tarc                 0.10910363
etarc                 0.17355416
continuation         0.20860641
branch_entropy       0.02733582
branch_negative_gap -0.00225851
```

The prospectively frozen update-space max-min rule evaluated all 239 candidates
and found 160 with every score above `1e-8`. Its selected candidate was the
single-certificate `branch_negative_gap` endpoint (mask 64), with exact scores

```text
risk                 0.21345470
uniform              0.20437283
tarc                 0.04260606
etarc                 0.14333172
continuation         0.19205189
branch_entropy       0.06250026
branch_negative_gap  0.04308808
```

Thus the minimum margin changed from `-0.00225851` to `0.04260606`. The rule
does not increase every already-positive score; it deliberately gives up TARC,
uniform, E-TARC, and continuation surplus to repair the actual bottleneck.
That is the expected behavior of a max-min anchor rather than a post-hoc story.

## Certificate

Fix model parameters, the seven loss gradients `c_j`, and AdamW state
`(m_{t-1}, v_{t-1}, t)`. Let `U_t(g)` be the exact positive displacement after
the frozen clipping and AdamW map, excluding the scalar learning rate, and let

```text
s_j(g) = <c_j / ||c_j||, U_t(g)> / ||U_t(g_baseline)||.
gamma_t = max_{g in F_t} min_j s_j(g).
```

For the selected D1F candidate, `gamma_t = 0.0426060566 > 1e-8`. Therefore
`<c_j, U_t(g)> > 0` for every registered loss. If each loss has a locally
Lipschitz gradient with constant `M_j`, the descent lemma gives

```text
L_j(theta - eta U)
  <= L_j(theta)
     - eta ||c_j|| ||U_ref|| gamma_t
     + (M_j / 2) eta^2 ||U||^2.
```

Hence all seven losses strictly decrease for any positive `eta` below the
minimum of the corresponding seven sufficient bounds. This is a first-order
local certificate; D1F intentionally did not execute the fourth update, so a
later matched retry must still verify the actual finite-step losses.

The result also repairs the non-emptiness argument used by route protection.
If this max-min anchor is included in the later candidate family and protected
scores are measured relative to it, the anchor itself has retention 1 and
target gain 0. The constrained family is therefore nonempty whenever
`gamma_t > 0`.

## What Changed

The gradient directions, AdamW state, clipping, seven certificates, subset
solver, homotopy fractions, and candidate count did not change. The old method
declared the full seven-certificate MCD direction to be the mandatory anchor
before exact optimizer scoring. D1F instead evaluates the same finite geometry
through the current AdamW map and chooses the candidate with maximum worst
exact score.

The finding is stronger than merely observing that the old anchor failed: at
this failure state, two thirds of the registered family (`160/239`) remained
strictly feasible. The issue was anchor selection, not exhaustion of the
candidate geometry.

## Limits and Next Test

This is one prospectively reconstructed state on an existing, category-
imbalanced source. It is not evidence of eight-step superiority or generated
behavior improvement, and it does not claim global optimality outside the 239
candidates. Exactly three replay updates were performed; the fourth update,
checkpoints, generation, Core100, and heldout all remained untouched.

The next justified experiment is a separately frozen seed-5 matched retry in
which both arms run the same eight blocks, while the treatment first constructs
this optimizer-space max-min anchor and then applies the existing protected
target selection relative to that anchor. It must retain the current D1
diagnostic and finite-step gates; no threshold relaxation is supported.
