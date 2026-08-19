# Step3 BARP-MCD D1: Seed-5 Matched Eight-Step Confirmation

## 1. Objective

Q1R established exact stateful AdamW execution for two consecutive updates on
disclosed seed-4 blocks. D1 tests whether the same registered method remains
active and protected across eight updates under a prospectively generated
seed-5 split. D1 also freezes final external adapters for later generated
behavior evaluation.

D1 is a fresh-seed algorithm confirmation over the existing approved source,
not a fresh-dataset confirmation. No Core100, heldout, or generated response is
opened in D1.

## 2. Prospective split

The 40 approved safety prompt pairs cover eight categories with frozen,
unequal prompt counts (`23/4/4/3/3/1/1/1`). For every category, D1 selects the
lexicographically smallest

```text
sha256("barp-d1-seed5-diagnostic|" + sample_id)
```

as the diagnostic pair. The remaining 32 pairs are sorted by

```text
sha256("barp-d1-seed5-training|" + sample_id)
```

and partitioned into eight ordered blocks of four. Risk rows use
`torch.randperm(256, generator=seed5)`; positions 0-31 form the eight training
blocks and positions 32-39 form the paired diagnostic block.

This split is frozen before any seed-5 model forward. The three singleton
categories therefore occur only in the paired diagnostic set; D1 does not
claim balanced-category training or fresh-dataset generalization.

## 3. Matched arms

- `raw_mcd`: full-seven raw MCD recomputed at every optimizer batch;
- `barp_mcd`: memory-bounded BARP finite-family search recomputed at every
  optimizer batch.

Both fresh models use identical seed-5 initialization, block order, AdamW
configuration, clipping, eval-mode gradient measurement, and objective code.
Each arm receives exactly eight updates, for 16 optimizer steps total.

The current-state AdamW map, 127 subset endpoints, Top-16 screen, 129 final
homotopy candidates, 95% protected floor, target bottleneck, and tie breakers
are unchanged from Q1R.

## 4. Diagnostic statistics

The eight category-paired diagnostic units are evaluated before and after
training. For objective `j` and unit `u`, define normalized BARP advantage

```text
A[u,j] = ((L0[u,j] - Lbarp[u,j]) - (L0[u,j] - Lraw[u,j]))
         / max(abs(L0[u,j]), 1e-8).
```

The target statistic is the unit-wise macro mean over E-TARC, branch entropy,
and branch negative-gap. The protected disadvantage is `-A[u,p]` for risk,
uniform, TARC, and continuation. D1 freezes 10,000 paired bootstrap resamples
and a two-sided exact sign test excluding exact ties.

## 5. Gates

D1 passes only if:

1. both arms execute eight matched finite updates from the same initial state;
2. every analytic current-state AdamW displacement matches the real update;
3. every raw anchor has strict common descent;
4. every BARP step retains at least 95% on all protected exact scores and has
   target bottleneck no worse than its current raw anchor;
5. at least six of eight BARP steps select a nonzero homotopy candidate;
6. every registered same-batch loss decreases after every selected update;
7. device headroom remains at least 10%;
8. each protected diagnostic has mean disadvantage at most 5%, bootstrap upper
   at most 7.5%, and worst-unit disadvantage at most 15%;
9. target diagnostic macro advantage has mean at least 0.2%, bootstrap lower
   above zero, and two-sided exact sign-test `p <= 0.05`; and
10. exactly one adapter checkpoint per arm is frozen outside the repository,
    inventoried by SHA-256, with no optimizer state or base-model weight.

The first failed gate fixes the route. Thresholds cannot be changed after any
seed-5 model forward.

## 6. Boundary

A pass authorizes only a separately frozen, checkpoint-hash-bound generated
safety and utility evaluation protocol. It does not authorize direct Core100
or heldout access, additional optimizer steps, or a paper-level claim.
