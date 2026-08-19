# Step3 BARP-MCD D1F: Optimizer-Update-Space Anchor Feasibility

## Objective

D1 stopped before seed-5 raw update four because the full gradient-space MCD
anchor had a non-positive exact score after the current AdamW map. D1F tests a
single mechanistic question: does the same fourth batch contain a strict common
descent anchor in the registered finite direction family when candidates are
ranked directly in optimizer update space?

This is a failure-state reconstruction and zero-update feasibility test at the
target state. It is not a training comparison or an effect claim.

## State reconstruction

D1F starts from the same seed-5 model and replays exactly the first three raw
updates on the same blocks. Each replayed update must match the frozen D1 trace
within `1e-7`. Exactly three optimizer steps are authorized. The fourth block
is measured but never applied.

## Finite anchor family

At the reconstructed fourth state, let `U_t(g)` be the exact positive AdamW
displacement under the current moments and clipping rule. For each of the 127
nonempty certificate subsets, D1F constructs the same minimum-norm simplex
direction used by MCD and scores

```text
s_j(g) = <c_hat_j, U_t(g)> / ||U_t(g_baseline)||.
```

All 127 endpoints are evaluated. The 16 endpoints with the largest
`min_j s_j` are screened, and seven fixed interior homotopies from the full-MCD
direction are evaluated for each. The complete registered family therefore has
`127 + 16 * 7 = 239` exact current-state AdamW candidates.

The optimizer-space anchor is

```text
g_anchor = argmax_g min_j s_j(g)
```

with target minimum, target mean, smaller homotopy fraction, smaller subset,
and lower bitmask as deterministic tie breakers.

## Certificate and route

If the selected margin is strictly greater than `1e-8`, every registered
first-order objective decreases for a sufficiently small positive learning
rate under the actual optimizer displacement. Because the selected anchor can
be included in any later protected family, it also supplies the missing
non-emptiness witness for relative-retention constraints.

D1F passes only if the frozen D1 full-anchor failure is reproduced, all three
replay updates match, all 239 candidates are evaluated, the selected margin is
above `1e-8`, no fourth update occurs, and no checkpoint or response is
written. A pass authorizes only a separately frozen matched-training retry
whose anchor is selected by this rule.
