# Step3 BARP-MCD D1R: Optimizer-Anchor Matched Confirmation

## Objective

D1F proved that the D1 failure state contained many strict common-descent
candidates after exact current-state AdamW scoring. D1R tests whether this
optimizer-space anchor construction remains feasible for eight updates and
whether route-protected target selection improves the frozen paired diagnostic
relative to the repaired anchor itself.

## Matched arms

Both arms start from identical seed-5 LoRA state and use the frozen D1 blocks:

- `raw_mcd`: at every step apply the optimizer-space max-min anchor;
- `barp_mcd`: construct the same anchor, then select the candidate with maximum
  target bottleneck subject to retaining at least 95% of every positive anchor
  score for risk, uniform, TARC, and continuation.

The anchor is selected from the D1F family: all 127 subset endpoints plus seven
interiors for each of the top-16 max-min endpoints, for 239 exact AdamW
candidates. The anchor itself is included in the protected family, proving
non-emptiness whenever its minimum score is positive.

Both arms receive eight updates. They share data order, initialization,
optimizer, clipping, and objective code. Exactly one final external LoRA
adapter is written per arm.

## Gates

D1R retains every D1 optimizer-map, finite-step, VRAM, paired diagnostic, and
checkpoint gate without relaxation. In addition, every per-step anchor must
have minimum exact score above `1e-8`, and the treatment must choose a candidate
different from its anchor on at least six of eight steps.

The eight category-paired diagnostic uses the same normalized BARP advantage,
10,000 paired bootstrap resamples, exact sign test, and protected-disadvantage
thresholds frozen in D1.

## Boundary

A pass authorizes only a separately frozen, checkpoint-hash-bound generated
safety and utility evaluation. It does not authorize Core100, heldout,
additional training, or a paper-level claim.
