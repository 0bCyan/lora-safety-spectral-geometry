# BARP-MCD D1X-C1 Conditional Qualification Interpretation

Decision: `STOP_BARP_MCD_D1X_C1_ROUTE_ACTIVITY_FAILED`.

## Clean Prospective Failure

C1 bound the sole post-selected descriptor `mask072_lambda0.500` before any
qualification forward. The descriptor combines E-TARC and branch negative-gap
at homotopy fraction 0.5. It had been locally feasible on three of four D1X
discovery proposals, but it was feasible on zero of four still-sealed
qualification proposals.

The exact max-min anchor remained strictly feasible. Its minimum seven-score
margins were `0.10592257`, `0.10396029`, `0.09588690`, and `0.10331841`. The
candidate route differed from the anchor in every block, so inactivity was not
caused by route identity.

## Active Constraint

TARC was the unique protected blocker in all four qualification blocks:

```text
block              risk       uniform    TARC       continuation
qualification_01   1.021214   1.012235   0.608200   1.065845
qualification_02   1.018765   1.011093   0.666351   1.060106
qualification_03   1.029246   0.997046   0.556484   1.046953
qualification_04   1.030599   1.023717   0.581381   1.083800
```

Each number is the candidate's exact zero-state AdamW protected score divided
by the positive anchor score. Risk, uniform, and continuation were retained or
improved, while TARC lost between `33.36%` and `44.35%` relative to the anchor,
far beyond the frozen 5% allowance.

This is stronger than a weak-effect failure. The route did not enter the
qualified feasible set on any new proposal, so target transfer was never
eligible to be measured as an applied route effect. The conditional policy
correctly fell back to the anchor four times, yielding target Delta zero and
preserving the protected objectives vacuously.

## Mechanism Conclusion

D1X discovery suggested that an E-TARC plus negative-gap route might transfer
when guarded by the optimizer-space anchor. C1 falsifies that hypothesis on
the untouched qualification blocks: activity itself was split-specific. The
repeated TARC blocker identifies the mechanism more sharply than the earlier
D1R aggregate failure. Under the current certificate definitions and 95%
anchor-retention rule, target-directed BARP routing is not a viable training
method.

The optimizer-space max-min anchor remains a valid positive result. It proves
existence of strict local common descent across all eight seed-6 proposals,
but the additional target route has now failed both fixed-route cross-fitting
and single-route prospective qualification. Further threshold relaxation or
descriptor search on these qualification blocks would be post-selection and
is not justified. The current BARP target-route line should close.

## Boundary

Independent artifact verification passed. C1 completed in `788.18` seconds
with `20.76%` VRAM headroom. Optimizer steps, checkpoint writes, and generated
responses were zero. The eight-pair diagnostic, Core100, and heldout remained
sealed. No training is authorized.
