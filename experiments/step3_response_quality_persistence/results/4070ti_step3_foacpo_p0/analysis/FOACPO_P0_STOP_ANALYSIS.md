# FOACPO P0 Stop Analysis

P0 is a valid `STOP_FOACPO_P0_NO_CALIBRATION_ROUTE`. All state-integrity,
gradient-coverage, and zero-optimizer checks passed. Mechanism-dev remained
sealed.

Every registered alpha produced positive calibration descent for both
factorized objectives. Quality cosine ranged from `0.01767552` to `0.01917736`;
safety cosine ranged from `0.02847646` to `0.02899998`. Every candidate exactly
retained the train protected floor of `0.95`, with `risk` limiting. The sole
failed eligibility condition was the frozen behavior-coordinate floor:
`lambda` ranged from `0.04888948` to `0.04914974`, below `0.05`.

The result cannot be reclassified by rounding or by changing the threshold.
It does, however, identify a mathematical defect in that threshold. The scalar
`lambda` is a coordinate on an anchor-to-endpoint segment, not an invariant
effect size. Reparameterizing the same ray changes the coordinate assigned to
an unchanged physical displacement; `lambda` alone also ignores endpoint
separation and observed objective gain.

A successor may therefore retain the exact 95% protection floor while replacing
the coordinate threshold with an attribution test: the candidate must differ
from the anchor and strictly improve both factorized calibration objectives
relative to that anchor. This is a stronger empirical mechanism requirement,
not a post-hoc lowering of the failed threshold. Its rule must be committed
before recomputing anchor-relative scores or opening mechanism-dev.

No optimizer step, checkpoint write, response generation, Core100 access, or
heldout access occurred. `optimizer_steps=0` and `training_authorized=false`.
