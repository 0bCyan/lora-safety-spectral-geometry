# FOACPO P0R Stop Analysis

P0R validly stopped at
`STOP_FOACPO_P0R_NO_ANCHOR_RELATIVE_PARETO_ROUTE`; mechanism-dev remained
sealed and no parameter changed.

The factorized behavior component was active. All three alpha routes improved
both calibration objectives over the optimizer anchor. The normalized
candidate-anchor displacement was about `0.068`; quality gain was
`0.00837–0.00988` cosine units and safety gain was `0.00837–0.00890`.

The failure was cross-block protection. Uniform, TARC, and continuation retained
`96%–97%` for every route. The calibration risk score of the train-only anchor
was already negative (`-0.00357293`). Every candidate worsened it, from
`-0.00018429` additional loss at alpha `0.25` to `-7.61e-7` at alpha `0.75`.
The frozen rule prohibited any worsening when the support anchor was
nonpositive, so no route qualified.

This distinguishes the two mechanisms: the independent quality/safety
preference estimator transfers, while a single-block max-min anchor does not
provide robust risk feasibility. A credible successor must construct the anchor
against both train and calibration protected blocks before selecting a fixed
route, and then use the still-sealed mechanism-dev block exactly once. It may
not round away the alpha `0.75` risk deficit or reclassify this stop.

`optimizer_steps=0`, `training_authorized=false`; Core100 and heldout remained
closed.
