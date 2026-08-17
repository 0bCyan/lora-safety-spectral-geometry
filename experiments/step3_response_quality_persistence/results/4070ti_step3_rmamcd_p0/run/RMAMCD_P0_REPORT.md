# NPT-MCD Seed-7 Zero-Step P0 Report

Route: `STOP_RMAMCD_P0_NO_MINIMUM_EFFECTIVE_ANGLE`.

Forty-eight unit-level protected fit gradients defined the projection
nullspace. The retained target-proposal norm fraction was `0.933645`.
Twelve one-pair eval-mode fit units selected the leftmost effective angular
dose, `theta=0.31434110` radians. Its minimum fit
protected retention was `0.6704081920508697`
and its fit target-floor gain over raw was
`0.063415`.

The candidate was fixed before all 20 calibration units. The registered
accept-or-raw policy uses the candidate only when the joint split-conformal
threshold is nonpositive and otherwise falls back to raw MCD. Its rank was
`20`, its policy-level finite-sample marginal
protection lower bound was
`0.952381`, and
the observed threshold was `-0.050000`. This guarantee is
for the complete policy over calibration and future draws; no coverage
conditional on candidate acceptance is claimed.

Across eight independent diagnostics, the mean candidate target minimum was
`0.055800` and
the macro relative finite-loss target gain over raw was `0.000000`.

All 24 finite-loss measurements were transient fixed-displacement applications
followed by exact state restoration. No optimizer or scheduler was created,
`optimizer_steps=0`, and no checkpoint, generated candidate, Core100 row, or
heldout row was opened. Even a pass authorizes only a separately frozen
`proxy_dev` triplet-data protocol, never training.
