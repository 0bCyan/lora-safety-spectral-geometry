# JCR-MCD Seed-5 Zero-Step P0 Report

Route: `STOP_JCRMCD_P0_CONSTRUCTION_UNCERTAINTY_INVALID`.

Four disjoint eight-pair construction supports calibrated seven jackknife
convex-hull radii without diagnostic gradients. The fixed 33-point path chose
`lambda=0.00000`. Its robust target
minimum was `-0.707420`
and its robust gain over the raw anchor was
`0.000000`.

The independent supported-domain panel covered
`6/56`
certificate vectors under the prospectively calibrated uncertainty sets. The
minimum observed-minus-certified protected residual slack was
`0.006867`. Coverage is a
hard gate; favorable outcomes outside the set do not repair a coverage failure.

Across eight independent diagnostic units, JCR-MCD's mean target minimum was
`0.046149`. The
macro relative finite-loss target gain over raw MCD was `0.000000`.

All 24 finite-loss measurements used transient direct applications of fixed
first-step displacements followed by exact state restoration. No optimizer or
scheduler was created, `optimizer_steps=0`, and no checkpoint, generated
candidate, Core100 row, or heldout row was opened. Even a pass authorizes only
a separately frozen `proxy_dev` triplet-data protocol, never training.
