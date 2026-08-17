# NPT-LR-MCD Seed-7 P0 Interpretation

## Decision

The frozen low-rank attempt returned:

`STOP_NPTLRMCD_P0_CONFORMAL_SELECTIVE_POLICY_FAILED`

The registered policy action is `raw_mcd`. No candidate, training, or
proxy-dev protocol is authorized.

## Low-rank geometry

The method reduced target selection from 36 unit-target covectors to three
equal-unit pooled target covectors. The protected/raw constraint Gram retained
rank 49. The maximum relative projection residual was `4.3262e-9`, the maximum
protected tangent cosine was `2.9306e-9`, and the tangent/raw norm ratio was
`1.0000000396`.

The affine simplex solution had two active pooled targets. Its KKT residual was
`1.0531e-8` and its primal-dual gap was `9.9206e-9`, both below the registered
`1e-7` limit.

The fixed-angle fit candidate passed the prospective fit gates:

| quantity | value |
| --- | ---: |
| fit protected retention minimum | 0.9510003214 |
| fit retention identity error maximum | 2.3682e-8 |
| raw fit target floor | 0.0122671819 |
| candidate fit target floor | 0.0841396533 |
| fit target-floor gain | 0.0718724714 |

Thus low-rank pooling weakened the fit effect relative to NPT-A-MCD but still
cleared the same non-weak target gates.

## Independent protection failure

Only 2 of 20 calibration units met joint 95% protected retention. The conformal
threshold was `4.0784448303`, the mean minimum retention was `0.4211130392`,
and three units had negative retention.

The worst case was again `calibration_12 / tarc`: raw TARC score
`0.0094674088` became `-0.0296182660`, for retention `-3.1284448303`.
`calibration_19` and `calibration_20` also had negative TARC retention. The
repeated TARC bottleneck under both 36-covector and three-covector target
searches rules out target-search dimension as the primary explanation.

## Target effect and finite-step boundary

The candidate retained a large target effect on all eight diagnostics:

| paired diagnostic | estimate | 95% bootstrap CI |
| --- | ---: | ---: |
| exact target-minimum gain over raw | 0.1256442 | [0.0940289, 0.1557600] |
| finite-loss target macro relative gain | 1.6179358 | [1.3273876, 1.9636437] |

Protected TARC transfer was worse than in NPT-A-MCD: its exact relative score
change averaged `-0.1399114`, and finite-loss reduction averaged `-0.3223421`.
The protected noninferiority gate therefore failed.

All seven finite-step objectives still had positive reductions on all eight
units. However, branch-entropy's mean actual-to-predicted ratio was `0.1926076`,
below the registered `0.25` lower bound, so finite-step common descent also
failed. This curvature mismatch is secondary because the conformal stop occurs
earlier.

## Mechanism conclusion

The evidence now separates target capacity from protection transfer:

1. Both high-rank and low-rank protected-null target searches produce strong,
   replicated target effects.
2. Exact orthogonality to all 48 fit protected covectors does not predict
   orthogonality to unseen protected covectors.
3. TARC is the stable transfer bottleneck, especially when raw TARC margins
   are small.

Further target-direction variants are not justified until the method models
out-of-fit protected sensitivity. The next candidate should introduce a
prospectively cross-fitted or diagonal/Mahalanobis protected metric with an
explicit future-gradient bound, then retain the same independent conformal
gate. Merely shrinking the target basis, changing the angle grid, or relaxing
95% protection would not address the observed mechanism.

## Boundary

`verify-run` passed with eight artifacts, `optimizer_steps=0`, exact trainable
state restoration, no checkpoint or generated text, and no Core100 or heldout
access.

The shared evaluator's inherited report title and manifest command were
corrected to the low-rank entry point after the run. No measurement, gate, or
route changed; artifact hashes were updated and `verify-run` was rerun.
