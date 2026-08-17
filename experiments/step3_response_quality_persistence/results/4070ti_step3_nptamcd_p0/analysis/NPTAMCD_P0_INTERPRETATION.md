# NPT-A-MCD Seed-7 P0 Interpretation

## Decision

The prospective run at protocol commit
`e0f0562b62edf8ccaa4c64943924e98a40d62223` returned:

`STOP_NPTAMCD_P0_CONFORMAL_SELECTIVE_POLICY_FAILED`

The registered action is `raw_mcd`. The affine candidate is not authorized for
proxy-dev work, training, generation, or deployment.

## Fit geometry and solver audit

The method solved the intended fit problem rather than repeating the preceding
NPT-MCD failure:

| quantity | value |
| --- | ---: |
| fixed angle | 0.3143411045 rad |
| fit protected retention minimum | 0.9510003311 |
| fit retention identity error maximum | 2.7600e-8 |
| raw fit target floor | 0.0122671819 |
| affine candidate fit target floor | 0.1568337773 |
| fit target-floor gain | 0.1445665954 |

The shared 49-constraint projector had numerical rank 49 and maximum relative
target-projection residual `4.8471e-9`. The tangent/raw norm ratio was
`0.9999999639`, the raw/tangent cosine was `-1.1742e-10`, and the maximum
protected cosine was `1.4257e-9`.

The 36-variable affine simplex dual used 12 active target covectors. Its
primal-dual gap and KKT residual were both `9.1561e-9`; the independently
recomputed candidate minimum differed from the solver prediction by
`2.7951e-9`. These numerical errors are far below the observed fit effect and
cannot explain the downstream failure.

## Calibration failure

Only 3 of 20 calibration units met the registered joint 95% protection rule.
The conformal threshold was `2.8914845941`, so the complete policy correctly
fell back to raw MCD. The minimum calibration protected retention was
`-1.9414845941`, and the mean minimum retention was `0.5990502970`.

The binding example was `calibration_12 / tarc`: its raw score was
`0.0094674088`, while the affine candidate score was `-0.0183808283`. Other
low-retention calibration cases were also TARC facets. This is not a numerical
projection leak on the fit constraints; it is failure of the 12-unit
protected subspace to transfer to unseen unit gradients, amplified by small
raw TARC margins.

## Independent diagnostic evidence

The candidate had a strong and consistent target effect on all eight
diagnostic units:

| paired diagnostic | estimate | 95% bootstrap CI | sign p |
| --- | ---: | ---: | ---: |
| exact target-minimum gain over raw | 0.0819396 | [0.0698187, 0.0933505] | 0.0078125 |
| finite-loss target macro relative gain | 0.9199260 | [0.7705819, 1.0896214] | 0.0078125 |

The minimum diagnostic candidate target score was `0.1052498`, versus
`0.0315178` for raw MCD. Thus the candidate is not weak.

Protection did not transfer with the target effect. Diagnostic TARC exact
scores had mean relative change `-0.0462623`, but the lower confidence bound
was `-0.1494074` and the worst unit was `-0.2721654`. TARC finite-loss
reduction had mean relative change `-0.1729387`. The registered protected
noninferiority gate therefore failed as expected.

## Mechanism conclusion

NPT-A-MCD separates two questions that the earlier scalar path confounded:

1. A high-dimensional protected-null direction with a large common target
   effect exists on the fit units.
2. The fit protected-nullspace is not a transferable protection certificate.

The next justified method should not add more target optimization or relax the
95% rule. It must control out-of-fit protected sensitivity, especially TARC.
A prospective follow-up should estimate a transferable protected metric from
fit-unit variation (for example a cross-fitted diagonal/Mahalanobis sensitivity
bound) and optimize the target tangent under that bound before the independent
20-unit conformal check. Calibration rows from this run may inform the new
exploratory design but may not be reused as its acceptance set.

## Boundary

`verify-run` passed with eight frozen artifacts. The run used zero optimizer
objects, zero scheduler objects, and `optimizer_steps=0`; it restored the
trainable state, wrote no checkpoint, generated no candidate text, and opened
neither Core100 nor heldout data.

The inherited shared evaluator initially wrote its base-runner name into the
report and reproduction manifest. Those two metadata fields were corrected to
the affine entry point after the run; artifact hashes, measurements, gates, and
route were unchanged, and `verify-run` was rerun successfully.
