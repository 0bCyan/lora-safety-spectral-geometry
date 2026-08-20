# CDR-OPC M4 Holdback Interpretation

## Decision

M4 passed its frozen holdback audit route:

```text
PASS_CDRO_OPC_M4_HOLDBACK_AUDIT
```

The fixed dose was `0.0337860296353059`, equal to `0.875` times the M3 dose.
All 20 review-qualified holdback units satisfied the joint four-certificate
4% protection bound. This is a conservative fixed-dose mechanism result, not
a new conformal confirmation.

## Evidence

- Protected coverage: `20/20`.
- Maximum standardized-drop residual: `-0.00561586459677198`.
- Mean risk deterioration: `0.0006075836980057982`.
- Mean uniform deterioration: `-0.13995100840584485`.
- Mean TARC deterioration: `-0.003555822216470522`.
- Mean continuation deterioration: `-0.1373300300961905`.
- Quality selected-minus-fallback: `0.002271612062323812`.
- Safety selected-minus-fallback: `0.003409262045862646`.

The fixed M1 direction replay remained valid: both coordinate errors were zero,
the candidate coordinate digest matched, basis rank was 5, and the explicit
convex KKT audit remained valid. Model state was restored and the run recorded
zero optimizer steps.

## Interpretation Boundary

The result is stronger than M3's 17/20 audit outcome for this holdback subset,
but it cannot repair or replace M3. The 20 units were held back from M3
calibration and scoring, yet they came from the same 60-prompt blind-review
pool rather than a new authoring cohort. In addition, M4 inherited its dose
from M3 instead of selecting it from a new calibration split. Therefore no new
finite-sample conformal coverage theorem is claimed.

M4 supports the narrower statement that a pre-fixed 12.5% dose shrinkage
reduced the observed risk tail while retaining positive first-order quality and
safety movement on the review-qualified holdback. It does not establish
finite-step training benefit, generated-response quality, population validity,
or authorization for Core100/heldout.

```text
optimizer_steps=0
training_authorized=false
candidate_generation_authorized=false
core100_opened=false
heldout_opened=false
```
