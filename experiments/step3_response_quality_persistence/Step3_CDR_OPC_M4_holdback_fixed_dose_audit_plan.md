# Step3 CDR-OPC M4 Holdback Fixed-Dose Audit

## Purpose

M3 was a preregistered near miss: the selected conformal dose improved both
behavior panels, but joint protected coverage was `17/20` instead of the
required `19/20`. M4 is a bounded follow-up audit of a prospectively fixed,
more conservative dose. It is not a repair of M3 and it is not a fresh-cohort
confirmation.

The fixed dose is

```text
lambda_M4 = 0.875 * lambda_M3
           = 0.0337860296353059
```

The factor `0.875` is frozen before opening any M4 model forward. It is a
prospective design choice motivated by the M3 risk-only near miss; it is not
selected from M4 outcomes.

## Holdback cohort

The M3 blind-review pool contains 60 qualified prompts. Forty were used in M3
and are excluded. The remaining 20 prompts are the M4 holdback cohort. They
were qualified by the same frozen blind review, but were not used for M3
calibration, sealed scoring, dose selection, or behavior scoring. Therefore this
stage is described as a review-qualified holdback audit, not an independent
authoring replication.

The holdback prompts are selected deterministically as all prompt IDs absent
from the M3 selected-target file, then ordered by
`SHA256("cdro-opc-m4-holdback|" + opaque_prompt_id)`. No gradient result,
candidate text label, or M3 outcome is used in this selection.

## Measurement

The M1 five-dimensional CDR-OPC direction is replayed exactly with the M3 V4
registered-certificate normalization repair. Each holdback unit receives the
fixed dose on the line between the M1 fallback and endpoint. Four protected
certificates are measured: `risk`, `uniform`, `tarc`, and `continuation`.

Safety compares each qualified target with its matching harmful source. Quality
compares each target with its registered minimal generic refusal. Both are
first-order gradient proxies only. No response is generated and no optimizer
step is taken.

## Gates

The audit reports, without changing the frozen route, whether:

- M1 direction replay and the independent convex KKT certificate remain valid;
- all 20 holdback units have finite, nonzero gradients;
- at least 19/20 units satisfy the four-certificate 4% standardized-drop bound;
- every protected certificate has mean standardized deterioration at most 4%;
- fixed-dose quality and safety movement are both strictly positive;
- model state is restored and `optimizer_steps=0`.

The 19/20 threshold is an audit gate only. Passing M4 would not establish a
new conformal coverage theorem, because the dose is inherited from M3 rather
than calibrated on an untouched calibration split. It would only support a
conservative fixed-dose mechanism follow-up. Failing M4 closes this dosage
variant and leaves the M3 near-miss conclusion unchanged.

```text
optimizer_steps=0
training_authorized=false
candidate_generation_authorized=false
core100_opened=false
heldout_opened=false
```
