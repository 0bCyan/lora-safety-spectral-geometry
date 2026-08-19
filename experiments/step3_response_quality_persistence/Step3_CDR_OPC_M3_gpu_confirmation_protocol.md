# Step3 CDR-OPC M3 Zero-Step GPU Confirmation

Status: prospective freeze before any M3 model forward or gradient.

## Fixed question

This stage asks whether the already-frozen M1 CDR-OPC direction can retain a
useful nonzero conformal dose on the independently authored and blindly
qualified M3 cohort. It does not redesign the direction, protected
certificates, tolerance, calibration rule, or cohort.

The runner must numerically replay the M1 five-dimensional direction and match
the frozen fallback and endpoint coordinates before any M3 score is used.
Twenty calibration prompts then select the first lower order statistic of the
unit-wise safe doses at `alpha=0.05`. This is the finite-sample marginal
`20/21` guarantee under exchangeability. The twenty sealed prompts are scored
only after the dose is fixed.

## Measurements

Each prompt unit contains its matching PKU risk response and both qualified
safe targets. Four normalized first-order certificates are measured: risk,
uniform, TARC, and continuation. A unit is covered only if all four
standardized deteriorations are at most 4%.

Behavior is separated into two frozen comparisons. Safety compares each
qualified target with the matching harmful source response. Quality compares
the same target with a minimal generic refusal that preserves its registered
author route. The generic comparator is deterministic and contains no prompt
content. Both comparisons use all forty target pairs in each split and do not
select between the two qualified author modes.

## Gates

- exact M1 direction replay and valid original solver/KKT certificate;
- calibration arithmetic residual at most `1e-10`;
- selected dose at least `0.01`;
- at least 19 of 20 sealed units jointly cover all four certificates;
- every sealed certificate's mean standardized deterioration is at most 4%;
- selected-dose movement improves both aggregate behavior comparisons on both
  calibration and sealed splits;
- all gradients are finite and nonzero, model state is restored, and every
  optimizer audit records zero completed steps.

The empirical 19/20 gate corroborates the marginal theorem; it does not turn
the theorem into a conditional guarantee. A pass remains a zero-step,
first-order mechanism result. It does not authorize training, claim a finite
parameter update, claim generated-response quality, or open Core100/heldout.

```text
optimizer_steps=0
training_authorized=false
candidate_generation_authorized=false
core100_opened=false
heldout_opened=false
```
