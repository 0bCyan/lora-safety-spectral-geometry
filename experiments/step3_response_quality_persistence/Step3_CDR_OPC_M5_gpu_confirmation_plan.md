# Step3 CDR-OPC M5 Fixed-Dose Zero-Step Confirmation

## Purpose

M5 tests the already frozen conservative dose `0.0337860296353059` on the
prompt-disjoint cohort accepted by the independent M5 blind review. The dose,
direction, protected certificates, score floor, and relative-drop tolerance
are inherited without tuning from M4/M3.

## Sequential opening rule

The runner loads one model and reconstructs the frozen M1 direction. It first
computes only the twenty M5 calibration units. The sealed-audit panel is
constructed only if every calibration gate passes: direction identity, solver
KKT audit, at least 19/20 protected units, aggregate protection, positive
quality and safety behavior, finite structural gradients, restored trainable
state, and zero completed optimizer steps.

If calibration fails, the route stops without computing sealed-audit gradients.
If calibration passes, the same unchanged model process evaluates the twenty
sealed-audit units at the same fixed dose. No result may tune the dose or alter
the direction.

## Claim boundary

This is a zero-step gradient-mechanism confirmation. It does not authorize a
parameter update, training, response generation, Core100 access, or heldout
access. A passing result authorizes interpretation and design of a separately
frozen finite-radius step protocol only.

```text
optimizer_steps=0
training_authorized=false
candidate_generation_authorized=false
core100_opened=false
heldout_opened=false
```
