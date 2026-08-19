# Step3 CDR-OPC M3 GPU Confirmation V3 Stable Replay

Status: prospective numeric repair after two fail-closed direction replays and
before any M3 calibration or sealed-audit gradient.

The M1 direction is reconstructed from NF4 weights with FP16 compute. V1 used
bitwise serialized-coordinate identity in addition to a `1e-10` coordinate
gate. V2 removed the bitwise condition but retained `1e-10`. Two formal
processes stopped before M3 scoring, while old-data-only diagnostic processes
replayed exactly once and then with maximum fallback/endpoint errors
`1.9298176717665733e-11` and `2.4607899051787285e-11`. The latter retained the
same rank-five basis, valid convex solver, primal residual
`1.128888649226667e-13`, and normalized stationarity residual
`3.332103598752877e-07`.

V3 sets the process-stable coordinate identity tolerance to `1e-6`. This is
200 times stricter than the frozen M1 `2e-4` basis/reconstruction tolerance
and ten times stricter than its `1e-5` KKT tolerance. Basis validity,
solver/KKT validity, and the reconstructed algorithm remain mandatory. The
runner prints all direction replay errors before the M3 gate, whether it passes
or stops.

No M3 target, split, protected certificate, scale floor, 4% allowance,
conformal rule, useful-dose floor, audit coverage gate, aggregate gate, or
behavior gate changes. V3 uses a new empty output namespace.

```text
prior_formal_m3_calibration_gradients=0
prior_formal_m3_sealed_audit_gradients=0
optimizer_steps=0
training_authorized=false
core100_opened=false
heldout_opened=false
```
