# Step3 CDR-OPC M3 GPU Confirmation V4 Registered Certificates

Status: prospective implementation repair after V3 opened calibration and
before sealed-audit scoring.

V3 passed the stable M1 direction gate with endpoint coordinate error
`1.2820467829716131e-09`, then stopped when the legacy normalization helper
required every one of seven gradient objectives to be nonzero. A calibration-
only diagnostic showed that all four protocol-registered protected gradients
were finite and nonzero on all 20 units. The unregistered auxiliary
`branch_entropy` gradient was zero on units `m3_cal_04` and `m3_cal_12`.

The M3 design, dose controller, and gates register only risk, uniform, TARC,
and continuation. V4 therefore subsets the gradient dictionary to exactly
those four names before M3 unit normalization. M1 direction reconstruction
continues to use its original seven-objective implementation. Missing or zero
registered certificates still fail closed.

This repair does not impute a gradient, drop a calibration unit, replace a
prompt, inspect sealed data, or change any score or threshold. V4 uses a new
empty output namespace and retains the V3 direction audit print.

```text
registered_protected_zero_counts=0
unregistered_branch_entropy_zero_count=2
sealed_audit_opened=false
optimizer_steps=0
training_authorized=false
```
