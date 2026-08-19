# Step3 CDR-OPC M3 Calibration Norm Diagnostic

Status: post-stop diagnostic after the V3 direction gate passed and the first
opened calibration unit failed protected-certificate normalization.

The diagnostic computes raw risk, uniform, TARC, and continuation certificate
norms for all twenty already-open calibration units. It does not reconstruct
the optimizer-space direction, select a dose, read or tokenize the sealed
split, create an optimizer, or authorize training. Its purpose is to determine
whether the zero gradient is isolated, certificate-specific, or systematic.

Any later method repair must be frozen after this diagnostic and may use only
the calibration evidence. Sealed-audit thresholds and membership remain
unchanged.

```text
sealed_audit_opened=false
optimizer_objects=0
optimizer_steps=0
training_authorized=false
```
