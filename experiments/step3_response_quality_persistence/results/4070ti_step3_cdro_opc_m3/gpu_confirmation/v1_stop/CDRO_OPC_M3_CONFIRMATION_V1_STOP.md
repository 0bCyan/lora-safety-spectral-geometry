# CDR-OPC M3 Confirmation V1 Stop

V1 stopped at the frozen M1 direction replay conjunction before any M3
calibration or sealed-audit gradient was computed. The fail-closed exception
did not persist which component differed, and no run output directory was
created.

An isolated diagnostic then reran only the already-open M1 construction. It
reproduced fallback and endpoint coordinates with maximum error `0.0`, matched
the endpoint SHA exactly, retained basis rank five, and passed the frozen
solver/KKT checks. This diagnostic is not an M3 retry and contains no new
cohort score.

V2 therefore repairs only the identity measurement: the frozen `1e-10`
coordinate tolerance is the hard mathematical gate, while bitwise JSON SHA
identity is retained as reported provenance. All experimental targets,
splits, dose rules, and effect gates remain unchanged.

```text
m3_calibration_gradients=0
m3_sealed_audit_gradients=0
optimizer_steps=0
training_authorized=false
```
