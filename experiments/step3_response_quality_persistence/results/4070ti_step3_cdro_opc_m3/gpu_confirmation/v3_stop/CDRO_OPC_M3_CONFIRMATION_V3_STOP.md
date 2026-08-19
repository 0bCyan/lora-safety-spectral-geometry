# CDR-OPC M3 Confirmation V3 Stop

V3 passed the fixed-direction replay gate: fallback error `3.83e-11`, endpoint
error `1.28e-9`, rank-five basis error `5.68e-8`, and a valid frozen solver.
It then stopped during calibration certificate normalization. The sealed split
was not opened.

The frozen calibration-only diagnostic found no zero or nonfinite gradient for
the four registered protected certificates across all 20 units. The only zero
gradient was the unregistered auxiliary `branch_entropy` objective on two
units. The legacy helper normalized all seven objectives even though M3 scores
only risk, uniform, TARC, and continuation.

V4 repairs this implementation mismatch by subsetting to the four registered
certificates before M3 unit normalization. It does not alter M1 direction
construction, calibration membership, the dose rule, or any outcome gate.

```text
sealed_audit_opened=false
optimizer_steps=0
training_authorized=false
```
