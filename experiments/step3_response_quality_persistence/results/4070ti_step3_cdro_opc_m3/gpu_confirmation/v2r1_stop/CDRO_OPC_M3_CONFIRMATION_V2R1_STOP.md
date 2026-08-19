# CDR-OPC M3 Confirmation V2r1 Stop

V2r1 stopped at the `1e-10` direction replay gate before any M3 calibration or
sealed-audit gradient. No run output directory was created. A subsequent
old-M1-data-only diagnostic passed that gate with fallback and endpoint errors
`1.93e-11` and `2.46e-11`, but differed at the serialized SHA level while all
basis, solver, primal, and KKT checks passed.

The discrepancy across fresh processes shows that `1e-10` is not a stable
identity threshold for this NF4/FP16 replay. V3 freezes `1e-6`, still 200 times
stricter than M1's `2e-4` reconstruction tolerance, and emits the component
audit before any new cohort score. No outcome-facing threshold changes.

```text
m3_calibration_gradients=0
m3_sealed_audit_gradients=0
optimizer_steps=0
training_authorized=false
```
