# Step3 CDR-OPC M3 GPU Confirmation V2 Transport Repair

Status: prospective repair after V1 stopped at direction replay and before any
M3 calibration or sealed-audit gradient.

V1 required both a maximum coordinate error no larger than `1e-10` and exact
SHA-256 identity of the JSON-serialized endpoint coordinates. The first run
stopped at this conjunction before scoring any M3 unit, but the fail-closed
exception did not preserve the two component results. A separate diagnostic
using only the already-open M1 inputs then reproduced both coordinate arrays
with maximum error zero, reproduced the endpoint SHA, and retained the frozen
solver/KKT certificate.

V2 separates mathematical identity from serialization identity. Its hard gate
is the already-frozen `1e-10` maximum error on both fallback and endpoint
coordinates, together with the unchanged basis and solver validity gates.
Bitwise coordinate SHA equality remains recorded as an audit field but is not
an additional mathematical condition. No tolerance, target, split, protected
certificate, dose rule, behavior comparator, or outcome threshold changes.

V2 remains fail-closed. A coordinate error above `1e-10` stops before any M3
unit is scored. Calibration still selects the dose before the sealed audit is
opened.

```text
v1_m3_calibration_gradients=0
v1_m3_sealed_audit_gradients=0
optimizer_steps=0
training_authorized=false
core100_opened=false
heldout_opened=false
```
