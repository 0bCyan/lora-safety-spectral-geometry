# Step3 CDR-OPC M3 GPU Confirmation V2r1 Storage Revision

Status: prospective path repair after V2 preflight and before any V2 model
load, forward, or gradient.

V2 inherited V1's output root. The preserved `v1_stop/` evidence therefore
caused the clean-start guard to classify the root as already containing run
artifacts. V2r1 changes only the output root and four output paths to the new
`gpu_confirmation/v2_run/` namespace. The candidate cohort, split, fixed
direction, numeric identity rule, protected scores, conformal dose rule,
behavior panels, and every threshold are byte-identical to V2.

```text
v2_model_loads=0
v2_m3_calibration_gradients=0
v2_m3_sealed_audit_gradients=0
optimizer_steps=0
training_authorized=false
```
