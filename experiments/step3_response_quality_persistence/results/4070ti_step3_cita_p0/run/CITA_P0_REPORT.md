# CITA-P0 Zero-Update Mechanism Report

Route: `PASS_CITA_P0_AUTHORIZE_MATCHED_SMOKE`.

## Registered Evidence

- Mean token-advantage bootstrap 95% CI: `[0.400041, 0.712199]`.
- Median positive-token fraction: `0.578784`.
- Median absolute CITA/sham rank correlation: `0.137466`.
- Calibration CITA-delta/uniform norm ratio: `0.146130`.
- Calibration CITA/uniform cosine: `0.990261`.
- CITA-delta/sham-delta cosine: `-0.293353`.
- Calibration/validation CITA-delta cosine: `0.680506`.
- Cross-split CITA-over-sham cosine margin: `1.154375`.
- Peak reserved VRAM headroom: `0.607865`.

## Interpretation Boundary

This run tests whether a fixed-budget, correct-prompt token-information ordering produces a non-degenerate and prompt-disjointly replicating LoRA gradient at the fresh seed-1 step-0 policy. It does not test generated behavior, safety, utility, or publication readiness. Fresh LoRA-A gradients may be structurally zero because LoRA-B is initialized to zero; all registered tensors must still have attached finite gradient records and the aggregate gradient must be non-zero.

No optimizer or scheduler was created, no parameter update or checkpoint was written, and the exact trainable-state hash remained unchanged. `optimizer_steps=0` and `training_authorized=false`.
