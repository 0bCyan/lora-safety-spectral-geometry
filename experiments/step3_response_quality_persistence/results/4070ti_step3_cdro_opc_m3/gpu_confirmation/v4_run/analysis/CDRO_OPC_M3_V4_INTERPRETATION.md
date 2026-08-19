# CDR-OPC M3 V4 Interpretation

## Decision

M3 is a preregistered near miss, not a confirmation. The selected dose was `0.03861261` and all four behavior panels improved, but sealed joint protection was `17/20` rather than the required `19/20`. The registered-certificate route is `STOP_CDRO_OPC_M3_AUDIT_PROTECTION`.

The raw runner route was `STOP_CDRO_OPC_M3_INVALID` because its legacy structural summary still included an unregistered auxiliary objective. All four registered protected gradients and all behavior gradients were finite and nonzero; correcting that label does not change the failed 17/20 outcome gate.

## Evidence

Calibration arithmetic residual was `4.163336342344337e-17`. The sealed failures were three risk-only exceedances above the 4% bound: `m3_audit_09` +0.568 percentage points, `m3_audit_35` +0.501 percentage points, `m3_audit_37` +0.107 percentage points. Aggregate standardized deterioration remained below 4% for every certificate.

The exact 95% binomial interval for 17/20 is `0.621` to `0.968`. Under p=20/21, P(X<=17) is `0.0672`. Thus this sample does not reject the marginal theorem at 5%, but it also does not pass the frozen empirical corroboration gate.

## Next Boundary

A hindsight dose near `0.03432` would cover 19/20, but using it now would leak sealed information. It may motivate a prospectively fixed shrinkage controller on a new cohort; it cannot repair M3 or authorize training.

```text
optimizer_steps=0
training_authorized=false
core100_opened=false
heldout_opened=false
```
