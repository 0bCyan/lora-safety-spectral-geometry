# Step3 PCR-M/PCR-MW Proxy-Dev Decision

## Status

- Mode: `exploratory_non_confirmatory`
- Frozen generation protocol commit: `83b437059f0d49cf7c4ea23c093dfb02c65a2b4b`
- Route: `STOP_PCR_FAMILY_PROXY_DEV_GATE_FAILED`
- Passing candidates: none
- Blind paired review authorized: no
- 64-step training authorized: no
- `core100` opened: no
- heldout opened: no

## Question

PCR-M and PCR-MW test whether a reviewed safety response can be made more prompt-conditional by scoring that same response under its correct prompt and two same-category mismatched prompts. PCR-M minimizes the mean calibrated violation. PCR-MW adds a temperature-0.1 smooth-worst term. The matched C0R arm reproduces the same trainer without a ranking backward pass.

The 16-step training tournament passed all execution-integrity gates. Final centered training-margin changes were `+0.590299` for PCR-M and `+0.591410` for PCR-MW. Those values were frozen as mechanism diagnostics only, before generation.

## Frozen Comparison

All arms used the same ordered 40-row `proxy_dev` split, greedy decoding, a 128-token limit, disabled thinking, and their frozen step16 adapters.

| Arm | Refusal | Redirect | Procedural | Mean tokens | Max-token hits | Unique rate |
|---|---:|---:|---:|---:|---:|---:|
| C0R reproduction | 0.650 | 0.125 | 0.225 | 38.475 | 1 | 1.000 |
| PCR-M | 0.575 | 0.125 | 0.200 | 42.275 | 2 | 1.000 |
| PCR-MW | 0.600 | 0.125 | 0.200 | 41.150 | 2 | 1.000 |

All three arms produced 40 non-empty responses, zero raw thinking-tag rows, and no exact normalized-template collisions.

## Paired Changes

Relative to the fresh same-trainer C0R:

| Candidate | Refusal gain/loss | Redirect gain/loss | Procedural gain/loss | Failed frozen checks |
|---|---:|---:|---:|---|
| PCR-M | 0 / 3 | 0 / 0 | 2 / 1 | refusal non-inferiority; strict primary improvement |
| PCR-MW | 0 / 2 | 0 / 0 | 2 / 1 | refusal non-inferiority; strict primary improvement |

The smooth-worst term reduced the refusal-marker regression from three paired rows to two, but did not remove it. Neither objective improved redirect behavior. The one-row net procedural improvement was below the prospectively frozen two-row strict-improvement requirement.

## Decision

Both candidates fail the relative proxy-dev gate. The PCR family stops at 16 steps under this formulation. Training-margin improvement cannot override a generated-behavior regression, so no blind review, longer training, `core100`, or heldout evaluation is authorized.

This result rejects the local hypothesis that same-response prompt-mismatch ranking, at the calibrated weights and 16-step setting tested here, is sufficient to improve generated safety behavior. It does not establish that prompt-conditional ranking is impossible in other formulations, models, seeds, or data regimes.

## Evidence Boundary

The marker rates are deterministic development triage rather than blinded safety judgments. The 40-row split has been used repeatedly for method development. Therefore this negative result is useful for method selection and mechanism-behavior falsification, but it is not a confirmatory paper claim.

Machine-readable decision: `pcr_mw_proxy_dev_gate_decision.json` (`47977cee403359ef69bf8a9264da2aca5bcf1c63957654a116526bcea099c0da`).
