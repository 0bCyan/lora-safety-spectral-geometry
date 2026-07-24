# Step3C-T3U Optimizer-Unit Implementation Freeze

## Decision

Route: `T3U_implementation_frozen_pending_smoke`

The committed T2R R3 route authorizes T3, but the parent 64-step trainer serialized only a first-micro-step gradient ratio. The prospective T3U amendment replaces that diagnostic with the required sum of four unweighted branch gradients on CPU float64. It does not change training losses, coefficients, RNG order, queue updates, clipping, optimizer behavior, eta, data, or endpoints.

## Verification

| Check | Result |
| --- | --- |
| T3 diagnostic unit tests | 3/3 pass |
| parent training common tests | 9/9 pass |
| T2R scalar diagnostic tests | 6/6 pass |
| T2R config hash unchanged | pass |
| eta unchanged | pass |
| previous R3 artifacts unchanged | pass |
| optimizer steps in this package | 0 |
| candidate/core100/heldout access | false/false/false |

The implementation records scalar norms, ratios, cosine, grouped energy, covered positions, hard-active counts, and stable hashes. Full gradient tensors are never serialized.

## Authorization

Only a fresh three-arm two-step smoke plus independent resume-from-step1 replay is authorized next. Full T3 training remains blocked until that smoke package passes every inherited numerical, determinism, queue, projection, parameter, draw, and VRAM gate and is committed and pushed.
