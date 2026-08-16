# PCR Serialization Transfer Audit

## Status

- Mode: `exploratory_non_confirmatory`
- Prospective protocol commit: `1799c2e`
- Route: `SERIALIZATION_NOT_SUFFICIENT_EXPLANATION_CLOSE_PCR`
- Optimizer steps: 0
- New training authorized: no
- `core100` opened: no
- Heldout opened: no

## Question

The frozen PCR training feature counts Qwen3's four-token forced empty no-think preamble as part of the assistant loss. Generation supplies that preamble as input. This audit asks whether the apparent prompt-conditional learning was concentrated in those forced tokens rather than in the freely generated response.

Before scoring, the protocol fixed three step16 models, four response sources, 40 correct prompts, two deterministic wrong prompts per row, token masks, 10,000 paired bootstrap draws, and route rules.

## Integrity

The audit produced 1,440 component-score rows and 480 prompt-margin rows:

- 3 models x 4 fixed response sources x 40 samples x 3 prompt roles = 1,440 scores.
- Every score was finite.
- The forced preamble was exactly 4 tokens in every sequence.
- The response-only component contained 17 to 270 tokens.
- Peak VRAM headroom was 63.4%.

## Shared-Response Transfer

The primary comparison holds each frozen proxy-dev reference response fixed and reports candidate-minus-C0R gain in the correct-versus-two-wrong prompt NLL margin.

| Candidate | Legacy combined gain, 95% CI | Forced-preamble gain, 95% CI | Response-only gain, 95% CI | Preamble minus response, 95% CI |
|---|---:|---:|---:|---:|
| PCR-M | +0.00557 [0.00396, 0.00720] | +0.00008 [-0.00029, 0.00043] | +0.00580 [0.00407, 0.00752] | -0.00572 [-0.00748, -0.00395] |
| PCR-MW | +0.00546 [0.00387, 0.00703] | +0.00012 [-0.00017, 0.00039] | +0.00569 [0.00398, 0.00738] | -0.00557 [-0.00729, -0.00382] |

Both candidates pass the prospectively frozen `TRANSFER_SURVIVES_RESPONSE_ONLY` rule. The transfer is in response tokens; the forced preamble contribution is near zero and its interval crosses zero.

Correct-prompt rank accuracy is 1.0 for C0R and both candidates on the shared reference responses, so rank accuracy is saturated. The paired NLL-margin contrast, rather than retrieval accuracy, provides the usable resolution.

## Generated-Response Check

On each candidate's own frozen generated responses, evaluated under the same response content for the candidate and C0R models:

| Candidate response source | Legacy combined gain, 95% CI | Forced-preamble gain, 95% CI | Response-only gain, 95% CI |
|---|---:|---:|---:|
| PCR-M | +0.04351 [0.03458, 0.05277] | +0.00008 [-0.00029, 0.00042] | +0.04947 [0.03872, 0.06033] |
| PCR-MW | +0.04524 [0.03610, 0.05473] | +0.00012 [-0.00017, 0.00039] | +0.05143 [0.04055, 0.06258] |

This secondary result points in the same direction but cannot override the failed behavior gate.

## Interpretation

The PCR objectives did change the model in their intended narrow sense: for fixed responses, both trained models assign a larger likelihood advantage to the correct prompt than C0R does, and that change survives after removing the forced no-think preamble from the scored tokens.

The prior generation gate nevertheless found lower refusal-marker rates, no redirect improvement, and no qualifying strict behavioral gain. Therefore the failure is not adequately explained by thinking-prefix leakage. Prompt identifiability and safety-policy quality are empirically separable in this setting. Improving the former alone was insufficient to improve the latter.

PCR remains closed. A defensible next method must optimize a behavior-proximal safety quantity or include a causal bridge from conditional scoring to decoding policy; it must not merely increase prompt-response identifiability, rescale PCR, or mask the four forced tokens and rerun.

## Boundary

The audit uses a repeatedly accessed 40-row development split and deterministic cross-prompt mismatches without semantic-category matching. It is a mechanism diagnostic, not a blinded safety evaluation or confirmatory paper claim.

Machine-readable decision: `serialization_transfer_decision.json` (`a0121e9ccce9a2edd404232705d16ff30201666580e257c6cb89d5574d737e07`).
