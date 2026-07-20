# Step3A Route Decision Report

## Status

Route: **A2**

Output concentration is not matched by target concentration; investigate optimization.

No model training was run. Heldout prompt and response content remain sealed.

## Review Execution

The planned new full-output recoding encountered a platform safety gateway when an independent reviewer opened the unsafe response packet. The session did not self-label, sanitize harmful evidence, or bypass the filter.

Output `mechanical_template` labels therefore reuse the frozen Step2J independent blind review, where 365 overlap rows had raw agreement `1.000` and Cohen's kappa `1.000`. Natural-v1 target structure received a new response-only independent review. Prompt relevance, naturalness, and leakage reuse the completed Step2J-B0 target review.

## Primary Reliability

| source | field | n | raw agreement | kappa | pass |
| --- | --- | ---: | ---: | ---: | --- |
| natural-v1 target response | mechanical_template | 80 | 0.912 | 0.789156626506024 | True |
| Step2J model output | mechanical_template | 365 | 1.000 | 1.000 | True |

Secondary target-structure fields failing the working reliability gate: boundary_act.

Failed secondary fields are descriptive only and are not used for route selection.

## Target Evidence

| metric | value |
| --- | ---: |
| independent prompt relevance | 1.000 |
| independent grammatical naturalness | 1.000 |
| independent harmful leakage | 0.000 |
| response-only mechanical-template share | 0.263 |

## Route Gates

| gate | observed | pass |
| --- | ---: | --- |
| target mechanical share `>=0.60` | 0.263 | False |
| constant20 step64 output mechanical share `>=0.50` | 0.720 | True |
| constant20 step64 redirect `<0.80` | 0.720 | True |

## Safety-Conditioning Check

For constant20 step64, all 72 clean-refusal responses were labeled mechanical and all 28 unsafe-compliance responses were labeled non-mechanical:

```text
P(mechanical | clean refusal) = 1.000
P(mechanical | unsafe compliance) = 0.000
```

Overall mechanical share is confounded with safety rate and must not be optimized directly. Step 3 promotion must use mechanical-template share conditional on clean-safe behavior; otherwise a less safe model could appear more diverse.

## Interpretation

Route A2 rejects immediate natural-v2 construction: independently reviewed targets are not mechanically concentrated enough to explain the output pattern. The next authorized test is the Step3B frozen decoding-attribution audit on the existing constant20 checkpoint. No training-time intervention is authorized until that audit separates decoding-limited diversity from training-distribution concentration.
