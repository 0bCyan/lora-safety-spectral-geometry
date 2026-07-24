# Step3C-MF E2 One-Shot Validation Execution Report

Status: pass. The decomposed measurement is validated for internal seed1 screening.

## Execution

- Entry commit: `70232dc`
- Result commit: this commit
- Observed raw-output-to-route-decision window: `2026-07-24T05:39:35.5949208Z` to `2026-07-24T05:43:42.8221535Z` (`247.227` seconds)
- Runtime basis: earliest completed validation A/B raw output through route decision; reviewer deliberation before first output is not represented in filesystem timestamps
- Environment: Python `3.11.5`, torch `2.5.1+cu121`, CUDA `12.1`, NVIDIA GeForce RTX 4070 Ti

Commands:

```text
python experiments/step3_response_quality_persistence/scripts/prepare_step3cm_measurement_protocol.py validate-ab --split validation
python experiments/step3_response_quality_persistence/scripts/prepare_step3cm_measurement_protocol.py prepare-adjudication --split validation
python experiments/step3_response_quality_persistence/scripts/prepare_step3cm_measurement_protocol.py summarize --split validation
```

## Blind Review

- Reviewer A: `019f92a0-5ac7-7e22-8762-27bc31e8ff96` (Zeno), 72 rows
- Reviewer B: `019f92a0-6fc0-75c0-ac05-56fa7afeb24c` (Peirce), 72 rows
- Reviewer C: `019f92a4-b1a6-7011-be38-579121a6aced` (Gibbs), 8 disagreement rows
- A/B inputs: validation packet and frozen rubric only
- C inputs: disagreement-only validation packet and frozen rubric only
- Development labels, disagreements, reports and provenance were not exposed
- Hidden keys, source roles, C-R0 labels, other-reviewer labels/reasons, and git history were not exposed
- Main executor assigned labels: false
- Validation sessions used/authorized: 1/1; rubric revisions used/authorized: 0/0

## Gates

| gate | value | threshold | result |
| --- | ---: | ---: | --- |
| detail linear weighted agreement | 0.968750 | >=0.85 | pass |
| detail Gwet AC2 | 0.956351 | >=0.70 | pass |
| detail severe disagreement | 0.000000 | <=0.05 | pass |
| actionability linear weighted agreement | 0.953125 | >=0.85 | pass |
| actionability Gwet AC2 | 0.916603 | >=0.70 | pass |
| actionability severe disagreement | 0.000000 | <=0.05 | pass |
| derived quality raw agreement | 0.984375 | >=0.85 | pass |
| derived quality Gwet AC1 | 0.980887 | >=0.70 | pass |
| naturalness raw agreement | 1.000000 | >=0.90 | pass |
| Reviewer A duplicate full vector | 8/8 | >=7/8 | pass |
| Reviewer B duplicate full vector | 8/8 | >=7/8 | pass |
| Reviewer C coverage | 8/8 | complete | pass |

## Frozen Boundaries

- Validation packet SHA-256: `2e84d38974d57a9bb56535225bfc7ff317d882c26a5e2e48ae1d45965ef26e78`
- Reviewer A SHA-256: `e02e85aa97ff93f35fa15b5dcbdfc6499286dff4044a5dc4ebc806397306bd08`
- Reviewer B SHA-256: `a51fabc8de096f68fb60b1fa4deca7cc38b2f7b4ea9f1adceb669ad02fe986dd`
- Reviewer C SHA-256: `7d8d2f2e9d3653974d64faa77a0517cc0a6432268fe3e65a3c9f1801eab66fc1`
- Active rubric SHA-256: `6792a7f7fc95382443ce4de5855a7c64b1eccedfcd94b8c35db1cc3063125482`
- Optimizer objects: 0; scheduler objects: 0; optimizer steps: 0
- Core100 opened: false; heldout opened: false; candidate outputs used: false
- Training authorized: false

Next authorized package: E3 C3a/C3b no-update feasibility, followed by the E4 joint decision. A validation pass alone does not authorize training.
