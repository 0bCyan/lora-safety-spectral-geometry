# Step3C-R C-R0 v2 Review and Route Report

Status: second-session gate failure; C-R0 closed.

## Provenance and Runtime

- Protocol commit: `c059b11`
- Result commit: this commit
- Reviewer A/B/C agent IDs: `019f9212-a787-7bf1-9935-632c9897925b`, `019f9212-bbd7-72c0-87e7-7cf5b6362e3e`, `019f9216-fd8e-78a1-b995-5677fb0b71dd`
- Reviewer A/B/C rows: 48 / 48 / 14
- Protocol-to-decision wall clock: 541.6 seconds
- Environment: Python 3.11.5, torch 2.5.1+cu121, CUDA 12.1, `NVIDIA GeForce RTX 4070 Ti`

## Reliability

| construct | n | raw agreement | Gwet AC1 | Cohen kappa |
| --- | ---: | ---: | ---: | ---: |
| pair preference | 40 | 0.800 | 0.720 | 0.6510359869138496 |
| response-side specificity | 80 | 0.963 | 0.957 | 0.7087378640776701 |
| naturalness sensitivity | 80 | 1.000 | 1.000 | None |

## Reversal and Position

- Reviewer A preference/specificity reversal: 8/8 / 8/8
- Reviewer B preference/specificity reversal: 8/8 / 8/8
- Reviewer A non-tie / position imbalance: 33 / 0.15151515151515155
- Reviewer B non-tie / position imbalance: 29 / 0.17241379310344823

## Gate Table

| gate | result |
| --- | --- |
| pair_preference_agreement_pass | fail |
| specificity_agreement_pass | pass |
| reviewer_a_preference_reversal_pass | pass |
| reviewer_b_preference_reversal_pass | pass |
| reviewer_a_specificity_reversal_pass | pass |
| reviewer_b_specificity_reversal_pass | pass |
| reviewer_a_non_tie_pass | pass |
| reviewer_b_non_tie_pass | pass |
| reviewer_a_position_bias_pass | pass |
| reviewer_b_position_bias_pass | pass |
| all_disagreements_adjudicated | pass |
| overall_pass | fail |

## Commands and Tests

```text
python experiments/step3_response_quality_persistence/scripts/run_step3cr_cr0_v2.py prepare
python experiments/step3_response_quality_persistence/scripts/run_step3cr_cr0_v2.py validate-ab
python experiments/step3_response_quality_persistence/scripts/run_step3cr_cr0_v2.py prepare-adjudication
python experiments/step3_response_quality_persistence/scripts/run_step3cr_cr0_v2.py summarize
python experiments/step3_response_quality_persistence/scripts/prepare_step3cr_cr0_protocol.py test-fixtures
```

Preparation validation passed: 48 rows, 40 canonical pairs, 8 exact reversed duplicates, fresh IDs/order, and canonical content equality with v1. Synthetic metric, reversal, EOL, hidden-field, and zero-variance fixtures passed.

## Frozen Hashes

- Packet: `b84f1608e24b5eedb919dfc99532e5d2b80546273aeef181a1a15703d4274eb7`
- Rubric: `e694e0eac983e0c73500bc567337c55f95b6a4ac07b805daa286cbeb61eb6cc6`
- Reviewer A: `ba29a562d8c614df379f2a778eff3c93e6f7b64ee733bbe3cd3527f181f07456`
- Reviewer B: `eb94d80548db9d18841f4bd22876e507704e2da6d220af9180a0bbd5838f9272`
- Reviewer C: `4df5db2f41addd4954a93b233bf8ac9475053313f9b118cb49009a41418dcd8c`
- Final display labels: `c8fd4718e93b2cac25c6005f03205b3cc1a3e79e1ca6197543fe8c9505fd5df7`
- Final canonical labels: `68a6bc2aeea5c87b6e77777a97561f9ae3c1db895b474c04584da4216a50fe7d`

## Route

Overall gate: **fail**. Next authorized package: `C-R0_closed_after_second_complete_failure`.

This was the single permitted complete rubric revision. No v1 review-label value was used to construct the v2 packet or assign a v2 label. Heldout remained closed; no optimizer or scheduler was created, and no training was run.
