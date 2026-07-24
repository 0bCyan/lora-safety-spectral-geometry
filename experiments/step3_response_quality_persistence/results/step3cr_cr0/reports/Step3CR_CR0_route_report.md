# Step3C-R C-R0 Review and Route Report

Status: first-session gate failure.

## Review Isolation

- Session: `step3cr_cr0_review_session_v1_20260724`
- Reviewer A rows: 48
- Reviewer B rows: 48
- Reviewer C disagreement rows: 7
- Canonical outcome denominator: 40
- Reversed duplicate QC rows: 8
- Main executor assigned labels: false

## Reliability

| construct | n | raw agreement | Gwet AC1 | Cohen kappa | pass |
| --- | ---: | ---: | ---: | ---: | --- |
| pair preference | 40 | 0.825 | 0.760 | 0.6785304247990813 | False |
| response-side specificity | 80 | 1.000 | 1.000 | undefined | True |

## Reversal and Position Gates

- Reviewer A preference/specificity consistency: 8/8, 8/8
- Reviewer B preference/specificity consistency: 8/8, 8/8
- Reviewer A non-tie count and position imbalance: 29, 0.17241379310344823
- Reviewer B non-tie count and position imbalance: 31, 0.3548387096774193
- All disagreement rows adjudicated: True

## Route

Overall C-R0 gate: **fail**.

Next authorized package: `one_complete_rubric_clarification_and_fresh_review_session`.

Heldout remained closed. No optimizer or scheduler was created, and no training was run.
