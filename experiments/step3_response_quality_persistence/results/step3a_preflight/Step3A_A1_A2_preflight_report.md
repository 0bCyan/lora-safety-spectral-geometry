# Step3A A1-A2 Preflight Report

## Status

A1 evidence/leakage audit: **pass**.

A2 deterministic target inventory: complete.

No new training was run. No heldout prompt or response content was opened.

## Split and Exposure Integrity

| check | result |
| --- | --- |
| target rows | 80 |
| unique training prompts | 40 |
| targets per prompt min/max | 2 / 2 |
| core100 IDs aligned across four cells | True |
| training/core100 overlap | 0 |
| core100/Step2G split-ID overlap | 0 |
| target/heldout-ID overlap | 0 |
| exposure rows | 80 |
| total safety draws | 256 |
| exposure min/max | 3 / 4 |
| exposure hash order matches targets | True |

The core100 development set is disjoint from natural-v1 safety training and from all Step2G split IDs. The script read only heldout IDs and hashes in the split manifest, not heldout contents.

## Natural v1 Deterministic Structure

| metric | value |
| --- | ---: |
| unique responses | 80 / 80 |
| exact duplicate rows beyond first | 0 |
| `I can't` openings | 40 |
| `I won't` openings | 40 |
| other openings | 0 |
| contains `I can help` | 72 |
| token count min/median/mean/max | 17 / 23.0 / 23.12 / 31 |

Natural v1 is exactly diverse at the string level but visibly narrow at the opening and redirect-phrase level. This is a decision-relevant diagnostic, not a subjective mechanical-template label and not proof of causal structure transfer.

## Decision

A1 passes and A2 supports proceeding to the preregistered independent fixed-codebook review. New training remains prohibited until reviewer reliability and the Step3A route decision are complete.
