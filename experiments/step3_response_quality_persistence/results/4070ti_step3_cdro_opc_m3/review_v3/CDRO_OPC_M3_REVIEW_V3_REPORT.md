# CDR-OPC M3 Review V3 Report

V3 passed the frozen calibration, reliability, and target-yield conjunction.
Two new context-isolated reviewers completed 180 rows each. Neither reviewer
read V1/V2 row labels, hidden maps, author modes, method results, gradients, or
the other review.

## Calibration and reliability

Both reviewers reproduced all 12 hidden primary duplicates exactly. Reviewer F
scored 11/12 or better on every applicable anchor field and exact expected
subvector; Reviewer G did the same. The closest gates were deliberately
reported rather than rounded away: F had 11/12 positive prompt binding and
11/12 clean-safety negatives; G had 11/12 mismatch detail and 11/12
clean-safety negatives.

On the 120 primary candidates, full-vector exact agreement, detail exact and
weighted agreement, Gwet AC2, derived-qualification raw agreement and Gwet
AC1, every component agreement, and the 2000-replicate prompt-cluster
bootstrap lower quantile were all `1.0`. Both reviewers qualified all 120
primaries, so primary negative agreement is undefined and is not claimed. V3
measured negative-boundary competence on explicit hidden generic, mismatch,
and clean-safety anchors instead.

There were zero derived-qualification disagreements. Reviewer C was therefore
forbidden and not launched.

## Cohort

All 60 prompts had both candidates qualified. The protocol-seeded SHA-256
ordering selected the first 40 prompts without using V1/V2 labels or any
gradient result. One-based even ranks form 20 calibration units and odd ranks
form 20 sealed-audit units, producing 80 selected targets.

```text
route=PASS_M3_TARGET_COHORT
qualified_candidates=120
eligible_prompts=60
selected_prompts=40
selected_targets=80
calibration_units=20
sealed_audit_units=20
reviewer_c_used=false
optimizer_steps=0
training_authorized=false
```

## Interpretation boundary

V3 is a post-failure measurement repair, not a fresh replication of the
candidate pool. Its evidential value comes from prospective V3 freezing,
role-hidden two-class anchors, new reviewers, label-independent cohort order,
and complete artifact preservation. It establishes a review-qualified cohort;
it does not establish CDR-OPC effectiveness or conformal-dose coverage. Those
claims require the next zero-step 4070 Ti gradient confirmation.
