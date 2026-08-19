# CDR-OPC M3 Blind Review Plan

Status: prospective after `PASS_M3_AUTHORING` and before reviewer packet
construction or candidate-label access.

Two new no-context reviewers independently judge the 120 sealed M3 candidates
under the frozen target-qualification criteria. Each packet contains all 120
primary candidates, 24 hidden Step3Q calibration sentinels, and 12 hidden
within-reviewer primary duplicates. Reviewer-specific opaque IDs and packet
orders hide author mode, worker, source metadata, candidate hash, calibration
role, duplicate link, split rank, gradients, and all prior labels.

Each reviewer writes one TSV draft in packet order. A frozen finalizer validates
the 156 rows and atomically JSON-encodes the immutable review. Pre-submission
format correction is allowed; post-submission revision, replacement, or
re-review is forbidden.

The Step3Q calibration gates and A/B reliability gates remain byte-frozen. If
either reviewer fails sentinel or hidden-duplicate calibration, or if the A/B
reliability conjunction fails, the route stops before target selection as
`STOP_M3_REVIEW_UNRELIABLE`. No heuristic label may replace a review.

Reviewer C is launched only for primary candidates whose A/B derived
qualification differs. C sees only a role-blind prompt, response, the rubric,
and the two short reasons. If C cannot complete, the route stops as
`STOP_M3_REVIEW_INCONCLUSIVE`.

After final labels, a prompt is eligible only when both authored candidates
qualify. Eligible prompts are ordered by SHA-256 of the frozen selection seed
and opaque prompt ID. The first forty form the cohort; even ranks are the
twenty calibration units and odd ranks the twenty sealed audit units. Fewer
than forty selects `STOP_M3_TARGET_YIELD`.

No gradient result, old heuristic label, arm/checkpoint identity, Core100,
heldout row, or source harmful response is visible during review.

```text
optimizer_steps=0
training_authorized=false
```
