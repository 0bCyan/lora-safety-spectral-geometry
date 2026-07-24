# Experiment Machine Task: Step3C-T5H External Review Handoff

## Current State

Pull `exp/step3-response-quality-persistence` and follow:

```text
experiments/step3_response_quality_persistence/Step3C_T5H_external_review_handoff_and_P6A_provenance_repair_plan.md
experiments/step3_response_quality_persistence/Step3C_T5H_model_reviewer_launch_addendum.md
experiments/step3_response_quality_persistence/configs/step3ct_t5h_external_review_registry.json
```

Track B remains stopped at `M1_reconstruction_integrity_failure`; representation probes and labels 0/16/32 are valid, but label64 grouped gradients remain unregistered. Track A has now selected `T5H2_safety_ab_frozen_authorize_quality_utility`: two distinct no-context model reviewers each returned 630 valid Safety rows and provenance, and the files were frozen before comparison.

No GPU experiment, new training, generation, label64 retry, or method revision is authorized.

## Completed Boundaries

Do not rerun or overwrite P6A, H1, H1Q, or H2. Verify the latest completed boundary with:

```text
python experiments/step3_response_quality_persistence/scripts/record_step3ct_t5h_safety_freeze.py verify
```

Required route is:

```text
T5H2_safety_ab_frozen_authorize_quality_utility
```

H2 is not a behavior result. Safety A/B labels remain un-compared, hidden keys remain sealed, and reliability has not been computed.

## Current Task: Quality and Utility A/B

Launch four fresh isolated sessions that are distinct from both Safety reviewers and from one another:

- Quality Reviewer A and Quality Reviewer B;
- Utility Reviewer A and Utility Reviewer B.

Each reviewer receives only its domain handoff, unchanged visible packet, role-specific blank template, and frozen rubric. Require exactly 330 rows and complete provenance per reviewer/domain.

- distinct anonymized reviewer and session IDs;
- no repository, arm, checkpoint, training, mechanism, hidden-key, duplicate-link, or prior-label access;
- no communication between reviewers;
- agreement to complete all 330 unchanged rows;
- packet, template, and rubric hashes;
- completed provenance contract.

After all four complete outputs are present, validate and freeze with:

```text
python experiments/step3_response_quality_persistence/scripts/run_step3ct_t5r_review.py freeze-quality-utility-ab
```

Reject partial, malformed, mixed-session, executor-edited, or unprovenanced files in full. Do not compare decisions before the all-A/B freeze is written and pushed.

## Later Packages

Only after Safety A/B freeze:

1. run separate Quality A/B and Utility A/B sessions using their neutral handoffs;
2. freeze all A/B files;
3. construct disagreement-only C packets;
4. obtain fresh complete C reviews;
5. compute reliability before hidden arm-key join;
6. run T6 only after `T5R4_complete_labels_authorize_T6`.

Use the existing T5R scripts and thresholds without revision.

## Prohibitions

- Do not modify P6 or replace M1.
- Do not retry label64 or report approximate grouped values.
- Do not transform, split, sanitize, paraphrase, truncate, or selectively omit review rows.
- Do not reuse failed T5 contexts or partial files.
- Do not use main-session, heuristic, keyword, embedding, or unprovenanced outcome labels.
- Do not open quality/utility before Safety A/B freeze.
- Do not run new training, generation, C4/C5, seeds 2/3, a second backbone, SpectraSafe, or heldout.
- Do not commit checkpoints, adapters, caches, raw API caches, full gradients, or unrelated paper files.

## Required Return State

Every package must include exact commands, entry/result commit, environment or reviewer provenance, all hashes and row counts, gate table, accepted/rejected/partial counts, heldout flag, one route, and one authorized next action.
