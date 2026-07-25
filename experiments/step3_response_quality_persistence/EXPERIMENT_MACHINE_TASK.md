# Experiment Machine Task: Step3C-T5H External Review Handoff

## Current State

Pull `exp/step3-response-quality-persistence` and follow:

```text
experiments/step3_response_quality_persistence/Step3C_T5H_external_review_handoff_and_P6A_provenance_repair_plan.md
experiments/step3_response_quality_persistence/Step3C_T5H_model_reviewer_launch_addendum.md
experiments/step3_response_quality_persistence/Step3C_T5H_quality_utility_model_reviewer_launch_addendum.md
experiments/step3_response_quality_persistence/configs/step3ct_t5h_external_review_registry.json
```

Track B remains stopped at `M1_reconstruction_integrity_failure`; representation probes and labels 0/16/32 are valid, but label64 grouped gradients remain unregistered. Track A has now selected `T5H2_safety_ab_frozen_authorize_quality_utility`: two distinct no-context model reviewers each returned 630 valid Safety rows and provenance, and the files were frozen before comparison.

No GPU experiment, new training, generation, label64 retry, or method revision is authorized.

## Completed Boundaries

Do not rerun or overwrite P6A, H1, H1Q, or H2. Verify the latest completed boundary with:

```text
python experiments/step3_response_quality_persistence/scripts/record_step3ct_t5h_quality_utility_freeze.py verify
```

Required route is:

```text
H3A1_all_ab_frozen_authorize_disagreement_c
```

H3 is not a behavior result. All six A/B files are frozen, hidden keys remain sealed, and reliability has not been computed.

## Current Task: Disagreement-Only Reviewer C

Only after the H3 commit is pushed, run:

```text
python experiments/step3_response_quality_persistence/scripts/run_step3ct_t5r_review.py prepare-c
```

This command may compare frozen decision fields only to identify disagreements. Each C packet must omit A/B labels, reasons, reviewer identities, vote counts, arms, checkpoints, hidden keys, and duplicate links. Freeze C packet and template hashes and exact row counts before launching reviewers.

Launch one fresh no-context Reviewer C session per domain. C sessions must be distinct from all six A/B sessions and may access only their domain's disagreement packet, blank C template, and frozen rubric. Require complete coverage of every C row and valid provenance.

## Later Packages

Only after complete C reviews:

1. compute reliability before hidden arm-key join;
2. stop at `T5R3_review_reliability_stop` if any frozen gate fails;
3. authorize T6 only through `T5R4_complete_labels_authorize_T6`;
4. run T6 without revising thresholds or candidate rules.

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
