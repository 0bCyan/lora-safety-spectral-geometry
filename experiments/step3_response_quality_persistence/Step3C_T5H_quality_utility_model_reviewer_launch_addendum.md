# Step3C-T5H Quality and Utility Model Reviewer Launch Addendum

## 1. Entry Boundary

Safety A/B is frozen under `T5H2_safety_ab_frozen_authorize_quality_utility`. No Safety A/B decisions, agreement metrics, hidden identities, or behavior endpoints have been exposed to the new reviewers.

This addendum authorizes four fresh no-context model sessions:

- Quality Reviewer A and Quality Reviewer B;
- Utility Reviewer A and Utility Reviewer B.

It does not revise any packet, rubric, schema, reliability threshold, candidate gate, or T6 statistic.

## 2. Independence and Model Provenance

All four sessions must have distinct reviewer IDs and session IDs, use `fork_context=false`, and remain separate from the completed Safety sessions. Exact runtime selections and self-reported model identities are frozen in `configs/step3ct_t5h_quality_utility_model_reviewer_registry.json`.

Quality A, Quality B, and Utility B self-reported the requested runtime model IDs. Utility A was launched with runtime selection `gpt-5.5` and self-reported the broader family `gpt-5`; both facts must remain visible in provenance.

Before H3Q is pushed, every reviewer must remain at zero repository access and zero study-row access. A model session receives exactly one complete attempt after H3Q. An incomplete, policy-blocked, malformed, or unprovenanced output is rejected in full and selects `T5R2_review_completion_or_schema_stop`.

## 3. Allowed Materials

After H3Q is pushed, each reviewer may read only its domain-specific handoff, unchanged visible packet, role-specific blank template, and frozen rubric.

Quality reviewers may access:

```text
review_handoffs/step3ct_t5h_quality_reviewer_instructions.md
t4/packets/quality/step3ct_t4_quality_packet.jsonl
t4/packets/quality/step3ct_t4_quality_reviewer_<a|b>_template.jsonl
review_rubrics/step3cm_decomposed_quality_rubric_v1.md
```

Utility reviewers may access:

```text
review_handoffs/step3ct_t5h_utility_reviewer_instructions.md
t4/packets/utility/step3ct_t4_utility_packet.jsonl
t4/packets/utility/step3ct_t4_utility_reviewer_<a|b>_template.jsonl
step2_mechanism_dynamics/review_rubrics/step2j_b1_utility_quality_rubric.md
```

Each reviewer may write only its own final label and provenance files plus an optional role-specific temporary file. Reviewers may not list or search the repository, inspect Git history, read Safety labels, open another domain, inspect another reviewer output, infer arms, access hidden keys or duplicate maps, or transform the packet.

## 4. Completion and Freeze

Each final output must contain exactly 330 unique opaque IDs in template order and the complete frozen domain schema. Provenance must satisfy `step3ct_t5r_common.validate_provenance`, including exact paths, hashes, timestamps, model/session identity, and every independence confirmation.

The main executor may perform mechanical schema, ID, hash, row-count, and provenance validation. It may not inspect label semantics, edit labels, compare decisions before the all-A/B freeze, or substitute labels.

After all four outputs validate, run:

```text
python experiments/step3_response_quality_persistence/scripts/run_step3ct_t5r_review.py freeze-quality-utility-ab
```

The resulting all-A/B freeze must be committed and pushed before any disagreement packet or reliability analysis is generated.

## 5. Scope

This launch authorizes Quality A/B and Utility A/B only. Reviewer C, A/B comparison, reliability, hidden-key join, T6, training, generation, method tuning, C4/C5, additional seeds, second backbone, and heldout remain unauthorized until their respective frozen gates pass.
