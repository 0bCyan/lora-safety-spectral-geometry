# Step3C-T5H Reviewer C Model Launch Addendum

## 1. Entry Boundary

All six Reviewer A/B outputs are frozen under `H3A1_all_ab_frozen_authorize_disagreement_c`. The frozen comparison produced three disagreement-only packets without joining arm, checkpoint, sample, hidden-key, or duplicate-link identities:

- Safety: 11 rows and 37 disputed decision fields;
- Quality: 69 rows and 76 disputed decision fields;
- Utility: 33 rows and 43 disputed decision fields.

These counts describe adjudication workload only. They are not reliability or behavior results.

## 2. Prospective Reviewer Qualification

This addendum authorizes three fresh no-context policy-compliant model sessions:

- Safety Reviewer C;
- Quality Reviewer C;
- Utility Reviewer C.

All three sessions use distinct reviewer IDs and session IDs, use `fork_context=false`, and are distinct from every frozen A/B session and failed T5 session. The requested runtime is `gpt-5.6-terra`; each session self-reported the broader GPT-5 family, and both facts remain visible in provenance.

Before H4Q is committed and pushed, every Reviewer C session must remain at zero repository access and zero study-row access. Each session receives exactly one unchanged domain packet after H4Q. A partial, blocked, malformed, or unprovenanced output is rejected in full and selects `T5R2_review_completion_or_schema_stop`.

## 3. Allowed Materials After H4Q

Each reviewer may access only its assigned disagreement packet, blank Reviewer C template, and frozen domain rubric.

Safety Reviewer C:

```text
results/4070ti_step3ct_seed1/t5r/packets/safety/step3ct_t5r_safety_reviewer_c_packet.jsonl
results/4070ti_step3ct_seed1/t5r/packets/safety/step3ct_t5r_safety_reviewer_c_template.jsonl
experiments/step2_mechanism_dynamics/review_rubrics/step2j_b1_safety_behavior_rubric.md
```

Quality Reviewer C:

```text
results/4070ti_step3ct_seed1/t5r/packets/quality/step3ct_t5r_quality_reviewer_c_packet.jsonl
results/4070ti_step3ct_seed1/t5r/packets/quality/step3ct_t5r_quality_reviewer_c_template.jsonl
review_rubrics/step3cm_decomposed_quality_rubric_v1.md
```

Utility Reviewer C:

```text
results/4070ti_step3ct_seed1/t5r/packets/utility/step3ct_t5r_utility_reviewer_c_packet.jsonl
results/4070ti_step3ct_seed1/t5r/packets/utility/step3ct_t5r_utility_reviewer_c_template.jsonl
experiments/step2_mechanism_dynamics/review_rubrics/step2j_b1_utility_quality_rubric.md
```

Paths under `results/` and `review_rubrics/` are relative to `experiments/step3_response_quality_persistence/`. Reviewers may write only their own final label and provenance files plus an optional role-specific temporary file. They may not list or search the repository, inspect Git history, open A/B labels or reasons, open another domain, inspect another reviewer output, access packet manifests or freeze files, infer arms, access hidden keys or duplicate maps, transform the packet, or use external retrieval.

## 4. Completion Contract

Safety, Quality, and Utility outputs must contain exactly 11, 69, and 33 unique opaque IDs, respectively, in template order and with the complete frozen domain schema. Every template field must be filled even when only a subset is named in `disagreement_fields`. Final adjudication later uses Reviewer C only for fields on which A and B disagreed; Reviewer C cannot revise an A/B agreement.

Each provenance file must satisfy `step3ct_t5r_common.validate_provenance`, including exact packet, template, rubric, output hashes, row counts, timestamps, model/session identity, and all independence confirmations.

The coordinator may perform mechanical hash, schema, ID, field-domain, row-count, and provenance checks. The coordinator may not inspect label semantics, edit labels, substitute labels, compute reliability before all C outputs freeze, or join outcome keys.

## 5. Scope and Next Gate

H4Q authorizes only one complete disagreement-only Reviewer C attempt per domain. It does not authorize reliability analysis, adjudicated labels, hidden-key join, T6, candidate selection, new training, generation, C4/C5, extra seeds, another backbone, method revision, or heldout access.

After all three complete C files validate, freeze their hashes and provenance before running the existing reliability and finalization path. Any completion or schema failure selects `T5R2_review_completion_or_schema_stop`. Any frozen reliability-gate failure selects `T5R3_review_reliability_stop`. Only `T5R4_complete_labels_authorize_T6` permits hidden-key join and T6.
