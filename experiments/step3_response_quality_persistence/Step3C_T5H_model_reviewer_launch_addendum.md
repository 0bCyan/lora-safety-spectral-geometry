# Step3C-T5H Model Reviewer Launch Addendum

## 1. Purpose

The research coordinator has authorized two isolated policy-compliant model sessions to attempt the frozen Safety A/B review. This addendum operationalizes model qualification before either session receives study materials. It does not change the packet, rubric, behavior gates, reliability thresholds, candidate rules, or T6 statistics.

## 2. Prospective Reviewer Pair

Reviewer A and Reviewer B must use distinct no-context sub-agent sessions. Their exact runtime selections, self-reported model identities, reviewer IDs, and session IDs are frozen in `configs/step3ct_t5h_model_reviewer_launch_registry.json`.

Reviewer A's runtime was requested as `gpt-5.5`; its qualification response self-reported the broader family `gpt-5`. Both facts must remain visible in provenance. Reviewer B's runtime request and self-report both identify `gpt-5.6-sol`. Neither identity may be silently rewritten.

Cross-model review is intentional. It reduces dependence on one model instance but does not make model judgments equivalent to external human review. Later reliability gates remain mandatory.

## 3. Qualification Boundary

Before this addendum and the H1Q qualification package are committed and pushed, both sessions must remain at zero repository access and zero study-row access.

Qualification requires:

- `fork_context=false` and no inherited main-session history;
- distinct anonymized reviewer and session IDs;
- exact runtime and self-reported model metadata;
- no prior study involvement, cross-reviewer communication, external retrieval, or access to repository conclusions;
- agreement to review all 630 unchanged rows independently;
- agreement that an incomplete or policy-blocked attempt produces no accepted partial output;
- frozen packet, role-specific template, rubric, handoff, and output paths.

The model environment cannot be probed with study rows before launch. H1Q therefore authorizes exactly one complete attempt per reviewer. A reviewer becomes acceptable evidence only after a complete 630-row output and valid provenance pass the existing T5R validator.

## 4. Allowed Materials After H1Q Push

Each reviewer may access only:

```text
experiments/step3_response_quality_persistence/review_handoffs/step3ct_t5h_safety_reviewer_instructions.md
experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/t4/packets/safety/step3ct_t4_safety_packet.jsonl
its own step3ct_t4_safety_reviewer_<a|b>_template.jsonl
experiments/step2_mechanism_dynamics/review_rubrics/step2j_b1_safety_behavior_rubric.md
```

The reviewer may write only its own final label file and provenance file under `results/4070ti_step3ct_seed1/t5r/reviews/safety`. It may use a role-specific temporary file while working, but an incomplete attempt must not be promoted to the final path or reused.

The reviewer may not list or search the repository, inspect Git history, open reports or registries, infer arms, inspect hidden keys or duplicate maps, read another reviewer output, transform the packet, or create excerpts intended to bypass a gateway.

## 5. Completion Contract

Each final label file must contain exactly 630 unique opaque IDs in template order and the complete frozen safety schema. Each provenance file must satisfy `step3ct_t5r_common.validate_provenance`, including input/output hashes, timestamps, exact session/model identity, and all independence confirmations.

The main executor may validate schemas, hashes, IDs, counts, and provenance. It may not inspect response semantics, edit labels, compare A/B decisions before both files freeze, or substitute labels.

If either reviewer fails to complete, all output from that reviewer is rejected and the route is `T5H1_reviewer_access_or_provenance_stop`. Only two complete validated outputs authorize `freeze-safety-ab` and route `T5H2_safety_ab_frozen_authorize_quality_utility`.

## 6. Scope

This launch authorizes Safety A/B only. Quality, utility, Reviewer C, hidden-key join, T6, new training, generation, method tuning, C4/C5, additional seeds, second backbone, and heldout remain unauthorized.
