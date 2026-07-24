# Experiment Machine Task: Step3C-T5H External Review Handoff

## Current State

Pull `exp/step3-response-quality-persistence` and follow:

```text
experiments/step3_response_quality_persistence/Step3C_T5H_external_review_handoff_and_P6A_provenance_repair_plan.md
experiments/step3_response_quality_persistence/configs/step3ct_t5h_external_review_registry.json
```

Track A is stopped at `T5R1_reviewer_access_stop` with zero accepted labels. Track B is stopped at `M1_reconstruction_integrity_failure`; representation probes and labels 0/16/32 are valid, but label64 grouped gradients remain unregistered.

No GPU experiment, new training, generation, label64 retry, or method revision is authorized.

## Package P6A

After the H0 implementation commit is pushed, run:

```text
python experiments/step3_response_quality_persistence/scripts/audit_step3ct_t5r_p6_eol.py run
python experiments/step3_response_quality_persistence/scripts/audit_step3ct_t5r_p6_eol.py verify
```

Required route:

```text
P6A1_eol_reconciled_content_unchanged
```

The audit must account for all 21 legacy P6 artifacts, match every committed HEAD blob, write only to `mechanism_completion/p6a`, and leave every P6 file byte-identical. Commit and push P6A before reviewer qualification.

If any mismatch cannot be explained by raw equality or LF-to-CRLF transformation, select `P6A0_non_eol_content_or_git_blob_mismatch` and stop.

## Reviewer Qualification

After P6A1, wait for a research coordinator to supply two independent safety reviewers. Preferred type is `external_human`.

Before either reviewer opens the packet, require:

- distinct anonymized reviewer and session IDs;
- no repository, arm, checkpoint, training, mechanism, hidden-key, duplicate-link, or prior-label access;
- no communication between reviewers;
- agreement to complete all 630 unchanged rows;
- packet, template, and rubric hashes;
- completed provenance contract.

An independent model session is eligible only with exact provider/model version and policy-compliant access to the complete unchanged packet. Do not probe a gateway with study rows.

If reviewers are unavailable, record `T5H0_waiting_for_qualified_reviewers` with zero packet access. This is not a method result.

## Safety A/B

Give each qualified reviewer only:

```text
review_handoffs/step3ct_t5h_safety_reviewer_instructions.md
the unchanged visible safety packet
its own blank A/B template
the frozen safety rubric
```

Place completed outputs and provenance at the paths expected by `run_step3ct_t5r_review.py`. Validate and freeze with:

```text
python experiments/step3_response_quality_persistence/scripts/run_step3ct_t5r_review.py freeze-safety-ab
```

Each output must contain exactly 630 unique rows. Reject partial, malformed, mixed-session, executor-edited, or unprovenanced files in full. Push the complete Safety A/B freeze before quality or utility review.

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
