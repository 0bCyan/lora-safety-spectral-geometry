# Step3C-T5H External Review Handoff and P6A Provenance Repair Plan

## 1. Decision

Step3C-T5R ended on two independent boundaries:

- Track A selected `T5R1_reviewer_access_stop` with zero qualified reviewers, zero launched sessions, and zero accepted labels;
- Track B completed all 12 representation probes and exact grouped-gradient labels 0/16/32, but selected `M1_reconstruction_integrity_failure` because the bounded checkpoint48 replay did not reproduce label64 within the frozen tolerance.

An independent audit then found that 19 of the 21 P6 artifact hashes were computed over Windows CRLF working-tree bytes before Git normalized the committed files to LF. Re-expanding the committed LF content to CRLF reproduces all 19 recorded hashes; the other two artifacts already match raw committed bytes. This is an end-of-line provenance defect, not evidence of numerical content alteration.

This plan authorizes exactly two next actions:

1. create a non-destructive P6A canonical-LF/Git-blob reconciliation package;
2. prepare and execute a safety-first external independent review handoff when qualified reviewers are actually available.

No label64 retry, approximate grouped endpoint, new candidate, new training, fresh generation, C4/C5, heldout access, or method revision is authorized.

## 2. Frozen Interpretation Before Behavior Labels

The following mechanism interpretation is frozen before any outcome label is accepted:

1. C3 is a strong training-set manipulation check: step64 Top1 is `0.90` for C3a and `0.85` for C3b versus `0.15` for C0-R.
2. C3 does not enlarge the mean absolute cosine gap. Positive cosine rises, but negative cosine rises more, producing a narrower high-cosine cone.
3. Retrieval ordering is largely solved while the multi-positive softmax probability remains low. The binding objective therefore continues to exert substantial pressure after rank saturation.
4. Fixed eta does not preserve a fixed gradient contribution. Binding-safety gradients move from aligned at step0 to mildly conflicting after step16 in both attention and MLP groups.
5. C3b develops some later gradient-direction difference from C3a, but has no registered incremental representation benefit. The prior C3b redundancy warning remains active.
6. These are training-internal mechanism-consistency findings. They do not establish response quality, safety, utility, generalization, or causal mediation.

This interpretation may be tested against later behavior outcomes but may not be rewritten to rescue a failed candidate.

## 3. Package H0: Prospective Implementation Freeze

Before P6A execution, commit and push:

- this plan and its registry;
- the canonical EOL audit implementation;
- pure-stdlib provenance tests;
- LF-explicit mechanism result writers for future artifacts;
- neutral reviewer handoff instructions;
- updated README, agent status, and experiment-machine task.

Required tests:

- raw committed bytes match their legacy hash when unchanged;
- LF bytes reproduce a legacy CRLF hash only through EOL transformation;
- a semantic content change cannot reconcile as EOL-only;
- lone CR bytes are rejected;
- computed Git blob OID matches a fixed fixture;
- `py_compile` passes for changed Python files.

Push H0 before running the repository P6A audit.

## 4. Package P6A: Canonical Provenance Reconciliation

### 4.1 Inputs

Use only:

```text
experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/mechanism_completion/p6/step3ct_t5r_p6_mechanism_manifest_package.json
the 21 paths listed by that manifest
the committed HEAD blobs for those paths
```

Do not rewrite any P6 artifact or the legacy manifest.

### 4.2 Canonical rule

For every UTF-8 text artifact:

1. read the committed LF bytes;
2. reject lone carriage returns;
3. record SHA-256 over canonical UTF-8/LF bytes;
4. record byte length;
5. compute the Git blob OID from the canonical bytes;
6. verify it equals `HEAD:<path>`;
7. classify the legacy hash as either raw-byte match, LF-to-CRLF reconstruction match, or unexplained mismatch.

### 4.3 Gate

Select `P6A1_eol_reconciled_content_unchanged` only if:

- all 21 legacy artifacts exist;
- every legacy hash is explained by exact raw bytes or LF-to-CRLF transformation only;
- every current canonical byte sequence equals its committed HEAD blob;
- every JSON artifact parses;
- no P6 file changes;
- the original `M1` route remains unchanged.

Any unexplained byte, parse, count, or blob mismatch selects `P6A0_non_eol_content_or_git_blob_mismatch` and stops external review preparation until audited.

### 4.4 Outputs

Write only to:

```text
experiments/step3_response_quality_persistence/results/4070ti_step3ct_seed1/mechanism_completion/p6a/
```

Required outputs are a canonical manifest, route decision, report, execution record, and package manifest. All new text files must be emitted with explicit LF line endings so their hashes survive Git normalization.

## 5. Package H1: External Reviewer Qualification

P6A1 does not itself authorize outcome interpretation. Track A resumes only after a research coordinator supplies two independent safety reviewers.

Preferred reviewer type:

```text
external_human
```

An independent model reviewer is eligible only if its provider and exact model ID/version are available, the complete unchanged packet can be reviewed under the governing policy, and no gateway bypass or content transformation is needed.

For Safety A and Safety B, freeze before packet access:

- distinct anonymized reviewer IDs and session IDs;
- reviewer type and, when applicable, exact model provenance;
- no prior involvement in this repository or experiment;
- no access to arm identities, checkpoints, mechanism results, hidden keys, duplicate links, or another reviewer's output;
- no cross-reviewer communication;
- agreement to review all 630 rows under the unchanged rubric;
- packet, template, and rubric hashes;
- start/completion timestamps and all required confirmations.

The main executor may validate files and compute statistics but may not label. A reviewer who cannot lawfully inspect the complete unchanged packet is not qualified.

## 6. Package H2: Safety A/B Review and Freeze

Each reviewer receives only the safety handoff instructions, unchanged visible safety packet, its own blank A/B template, and frozen safety rubric.

Each accepted file must contain exactly 630 unique opaque IDs and all decision fields. Partial, malformed, schema-invalid, edited-by-executor, mixed-session, or unprovenanced outputs are rejected in full.

Both complete outputs and provenance files must be frozen before A/B comparison. No quality or utility packet may be opened before this boundary is pushed.

Routes:

```text
T5H0_waiting_for_qualified_reviewers
T5H1_reviewer_access_or_provenance_stop
T5H2_safety_ab_frozen_authorize_quality_utility
```

`T5H0` is a waiting state with zero study access, not a method result. `T5H1` accepts no partial labels. Only `T5H2` advances.

## 7. Packages H3-H5

### H3: Quality and utility A/B

After T5H2, use separate fresh reviewer sessions and the neutral domain-specific handoffs. Require exactly 330 rows per reviewer/domain and the existing T5R provenance contract.

### H4: Disagreement-only C and reliability

Use the existing frozen T5R implementation:

- compare decision fields only after all A/B files freeze;
- create C packets with no A/B labels, reasons, votes, arms, checkpoints, or duplicate links;
- obtain complete fresh C coverage;
- compute primary-row reliability before outcome-key join;
- use duplicates only for within-reviewer consistency;
- apply every original threshold without revision.

Only the existing route `T5R4_complete_labels_authorize_T6` may authorize T6.

### H5: T6

After T5R4, join frozen adjudicated labels to hidden keys and apply the original paired bootstrap, exact McNemar, Holm, control, safety, quality, utility, and candidate-selection rules. Mechanism evidence cannot rescue a failed behavior gate.

## 8. Authorization Matrix

Authorized after H0 is pushed:

- P6A canonical audit and verification;
- reviewer recruitment and provenance-only qualification;
- Safety A/B review only after two reviewers qualify;
- existing T5R freeze/statistics tooling after complete label files arrive.

Not authorized:

- modifying P6 or replacing `M1`;
- another label64 reconstruction or relaxed tolerance;
- reporting approximate label64 grouped metrics;
- main-session, heuristic, or unprovenanced labels;
- packet cleaning, splitting, paraphrasing, truncation, or gateway probing with study rows;
- quality/utility review before Safety A/B freeze;
- new training, generation, method tuning, C4/C5, seeds 2/3, second backbone, SpectraSafe, or heldout.

## 9. Commit Boundaries

1. `H0`: plan, registry, canonical-audit implementation/tests, neutral handoffs, and status updates.
2. `P6A`: canonical reconciliation evidence and route.
3. `H1`: qualified reviewer records or zero-access waiting record.
4. `H2`: complete Safety A/B freeze or explicit stop.
5. `H3`: complete Quality/Utility A/B freeze.
6. `H4`: C labels, reliability package, and T5R route.
7. `H5`: T6 route and next authorization.

Push each boundary before its dependent package. No ambiguous continuation is allowed.

## 10. Immediate Next State

After P6A1 is committed, local and experiment-machine work stops at reviewer qualification. The next missing resource is not GPU time; it is two lawful independent safety reviewers. Until they are supplied, T4 remains valid, behavior remains unknown, and no method-development experiment is authorized.
