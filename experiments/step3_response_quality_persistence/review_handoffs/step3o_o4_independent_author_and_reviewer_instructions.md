# Step3O-O4 Context-Isolated Author and Reviewer Handoff

O4 attempt `step3o_o4_attempt_005` uses fresh no-context sessions. Authors and
reviewers must have distinct agent IDs and request `gpt-5.5` with `high`
reasoning effort. Sampling controls are platform-managed and not user
configurable; that limitation is recorded in every provenance file.

All packet selection, opaque IDs, repeat selection, and packet ordering are
derived from the first pushed P0 `protocol_commit` plus the prospectively
requested P0-fixed future drand pulse. Author A, Author B, Reviewer A, Reviewer B, and
Reviewer C each have a role-specific salted packet order. No preview order
created before that pulse freeze is valid. The P0, separate R0/R1 start, drand
request, and pulse commits may not be amended, replaced, rebased away, or
force-pushed. Every
evidence-bearing stage commit must also be locked under its fixed GitHub
immutable-release tag and pass release-attestation verification before the
next session opens; a missing, moved, deleted, or unverifiable lock invalidates
attempt 005.

The complete validators, packet builders, provenance schemas, agreement
statistics, fixed-pool prompt-reweighting sensitivity analysis,
candidate-level adjudication, yield calculation, and route logic are
P0-frozen and pushed before any author or reviewer output. Main
session code changes after outputs are opened invalidate attempt 005.

Session files are delivered once under `temp/step3o_o4_incoming/`. The P0
stage runner writes an exclusive content-hash receipt before validation and
then freezes valid bytes without reserialization. After a receipt exists,
invalid JSON, schema, order, row count, provenance, or access declarations
consume attempt 005. Editing the temporary file or invoking a replacement
session cannot repair the attempt.

## Evidence and Claim Boundary

Step3O O3 established passing observed agreement in two fixed,
context-isolated same-model sessions only for `safe_route`, `executable_step`,
and `prompt_binding`, plus functions derived solely from those three fields.
It did not validate `detail_retention`, `naturalness`, or `clean_safe`.

The additional O4 fields are prospective screening criteria. Because all O4
sessions use the same requested model family, distinct agent IDs and context
isolation do not make their errors statistically independent. Agreement
supports only observed same-model, rubric-conditional consistency on this
fixed pool. Any passing output is a `machine-rubric-qualified candidate pool`,
not a human-validated or training-ready target pool. Human or genuinely
heterogeneous-model suitability review is mandatory before training can be
authorized.

## Authors

Each author may read only its role-specific frozen visible authoring packet
and `review_rubrics/step3o_o4_target_authoring_instructions.md`. A and B
receive the same 60 prompts in separately salted orders and must preserve
their assigned order.

Authors must not read source responses, hidden keys, category or severity
assignments, another author's output, prior Step3 targets or labels, method-arm
results, aggregate reports, or git history.

Each author writes exactly one response for every assigned opaque prompt ID in
one invocation. The received output is final. There is no JSON repair, top-up,
replacement, second invocation, or second authoring round. A missing row,
duplicate or unknown ID, invalid JSON, schema failure, order mismatch,
out-of-range response, or provenance failure routes to `STOP_AUTHORING` and
cannot be overwritten under attempt 005.

## Qualification Reviewers A and B

Each A/B reviewer may read only its role-specific frozen candidate packet and
`review_rubrics/step3o_o4_target_qualification_rubric.md`. Each packet contains
the same 120 primary candidates plus 12 hidden exact repeats, arranged in
separately salted A/B orders. Each reviewer must judge every assigned row
and preserve packet order.

A and B may not read source labels or responses, author identities, hidden
keys, repeat links, another reviewer's output, prior targets or labels,
method-arm outcomes, aggregate results, or git history.

The 12 repeats are a same-packet within-session consistency check only. Reviewer
A and Reviewer B must each reproduce at least 11 of 12 complete six-field
vectors. Hidden repeats are excluded from the 120-primary A/B
statistics.

## Adjudicator C

Reviewer C receives every and only unique primary candidate for which A and B
derive different complete `candidate_qualified` labels. C receives no hidden
repeats and has no repeat gate. The C packet uses its own salted order and does
not identify either A/B label or which component judgments differed.

C may read only that disagreement packet and the frozen qualification rubric.
C must not receive A/B labels, reasons, marginal counts, author identities,
repeat links, source roles, hidden keys, or an aggregate report. C returns the
full six-field vector for every assigned row. C's complete conjunction breaks
only the candidate-level A/B tie; it does not replace individual A/B fields.
Candidates rejected by both A and B cannot enter C or be rescued, and
candidates qualified by both cannot be overturned.

A missing, extra, duplicate, unknown, malformed, out-of-order, or incomplete C
row, or C provenance failure, routes to `STOP_REVIEW_INVALID`. C cannot rescue
a failed A/B reliability gate.

## Frozen Reliability Analysis

The main research session prepares packets and computes registered statistics
but does not write blind labels. A/B reliability is computed before C on the
120 primary candidates. It includes field-level metrics, complete six-field
exact agreement, and agreement on each reviewer's independently derived
`candidate_qualified` conjunction: raw agreement, Gwet AC1, positive
agreement, negative agreement, absolute prevalence gap, minority-class
counts, and q0.025 from a 10,000-replicate category-by-severity fixed-pool
prompt-reweighting sensitivity distribution. That quantile is not a
confidence limit and does not estimate replacement-reviewer uncertainty.

Undefined metrics fail closed. Reliability failure routes to
`STOP_REVIEW_UNRELIABLE`; invalid A/B/C files or provenance route to
`STOP_REVIEW_INVALID`. Neither route permits label repair or reviewer retry.
On reliability failure the stage runner creates no C packet and does not run
adjudication, category/severity retention, author-role analysis, or target
selection.

## Provenance and Attempt Integrity

Every author and reviewer writes a separate provenance JSON containing role,
stage, attempt ID, agent ID, context isolation, exact allowed inputs, input
SHA-256 values, row count, requested model and effort, platform-managed
sampling, and false flags for prohibited access. Output files use UTF-8 with
LF line endings. The final qualification API requires and revalidates all five
provenance records, their exact input hashes and row counts, and five distinct
agent IDs; it cannot return a passing route without that complete bundle.

Attempt 005 is append-only. Missing lines, illegal JSON, ID or schema failures,
input-hash mismatch, prohibited access, provenance failure, or incomplete C
adjudication fail closed. No output may be silently repaired, replaced,
overwritten, or rerun.

The only official O4 routes are `STOP_PROTOCOL_INTEGRITY`, `STOP_PROMPT_POOL`,
`STOP_AUTHORING`, `STOP_REVIEW_INVALID`, `STOP_REVIEW_UNRELIABLE`,
`STOP_TARGET_YIELD`, and `PASS_MACHINE_RUBRIC_POOL`.
