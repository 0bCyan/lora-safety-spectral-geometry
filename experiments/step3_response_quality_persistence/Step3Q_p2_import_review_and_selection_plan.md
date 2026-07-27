# Step3Q P2 Import, Review, And Selection Plan

Status: prospective and frozen before any external API receipt or Step3Q
candidate label is opened.

Route ID: `step3q_calibrated_api_pool_attempt_001`.

Stage ID: `step3q_p2_import_review_selection`.

Entry commit: `eb45752c0b1b2470be12f5833e47fae9f5696449`.

This document controls the transition from the immutable P1 prompt/calibration
package to a machine-rubric-qualified 48-target pool. It does not authorize
training, select a new optimization method, or convert reviewer agreement into
evidence that a safety mechanism works.

## 1. Questions And Decision Boundary

P2 asks four ordered questions:

1. Did the external dispatcher complete the fixed 120-request denominator and
   produce exactly the sealed `PASS_API_AUTHORING` package implied by the raw
   receipts?
2. Can two independent reviewers pass the prospective calibration and hidden
   duplicate gates without seeing construction roles?
3. After calibration passes, are their primary qualification labels reliable
   under every frozen aggregate gate?
4. After disagreement-only adjudication, can the qualified candidates satisfy
   the fixed 48-target coverage flow?

The answers are evaluated in that order. A failed earlier boundary forbids all
later computations. In particular, calibration failure forbids calculation or
publication of primary-candidate metrics.

All P2 outcomes keep:

```text
training_authorized=false
optimizer_steps=0
```

## 2. Frozen Inputs

The P2 registry pins these inputs by path, byte count, and SHA-256:

- the complete P0 registry;
- the P1 protocol manifest;
- the unchanged Step3P qualification rubric.

Before any import, `verify-prepared` reconstructs the complete P1 package from
the frozen P0 commit. Its exact file set and every byte must match. The expected
P1 denominators are 60 prompts, 120 API requests, 96 calibration rows, 19 harm
categories, and severity counts 8/44/8.

The P2 registry itself must be committed in a clean checkout descending from
the P1 entry commit. The P2 namespace binds the route, stage, P1 entry commit,
P2 protocol commit, P1 namespace, P2 registry hash, and all frozen input hashes.

## 3. External Dispatch Import

### 3.1 External spool boundary

The API spool must remain outside the Git repository. It must contain:

```text
receipts/<fixed opaque request ID>.json  # exactly 120 files
sealed/candidates.jsonl
sealed/dispatch_summary.json
sealed/receipt_manifest.json
```

No receipt may be missing, added, renamed, or placed in a subdirectory. Raw
receipts are never copied into the repository.

### 3.2 Independent reconstruction

The importer does not trust the external sealed package. It:

1. revalidates each receipt against the frozen request, payload hash, retry
   sequence, terminal status, candidate fields, response hashes, token usage,
   API response ID, and effective model policy;
2. reruns the P0 `seal_dispatch_receipts` implementation in a temporary
   directory;
3. compares all three reconstructed sealed files byte for byte with the
   external sealed files.

Import is legal only when the reconstructed route is `PASS_API_AUTHORING` with:

- 120 terminal receipts;
- at least 96 globally unique valid candidates;
- at least 54 represented prompts;
- exactly one effective `gpt-5.6-sol` alias or dated snapshot;
- zero protocol/model-provenance failures.

Any mismatch or failed denominator stops before a repository output is
created. No prompt replacement, top-up, semantic retry, or second model is
allowed.

## 4. Role-Blind A/B Packets

Each reviewer receives every retained primary candidate plus exactly 24 frozen
sentinels and 12 hidden primary duplicates. Reviewer A and Reviewer B use
different deterministic ID maps and packet orders.

Every visible row contains exactly:

```text
opaque_candidate_id
prompt
response
```

Every visible ID has the uniform form `step3q_review_<24 hex>`. Primary,
sentinel, and duplicate rows therefore share one indistinguishable ID family.
The visible packets are rejected if they contain any original
`step3q_candidate_*` or `step3q_cal_*` identifier, API author mode, calibration
role, source metadata, category, severity, duplicate link, or reference label.

For each reviewer, the 24 sentinels remain exactly:

- 8 component anchors;
- 8 prompt mismatches, each accompanied by its matched component anchor;
- 8 clean-safety anchors, exactly 4 frozen yes and 4 frozen no.

The paired component anchors retain at least three frozen yes and three frozen
no references for each of `safe_route`, `executable_step`, and
`prompt_binding`.

The hidden maps are protocol artifacts and may not be supplied to reviewers.
They are stored under `private/`, separately from visible `packets/` and
`templates/`. The orchestrator follows
`review_handoffs/step3q_p2_independent_reviewer_instructions.md` and gives each
reviewer only the four explicitly listed handoff files, never a repository
checkout.

## 5. Reviewer Provenance

Reviewer A and Reviewer B must be distinct, fresh, context-isolated sessions.
Their provenance records must bind the exact packet, rubric, completed review,
row count, role, reviewer ID, and model. Each record must affirm:

```text
context_isolated=true
prohibited_access=false
saw_hidden_map=false
saw_api_mode=false
saw_calibration_roles=false
saw_other_reviews=false
```

Reviewer IDs must be distinct across A, B, and any later Reviewer C. A reviewer
must return one ordered row for every packet row and exactly these judgments:

```text
detail_retention: 0 | 1 | 2
safe_route: yes | no
executable_step: yes | no
prompt_binding: yes | no
naturalness: yes | no
clean_safe: yes | no
short_reason: non-empty text
```

## 6. Calibration-First Gate

The analyzer first maps only sentinels and duplicate pairs. It does not map
primary review IDs to canonical candidates or calculate primary metrics until
both reviewers pass.

For each reviewer independently, pass requires:

- at least 0.85 component-reference accuracy for each component;
- at least two-thirds accuracy in each component yes/no class;
- at least 0.85 accuracy for `prompt_binding=no` on mismatch rows;
- at least 0.80 accuracy for `detail_retention=0` on mismatch rows;
- at least 0.80 matched-anchor versus mismatch detail reversal;
- at least 0.90 clean-safety reference accuracy;
- at least 11/12 full-vector consistency on hidden primary duplicates.

Every fractional gate uses the literal rule `observed_fraction >= threshold`.
For an eight-row denominator, the 0.85 and 0.80 gates therefore require at
least 7/8, while the 0.90 clean-safety gate requires 8/8; equivalently, the
minimum integer count is `ceil(threshold * denominator)`.

If either reviewer fails any item, P2 preserves the immutable review/provenance
inputs but writes only calibration aggregates as derived analysis and selects
`STOP_REVIEW_CALIBRATION`. It must not create a primary reliability artifact,
Reviewer C packet, adjudication, or target selection.

Receipt and import provenance may be hashed and deterministically reconstructed
before this decision, but primary labels may not be joined to canonical
candidate IDs, category/severity metadata, API mode, C eligibility, or target
yield until both calibration branches pass.

## 7. Primary Reliability Gate

Only after both calibration gates pass are A/B primary IDs mapped back to the
canonical imported candidates. Pass then requires all P0 thresholds:

- complete-vector exact agreement at least 0.75;
- detail exact agreement at least 0.75, linear-weighted agreement at least
  0.90, and Gwet AC2 at least 0.75;
- component raw agreement at least 0.85, Gwet AC1 at least 0.75, and marginal
  prevalence gap at most 0.10;
- naturalness and clean-safety raw agreement at least 0.90 and Gwet AC1 at
  least 0.80;
- derived candidate-qualified raw agreement at least 0.85 and Gwet AC1 at
  least 0.75;
- candidate-qualified positive and negative agreement at least 0.70,
  prevalence gap at most 0.10, and at least 12 pooled ratings for each class;
- 2,000-replicate prompt-cluster bootstrap Gwet AC1 2.5th percentile at least
  0.60.

For the detail-qualified boundary and each of the three components,
positive/negative agreement at least 0.70 is required only when the class has
at least 12 pooled primary ratings. An unsupported class is explicitly
reported and cannot support a class-specific claim, but does not alone stop
the route.

That unsupported-class exception does not apply to the derived
candidate-qualified construct. Fewer than 12 pooled candidate-qualified yes
ratings or fewer than 12 pooled no ratings is itself a primary-gate failure.

Failure selects `STOP_REVIEW_UNRELIABLE` and forbids Reviewer C and final
selection.

## 8. Reviewer C Gate

When and only when A/B calibration and reliability pass, Reviewer C receives
every and only primary candidate whose derived A/B `candidate_qualified`
labels disagree. C receives a new role-blind `step3q_review_*` map and cannot
see A/B labels or IDs.

Reviewer C cannot rescue an A/B both-no candidate and cannot reject an A/B
both-yes candidate. A malformed, incomplete, non-isolated, or over-broad C
review prevents finalization and corresponds to `STOP_ADJUDICATION_INVALID`.
An empty disagreement packet requires no C session and rejects unsolicited C
input.

## 9. Fixed 48-Target Flow

A candidate qualifies only when its final vector has detail 2 and all five
binary fields equal to yes. A deterministic maximum-flow selector then seeks
exactly 48 targets with:

- one candidate per prompt;
- globally unique responses under the frozen
  NFKC/casefold/whitespace-collapse normalization;
- all 19 frozen harm categories represented at least once;
- severity minima 5/28/5 for levels 1/2/3;
- no API-mode quota, preference, or label-dependent tie-break.

Category coverage uses each prompt's pre-API `assigned_category` from the
frozen category-severity slot table. Multi-label source categories are retained
for audit but cannot be opportunistically reassigned to make the final flow
pass. Response content, historical hidden labels, API mode, reviewer reasons,
and post-review interpretation cannot define or change coverage.

The protocol-bound hash order resolves feasible ties. Failure selects
`STOP_TARGET_YIELD`; thresholds are not relaxed and no new API call is made.
Success selects `PASS_CALIBRATED_API_POOL`, which is still only a qualified data
pool.

## 10. Immutable Outputs

`import-dispatch`, `analyze-ab`, and `finalize` each create a new immutable
bundle. Existing output roots cannot be overwritten. Every bundle carries an
exact artifact list with byte counts and SHA-256 hashes; added, missing, or
modified files invalidate later stages.

The prospective P2 implementation is anchored by tag
`step3q-calibrated-api-attempt-001-p2-protocol`. Because a commit cannot contain
its own SHA, freezing uses two commits: the tagged implementation commit first,
then one child lock commit that changes only the P2 registry from a pending
placeholder to the exact tagged commit SHA. Import must run from that clean lock
commit; the registry-locked SHA and local tag target must agree. Before
`analyze-ab` or `finalize`, the preceding stage bundle must be committed in a
clean descendant checkout while every critical P2 script still matches the
registry-locked implementation commit and the locked registry matches HEAD.

Self-consistent manifests are not accepted as execution proof. Both downstream
commands require the preserved external spool, rerun all 120 receipt checks,
reconstruct the complete import bundle, and compare it byte for byte. Finalize
also recomputes the complete A/B calibration/reliability outcome from its
committed review inputs and compares every byte before accepting
`PASS_AB_REVIEW`.

The imported repository package includes the three verified sealed artifacts,
enriched hidden primary metadata, A/B packets and hidden maps, templates, and
an import manifest. It records raw-receipt hashes but not raw receipt bodies.

On calibration failure, the absence of `primary/ab_reliability.json` is a
required protocol property, not an incomplete result.

## 11. Commands And Legal Next Boundaries

Before external results exist:

```powershell
python experiments/step3_response_quality_persistence/scripts/step3q_p2_review.py preflight
```

After the external dispatcher has produced a sealed PASS spool:

```powershell
python experiments/step3_response_quality_persistence/scripts/step3q_p2_review.py import-dispatch --spool-root <outside-repository-spool>
```

After two valid independent A/B review packages exist:

```powershell
python experiments/step3_response_quality_persistence/scripts/step3q_p2_review.py analyze-ab `
  --spool-root <outside-repository-spool> `
  --review-a <reviewer-a.jsonl> --provenance-a <reviewer-a-provenance.json> `
  --review-b <reviewer-b.jsonl> --provenance-b <reviewer-b-provenance.json>
```

After an opened non-empty C packet receives a valid independent review:

```powershell
python experiments/step3_response_quality_persistence/scripts/step3q_p2_review.py finalize `
  --spool-root <outside-repository-spool> `
  --review-c <reviewer-c.jsonl> --provenance-c <reviewer-c-provenance.json>
```

For an empty C packet, run `finalize` without C arguments.

## 12. Route Table

| Boundary | Frozen condition | Route |
| --- | --- | --- |
| import | P1, receipt denominator, receipt semantics, sealed byte reconstruction, model provenance, or API-yield failure | no P2 import; upstream `STOP_API_YIELD` remains controlling |
| A/B calibration | either reviewer fails any sentinel or 11/12 duplicate gate | `STOP_REVIEW_CALIBRATION` |
| A/B primary | any registered aggregate primary gate fails | `STOP_REVIEW_UNRELIABLE` |
| C | C packet, provenance, separation, or disagreement-only coverage is invalid | `STOP_ADJUDICATION_INVALID` |
| final | fixed 48-target flow is infeasible | `STOP_TARGET_YIELD` |
| final | all ordered gates pass | `PASS_CALIBRATED_API_POOL` |

Even the final pass does not authorize an optimizer step. A later prospective
training-suitability package must separately freeze leakage checks, tokenizer
and assistant-only loss masking, response-template and API-mode concentration
audits, independent suitability review, the method comparison, concurrent
baseline, smoke criteria, seeds, stopping rules, and safety/quality/utility
endpoints.
