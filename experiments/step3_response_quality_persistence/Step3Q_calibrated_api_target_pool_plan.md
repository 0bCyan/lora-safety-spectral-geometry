# Step3Q Calibrated API Target-Pool Plan

Status: prospective P0 protocol before fresh prompt materialization, API
authoring, calibration-sentinel exposure, primary review, or optimizer steps.

Route ID: `step3q_calibrated_api_pool_attempt_001`.

Entry evidence: Step3P attempt 001 ended at commit
`04a8f72bd062c7660150c08572b4a626eea1b3cf` with route
`STOP_REVIEW_UNRELIABLE`, no Reviewer C, no hidden-stratum yield analysis,
`training_authorized=false`, and `optimizer_steps=0`.

## 1. Decision And Scope

Step3Q does not repair or rerun Step3P. Its 92 primary candidates, A/B labels,
and five failed gates remain frozen and are excluded from all Step3Q target
selection and confirmatory review.

Step3Q asks two narrower questions:

1. Can hidden, prospectively frozen calibration sentinels establish that each
   primary reviewer session can apply the unchanged qualification rubric to
   positive and negative component classes, without requiring a naturally
   high-quality candidate pool to contain artificial failures?
2. Can one controlled structured-API dispatcher produce enough fresh,
   independently screened prompt-specific safe responses to freeze a
   48-target pool under fixed category, severity, uniqueness, and no-top-up
   constraints?

This is a data and measurement qualification route. It is not a training
result, a behavioral gain, a mechanism intervention, human construct
validation, or evidence that PCSR, TRIM, SpectraSafe, or any successor method
works.

## 2. Why Step3P Cannot Simply Be Relaxed

Step3P's final qualification conjunction was reproducible: raw agreement was
0.891, Gwet AC1 was 0.792, positive agreement was 0.861, negative agreement
was 0.911, and both reviewers reproduced 12/12 hidden duplicates. The route
still stopped because prompt binding had zero negative ratings, the strict
detail boundary had only three negative ratings, and complete-vector agreement
was 68/92 against a 69/92 minimum.

Removing those gates after opening the labels would be outcome-driven. Step3Q
instead changes the design prospectively on completely fresh primary prompts.
Class-conditional reviewer competence is measured on hidden calibration
sentinels with deliberately supplied contrast. Natural primary candidates
retain aggregate component, complete-vector, final-conjunction, duplicate,
and prompt-cluster gates. A primary class-specific gate is applied only when
its registered minimum class support is present; otherwise its evidence comes
from the already required hidden sentinels and the primary result is reported
as one-class, not as a two-class reliability claim.

## 3. Immutable Boundaries

The following inputs are historical evidence only and are hash-pinned in the
registry:

- the 1,000-row `train_risk.jsonl` source;
- the O4 source/exclusion audit and 60-prompt hidden authoring key;
- all 92 Step3P primary candidates, representing 87 exposed prompts;
- the O3 120-row external actionability validation packet, hidden key, and
  final labels;
- the O3 180-row transport clean-safety packet, hidden key, and final labels;
- the unchanged Step3P qualification rubric.

No Step3P primary row may become a Step3Q calibration sentinel or target. No
O4/O4R prompt may be reused. The four delivered O4R shards remain excluded.
No old response label, review reason, category yield, or candidate role may be
used to alter a Step3Q primary result.

Core100 and heldout remain sealed. Source harmful responses are never released
to the authoring API. Only prompt text and an opaque request ID are sent.

## 4. Q0: Calibration Challenge

The local builder deterministically freezes a 96-row hidden-reference bank.
All visible rows contain only `opaque_candidate_id`, `prompt`, and `response`.

### 4.1 Component anchors

Forty-eight rows are selected from the completed O3 external validation. A
deficit-first deterministic selector guarantees at least 12 frozen yes and 12
frozen no reference labels for each of `safe_route`, `executable_step`, and
`prompt_binding`. The old final labels are reference-codebook evidence only;
they are not human ground truth.

### 4.2 Prompt-binding interventions

Twenty-four selected component anchors receive a response from a different
assigned BeaverTails category under a deterministic one-to-one derangement.
The prompt remains fixed. These rows are controlled mismatch interventions,
not naturally sampled bad answers. They provide prospective negative contrast
for `prompt_binding` and `detail_retention` and support a within-pair reversal
test.

### 4.3 Clean-safety anchors

Twenty-four rows are selected from the completed O3 transport clean-safety
screen: exactly 12 frozen yes and 12 frozen no labels.

### 4.4 Reviewer packets

Each A/B primary-review packet later receives exactly eight component anchors,
eight prompt-mismatch interventions, and eight clean-safety anchors from this
bank, plus 12 hidden duplicates of primary candidates. Every selected mismatch
is bundled with its own matched component anchor inside the same review packet;
otherwise the registered within-pair detail-reversal test would be undefined.
The eight paired anchors are selected under a reviewer-specific combinatorial
constraint requiring at least three frozen yes and three frozen no references
for each of `safe_route`, `executable_step`, and `prompt_binding`. The eight
clean-safety sentinels are exactly four frozen yes and four frozen no references.
Sentinel roles, historical IDs, construction roles, reference labels, and
duplicate links are hidden. Reviewer A and B receive independently selected and
ordered sentinel packets and distinct opaque duplicate IDs.

The sentinel gate is evaluated before any primary candidate label or hidden
category key is opened. Failure selects `STOP_REVIEW_CALIBRATION`; Reviewer C,
candidate adjudication, and target selection remain forbidden.

## 5. Q1: Fresh Prompt Freeze

The prompt source remains the hash-pinned 1,000-row PKU risk training pool.
Step3Q excludes the union of:

1. every O4 registered, incidental, and restricted lexical exclusion;
2. all 60 O4/O4R authoring prompts;
3. all 87 Step3P-exposed prompt sample IDs;
4. prompts outside the frozen 8-96 whitespace-token envelope.

The P0 implementation and registry are committed and pushed before the
`prepare` command may materialize prompts. The deterministic namespace binds
the route ID, Step3P terminal commit, P0 commit, source hash, and every frozen
exclusion artifact hash. P0 may run aggregate feasibility preflight but may
not print or write selected sample IDs or prompt text.

Exactly 60 unique prompts are matched to the unchanged O4 category-severity
slot table: all 19 categories receive at least three prompts, severity counts
are exactly 8/44/8, and multi-category rows may satisfy only one assigned slot.
No replacement or top-up is permitted after prompt materialization.

## 6. Q2: Controlled Structured-API Authoring

The sole authoring route is the OpenAI Responses API at
`https://api.openai.com/v1/responses`, using `gpt-5.6-sol`, medium reasoning,
strict JSON-schema output, `store=false`, no tools, no prior-response state,
disabled truncation, and a fixed maximum-output budget. The model choice was
frozen from the official OpenAI model guidance available on 2026-07-27.

Each prompt receives exactly two independent requests:

- `specific_redirect`: retain the decisive prompt details and give a concrete
  safe alternative or authorized path;
- `risk_mitigation`: focus on prevention, verification, reporting, support,
  or harm reduction with an identifiable first action.

The two requests share no model state and cannot see each other. Their role is
hidden from reviewers. A valid response must be 40-160 whitespace tokens,
match the strict response schema, contain no extra fields, and be globally
unique after NFKC/casefold/whitespace normalization.

The dispatcher uses a fixed request order and at most four concurrent calls.
It may retry only registered transport or rate-limit statuses, at most three
attempts total. A completed, refused, invalid, or otherwise semantically
returned response is never regenerated. A malformed HTTP-200 body is one
terminal semantic failure, not a retry opportunity. Every fixed request
receives one terminal receipt. Existing receipts are reused only after their
route, request ID, request hash, payload hash, attempt sequence, terminal
status, candidate fields, and response hashes are revalidated. Receipts are
immutable; missing requests cannot be silently skipped, and receipts outside
the fixed denominator invalidate the spool.

The returned effective model must be either the exact requested
`gpt-5.6-sol` alias or its dated snapshot form
`gpt-5.6-sol-YYYY-MM-DD`. All valid candidates in the fixed dispatch must use
one and only one effective model. Any model-family mismatch or mixed effective
model set fails API provenance and prevents an authoring pass.

Refusals and invalid responses remain denominator failures rather than
terminating the whole route. Authoring passes only with at least 96 valid
unique candidates over at least 54 of the fixed 60 prompts. Failure selects
`STOP_API_YIELD`; no prompts or calls are added.

The API key is read only from `OPENAI_API_KEY`, never written to disk, and is
not available in P0. P0 therefore authorizes implementation and dry-run tests
only, not an external call.

Before dispatch, `verify-prepared` deterministically reconstructs the complete
P1 package from the P0 commit and all frozen input hashes. Its file set and
every byte of the diagnostics, 96-row calibration bank, hidden references,
sentinel allocations, 60 prompts, 120 requests, and manifest must match. A
missing, added, or one-byte-modified artifact stops before network access.

## 7. Q3: Primary Review

The unchanged qualification fields are:

```text
detail_retention: 0 | 1 | 2
safe_route: yes | no
executable_step: yes | no
prompt_binding: yes | no
naturalness: yes | no
clean_safe: yes | no
```

A candidate qualifies only at detail 2 with all five binary fields equal to
yes. A/B reviewers are distinct, context-isolated sessions, blind to API mode,
source metadata, prompt category, severity, hidden sentinels, duplicates, and
the other reviewer.

### 7.1 Sentinel gate

For each reviewer independently:

- component-anchor reference accuracy is at least 0.85 for each of the three
  O3 components;
- within each component, yes-reference and no-reference accuracy are each at
  least two thirds, with at least three frozen references from each class;
- mismatch `prompt_binding=no` accuracy is at least 0.85;
- mismatch `detail_retention=0` accuracy is at least 0.80;
- matched-versus-mismatched detail reversal holds for at least 0.80 of pairs;
- clean-safety reference accuracy is at least 0.90;
- hidden-primary full-vector duplicate consistency is at least 11/12.

### 7.2 Primary gate

After the sentinel gate passes, primary candidates require:

- complete-vector exact agreement at least 0.75;
- derived candidate-qualified raw agreement at least 0.85 and Gwet AC1 at
  least 0.75;
- candidate-qualified positive and negative agreement at least 0.70, marginal
  prevalence gap at most 0.10, and at least 12 pooled ratings per class;
- prompt-cluster bootstrap 2.5th percentile at least 0.60;
- detail/component raw agreement at least 0.85, Gwet AC1/AC2 at least 0.75,
  and prevalence gap at most 0.10;
- naturalness and clean-safety raw agreement at least 0.90 and Gwet AC1 at
  least 0.80.

For detail and the three components, positive/negative agreement is required
at 0.70 only for a class with at least 12 pooled primary ratings. A class below
12 is reported as unsupported on the primary distribution and cannot support a
class-specific claim; it does not fail the route if every sentinel gate and
all aggregate primary gates pass.

If and only if calibration and primary reliability both pass, Reviewer C sees
every and only primary candidate whose A/B derived qualification labels
disagree. C cannot rescue A/B both-no candidates or reject A/B both-yes
candidates.

## 8. Q4: Fixed 48-Target Gate

After valid adjudication, a deterministic maximum-flow selector chooses
exactly 48 candidates with:

- one candidate per prompt;
- globally unique normalized responses;
- every one of 19 categories represented at least once;
- severity minima 5/28/5 for levels 1/2/3;
- no API-mode quota or post-label preference.

When two candidates for one prompt qualify, a protocol-bound hash rank chooses
the candidate. Reviewer labels determine eligibility but never the tie-break.
Failure selects `STOP_TARGET_YIELD`; thresholds are not relaxed and no calls
are added.

## 9. Training-Suitability Boundary

Even `PASS_CALIBRATED_API_POOL` produces only a machine-rubric-qualified pool.
Before any optimizer step, a separate prospective package must pass:

1. tokenizer and assistant-only loss-mask reconstruction;
2. exact training/evaluation/prompt-exposure leakage audit;
3. response-template concentration and API-mode concentration audit;
4. independent human or genuinely heterogeneous training-suitability review;
5. a frozen method, concurrent baseline, smoke test, seed schedule, stopping
   rules, safety endpoint, quality endpoint, and utility boundary.

No Step3Q stage selects or validates a new optimization method.

## 10. Route Table

| boundary | condition | route |
| --- | --- | --- |
| P0/P1 | source, hash, exclusion, schema, P0 identity, deterministic reconstruction, or slot feasibility failure | `STOP_SOURCE_PROTOCOL` |
| API | incomplete receipt set, invalid provenance, fewer than 96 valid unique candidates, or fewer than 54 represented prompts | `STOP_API_YIELD` |
| A/B sentinels | any frozen reviewer calibration or hidden-duplicate gate fails | `STOP_REVIEW_CALIBRATION` |
| A/B primary | malformed review, provenance failure, or any frozen aggregate primary gate fails | `STOP_REVIEW_UNRELIABLE` |
| C | disagreement packet or provenance invalid | `STOP_ADJUDICATION_INVALID` |
| final | 48-target flow or coverage constraints fail | `STOP_TARGET_YIELD` |
| final | all gates pass | `PASS_CALIBRATED_API_POOL` |

Every route records `training_authorized=false` and `optimizer_steps=0`.
