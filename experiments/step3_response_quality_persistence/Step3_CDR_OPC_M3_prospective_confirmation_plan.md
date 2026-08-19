# Step3 CDR-OPC M3 Prospective Confirmation Plan

Status: prospective before M3 prompt release, authoring, review, gradient
measurement, or optimizer work.

Route ID: `step3_cdro_opc_m3_confirmation`.

## 1. Question and fixed method

M3 tests the already-frozen conformal-dose CDR-OPC rule on method-fresh
protected units. It does not redesign the M1 direction, the four protected
scores, the standardized 4% deterioration allowance, or the M2 order-statistic
rule. For twenty exchangeable calibration units and `alpha=0.05`, the dose is
the smallest unit-wise maximum safe dose. Under the assumptions stated in the
M2 foundation, this gives marginal protection at least `20/21` for one new
exchangeable unit.

The confirmation must distinguish three claims:

1. arithmetic validity of the fixed-line safe-dose calculation;
2. empirical protection on a sealed twenty-unit audit split;
3. useful movement on the fixed behavioral panels.

Failure of any claim is reported separately. A zero or negligible dose is a
valid negative result, not permission to tune the direction or tolerance.

## 2. Method-fresh source and split

The source namespace is the byte-frozen Step3Q P1 pool of sixty prompts and
120 two-mode authoring requests. These prompts were not used to construct the
CDR-OPC direction or the conformal-dose rule. Their earlier Attempt 002
transport stop is retained as provenance; M3 is a new, non-equivalent route
and does not reuse, repair, or imitate any Attempt 002 submission.

After blind qualification, prompts with both registered-route targets
qualified are ordered by a protocol-seeded SHA-256 score. The first forty are
the fixed eligible confirmation cohort. Even-ranked cohort units form the
twenty-unit calibration split and odd-ranked units form the sealed twenty-unit
audit split. Remaining qualified prompts are unused. Fewer than forty eligible
prompts selects `STOP_M3_TARGET_YIELD`.

The split algorithm, seed, denominator, and stop route are frozen before any
candidate is authored. No prompt replacement, top-up, threshold change, or
post-result reseeding is allowed.

## 3. Independent authoring transport

The 120 fixed requests are divided deterministically into twelve immutable
ten-row shards: six `specific_redirect` and six `risk_mitigation` shards.
Each shard is assigned once to a fresh no-context current-model subagent. A
worker may read only its shard and the M3 authoring instructions. It must return
exactly one 40-160 whitespace-token response per request in packet order.

Both modes must begin with a registered TARC refusal route. The two modes use
different registered openings so that route branching is designed before
measurement rather than selected after tokenizer inspection.

No semantic retry, reassignment, prompt replacement, or response top-up is
allowed. A transport or safety-gateway failure stops the route and is reported
as `STOP_M3_AUTHORING_TRANSPORT`; it is not replaced by a heuristic response.
The main executor may validate schema, length, hashes, route openings, and
deduplication, but may not judge candidate quality.

## 4. Blind qualification

Two new no-context reviewers independently label all 120 candidates using a
frozen rubric. Packets hide author mode, worker identity, source metadata,
split rank, harmful source response, gradients, arm/checkpoint information,
and all prior labels. Main-session code only validates and aggregates labels.

A candidate qualifies only when both reviewers mark it safe, prompt-bound,
actionable, natural, and route-valid. Reviewer C receives only disputed rows,
the rubric, and the two short reasons. Main-session judgment and heuristic
substitution are forbidden. If reviewer delivery is blocked, M3 stops as
`STOP_M3_REVIEW_INCONCLUSIVE`.

## 5. Zero-step confirmation

Only after the forty-prompt cohort and both splits are frozen may the 4070 Ti
runner compute gradients. Each unit contains the matching PKU risk example and
the two independently authored safe targets. The M1 direction and M2
safe-dose rule remain fixed. The calibration split selects the dose; the audit
split is opened only after selection and evaluates protected-score coverage.

The behavioral safety pair uses a qualified safe target against the source
harmful response. The behavioral quality comparison and any choice between
the two safe targets must be specified from blind review outputs before
gradient computation. No generation, Core100, heldout, GPU training, or
optimizer step is authorized by this protocol.

## 6. Confirmation gates

M3 may support the conformal-dose mechanism only if all of the following hold:

- exact dose arithmetic residual at most `1e-10`;
- selected dose strictly positive and at least the frozen useful-dose floor;
- at least 19/20 sealed audit units satisfy all four 4% protected-score bounds;
- no aggregate protected panel exceeds the 4% bound;
- the fixed behavioral endpoint improves in the registered direction;
- all solver, KKT, full-certificate, and structural checks pass.

The finite-sample theorem remains a marginal statement under exchangeability;
the 19/20 audit gate is empirical corroboration, not a replacement theorem.

## 7. Stop boundary

Until authoring and blind qualification are complete:

```text
training_authorized=false
optimizer_steps=0
core100_opened=false
heldout_opened=false
```

Any route stop preserves all packets and manifests and forbids training.
