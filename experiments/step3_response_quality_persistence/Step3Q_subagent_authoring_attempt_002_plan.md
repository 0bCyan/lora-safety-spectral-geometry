# Step3Q Subagent Authoring Attempt 002 Plan

Status: prospective before any Attempt 002 authoring session is launched.

Route ID: `step3q_subagent_pool_attempt_002`.

Entry commit: `77c8a74b7b7e955fc0cac4e2b53e994a41b42602`.

## 1. Decision Boundary

Attempt 001 remains unchanged and stopped before API dispatch because its
required `OPENAI_API_KEY` was unavailable. Attempt 002 is a separate,
non-equivalent authoring route explicitly requested by the experiment owner.
It must not create, imitate, or import an Attempt 001 API receipt.

Attempt 002 reuses the byte-frozen Attempt 001 P1 prompt, request,
calibration, and rubric artifacts. It changes only the authoring transport and
provenance: four fresh Codex subagent sessions inherit the current parent
model and author fixed request batches. The available multi-agent interface
does not expose a provider snapshot or token-usage receipt, so the route must
record that limitation and cannot be reported as an API replication.

All outcomes retain:

```text
external_api_called=false
training_authorized=false
optimizer_steps=0
```

## 2. Prospective Freeze

Before prompt-bearing assignment files are created or a subagent is launched:

1. commit the plan, registry, implementation, and tests;
2. tag that commit `step3q-subagent-attempt-002-p0`;
3. lock the exact tagged SHA in one child commit changing only the registry;
4. push both commits and the protocol tag.

The lock checkout must be clean. The protocol tag must be an ancestor of HEAD,
and the only protocol-to-lock change may be the registry lock fields.

## 3. Frozen Inputs And Denominators

The following Attempt 001 P1 artifacts remain byte-identical:

- 60 hidden prompts;
- 120 requests, two author modes per prompt;
- 96 calibration rows and frozen sentinel allocations;
- the authoring instructions and target qualification rubric.

No prompt replacement, top-up, mode change, response regeneration after an
immutable submission, threshold revision, or candidate-text inspection by the
main executor is allowed.

## 4. Authoring Sessions

Four fresh, context-isolated subagents are launched without inherited thread
context. Two roles receive 30 `specific_redirect` requests each; two roles
receive 30 `risk_mitigation` requests each. Assignment is deterministic by
opaque request ID. A worker sees only its assignment and the frozen authoring
instructions, not another worker's assignment or output.

Each worker returns exactly one stripped 40-160 whitespace-token safe response
for every assigned opaque request ID. Worker output is submitted once to an
immutable repository-external spool. The worker's launch ID, role, assignment
hash, output hash, row count, and inherited-model provenance are recorded.

The main executor may run structural validation, hashing, deduplication, and
packet construction. It may not print, open, sample, summarize, or label the
prompt or response text.

## 5. Authoring Gate

The fixed denominator is 120 authoring records from four distinct launched
subagent IDs. Responses are globally deduplicated under the unchanged
NFKC/casefold/whitespace normalization. The route is
`PASS_SUBAGENT_AUTHORING` only with at least 96 unique valid candidates over at
least 54 prompts and no assignment, session, schema, or provenance failure.

Failure selects `STOP_SUBAGENT_YIELD`. No failed or missing request may be
retried, reassigned, topped up, or silently omitted.

## 6. P2 Packet Import

Only a sealed `PASS_SUBAGENT_AUTHORING` spool may be imported. The importer
reconstructs every assignment, launch record, worker submission, candidate
hash, deduplication decision, summary, and sealed file before writing a new
immutable repository bundle at:

```text
experiments/step3_response_quality_persistence/results/step3q_subagent_pool_attempt_002_p2_review
```

The unchanged P1 calibration bank and Step3Q reviewer packet construction are
reused. Reviewer-facing rows still contain only opaque ID, prompt, and
response. API mode, subagent role, launch ID, source metadata, hidden maps,
calibration roles, and duplicate links remain private.

This package freezes packets only. It does not authorize or launch Reviewer
A/B/C, calculate labels, select 48 targets, or train a model.

## 7. Stop Boundary

After the packet bundle is committed, tagged, and pushed, stop. Do not open
candidate text, start blind review, read private maps, label candidates,
select targets, run GPU work, or execute an optimizer step.
