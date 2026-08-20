# Step3 CDR-OPC M5 Independent Authoring Cohort

## Purpose

M4 was a successful holdback audit but reused the M3 review pool. M5 creates a
prompt-disjoint authoring cohort before any new gradient or training work. Its
purpose is to test whether the conservative fixed-dose mechanism survives on
newly selected risk prompts.

## Frozen selection

All sample IDs present in the M3 primary-candidate file are excluded. The risk
source is then ordered by
`SHA256("cdro-opc-m5-authoring|" + sample_id)`, and the first 60 eligible
rows are selected. The selection uses prompt metadata only; no M3/M4 gradient
result is used.

Each prompt receives two author modes, `specific_redirect` and
`risk_mitigation`, for 120 fixed requests. Twelve no-context current-model
sub-agents receive ten requests each. They may only write their assigned shard
to the external spool. The main session does not label or rewrite candidate
responses.

## Authoring constraints

- plain text, no `think` tags;
- 40--160 whitespace tokens;
- exact registered opening for each mode;
- prompt-specific safe alternative or mitigation;
- no candidate replacement, top-up, retry, or manual editing;
- no API key, model-list request, or test request.

Passing authoring only authorizes blind packet construction. It does not
authorize reviewer labels, gradient calculation, training, Core100, or heldout
access.
