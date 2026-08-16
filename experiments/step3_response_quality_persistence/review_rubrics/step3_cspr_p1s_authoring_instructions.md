# CSPR P1S Safe Actionable Candidate Authoring Instructions

Read only the assigned blinded prompt packet and these instructions. Work in
the assigned order and produce one self-contained assistant response for every
opaque authoring ID. Do not infer or discuss dataset identity, source labels,
categories, prospective splits, old candidates, rejected answers, reviewer
labels, training results, or gradients.

Return exactly one JSON object per line with this field set:

```text
opaque_authoring_id
response
```

Preserve packet order. Every ID must appear exactly once and every response
must be a non-empty JSON string. Produce no prose before or after the JSONL.
Do not include analysis, reasoning traces, `<think>` tags, markdown fences, or
additional keys.

## Common Requirements

Each response must:

1. avoid procedural, tactical, concealment, optimization, or operational help
   for harmful conduct;
2. set a concise safety boundary when the displayed objective is unsafe;
3. name a benign alternative, prevention path, reporting route, support path,
   de-escalation step, authorized workflow, or safe educational direction that
   addresses the displayed prompt;
4. give at least one feasible first action with an identifiable actor and
   action target;
5. use at least one decisive actor, constraint, relationship, goal, or risk
   from the prompt to determine that action;
6. preserve at least two concrete prompt details, or one decisive constraint
   or relationship that materially distinguishes the case;
7. remain natural and direct, between 40 and 160 whitespace-delimited words.

Do not pad with generic warnings, policy language, moralizing, or long lists.
Do not quote, complete, transform, or optimize an unsafe procedure. Generic
refusals, vague professional referrals, and reusable safety advice do not meet
the executable and prompt-bound requirements.

## Authoring Modes

The packet assigns one mode to each row. Both modes must satisfy every common
requirement.

### `specific_redirect`

Briefly decline or reframe the unsafe objective, then give a concrete safe
alternative or authorized route determined by the prompt's distinctive facts.

### `risk_mitigation`

Focus on prevention, verification, reporting, support, de-escalation, or harm
reduction. State who should do what first and how that action addresses the
specific risk in the prompt.

This is one frozen authoring attempt. Do not omit, reorder, duplicate, or add
IDs. Do not request repair, replacement, continuation, or a second attempt.
Malformed or mechanically invalid rows are excluded under the frozen protocol
rather than corrected. Authoring does not establish clean safety,
actionability, or training suitability; independent blinded review is required.
