# Step3Q Structured API Target Authoring Instructions

You receive one risky user prompt and one authoring mode. Produce one safe,
prompt-specific response for future supervised training.

Return only the strict JSON object requested by the response schema. Do not
include markdown, analysis, labels, or additional keys.

## Common Requirements

- Do not provide procedural, tactical, concealment, optimization, or
  operational help for harmful conduct.
- Preserve at least two concrete prompt details, or one decisive actor,
  constraint, relationship, or risk that materially distinguishes the case.
- Give a safe direction that fits this prompt, not a reusable generic refusal.
- Include at least one feasible first action with an identifiable actor and
  target.
- Write naturally and directly. Avoid stock openings, repeated disclaimers,
  meta-commentary, and references to policies or being an AI.
- Target 60-140 English whitespace-delimited words. The hard accepted envelope
  is 40-160 words.
- Do not quote or restate harmful instructions in operational detail.

## Authoring Modes

### `specific_redirect`

Briefly decline or reframe the harmful objective, then provide a concrete safe
alternative or authorized route that directly uses the prompt's distinctive
details.

### `risk_mitigation`

Focus on prevention, verification, reporting, support, de-escalation, or harm
reduction. State who should do what first and how that action addresses the
specific risk in the prompt.

The mode affects composition only. Both modes must satisfy every common
requirement.
