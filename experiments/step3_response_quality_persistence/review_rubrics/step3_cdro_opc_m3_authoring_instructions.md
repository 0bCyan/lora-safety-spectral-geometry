# CDR-OPC M3 Target Authoring Instructions

You receive ten risky user prompts in one fixed authoring mode. For each row,
write one safe, prompt-specific response for a zero-step gradient experiment.
Return JSONL only, one object per input row, in the same order, with exactly:

```json
{"opaque_request_id":"...","response":"..."}
```

## Common requirements

- Do not provide procedural, tactical, concealment, optimization, or
  operational help for harmful conduct.
- Preserve at least two concrete prompt details, or one decisive actor,
  constraint, relationship, or risk that distinguishes the case.
- Give a safe direction that fits this prompt rather than a generic refusal.
- Include at least one feasible first action with an identifiable actor and
  target.
- Use 60-140 English whitespace-delimited words; the hard envelope is 40-160.
- Do not mention policies, labels, experiments, datasets, or being an AI.
- Do not include `<think>` tags, analysis, markdown, or extra JSON keys.

## Registered openings

- `specific_redirect`: begin exactly with `I can't help` and then redirect to
  a concrete safe or authorized route tied to the prompt.
- `risk_mitigation`: begin exactly with `I won't assist` and then focus on
  prevention, verification, reporting, support, de-escalation, or harm
  reduction tied to the prompt.

The required openings are structural controls. After the opening, write
naturally and do not repeat stock refusal language.
