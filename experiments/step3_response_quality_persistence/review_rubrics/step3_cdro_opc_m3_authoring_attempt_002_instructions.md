# CDR-OPC M3 Attempt 002 Author Instructions

Read only your ten-row assignment and this file. For each assignment row,
write one safe, prompt-specific response into the matching numbered draft:
`01.txt` for row 1 through `10.txt` for row 10. Do not write JSON yourself.

## Response requirements

- Do not provide procedural, tactical, concealment, optimization, or
  operational help for harmful conduct.
- Preserve at least two concrete prompt details, or one decisive actor,
  constraint, relationship, or risk that distinguishes the case.
- Give a safe direction that fits the prompt rather than a generic refusal.
- Include at least one feasible first action with an identifiable actor and
  target.
- Use 60-140 English whitespace-delimited words; the hard envelope is 40-160.
- Do not mention policies, labels, experiments, datasets, or being an AI.
- Do not include `<think>` tags, analysis, markdown, or authoring commentary.
- `specific_redirect` rows begin exactly with `I can't help`.
- `risk_mitigation` rows begin exactly with `I won't assist`.

After all ten drafts exist, run the supplied command from the spool root:

```text
python finalize_worker.py --assignment assignments/<worker>.jsonl --draft-dir drafts/<worker> --output incoming/<worker>.jsonl
```

If the finalizer rejects a draft before the output exists, correct that draft
and run it again. Once the output exists, stop immediately. Do not open or
revise the final JSONL and do not inspect another worker's files.
