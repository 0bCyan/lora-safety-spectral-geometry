# OACPO D0 Safe-Response Authoring Rubric

You are an author, not a reviewer. For every row, write one response to the visible prompt.

The response must:

1. Refuse or clearly redirect the harmful request without supplying actionable harmful details.
2. Preserve the prompt's concrete context instead of using a generic policy statement.
3. Offer one to three practical, legal, safe next steps that are useful for the prompt's situation.
4. Be natural English, self-contained, and normally 40 to 120 words.
5. Avoid mentioning this rubric, model policy, hidden data, labels, or the authoring process.
6. Avoid `<think>` tags, headings such as "Safe alternative", canned disclaimers, and repeated boilerplate.

Return exactly one JSON object per input row, in the original order, with this schema:

```json
{"author_id":"unchanged opaque ID","authored_response":"completed response"}
```

Do not omit rows. Do not add fields. Do not inspect any repository file other than the assigned packet, this rubric, and the blank output template.
