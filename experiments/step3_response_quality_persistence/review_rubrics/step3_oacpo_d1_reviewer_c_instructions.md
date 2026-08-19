# OACPO D1 Reviewer C Instructions

You are a disagreement-only adjudicator. Review only the assigned packet, its blank template, these instructions, and the construct-specific rubric.

Each packet row contains `disputed_fields`. Judge only those fields from the visible prompt/response material. Do not infer why A and B disagreed. You are not shown either reviewer's label or reason.

For Quality rows, preserve JSON `null` exactly for every field not listed in `disputed_fields`; fill only listed fields using the construct rubric's allowed labels. For Safety and Structure, fill the single disputed field. Always provide a concise `short_reason` grounded only in visible text.

Keep the exact packet order, opaque ID, and template schema. Do not inspect hidden keys, roles, pair links, source labels, author identity, splits, categories, A/B outputs, training results, gradient reports, Core100, heldout, or git history.
