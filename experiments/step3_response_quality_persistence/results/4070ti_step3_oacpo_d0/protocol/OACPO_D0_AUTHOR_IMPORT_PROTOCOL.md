# OACPO D0 Author Import Protocol

Status: prospective before the main session accesses any author output.

This stage checks only the mechanical gates frozen in the base D0 protocol: 56 rows, exact ID set and order, exact two-field schema, nonempty 40–120-word responses, no thinking tags, and 56 distinct normalized response strings. It canonicalizes whitespace at file boundaries but does not rewrite response content.

The importer does not compare responses with one another semantically, join source answers or roles, assign safety or quality labels, or authorize a gradient. Passing import authorizes only a separately frozen D1 blind-review protocol. The main session must not manually edit an answer to make it pass.

`optimizer_steps=0`, `training_authorized=false`, Core100 remains closed, and heldout remains closed.
